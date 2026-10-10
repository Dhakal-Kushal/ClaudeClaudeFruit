import math
from queue import Empty
from time import monotonic
import cv2
import mediapipe as mp
import numpy as np
import json

from core.camera import Camera
from core.calibration import CalibrationSample

try:
    from core.mouse_controller import MouseController
except ImportError:
    try:
        from mouse_controller import MouseController
    except ImportError:
        MouseController = None


HEAD_INDICES = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array([
  (   0.0,    0.0,    0.0),   
  (   0.0,  330.0,   65.0),   
  (-225.0, -170.0,  135.0),   
  ( 225.0, -170.0,  135.0),   
  (-150.0,  150.0,  125.0),   
  ( 150.0,  150.0,  125.0),   
], dtype=np.float64)

AXIS_POINTS = np.array([
  (100, 0, 0),
  (0, 100, 0),
  (0, 0, -100),
], dtype=np.float64)


def rotation_matrix_to_angles(R):
  sy = math.hypot(R[0, 0], R[1, 0])
  if sy > 1e-6:
      pitch = math.atan2(R[2, 1], R[2, 2])
      yaw = math.atan2(-R[2, 0], sy)
      roll = math.atan2(R[1, 0], R[0, 0])
  else:
      pitch = math.atan2(-R[1, 2], R[1, 1])
      yaw = math.atan2(-R[2, 0], sy)
      roll = 0.0
  return np.degrees([pitch, yaw, roll])


def estimate_head_pose(landmarks, w, h, camera_matrix, dist_coeffs, prev_rvec=None, prev_tvec=None):
  image_points = np.array(
      [[landmarks[i].x * w, landmarks[i].y * h] for i in HEAD_INDICES],
      dtype=np.float64,
  )

  if prev_rvec is None:
      ok, rvec, tvec = cv2.solvePnP(
          MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
          flags=cv2.SOLVEPNP_ITERATIVE,
      )
  else:
      ok, rvec, tvec = cv2.solvePnP(
          MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
          rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
      )
  if not ok:
      return None

  rmat, _ = cv2.Rodrigues(rvec) 
  pitch, yaw, roll = rotation_matrix_to_angles(rmat)
  return pitch, yaw, roll, rvec, tvec


def _create_face_mesh():
    if not hasattr(mp, "solutions"):
        raise RuntimeError("This tracker needs the FaceMesh runtime. Install requirements.txt.")
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def eyes_are_open(landmarks, width, height, threshold=0.18):
    for indices in ((33, 160, 158, 133, 153, 144), (263, 387, 385, 362, 380, 373)):
        points = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in indices]
        )
        horizontal = np.linalg.norm(points[0] - points[3])
        if horizontal <= 1e-6:
            return False
        vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
        if vertical / (2 * horizontal) < threshold:
            return False
    return True


class VisionProcessor:
    def __init__(self, face_mesh=None):
        self.face_mesh = face_mesh if face_mesh is not None else _create_face_mesh()
        self.prev_rvec = None
        self.prev_tvec = None

    def process(self, frame, camera_matrix, dist_coeffs, timestamp=None):
        timestamp = monotonic() if timestamp is None else timestamp
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)
        invalid = CalibrationSample(
            0, 0, 0, 0, 0, eyes_open=False, tracking_valid=False, timestamp=timestamp
        )
        if not result.multi_face_landmarks:
            self.prev_rvec = self.prev_tvec = None
            return invalid
        landmarks = result.multi_face_landmarks[0].landmark
        if len(landmarks) < 478:
            self.prev_rvec = self.prev_tvec = None
            return invalid
        pupils = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in (468, 473)]
        )
        if (
            not np.isfinite(pupils).all()
            or (pupils < 0).any()
            or (pupils[:, 0] >= width).any()
            or (pupils[:, 1] >= height).any()
        ):
            self.prev_rvec = self.prev_tvec = None
            return invalid
        pose = estimate_head_pose(
            landmarks, width, height, camera_matrix, dist_coeffs,
            self.prev_rvec, self.prev_tvec,
        )
        if pose is None or not np.isfinite(pose[:3]).all():
            self.prev_rvec = self.prev_tvec = None
            return invalid
        pitch, yaw, roll, self.prev_rvec, self.prev_tvec = pose
        eye_x, eye_y = pupils.mean(axis=0)
        return CalibrationSample(
            float(eye_x), float(eye_y), float(pitch), float(yaw), float(roll),
            eyes_open=eyes_are_open(landmarks, width, height),
            timestamp=timestamp,
        )

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def process_frames(frames, adapter, stop_event, errors, processor_factory=None):
    factory = processor_factory if processor_factory is not None else VisionProcessor
    try:
        with factory() as processor:
            while not stop_event.is_set():
                try:
                    timestamp, frame, camera_matrix, dist_coeffs = frames.get(timeout=0.1)
                except Empty:
                    continue
                sample = processor.process(frame, camera_matrix, dist_coeffs, timestamp)
                adapter.publish(
                    sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll,
                    eyes_open=sample.eyes_open, tracking_valid=sample.tracking_valid,
                    timestamp=sample.timestamp,
                )
    except Exception as error:
        errors.put(f"Vision processing failed: {error}")
        stop_event.set()


def run_preview():
    face_mesh = _create_face_mesh()
    try:
        cam = Camera()
        print("Real Time Capturing (press ESC to quit, C to calibrate baseline)")
        mouse_handler = MouseController(tilt_threshold=15.0, freeze_duration=0.5) if MouseController else None

        session_landmarks = []
        session_angles = []

        BASELINE_FILE = "calibration_profile.json"
        BASELINE_KEY = "head_profile"
        CALIB_FRAMES = 120 
        baseline = None
        calibrating = False
        calib_buffer = []

        try:
          with open(BASELINE_FILE) as f:
              saved = json.load(f)[BASELINE_KEY]
          baseline = np.array([saved["pitch"], saved["yaw"], saved["roll"]], dtype=float)
        except (OSError, ValueError, TypeError, KeyError):
          baseline = None

        if baseline is not None:
          print(f"Loaded baseline: pitch={baseline[0]:.1f} yaw={baseline[1]:.1f} roll={baseline[2]:.1f}")
        else:
          print("No baseline yet. Look straight at the camera and press C.")

        camera_matrix = cam.camera_matrix
        dist_coeffs = cam.dist_coeffs
        prev_rvec, prev_tvec = None, None

        for frame in cam.frames():
          h, w, _ = frame.shape

          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_mesh.process(rgb_frame)

          if results.multi_face_landmarks:
              landmarks = results.multi_face_landmarks[0].landmark

              pose = estimate_head_pose(landmarks, w, h, camera_matrix, dist_coeffs, prev_rvec, prev_tvec)
              if pose is not None:
                  pitch, yaw, roll, rvec, tvec = pose
                  prev_rvec, prev_tvec = rvec, tvec

                  session_angles.append((pitch, yaw, roll))
                  
                  if calibrating:
                      calib_buffer.append((pitch, yaw, roll))
                      if len(calib_buffer) % 10 == 0:
                          print(f"Calibrating... {len(calib_buffer)}/{CALIB_FRAMES}")
                      if len(calib_buffer) >= CALIB_FRAMES:
                          baseline = np.median(calib_buffer, axis=0)
                          try:
                              with open(BASELINE_FILE) as f:
                                  data = json.load(f)
                              if not isinstance(data, dict):
                                  data = {}
                          except (OSError, ValueError):
                              data = {}
                          data[BASELINE_KEY] = {"pitch": float(baseline[0]), "yaw": float(baseline[1]), "roll": float(baseline[2])}
                          with open(BASELINE_FILE, "w") as f:
                              json.dump(data, f, indent=2)
                          calibrating = False
                          print(f"Baseline saved: pitch={baseline[0]:.1f} yaw={baseline[1]:.1f} roll={baseline[2]:.1f}")

                  if baseline is not None and not calibrating:
                      d_pitch, d_yaw, d_roll = (np.array([pitch, yaw, roll]) - baseline)
                      if mouse_handler:
                          mouse_handler.update(d_pitch, d_yaw, d_roll)

                  for i in HEAD_INDICES:
                      cv2.circle(frame, (int(landmarks[i].x * w), int(landmarks[i].y * h)), 3, (0, 0, 255), -1)

                  axes_2d, _ = cv2.projectPoints(AXIS_POINTS, rvec, tvec, camera_matrix, dist_coeffs)
                  nose = (int(landmarks[1].x * w), int(landmarks[1].y * h))
                  for pt, color in zip(axes_2d.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                      cv2.line(frame, nose, (int(pt[0]), int(pt[1])), color, 2)

                  cv2.putText(frame, f"Pitch: {pitch:6.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                  cv2.putText(frame, f"Yaw:   {yaw:6.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                  cv2.putText(frame, f"Roll:  {roll:6.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          else:
              prev_rvec, prev_tvec = None, None

          cv2.imshow("Real-Time Head Pose", frame)
          key = cv2.waitKey(5) & 0xFF
          if key == 27:
              break
          elif key == ord("c"):
              calibrating = True
              calib_buffer = []
              print("Calibrating... hold still and look straight at the camera.")

        cam.release()
        cv2.destroyAllWindows()

        if session_angles:
          angles = np.array(session_angles)
          np.save("live_head_angles.npy", angles)
          print(f"Saved {angles.shape[0]} frames to 'live_head_angles.npy'")
        else:
          print("No angles recorded.")
    finally:
        face_mesh.close()


if __name__ == "__main__":
    run_preview()
