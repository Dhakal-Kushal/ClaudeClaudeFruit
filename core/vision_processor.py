import math
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from camera import Camera   # NEW: shared camera file


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
  max_num_faces=1,
  refine_landmarks=True,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5,
)


# nose tip, chin, left eye outer, right eye outer, left mouth corner, right mouth corner
# ("left"/"right" = left/right side of the IMAGE, which is how MediaPipe orders them)
HEAD_INDICES = [1, 152, 33, 263, 61, 291]


# Generic 3D face model (arbitrary units, only the proportions matter).
# Axes match OpenCV's camera frame: +x = right, +y = down, +z = away from the camera.
# So a face looking straight at the camera is the identity rotation (0, 0, 0).
MODEL_POINTS = np.array([
  (   0.0,    0.0,    0.0),   # nose tip
  (   0.0,  330.0,   65.0),   # chin
  (-225.0, -170.0,  135.0),   # left eye outer corner
  ( 225.0, -170.0,  135.0),   # right eye outer corner
  (-150.0,  150.0,  125.0),   # left mouth corner
  ( 150.0,  150.0,  125.0),   # right mouth corner
], dtype=np.float64)


# Axes drawn from the nose tip: x (red), y (green), forward out of the face (blue)
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
  else:  # gimbal lock
      pitch = math.atan2(-R[1, 2], R[1, 1])
      yaw = math.atan2(-R[2, 0], sy)
      roll = 0.0
  return np.degrees([pitch, yaw, roll])


def estimate_head_pose(landmarks, w, h, camera_matrix, dist_coeffs, prev_rvec=None, prev_tvec=None):
  """Return (pitch, yaw, roll, rvec, tvec) or None if solvePnP fails."""
  # solvePnP needs 2D PIXEL coordinates, so scale the normalized landmarks.
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
      # Start from last frame's pose -> smoother and more stable
      ok, rvec, tvec = cv2.solvePnP(
          MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
          rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
      )
  if not ok:
      return None

  rmat, _ = cv2.Rodrigues(rvec)           # 3x1 rotation vector -> 3x3 matrix
  pitch, yaw, roll = rotation_matrix_to_angles(rmat)
  return pitch, yaw, roll, rvec, tvec


cam = Camera()   # NEW: replaces cv2.VideoCapture(0). Use Camera(width=1920, height=1080) for a bigger size.
print("Real Time Capturing (press ESC to quit, C to calibrate baseline)")


session_landmarks = []   # (6, 3) normalized landmarks per frame
session_angles = []      # (pitch, yaw, roll) per frame


BASELINE_FILE = "calibration_profile.json"
BASELINE_KEY = "head_profile"    # this script only reads/writes this element of the JSON
CALIB_FRAMES = 120        # frames averaged into the baseline (~4 seconds)
baseline = None          # [pitch, yaw, roll] in degrees
calibrating = False
calib_buffer = []

# Load the baseline only if the "head" element exists and holds pitch, yaw and roll
try:
  with open(BASELINE_FILE) as f:
      saved = json.load(f)[BASELINE_KEY]
  baseline = np.array([saved["pitch"], saved["yaw"], saved["roll"]], dtype=float)
except (OSError, ValueError, TypeError, KeyError):   # no file, bad JSON, missing "head", wrong contents
  baseline = None

if baseline is not None:
  print(f"Loaded baseline: pitch={baseline[0]:.1f} yaw={baseline[1]:.1f} roll={baseline[2]:.1f}")
else:
  print("No baseline yet. Look straight at the camera and press C.")

camera_matrix = cam.camera_matrix   # NEW: comes from the camera file
dist_coeffs = cam.dist_coeffs       # NEW: comes from the camera file
prev_rvec, prev_tvec = None, None


# NEW: the camera file hands out already-flipped frames, so the cap.read / flip lines are gone
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
          print(f"Pitch: {pitch:6.1f} | Yaw: {yaw:6.1f} | Roll: {roll:6.1f}")

          if calibrating:
              calib_buffer.append((pitch, yaw, roll))
              if len(calib_buffer) % 10 == 0:
                  print(f"Calibrating... {len(calib_buffer)}/{CALIB_FRAMES}")
              if len(calib_buffer) >= CALIB_FRAMES:
                  baseline = np.median(calib_buffer, axis=0)
                  # Read what's already in the file so other elements (like a friend's "eye") are kept
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
              print(f"dPitch: {d_pitch:6.1f} | dYaw: {d_yaw:6.1f} | dRoll: {d_roll:6.1f}")

          # Draw the 6 landmarks
          for i in HEAD_INDICES:
              cv2.circle(frame, (int(landmarks[i].x * w), int(landmarks[i].y * h)), 3, (0, 0, 255), -1)

          # Draw 3D axes from the nose tip
          axes_2d, _ = cv2.projectPoints(AXIS_POINTS, rvec, tvec, camera_matrix, dist_coeffs)
          nose = (int(landmarks[1].x * w), int(landmarks[1].y * h))
          for pt, color in zip(axes_2d.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
              cv2.line(frame, nose, (int(pt[0]), int(pt[1])), color, 2)

          cv2.putText(frame, f"Pitch: {pitch:6.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          cv2.putText(frame, f"Yaw:   {yaw:6.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          cv2.putText(frame, f"Roll:  {roll:6.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  else:
      prev_rvec, prev_tvec = None, None   # face lost: reset the initial guess

  cv2.imshow("Real-Time Head Pose", frame)
  key = cv2.waitKey(5) & 0xFF
  if key == 27:
      break
  elif key == ord("c"):
      calibrating = True
      calib_buffer = []
      print("Calibrating... hold still and look straight at the camera.")


cam.release()   # NEW: replaces cap.release()
cv2.destroyAllWindows()


if session_angles:
  angles = np.array(session_angles)        # shape (frames, 3)
  np.save("live_head_angles.npy", angles)
  angles = np.load("live_head_angles.npy")
  pitch, yaw, roll = angles[:, 0], angles[:, 1], angles[:, 2]   # degrees
  print(f"Saved {angles.shape[0]} frames to 'live_head_angles.npy'")
else:
  print("No angles recorded.")