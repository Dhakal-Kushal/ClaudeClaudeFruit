from queue import Empty
from time import monotonic
import math
import cv2
import mediapipe as mp
import numpy as np
from core import mouse_controller
from core.camera import Camera
from core.calibration import CalibrationSample

HEAD_INDICES = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, 330.0, 65.0), (-225.0, -170.0, 135.0),
    (225.0, -170.0, 135.0), (-150.0, 150.0, 125.0), (150.0, 150.0, 125.0),
], dtype=np.float64)
MODEL_POINTS_MIRRORED = MODEL_POINTS.copy()
MODEL_POINTS_MIRRORED[:, 0] *= -1
AXIS_POINTS = np.array([(100, 0, 0), (0, 100, 0), (0, 0, -100)], dtype=np.float64)


def rotation_matrix_to_angles(rotation):
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def angles_to_rotation_matrix(angles):
    """Inverse of the Z-Y-X Euler convention used by the pose estimator."""
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("Initial head pose needs three finite angles")
    pitch, yaw, roll = np.deg2rad(angles)
    cx, cy, cz = np.cos([pitch, yaw, roll])
    sx, sy, sz = np.sin([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def estimate_head_pose(landmarks, width, height, camera_matrix=None, dist_coeffs=None,
                       prev_rvec=None, prev_tvec=None, mirrored=False, initial_angles=None):
    if camera_matrix is None:
        camera_matrix = np.array(
            [[width, 0, width / 2.0],
             [0, width, height / 2.0],
             [0, 0, 1.0]],
            dtype=np.float64,
        )
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    model_points = MODEL_POINTS_MIRRORED if mirrored else MODEL_POINTS
    image_points = np.array(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in HEAD_INDICES],
        dtype=np.float64,
    )
    if prev_rvec is None and initial_angles is not None:
        rotation = angles_to_rotation_matrix(initial_angles)
        rotated = model_points @ rotation.T
        normalized = cv2.undistortPoints(image_points.reshape(-1, 1, 2),
                                         camera_matrix, dist_coeffs).reshape(-1, 2)
        # For fixed rotation, perspective equations are linear in translation.
        # Estimate translation from this frame rather than reusing old distance.
        equations = np.zeros((len(model_points) * 2, 3))
        equations[0::2, 0] = equations[1::2, 1] = 1
        equations[0::2, 2] = -normalized[:, 0]
        equations[1::2, 2] = -normalized[:, 1]
        rhs = (normalized * rotated[:, 2:3] - rotated[:, :2]).reshape(-1)
        translation, _, rank, _ = np.linalg.lstsq(equations, rhs, rcond=None)
        if rank == 3 and np.isfinite(translation).all():
            prev_rvec = cv2.Rodrigues(rotation)[0]
            prev_tvec = translation.reshape(3, 1)
    if prev_rvec is None:
        ok, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_angles(rotation)
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
    for indices in ((33, 160, 158, 133, 153, 144), (263, 387, 385, 362, 380, 373)):   # doing the math for each indice => left and right eye
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

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (263, 387, 385, 362, 380, 373)

def eye_ratio(landmarks, indices, width, height):   
    """Lid gap divided by eye width. Near 0 = upper lid has reached the lower lid (closed)."""
    points = np.array(
        [(landmarks[index].x * width, landmarks[index].y * height) for index in indices]  # Continously grab the x and y coordinates, assign it to a numpy array
    )
    horizontal = np.linalg.norm(points[0] - points[3]) 
    if horizontal <= 1e-6:
        return None
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    return vertical / (2 * horizontal)

def check_blink(landmarks, width, height, threshold=0.18):
    """
    Returns True if both eyes are closed (blinking).
    Explicit rule: Blinking must be detected to be ignored for clicks and freeze cursor.
    """
    left = eye_ratio(landmarks, LEFT_EYE, width, height)
    right = eye_ratio(landmarks, RIGHT_EYE, width, height)
    if left is None or right is None:
        return False
    if (left < threshold) and (right < threshold):
        return True

    return False


def check_winks(landmarks, width, height, threshold=0.18, diff_threshold=0.04):
    """
    Returns (left_is_winking, right_is_winking).
    Explicit rule: Blinking (both eyes closed) is ignored and not counted as a wink.
    Winking requires one eye closed, the other open, and a clear contrast between eyes.
    """
    left = eye_ratio(landmarks, LEFT_EYE, width, height)
    right = eye_ratio(landmarks, RIGHT_EYE, width, height)
    
    if left is None or right is None:   # no eye data exist for the right or left; return false
        return False, False             # can not determine an eye wink, without one of the eyes

    # Explicit rule: Both eyes closed is a blink, not a wink!
    if left < threshold and right < threshold:
        return False, False
        
    left_wink = (left < threshold) and (right >= threshold) and ((right - left) >= diff_threshold)
    right_wink = (right < threshold) and (left >= threshold) and ((left - right) >= diff_threshold)
    
    return left_wink, right_wink

EYE_LIDS = {
    468: ((160, 158), (153, 144)),
    473: ((387, 385), (380, 373)),
}


def eyelid_vertical_compensation(eye_y: float, d_upper: float, d_lower: float, gain: float = 0.15) -> float:
    """Compensate downward iris center bias caused by upper eyelid occlusion.

    When looking down, the upper eyelid descends and partially covers the upper
    iris rim, biasing MediaPipe's visible-iris centroid downward. If the ratio
    of upper-lid-to-pupil distance over lower-lid-to-pupil distance exceeds a
    downward gaze threshold, apply a smooth non-linear correction factor.
    """
    if eye_y <= 0 or d_upper <= 0 or d_lower <= 0:
        return 0.0
    opening = d_upper + d_lower
    if opening <= 1e-6:
        return 0.0
    asymmetry = (d_upper - d_lower) / opening
    if asymmetry <= 0.2:
        return 0.0
    correction = gain * eye_y * ((asymmetry - 0.2) / 0.8) ** 2
    return float(correction)


def normalized_iris_position(landmarks, width, height, compensate_eyelids: bool = False) -> tuple[float, float]:
    """Return binocular iris displacement in eye widths, along eye-local axes.

    Positive x points toward image right; positive y points below the eye.
    Corner midpoints and widths remove image translation and uniform scaling.
    The local axes compensate for roll while the eyes remain roughly upright.
    When compensate_eyelids is True, downward eyelid occlusion bias is corrected.
    """
    if len(landmarks) < 478 or not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("A complete face mesh and valid frame dimensions are required")

    positions = []
    # Keep each iris paired with its own corners, including on mirrored frames.
    for iris_index, corner_a, corner_b in ((468, 33, 133), (473, 362, 263)):
        points = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height)
             for index in (iris_index, corner_a, corner_b)],
            dtype=float,
        )
        if not np.isfinite(points).all() or (points < 0).any() or (points >= [width, height]).any():
            raise ValueError("Eye landmarks must be finite and inside the frame")

        pupil, start, end = points
        axis = end - start
        eye_width = np.linalg.norm(axis)
        if eye_width <= 1e-6:
            raise ValueError("Eye corners do not define a usable eye width")
        horizontal = axis / eye_width
        if horizontal[0] < 0:
            horizontal = -horizontal
        vertical = np.array([-horizontal[1], horizontal[0]])
        displacement = (pupil - (start + end) / 2) / eye_width
        disp_x = float(displacement @ horizontal)
        disp_y = float(displacement @ vertical)
        if compensate_eyelids and iris_index in EYE_LIDS:
            upper_indices, lower_indices = EYE_LIDS[iris_index]
            upper_pts = np.array(
                [(landmarks[i].x * width, landmarks[i].y * height) for i in upper_indices]
            )
            lower_pts = np.array(
                [(landmarks[i].x * width, landmarks[i].y * height) for i in lower_indices]
            )
            if np.isfinite(upper_pts).all() and np.isfinite(lower_pts).all():
                upper_mid = upper_pts.mean(axis=0)
                lower_mid = lower_pts.mean(axis=0)
                d_upper = float(((pupil - upper_mid) / eye_width) @ vertical)
                d_lower = float(((lower_mid - pupil) / eye_width) @ vertical)
                disp_y -= eyelid_vertical_compensation(disp_y, d_upper, d_lower)
        # Normalize both components by width; eyelid opening varies with blinks.
        positions.append((disp_x, disp_y))

    result = np.mean(positions, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Eye landmarks produced invalid gaze coordinates")
    return float(result[0]), float(result[1])


class VisionProcessor:
    """Publish eye-relative gaze and solvePnP head angles for each camera frame."""

    def __init__(self, face_mesh=None, mirrored=True, initial_head=None, compensate_eyelids: bool = False, min_wink_frames: int = 1):
        self.initial_angles = None
        if initial_head is not None:
            self.initial_angles = [initial_head[key] for key in ("pitch", "yaw", "roll")]
            angles_to_rotation_matrix(self.initial_angles)
        self.face_mesh = face_mesh if face_mesh is not None else _create_face_mesh()
        self.mirrored = mirrored
        self.compensate_eyelids = compensate_eyelids
        self.prev_rvec = None
        self.prev_tvec = None
        self.left_winking = False
        self.right_winking = False
        self.last_pupil_midpoint = None
        self.is_blinking = False
        self.min_wink_frames = min_wink_frames
        self._left_wink_counter = 0
        self._right_wink_counter = 0

    def process(self, frame, camera_matrix=None, dist_coeffs=None, timestamp=None):
        timestamp = monotonic() if timestamp is None else timestamp
        self.left_winking = False
        self.right_winking = False
        self.last_pupil_midpoint = None
        self.is_blinking = False
        
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)
        
        invalid = CalibrationSample(
            0, 0, 0, 0, 0, eyes_open=False, tracking_valid=False, timestamp=timestamp,
            left_winking=False, right_winking=False, is_blinking=False,
        )
        if not result.multi_face_landmarks:
            self.prev_rvec = self.prev_tvec = None
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            return invalid
            
        landmarks = result.multi_face_landmarks[0].landmark
        if len(landmarks) < 478:
            self.prev_rvec = self.prev_tvec = None
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            return invalid
            
        try:
            eye_x, eye_y = normalized_iris_position(landmarks, width, height, compensate_eyelids=self.compensate_eyelids)
        except ValueError:
            self.prev_rvec = self.prev_tvec = None
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            return invalid

        open_state = eyes_are_open(landmarks, width, height)
        pose = estimate_head_pose(
            landmarks, width, height, camera_matrix, dist_coeffs,
            self.prev_rvec, self.prev_tvec, mirrored=self.mirrored,
            initial_angles=self.initial_angles,
        )
        if pose is None or not np.isfinite(pose[:3]).all():
            self.prev_rvec = self.prev_tvec = None
            return invalid

        pitch, yaw, roll, self.prev_rvec, self.prev_tvec = pose
        pupils = np.array(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in (468, 473)]
        )
        self.last_pupil_midpoint = tuple(float(value) for value in pupils.mean(axis=0))
        
        # Check blink and winks
        self.is_blinking = check_blink(landmarks, width, height)
        raw_left_wink, raw_right_wink = check_winks(landmarks, width, height)

        if self.is_blinking:
            # Explicit rule: Blink detected! Reset wink counters, ignore winks.
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            self.left_winking = False
            self.right_winking = False
        else:
            if raw_left_wink:
                self._left_wink_counter += 1
                self._right_wink_counter = 0
            elif raw_right_wink:
                self._right_wink_counter += 1
                self._left_wink_counter = 0
            else:
                self._left_wink_counter = 0
                self._right_wink_counter = 0

            self.left_winking = self._left_wink_counter >= self.min_wink_frames
            self.right_winking = self._right_wink_counter >= self.min_wink_frames
        
        return CalibrationSample(
            float(eye_x), float(eye_y), float(pitch), float(yaw), float(roll),
            eyes_open=open_state,
            tracking_valid=True,
            timestamp=timestamp,
            left_winking=self.left_winking,
            right_winking=self.right_winking,
            is_blinking=self.is_blinking,
        )

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc=None, traceback=None):
        self.close()

def process_frames(frames, adapter, stop_event, errors, processor_factory=None, freeze_duration=0.5):
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
                    left_winking=getattr(sample, "left_winking", False),
                    right_winking=getattr(sample, "right_winking", False),
                    is_blinking=getattr(sample, "is_blinking", False),
                )
    except Exception as error:
        errors.put(f"Vision processing failed: {error}")
        stop_event.set()

def run_preview(freeze_duration=0.5):
    processor = VisionProcessor()
    my_mouse = mouse_controller.MouseController(freeze_duration=freeze_duration)
    
    try:
        cam = Camera()
        print("Real Time Capturing (press ESC to quit)")

        camera_matrix = cam.camera_matrix
        dist_coeffs = cam.dist_coeffs

        for frame in cam.frames():
            sample = processor.process(frame, camera_matrix, dist_coeffs)
            if sample.tracking_valid:
                pupil_point = tuple(int(value) for value in processor.last_pupil_midpoint)
                cv2.circle(frame, pupil_point, 4, (0, 255, 0), -1)
                pitch, yaw, roll = sample.pitch, sample.yaw, sample.roll

                if processor.prev_rvec is not None and processor.prev_tvec is not None:
                    axes_2d, _ = cv2.projectPoints(AXIS_POINTS, processor.prev_rvec, processor.prev_tvec, camera_matrix, dist_coeffs)
                    origin_pt = pupil_point
                    for pt, color in zip(axes_2d.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.line(frame, origin_pt, (int(pt[0]), int(pt[1])), color, 2)

                cv2.putText(frame, f"Pitch: {pitch:6.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw:   {yaw:6.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll:  {roll:6.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if sample.tracking_valid:
                my_mouse.winkClick(
                    processor.left_winking,
                    processor.right_winking,
                    eyes_open=sample.eyes_open,
                    timestamp=sample.timestamp,
                    is_blinking=processor.is_blinking,
                )
            else:
                my_mouse._pause(cancel_gestures=True)

            cv2.imshow("Real-Time Tracking", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
    finally:
        processor.close()


if __name__ == "__main__":
    run_preview()
