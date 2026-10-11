from queue import Empty
from time import monotonic
import cv2
import mediapipe as mp
import numpy as np
from core import mouse_controller
from core.camera import Camera
from core.calibration import CalibrationSample

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

def normalized_iris_position(landmarks, width, height):
    eye_regions = (
        ((33, 133), (160, 158), (153, 144)),
        ((263, 362), (387, 385), (380, 373)),
    )
    pupils = sorted(
        ((landmarks[index].x * width, landmarks[index].y * height) for index in (468, 473)),
        key=lambda point: point[0],
    )
    positions = []
    for pupil, (corner_indices, upper_indices, lower_indices) in zip(pupils, eye_regions):
        corners = sorted(
            ((landmarks[index].x * width, landmarks[index].y * height) for index in corner_indices),
            key=lambda point: point[0],
        )
        upper = np.mean(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in upper_indices],
            axis=0,
        )
        lower = np.mean(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in lower_indices],
            axis=0,
        )
        eye_width = corners[1][0] - corners[0][0]
        eye_height = lower[1] - upper[1]
        if eye_width <= 1e-6 or eye_height <= 1e-6:
            raise ValueError("Eye landmarks do not define a usable eye region")
        positions.append(
            (
                (pupil[0] - corners[0][0]) / eye_width,
                (pupil[1] - (upper[1] + lower[1]) / 2) / eye_height,
            )
        )
    result = np.mean(positions, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Eye landmarks produced invalid gaze coordinates")
    return float(result[0]), float(result[1])


class VisionProcessor:
    def __init__(self, face_mesh=None, mirrored=True, min_wink_frames=2):
        self.face_mesh = face_mesh if face_mesh is not None else _create_face_mesh()
        self.mirrored = mirrored
        self.prev_rvec = None
        self.prev_tvec = None
        self.left_winking = False
        self.right_winking = False
        self.is_blinking = False
        self.min_wink_frames = min_wink_frames
        self._left_wink_counter = 0
        self._right_wink_counter = 0

    def process(self, frame, camera_matrix=None, dist_coeffs=None, timestamp=None):
        timestamp = monotonic() if timestamp is None else timestamp
        self.left_winking = False
        self.right_winking = False
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
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            return invalid

        open_state = eyes_are_open(landmarks, width, height)
        eye_x, eye_y = pupils.mean(axis=0)
        
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
            float(eye_x), float(eye_y), 0.0, 0.0, 0.0,
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
                cv2.circle(frame, (int(sample.eye_x), int(sample.eye_y)), 4, (0, 255, 0), -1)

            my_mouse.winkClick(
                processor.left_winking,
                processor.right_winking,
                is_blinking=processor.is_blinking,
            )

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