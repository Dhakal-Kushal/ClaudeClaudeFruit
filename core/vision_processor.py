from queue import Empty
from time import monotonic
import cv2
import mediapipe as mp
import numpy as np

# Correct import for running from main.py
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

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (263, 387, 385, 362, 380, 373)

def eye_ratio(landmarks, indices, width, height):
    """Lid gap divided by eye width. Near 0 = upper lid has reached the lower lid (closed)."""
    points = np.array(
        [(landmarks[index].x * width, landmarks[index].y * height) for index in indices]
    )
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal <= 1e-6:
        return None
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    return vertical / (2 * horizontal)

def check_winks(landmarks, width, height, threshold=0.18):
    """Returns (left_is_winking, right_is_winking)"""
    left = eye_ratio(landmarks, LEFT_EYE, width, height)
    right = eye_ratio(landmarks, RIGHT_EYE, width, height)
    
    if left is None or right is None:
        return False, False
        
    left_wink = left < threshold and right >= threshold
    right_wink = right < threshold and left >= threshold
    
    return left_wink, right_wink

class VisionProcessor:
    def __init__(self, face_mesh=None):
        self.face_mesh = face_mesh if face_mesh is not None else _create_face_mesh()
        self.left_winking = False  
        self.right_winking = False 

    def process(self, frame, camera_matrix, dist_coeffs, timestamp=None):
        timestamp = monotonic() if timestamp is None else timestamp
        self.left_winking = False
        self.right_winking = False
        
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.face_mesh.process(rgb)
        
        invalid = CalibrationSample(
            0, 0, 0, 0, 0, eyes_open=False, tracking_valid=False, timestamp=timestamp
        )
        if not result.multi_face_landmarks:
            return invalid
            
        landmarks = result.multi_face_landmarks[0].landmark
        if len(landmarks) < 478:
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
            return invalid
            
        eye_x, eye_y = pupils.mean(axis=0)
        
        # Check both eyes
        self.left_winking, self.right_winking = check_winks(landmarks, width, height)
        
        return CalibrationSample(
            float(eye_x), float(eye_y), 0.0, 0.0, 0.0,
            eyes_open=eyes_are_open(landmarks, width, height),
            timestamp=timestamp,
        )

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def process_frames(frames, adapter, stop_event, errors, processor_factory=None):
    factory = processor_factory if processor_factory is not None else VisionProcessor
    
    # 1. Build the mouse object here for the main app pipeline
    my_mouse = mouse_controller.MouseController()
    
    try:
        with factory() as processor:
            while not stop_event.is_set():
                try:
                    timestamp, frame, camera_matrix, dist_coeffs = frames.get(timeout=0.1)
                except Empty:
                    continue
                sample = processor.process(frame, camera_matrix, dist_coeffs, timestamp)
                
                # 2. Fire the clicker in the main app
                my_mouse.winkClick(processor.left_winking, processor.right_winking)
                
                adapter.publish(
                    sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll,
                    eyes_open=sample.eyes_open, tracking_valid=sample.tracking_valid,
                    timestamp=sample.timestamp,
                )
    except Exception as error:
        errors.put(f"Vision processing failed: {error}")
        stop_event.set()

def run_preview():
    processor = VisionProcessor()
    my_mouse = mouse_controller.MouseController()
    
    try:
        cam = Camera()
        print("Real Time Capturing (press ESC to quit)")

        camera_matrix = cam.camera_matrix
        dist_coeffs = cam.dist_coeffs

        for frame in cam.frames():
            sample = processor.process(frame, camera_matrix, dist_coeffs)
            if sample.tracking_valid:
                cv2.circle(frame, (int(sample.eye_x), int(sample.eye_y)), 4, (0, 255, 0), -1)
            
            my_mouse.winkClick(processor.left_winking, processor.right_winking)

            cv2.imshow("Real-Time Eye Tracking", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
    finally:
        processor.close()

if __name__ == "__main__":
    run_preview()