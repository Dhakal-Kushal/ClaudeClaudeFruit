import sys

import cv2
import numpy as np


class Camera:
    def __init__(self, index=0, width=None, height=None, flip=True):

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. Check that it is plugged in, "
                "not being used by another app, and try a different index."
            )

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.flip = flip
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            success, test_frame = self._cap.read()
            if success and test_frame is not None:
                self.height, self.width = test_frame.shape[:2]

        # Approximate intrinsics for solvePnP (focal length ~ image width, center
        # principal point). Ignore these if you don't need them.
        self.camera_matrix = np.array(
            [[self.width, 0, self.width / 2],
             [0, self.width, self.height / 2],
             [0, 0, 1]], dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1))   # assume no lens distortion

    def read(self):
        """Return the next frame, or None if the camera gave nothing."""
        success, frame = self._cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1) if self.flip else frame

    def frames(self):
        """Yield frames until the camera stops giving them."""
        while self._cap.isOpened():
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with Camera(index) as cam:
        print(f"Camera {index} opened at {cam.width}x{cam.height}. Press ESC to quit.")
        for frame in cam.frames():
            cv2.imshow("Camera test", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

