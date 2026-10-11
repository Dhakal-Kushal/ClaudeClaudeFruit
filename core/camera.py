import subprocess
import sys

import cv2
import numpy as np


def get_friendly_camera_names():
    """Retrieve friendly hardware names for camera devices on Windows."""
    if not sys.platform.startswith("win"):
        return []
    try:
        ps_cmd = (
            'Get-CimInstance Win32_PnPEntity | '
            'Where-Object { $_.PNPClass -eq "Camera" -or $_.PNPClass -eq "Image" } | '
            'ForEach-Object { $_.Name }'
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def list_available_cameras(max_to_test=6):
    """Scan and return a list of available webcams with index, name, and resolution.

    Returns:
        List of dicts: [{'index': int, 'name': str, 'width': int, 'height': int, 'label': str}, ...]
    """
    try:
        old_level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        old_level = None

    friendly_names = get_friendly_camera_names()
    cameras = []

    try:
        for index in range(max_to_test):
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            success, test_frame = cap.read()
            if success and test_frame is not None and (w <= 0 or h <= 0):
                h, w = test_frame.shape[:2]
            cap.release()

            if index < len(friendly_names):
                name = friendly_names[index]
            else:
                name = f"Webcam {index}"

            res_str = f" ({w}x{h})" if w > 0 and h > 0 else ""
            label = f"Camera {index}: {name}{res_str}"
            cameras.append({
                "index": index,
                "name": name,
                "width": w,
                "height": h,
                "label": label,
            })
    finally:
        if old_level is not None:
            try:
                cv2.utils.logging.setLogLevel(old_level)
            except Exception:
                pass

    return cameras


class Camera:
    @classmethod
    def list_available_cameras(cls, max_to_test=6):
        return list_available_cameras(max_to_test)

    def __init__(self, index=0, width=None, height=None, flip=True, *, backend=None, fps=None, fourcc=None):

        backend = (cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY) if backend is None else backend
        self.index = index
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(
                f"Could not open camera {index}. Check that it is plugged in, "
                "not being used by another app, and try a different index."
            )

        if fourcc is not None:
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

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
        if frame is None:
            return None
        self.height, self.width = frame.shape[:2]
        self.camera_matrix = np.array(
            [[self.width, 0, self.width / 2], [0, self.width, self.height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        return cv2.flip(frame, 1) if self.flip else frame

    def get_metadata(self):
        """Negotiated settings; dimensions reflect the last successful frame."""
        def optional_property(prop):
            value = float(self._cap.get(prop))
            return value if np.isfinite(value) and value > 0 else None

        return {
            "index": self.index, "flip": self.flip,
            "width": self.width, "height": self.height,
            "backend": int(self._cap.get(cv2.CAP_PROP_BACKEND)),
            "fps": optional_property(cv2.CAP_PROP_FPS),
            "fourcc": optional_property(cv2.CAP_PROP_FOURCC),
        }

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

    def __exit__(self, exc_type, exc_value, traceback):
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

