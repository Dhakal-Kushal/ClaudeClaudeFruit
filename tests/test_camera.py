import subprocess
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.camera import Camera, get_friendly_camera_names, list_available_cameras


class CameraTests(unittest.TestCase):
    @patch("core.camera.cv2.VideoCapture")
    def test_actual_frame_dimensions_and_unknown_optional_settings_are_reported(self, capture):
        cap = capture.return_value
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {3: 640, 4: 480, 42: 700}.get(prop, 0)
        cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        with Camera(index=2, backend=700, fps=30) as camera:
            camera.read()
            metadata = camera.get_metadata()
            self.assertEqual((metadata["width"], metadata["height"]), (1280, 720))
            self.assertEqual(metadata["backend"], 700)
            self.assertIsNone(metadata["fps"])
            self.assertIsNone(metadata["fourcc"])
            self.assertEqual(camera.camera_matrix[0, 0], 1280)
        cap.release.assert_called_once()
        capture.assert_called_once_with(2, 700)

    @patch("core.camera.cv2.VideoCapture")
    def test_camera_init_and_read(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        cam = Camera(index=1, width=1280, height=720, flip=True)
        self.assertEqual(cam.width, 640)
        self.assertEqual(cam.height, 480)
        mock_cap.set.assert_any_call(3, 1280)
        mock_cap.set.assert_any_call(4, 720)

        frame = cam.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

        cam.release()
        mock_cap.release.assert_called_once()

    @patch("core.camera.cv2.VideoCapture")
    def test_camera_init_failure_raises_runtime_error(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        with self.assertRaises(RuntimeError):
            Camera(index=5)

    @patch("core.camera.subprocess.run")
    def test_get_friendly_camera_names(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="Integrated Webcam\r\nOBS Virtual Camera\r\n"
        )
        names = get_friendly_camera_names()
        self.assertEqual(names, ["Integrated Webcam", "OBS Virtual Camera"])

    @patch("core.camera.subprocess.run", side_effect=Exception("PowerShell unavailable"))
    def test_get_friendly_camera_names_handles_exception(self, mock_run):
        names = get_friendly_camera_names()
        self.assertEqual(names, [])

    @patch("core.camera.get_friendly_camera_names", return_value=["Webcam HD", "External Cam"])
    @patch("core.camera.cv2.VideoCapture")
    def test_list_available_cameras(self, mock_cap_cls, mock_names):
        def make_cap(index, backend=None):
            cap = MagicMock()
            if index in (0, 1):
                cap.isOpened.return_value = True
                cap.get.side_effect = lambda prop: 640.0 if prop == 3 else 480.0
                cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                cap.isOpened.return_value = False
            return cap

        mock_cap_cls.side_effect = make_cap

        cameras = list_available_cameras(max_to_test=3)
        self.assertEqual(len(cameras), 2)
        self.assertEqual(cameras[0]["index"], 0)
        self.assertEqual(cameras[0]["name"], "Webcam HD")
        self.assertEqual(cameras[0]["label"], "Camera 0: Webcam HD (640x480)")
        self.assertEqual(cameras[1]["index"], 1)
        self.assertEqual(cameras[1]["name"], "External Cam")
        self.assertEqual(cameras[1]["label"], "Camera 1: External Cam (640x480)")

    @patch("core.camera.list_available_cameras")
    def test_camera_class_static_method(self, mock_list):
        mock_list.return_value = [{"index": 0}]
        res = Camera.list_available_cameras(max_to_test=2)
        self.assertEqual(res, [{"index": 0}])
        mock_list.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
