import unittest
from unittest.mock import patch, MagicMock

from ui.startup_dialog import StartupDialog, prompt_startup_choice


class StartupDialogTests(unittest.TestCase):
    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    def test_dialog_buttons_set_expected_choice(self, mock_h, mock_w):
        with patch("customtkinter.CTk.destroy"):
            from test_calibration import validated_profile
            dialog = StartupDialog(profile=validated_profile(), camera_list=[])
            self.assertEqual(dialog.choice, "exit")

            dialog._choose_track()
            self.assertEqual(dialog.choice, "track")

            dialog._choose_calibrate()
            self.assertEqual(dialog.choice, "calibrate")

            dialog._choose_exit()
            self.assertEqual(dialog.choice, "exit")

    @patch("ui.startup_dialog.StartupDialog")
    def test_prompt_startup_choice_runs_mainloop_and_returns_choice(self, mock_dialog_cls):
        mock_instance = MagicMock()
        mock_instance.choice = "track"
        mock_dialog_cls.return_value = mock_instance

        choice = prompt_startup_choice(profile={"dummy": 1})
        self.assertEqual(choice, "track")
        mock_instance.mainloop.assert_called_once()

    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    def test_camera_selection_and_profile_default(self, mock_h, mock_w):
        custom_cameras = [
            {"index": 0, "name": "Webcam 0", "width": 640, "height": 480, "label": "Camera 0: Webcam 0 (640x480)"},
            {"index": 1, "name": "USB Cam 1", "width": 1920, "height": 1080, "label": "Camera 1: USB Cam 1 (1920x1080)"},
        ]
        with patch("customtkinter.CTk.destroy"):
            profile = {"monitor": {"width": 1920, "height": 1080}, "camera": {"index": 1}}
            dialog = StartupDialog(profile=profile, camera_list=custom_cameras)
            self.assertEqual(dialog.camera_index, 1)

            dialog._on_camera_selected("Camera 0: Webcam 0 (640x480)")
            self.assertEqual(dialog.camera_index, 0)

    @patch("ui.startup_dialog.list_available_cameras")
    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    def test_rescan_cameras_updates_dropdown(self, mock_h, mock_w, mock_list):
        with patch("customtkinter.CTk.destroy"):
            initial_cameras = [{"index": 0, "name": "Cam 0", "width": 640, "height": 480, "label": "Camera 0: Cam 0"}]
            dialog = StartupDialog(profile={}, camera_list=initial_cameras)
            self.assertEqual(len(dialog.cameras), 1)

            rescanned = [
                {"index": 0, "name": "Cam 0", "width": 640, "height": 480, "label": "Camera 0: Cam 0"},
                {"index": 1, "name": "Cam 1", "width": 1280, "height": 720, "label": "Camera 1: Cam 1"},
            ]
            mock_list.return_value = rescanned
            dialog._rescan_cameras()
            self.assertEqual(len(dialog.cameras), 2)
            self.assertEqual(dialog.camera_dropdown.cget("values"), ["Camera 0: Cam 0", "Camera 1: Cam 1"])

    @patch("ui.startup_dialog.StartupDialog")
    def test_prompt_startup_choice_returns_camera_when_requested(self, mock_dialog_cls):
        mock_instance = MagicMock()
        mock_instance.choice = "track"
        mock_instance.camera_index = 2
        mock_dialog_cls.return_value = mock_instance

        choice, cam_idx = prompt_startup_choice(profile={"dummy": 1}, return_camera=True)
        self.assertEqual(choice, "track")
        self.assertEqual(cam_idx, 2)

    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    def test_dialog_without_profile_blocks_track_and_shows_warning(self, mock_h, mock_w):
        with patch("customtkinter.CTk.destroy"):
            dialog = StartupDialog(profile=None)
            self.assertEqual(dialog.choice, "exit")

            dialog._choose_track()
            self.assertEqual(dialog.choice, "exit")
            self.assertIn("Calibration file is not full yet", dialog.status_label.cget("text"))

            dialog._choose_calibrate()
            self.assertEqual(dialog.choice, "calibrate")


class CalibrationWindowTests(unittest.TestCase):
    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    @patch("ui.calibration_window._physical_screen_size", return_value=(1920, 1080))
    def test_calibration_window_auto_start_withdraws_and_starts(self, mock_phys, mock_h, mock_w):
        from ui.calibration_window import CalibrationWindow

        with patch.object(CalibrationWindow, "withdraw") as mock_withdraw, \
             patch.object(CalibrationWindow, "start_calibration") as mock_start, \
             patch.object(CalibrationWindow, "close"):
            window = CalibrationWindow(auto_start=True)
            mock_withdraw.assert_called()
            self.assertIsNotNone(window._start_id)
            self.assertIsNone(window._maximize_id)
            window.cancel_calibration()
            self.assertTrue(window.cancelled)
            window.close()
