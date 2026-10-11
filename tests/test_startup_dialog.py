import unittest
from unittest.mock import patch, MagicMock

from ui.startup_dialog import StartupDialog, prompt_startup_choice


class StartupDialogTests(unittest.TestCase):
    @patch("customtkinter.CTk.winfo_screenwidth", return_value=1920)
    @patch("customtkinter.CTk.winfo_screenheight", return_value=1080)
    def test_dialog_buttons_set_expected_choice(self, mock_h, mock_w):
        with patch("customtkinter.CTk.destroy"):
            dialog = StartupDialog(profile={"monitor": {"width": 1920, "height": 1080}})
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
