import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np

from core.mouse_controller import MouseController
from core.vision_processor import check_winks, eye_ratio, LEFT_EYE, RIGHT_EYE


def create_mock_eye_landmarks(left_open=True, right_open=True):
    """
    Helper to generate mock landmark points for left and right eyes.
    Indices:
      LEFT_EYE: (33, 160, 158, 133, 153, 144)
      RIGHT_EYE: (263, 387, 385, 362, 380, 373)
    """
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]

    # Left eye points (horizontal span: 0.2 to 0.4 -> width = 0.2)
    # Open: height = 0.1 -> ratio = (0.05 + 0.05) / (2 * 0.2) = 0.25 (> 0.18)
    # Closed: height = 0.0 -> ratio = 0.0 (< 0.18)
    left_dy = 0.05 if left_open else 0.0
    left_points = {
        33: (0.2, 0.5),
        160: (0.25, 0.5 - left_dy),
        158: (0.35, 0.5 - left_dy),
        133: (0.4, 0.5),
        153: (0.35, 0.5 + left_dy),
        144: (0.25, 0.5 + left_dy),
    }
    for idx, (x, y) in left_points.items():
        landmarks[idx] = SimpleNamespace(x=x, y=y)

    # Right eye points (horizontal span: 0.6 to 0.8 -> width = 0.2)
    right_dy = 0.05 if right_open else 0.0
    right_points = {
        263: (0.6, 0.5),
        387: (0.65, 0.5 - right_dy),
        385: (0.75, 0.5 - right_dy),
        362: (0.8, 0.5),
        380: (0.75, 0.5 + right_dy),
        373: (0.65, 0.5 + right_dy),
    }
    for idx, (x, y) in right_points.items():
        landmarks[idx] = SimpleNamespace(x=x, y=y)

    return landmarks


class TestWinkDetection(unittest.TestCase):
    def setUp(self):
        self.width = 640
        self.height = 480

    def test_both_eyes_open(self):
        landmarks = create_mock_eye_landmarks(left_open=True, right_open=True)
        left_wink, right_wink = check_winks(landmarks, self.width, self.height)
        self.assertFalse(left_wink, "Left eye should not be winking when both eyes are open")
        self.assertFalse(right_wink, "Right eye should not be winking when both eyes are open")

    def test_left_wink_detected(self):
        landmarks = create_mock_eye_landmarks(left_open=False, right_open=True)
        left_wink, right_wink = check_winks(landmarks, self.width, self.height)
        self.assertTrue(left_wink, "Left wink should be detected when left eye is closed and right eye is open")
        self.assertFalse(right_wink, "Right wink should not be detected when right eye is open")

    def test_right_wink_detected(self):
        landmarks = create_mock_eye_landmarks(left_open=True, right_open=False)
        left_wink, right_wink = check_winks(landmarks, self.width, self.height)
        self.assertFalse(left_wink, "Left wink should not be detected when left eye is open")
        self.assertTrue(right_wink, "Right wink should be detected when right eye is closed and left eye is open")

    def test_blink_both_eyes_closed_not_detected_as_wink(self):
        landmarks = create_mock_eye_landmarks(left_open=False, right_open=False)
        left_wink, right_wink = check_winks(landmarks, self.width, self.height)
        self.assertFalse(left_wink, "Blink (both eyes closed) should not trigger left wink")
        self.assertFalse(right_wink, "Blink (both eyes closed) should not trigger right wink")

    def test_eye_ratio_zero_width_handles_gracefully(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        ratio = eye_ratio(landmarks, LEFT_EYE, self.width, self.height)
        self.assertIsNone(ratio)
        left_wink, right_wink = check_winks(landmarks, self.width, self.height)
        self.assertFalse(left_wink)
        self.assertFalse(right_wink)


class TestMouseControllerWinkClick(unittest.TestCase):
    def setUp(self):
        self.controller = MouseController()

    @patch.object(MouseController, "_trigger_click")
    def test_left_wink_triggers_left_click(self, mock_trigger):
        self.controller.winkClick(left_winking=True, right_winking=False)
        mock_trigger.assert_called_once_with("left")
        self.assertTrue(self.controller.wink_debounce_flag)

    @patch.object(MouseController, "_trigger_click")
    def test_right_wink_triggers_double_left_click(self, mock_trigger):
        self.controller.winkClick(left_winking=False, right_winking=True)
        mock_trigger.assert_called_once_with("double_left")
        self.assertTrue(self.controller.wink_debounce_flag)

    @patch.object(MouseController, "_trigger_click")
    def test_debounce_prevents_multiple_clicks_while_holding_wink(self, mock_trigger):
        # Frame 1: Initial left wink -> click fires
        self.controller.winkClick(left_winking=True, right_winking=False)
        self.assertEqual(mock_trigger.call_count, 1)

        # Frame 2-5: User still winking in subsequent camera frames -> no new clicks
        for _ in range(4):
            self.controller.winkClick(left_winking=True, right_winking=False)
        self.assertEqual(mock_trigger.call_count, 1, "Click should not repeat while wink is held")

    @patch.object(MouseController, "_trigger_click")
    def test_release_both_eyes_resets_debounce(self, mock_trigger):
        # 1. Left wink fires
        self.controller.winkClick(left_winking=True, right_winking=False)
        self.assertEqual(mock_trigger.call_count, 1)

        # 2. Both eyes open (release wink)
        self.controller.winkClick(left_winking=False, right_winking=False)
        self.assertFalse(self.controller.wink_debounce_flag, "Debounce flag should reset when eyes reopen")

        # 3. Left wink fires again
        self.controller.winkClick(left_winking=True, right_winking=False)
        self.assertEqual(mock_trigger.call_count, 2)
        mock_trigger.assert_called_with("left")

    @patch.object(MouseController, "_trigger_click")
    def test_alternating_winks_with_release(self, mock_trigger):
        # Left wink
        self.controller.winkClick(left_winking=True, right_winking=False)
        self.assertEqual(mock_trigger.call_count, 1)
        self.assertEqual(mock_trigger.call_args[0][0], "left")

        # Open eyes
        self.controller.winkClick(left_winking=False, right_winking=False)

        # Right wink (double left click)
        self.controller.winkClick(left_winking=False, right_winking=True)
        self.assertEqual(mock_trigger.call_count, 2)
        self.assertEqual(mock_trigger.call_args[0][0], "double_left")

    @patch.object(MouseController, "_trigger_click")
    def test_neither_winking_does_not_fire_click(self, mock_trigger):
        self.controller.winkClick(left_winking=False, right_winking=False)
        mock_trigger.assert_not_called()
        self.assertFalse(self.controller.wink_debounce_flag)

    @patch("core.mouse_controller.ctypes.windll.user32.mouse_event")
    @patch("core.mouse_controller.mouse.press", side_effect=Exception("Simulated mouse library failure"))
    @patch("core.mouse_controller.mouse.double_click", side_effect=Exception("Simulated mouse library failure"))
    def test_windows_events_double_left_click_fallback(self, mock_double_click, mock_press, mock_mouse_event):
        # Trigger double_left click in fallback mode
        self.controller._trigger_click("double_left")

        # Must execute: LEFTDOWN (0x0002), LEFTUP (0x0004), LEFTDOWN (0x0002), LEFTUP (0x0004)
        expected_calls = [
            unittest.mock.call(0x0002, 0, 0, 0, 0),
            unittest.mock.call(0x0004, 0, 0, 0, 0),
            unittest.mock.call(0x0002, 0, 0, 0, 0),
            unittest.mock.call(0x0004, 0, 0, 0, 0),
        ]
        self.assertEqual(mock_mouse_event.call_args_list, expected_calls)


class TestWinkClickingIntegration(unittest.TestCase):
    def setUp(self):
        self.controller = MouseController()
        self.width = 640
        self.height = 480

    @patch.object(MouseController, "_trigger_click")
    def test_full_pipeline_blink_vs_wink(self, mock_trigger):
        # 1. Normal frame: both eyes open
        landmarks_open = create_mock_eye_landmarks(left_open=True, right_open=True)
        lw, rw = check_winks(landmarks_open, self.width, self.height)
        self.controller.winkClick(lw, rw)
        mock_trigger.assert_not_called()

        # 2. Blink: both eyes closed
        landmarks_blink = create_mock_eye_landmarks(left_open=False, right_open=False)
        lw, rw = check_winks(landmarks_blink, self.width, self.height)
        self.controller.winkClick(lw, rw)
        mock_trigger.assert_not_called()  # Blinking must not trigger clicks

        # 3. Eyes reopen
        lw, rw = check_winks(landmarks_open, self.width, self.height)
        self.controller.winkClick(lw, rw)
        mock_trigger.assert_not_called()

        # 4. Intentional Left Wink (left closed, right open)
        landmarks_left_wink = create_mock_eye_landmarks(left_open=False, right_open=True)
        lw, rw = check_winks(landmarks_left_wink, self.width, self.height)
        self.controller.winkClick(lw, rw)
        mock_trigger.assert_called_once_with("left")

        # 5. Holding left wink over 3 frames (debounce must prevent re-firing)
        for _ in range(3):
            self.controller.winkClick(lw, rw)
        self.assertEqual(mock_trigger.call_count, 1)

        # 6. Reopen eyes
        lw, rw = check_winks(landmarks_open, self.width, self.height)
        self.controller.winkClick(lw, rw)

        # 7. Intentional Right Wink (right closed, left open) -> Double left click
        landmarks_right_wink = create_mock_eye_landmarks(left_open=True, right_open=False)
        lw, rw = check_winks(landmarks_right_wink, self.width, self.height)
        self.controller.winkClick(lw, rw)
        self.assertEqual(mock_trigger.call_count, 2)
        self.assertEqual(mock_trigger.call_args[0][0], "double_left")


if __name__ == "__main__":
    unittest.main()

