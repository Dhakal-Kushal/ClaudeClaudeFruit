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
        self.now = 100.0
        self.controller = MouseController(mouse_module=Mock(), clock=lambda: self.now)
        self.controller._trigger_click = Mock()

    def frame(self, left=False, right=False, eyes_open=False, dt=.05):
        self.now += dt
        self.controller.winkClick(left, right, eyes_open=eyes_open, timestamp=self.now)

    def hold(self, side="left", count=5):
        for _ in range(count):
            self.frame(left=side == "left", right=side == "right")

    def reopen(self):
        for _ in range(7):
            self.frame(eyes_open=True)

    def test_left_and_right_bindings_after_hold_and_reopening(self):
        self.hold()
        self.controller._trigger_click.assert_not_called()
        self.reopen()
        self.controller._trigger_click.assert_called_once_with("left")
        self.hold("right")
        self.reopen()
        self.assertEqual(self.controller._trigger_click.call_count, 2)
        self.controller._trigger_click.assert_called_with("double_left")

    def test_holding_wink_does_not_repeat_clicks(self):
        self.hold(count=30)
        self.controller._trigger_click.assert_not_called()
        self.reopen()
        self.reopen()
        self.controller._trigger_click.assert_called_once_with("left")

    def test_brief_wink_or_too_few_frames_does_not_click(self):
        self.frame(left=True)
        self.reopen()
        self.controller._trigger_click.assert_not_called()
        self.frame(left=True)
        self.frame(left=True, dt=.2)
        self.reopen()
        self.controller._trigger_click.assert_not_called()

    def test_bilateral_blink_cancels_even_qualified_wink(self):
        self.hold()
        self.frame()
        self.hold()
        self.reopen()
        self.controller._trigger_click.assert_not_called()
        self.hold()
        self.reopen()
        self.controller._trigger_click.assert_called_once_with("left")

    def test_short_reopening_or_switching_eyes_cancels_gesture(self):
        self.hold()
        self.frame(eyes_open=True)
        self.hold()
        self.reopen()
        self.controller._trigger_click.assert_not_called()
        self.hold()
        self.hold("right")
        self.reopen()
        self.controller._trigger_click.assert_not_called()

    def test_head_and_wink_share_cooldown(self):
        self.controller.cooldown = 2
        self.controller.headClick(-18, timestamp=self.now)
        self.now += .2
        self.controller.headClick(-18, timestamp=self.now)
        self.controller._trigger_click.assert_called_once_with("left")
        self.hold("right")
        self.reopen()
        self.controller._trigger_click.assert_called_once()
        self.now += 2
        self.hold("right")
        self.reopen()
        self.controller._trigger_click.assert_called_with("double_left")

    def test_profile_hold_frame_count_and_settle_are_honored(self):
        self.controller.hold_seconds = .3
        self.controller.closed_eye_frames = 8
        self.controller.settle_seconds = .4
        self.hold(count=5)
        self.reopen()
        self.controller._trigger_click.assert_not_called()
        # Complete reopening before starting a second attempt.
        self.reopen()
        self.hold(count=9)
        self.reopen()
        self.controller._trigger_click.assert_not_called()
        self.reopen()
        self.controller._trigger_click.assert_called_once_with("left")

    @patch("core.mouse_controller.ctypes.windll.user32.mouse_event")
    @patch("core.mouse_controller.mouse.press", side_effect=Exception("Simulated mouse library failure"))
    @patch("core.mouse_controller.mouse.double_click", side_effect=Exception("Simulated mouse library failure"))
    def test_windows_events_double_left_click_fallback(self, mock_double_click, mock_press, mock_mouse_event):
        self.controller.mouse = __import__("mouse")
        # Trigger double_left click in fallback mode
        MouseController._trigger_click(self.controller, "double_left")

        # Must execute: LEFTDOWN (0x0002), LEFTUP (0x0004), LEFTDOWN (0x0002), LEFTUP (0x0004)
        expected_calls = [
            unittest.mock.call(0x0002, 0, 0, 0, 0),
            unittest.mock.call(0x0004, 0, 0, 0, 0),
            unittest.mock.call(0x0002, 0, 0, 0, 0),
            unittest.mock.call(0x0004, 0, 0, 0, 0),
        ]
        self.assertEqual(mock_mouse_event.call_args_list, expected_calls)


class TestWinkClickingIntegration(unittest.TestCase):
    def test_landmarks_distinguish_blink_reopening_from_deliberate_wink(self):
        now = 100.0
        controller = MouseController(mouse_module=Mock(), clock=lambda: now)
        controller._trigger_click = Mock()
        def frames(left_open, right_open, count):
            nonlocal now
            landmarks = create_mock_eye_landmarks(left_open, right_open)
            for _ in range(count):
                now += .05
                left, right = check_winks(landmarks, 640, 480)
                controller.winkClick(left, right, eyes_open=left_open and right_open, timestamp=now)
        frames(True, True, 7)
        frames(False, True, 1)
        frames(False, False, 2)
        frames(False, True, 1)
        frames(True, True, 7)
        controller._trigger_click.assert_not_called()
        frames(False, True, 5)
        frames(True, True, 7)
        controller._trigger_click.assert_called_once_with("left")
        frames(True, False, 5)
        frames(True, True, 7)
        controller._trigger_click.assert_called_with("double_left")


if __name__ == "__main__":
    unittest.main()
