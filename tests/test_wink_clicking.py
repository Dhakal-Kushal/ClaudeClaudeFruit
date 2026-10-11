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

from core.calibration import CalibrationSample
from core.mouse_controller import MouseController
from core.vision_processor import (
    VisionProcessor,
    check_blink,
    check_winks,
    eye_ratio,
    LEFT_EYE,
    RIGHT_EYE,
)


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

    def test_wink_clicks_share_cooldown(self):
        self.controller.cooldown = 2
        self.hold("left")
        self.reopen()
        self.controller._trigger_click.assert_called_once_with("left")
        self.hold("right")
        self.reopen()
        self.assertEqual(self.controller._trigger_click.call_count, 1)
        self.now += 2
        self.hold("right")
        self.reopen()
        self.assertEqual(self.controller._trigger_click.call_count, 2)
        self.controller._trigger_click.assert_called_with("double_left")

    def test_head_click_is_disabled_and_does_not_click(self):
        self.controller.headClick(-18, timestamp=self.now)
        self.controller.headClick(18, timestamp=self.now)
        self.controller._trigger_click.assert_not_called()

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
    def setUp(self):
        self.width = 640
        self.height = 480

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

    @patch.object(MouseController, "_trigger_click")
    def test_blink_freezes_mouse_for_half_second_and_never_clicks(self, mock_trigger):
        controller = MouseController(freeze_duration=0.5)
        self.assertFalse(controller.is_frozen)

        # Trigger a blink
        controller.winkClick(left_winking=False, right_winking=False, is_blinking=True)
        mock_trigger.assert_not_called()
        self.assertTrue(controller.is_frozen, "Mouse should be frozen after a blink occurs")
        self.assertTrue(controller.wink_debounce_flag, "Debounce flag should be set on blink")

        # Calling move() while frozen must not move cursor
        mock_mouse = Mock()
        controller.mouse = mock_mouse
        controller.move(500, 500)
        mock_mouse.move.assert_not_called()

        # Any wink during blink freeze duration is ignored
        controller.winkClick(left_winking=True, right_winking=False, is_blinking=False)
        mock_trigger.assert_not_called()

    def test_check_blink_detects_both_eyes_closed(self):
        landmarks_open = create_mock_eye_landmarks(left_open=True, right_open=True)
        self.assertFalse(check_blink(landmarks_open, self.width, self.height))

        landmarks_left_wink = create_mock_eye_landmarks(left_open=False, right_open=True)
        self.assertFalse(check_blink(landmarks_left_wink, self.width, self.height))

        landmarks_right_wink = create_mock_eye_landmarks(left_open=True, right_open=False)
        self.assertFalse(check_blink(landmarks_right_wink, self.width, self.height))

        landmarks_blink = create_mock_eye_landmarks(left_open=False, right_open=False)
        self.assertTrue(check_blink(landmarks_blink, self.width, self.height))

    @patch.object(MouseController, "_trigger_click")
    def test_update_with_blink_sample_freezes_mouse_and_does_not_click(self, mock_trigger):
        controller = MouseController(freeze_duration=0.5)
        blink_sample = CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, eyes_open=False, is_blinking=True)
        controller.update(blink_sample)

        mock_trigger.assert_not_called()
        self.assertTrue(controller.is_frozen, "MouseController should freeze on blink sample")

    def test_subthreshold_asymmetry_does_not_trigger_wink(self):
        # Create landmarks where both eyes are slightly open with tiny EAR difference (noise / jitter)
        # diff < diff_threshold (0.04)
        landmarks = create_mock_eye_landmarks(left_open=True, right_open=True)
        # Default height ratio is 0.25 for both eyes
        lw, rw = check_winks(landmarks, self.width, self.height)
        self.assertFalse(lw)
        self.assertFalse(rw)


class TestVisionProcessorWinkAndBlinkDetection(unittest.TestCase):
    """Tests the VisionProcessor multi-frame temporal filter for blinks and winks."""

    def setUp(self):
        self.mock_mesh = Mock()
        self.processor = VisionProcessor(face_mesh=self.mock_mesh, min_wink_frames=2)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _feed_frame(self, left_open=True, right_open=True):
        landmarks = create_mock_eye_landmarks(left_open=left_open, right_open=right_open)
        self.mock_mesh.process.return_value = SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
        )
        return self.processor.process(self.frame)

    def test_blink_detected_immediately_in_vision_processor(self):
        sample = self._feed_frame(left_open=False, right_open=False)
        self.assertTrue(sample.is_blinking)
        self.assertFalse(sample.left_winking)
        self.assertFalse(sample.right_winking)
        self.assertFalse(sample.eyes_open)

    def test_left_wink_requires_min_wink_frames(self):
        # Frame 1: left wink started -> counter becomes 1 (< 2 frames), so not yet active
        sample1 = self._feed_frame(left_open=False, right_open=True)
        self.assertFalse(sample1.left_winking, "Transient 1-frame left wink must not trigger prematurely")
        self.assertFalse(sample1.right_winking)
        self.assertFalse(sample1.is_blinking)

        # Frame 2: left wink held -> counter becomes 2 (>= 2 frames), active!
        sample2 = self._feed_frame(left_open=False, right_open=True)
        self.assertTrue(sample2.left_winking, "Left wink must trigger after reaching min_wink_frames")
        self.assertFalse(sample2.right_winking)
        self.assertFalse(sample2.is_blinking)

        # Frame 3: left wink continued -> remains active
        sample3 = self._feed_frame(left_open=False, right_open=True)
        self.assertTrue(sample3.left_winking)

        # Frame 4: eyes reopened -> counter reset, left wink deactivated immediately
        sample4 = self._feed_frame(left_open=True, right_open=True)
        self.assertFalse(sample4.left_winking)
        self.assertFalse(sample4.right_winking)
        self.assertFalse(sample4.is_blinking)

    def test_right_wink_requires_min_wink_frames(self):
        # Frame 1: right wink started -> counter becomes 1 (< 2 frames), so not yet active
        sample1 = self._feed_frame(left_open=True, right_open=False)
        self.assertFalse(sample1.right_winking, "Transient 1-frame right wink must not trigger prematurely")
        self.assertFalse(sample1.left_winking)
        self.assertFalse(sample1.is_blinking)

        # Frame 2: right wink held -> counter becomes 2 (>= 2 frames), active!
        sample2 = self._feed_frame(left_open=True, right_open=False)
        self.assertTrue(sample2.right_winking, "Right wink must trigger after reaching min_wink_frames")
        self.assertFalse(sample2.left_winking)
        self.assertFalse(sample2.is_blinking)

        # Frame 3: right wink continued -> remains active
        sample3 = self._feed_frame(left_open=True, right_open=False)
        self.assertTrue(sample3.right_winking)

        # Frame 4: eyes reopened -> counter reset, right wink deactivated immediately
        sample4 = self._feed_frame(left_open=True, right_open=True)
        self.assertFalse(sample4.left_winking)
        self.assertFalse(sample4.right_winking)
        self.assertFalse(sample4.is_blinking)

    def test_blink_resets_accumulated_wink_counters(self):
        # Accumulate 1 frame of left wink
        sample1 = self._feed_frame(left_open=False, right_open=True)
        self.assertEqual(self.processor._left_wink_counter, 1)

        # User blinks (both eyes close)
        sample2 = self._feed_frame(left_open=False, right_open=False)
        self.assertTrue(sample2.is_blinking)
        self.assertFalse(sample2.left_winking)
        self.assertEqual(self.processor._left_wink_counter, 0, "Blink must reset left wink counter")

        # Accumulate 1 frame of right wink
        sample3 = self._feed_frame(left_open=True, right_open=False)
        self.assertEqual(self.processor._right_wink_counter, 1)

        # User blinks again
        sample4 = self._feed_frame(left_open=False, right_open=False)
        self.assertTrue(sample4.is_blinking)
        self.assertFalse(sample4.right_winking)
        self.assertEqual(self.processor._right_wink_counter, 0, "Blink must reset right wink counter")


class TestCompleteConditionsAndTransitions(unittest.TestCase):
    """Exhaustive tests for blink, left wink, and right wink conditions and transitions."""

    def setUp(self):
        self.controller = MouseController(freeze_duration=0.5)

    @patch.object(MouseController, "_trigger_click")
    def test_simultaneous_winks_treated_as_blink(self, mock_trigger):
        # If both left and right wink flags are somehow True simultaneously,
        # it must be treated as a blink: freeze mouse, fire NO clicks
        self.controller.winkClick(left_winking=True, right_winking=True, is_blinking=False)
        mock_trigger.assert_not_called()
        self.assertTrue(self.controller.is_frozen)
        self.assertTrue(self.controller.wink_debounce_flag)

    @patch.object(MouseController, "_trigger_click")
    def test_left_wink_condition_does_not_trigger_right_click_or_double_click(self, mock_trigger):
        # Left wink must specifically only trigger "left" (single left click)
        self.controller.winkClick(left_winking=True, right_winking=False, is_blinking=False)
        mock_trigger.assert_called_once_with("left")
        self.assertNotEqual(mock_trigger.call_args[0][0], "double_left")
        self.assertNotEqual(mock_trigger.call_args[0][0], "right")

    @patch.object(MouseController, "_trigger_click")
    def test_right_wink_condition_does_not_trigger_single_left_click(self, mock_trigger):
        # Right wink must specifically only trigger "double_left"
        self.controller.winkClick(left_winking=False, right_winking=True, is_blinking=False)
        mock_trigger.assert_called_once_with("double_left")
        self.assertNotEqual(mock_trigger.call_args[0][0], "left")

    @patch.object(MouseController, "_trigger_click")
    def test_full_lifecycle_neutral_leftwink_neutral_rightwink_neutral_blink(self, mock_trigger):
        # 1. Neutral (eyes open)
        self.controller.winkClick(False, False, is_blinking=False)
        self.assertFalse(self.controller.is_frozen)
        self.assertEqual(mock_trigger.call_count, 0)

        # 2. Left wink held across 5 frames -> exactly 1 left click fired
        for _ in range(5):
            self.controller.winkClick(True, False, is_blinking=False)
        self.assertEqual(mock_trigger.call_count, 1)
        self.assertEqual(mock_trigger.call_args[0][0], "left")

        # 3. Reopen eyes -> resets debounce
        self.controller.winkClick(False, False, is_blinking=False)
        self.assertFalse(self.controller.wink_debounce_flag)

        # 4. Right wink held across 5 frames -> exactly 1 double_left click fired
        for _ in range(5):
            self.controller.winkClick(False, True, is_blinking=False)
        self.assertEqual(mock_trigger.call_count, 2)
        self.assertEqual(mock_trigger.call_args[0][0], "double_left")

        # 5. Reopen eyes
        self.controller.winkClick(False, False, is_blinking=False)

        # 6. Blink -> mouse frozen, no extra clicks
        self.controller.winkClick(False, False, is_blinking=True)
        self.assertEqual(mock_trigger.call_count, 2)
        self.assertTrue(self.controller.is_frozen)


if __name__ == "__main__":
    unittest.main()

