import time
import copy
from queue import Queue
import unittest
from unittest.mock import MagicMock

import numpy as np

from core.calibration import CalibrationSample
from core.gaze_mapper import GazeMapper
from core.mouse_controller import MouseController
from core.math_filters import KalmanFilter2D


SAMPLE_PROFILE = {
    "monitor": {"width": 1920, "height": 1080, "x": 0, "y": 0},
    "screen_targets": {
        "top_left": [0, 0],
        "top_right": [1919, 0],
        "bottom_right": [1919, 1079],
        "bottom_left": [0, 1079],
        "center": [960, 540],
    },
    "eye_targets": {
        "top_left": [200.0, 100.0],
        "top_right": [400.0, 100.0],
        "bottom_right": [400.0, 300.0],
        "bottom_left": [200.0, 300.0],
        "center": [300.0, 200.0],
    },
    "neutral_head": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "gesture_settings": {
        "trigger_degrees": 15.0,
        "release_degrees": 7.5,
        "cooldown_ms": 300,
    },
}


class GazeMapperTests(unittest.TestCase):
    def setUp(self):
        self.mapper = GazeMapper(SAMPLE_PROFILE)

    def test_center_mapping(self):
        sx, sy = self.mapper.map_coordinates(300.0, 200.0)
        self.assertEqual((sx, sy), (960.0, 540.0))

    def test_eye_relative_profile_does_not_collapse_small_movements_to_center(self):
        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["mapping"] = {"coordinate_space": "eye_relative_v1"}
        profile["eye_targets"] = {
            name: [(point[0] - 300) / 1000, (point[1] - 200) / 1000]
            for name, point in profile["eye_targets"].items()
        }
        mapper = GazeMapper(profile)
        self.assertEqual(mapper.map_coordinates(0, 0), (960.0, 540.0))
        self.assertEqual(mapper.map_coordinates(-0.05, -0.05), (480.0, 270.0))
        self.assertEqual(mapper.map_coordinates(0.1, -0.1), (1919.0, 0.0))

    def test_corner_mappings(self):
        self.assertEqual(self.mapper.map_coordinates(200.0, 100.0), (0.0, 0.0))
        self.assertEqual(self.mapper.map_coordinates(400.0, 100.0), (1919.0, 0.0))
        self.assertEqual(self.mapper.map_coordinates(400.0, 300.0), (1919.0, 1079.0))
        self.assertEqual(self.mapper.map_coordinates(200.0, 300.0), (0.0, 1079.0))

    def test_intermediate_interpolation(self):
        sx, sy = self.mapper.map_coordinates(250.0, 150.0)
        self.assertEqual((sx, sy), (480.0, 270.0))

    def test_boundary_clamping(self):
        sx, sy = self.mapper.map_coordinates(100.0, 50.0)
        self.assertEqual((sx, sy), (0.0, 0.0))
        sx, sy = self.mapper.map_coordinates(500.0, 400.0)
        self.assertEqual((sx, sy), (1919.0, 1079.0))

    def test_reaches_monitor_edges_beyond_inset_targets(self):
        profile_with_insets = {
            "monitor": {"width": 1920, "height": 1080, "x": 0, "y": 0},
            "screen_targets": {
                "top_left": [24, 24],
                "top_right": [1895, 24],
                "bottom_right": [1895, 1055],
                "bottom_left": [24, 1055],
                "center": [960, 540],
            },
            "eye_targets": {
                "top_left": [200.0, 100.0],
                "top_right": [400.0, 100.0],
                "bottom_right": [400.0, 300.0],
                "bottom_left": [200.0, 300.0],
                "center": [300.0, 200.0],
            },
        }
        mapper = GazeMapper(profile_with_insets)
        # Eye at top-right dot gets inset coordinate 1895
        sx, sy = mapper.map_coordinates(400.0, 100.0)
        self.assertEqual(sx, 1895.0)
        # Eye looking past top-right dot can reach the last display pixel.
        sx, sy = mapper.map_coordinates(450.0, 100.0)
        self.assertEqual(sx, 1919.0)

    def test_collinear_calibration_is_rejected_instead_of_collapsing_to_center(self):
        profile_zero_delta = {
            "monitor": {"width": 1920, "height": 1080, "x": 0, "y": 0},
            "screen_targets": SAMPLE_PROFILE["screen_targets"],
            "eye_targets": {
                "top_left": [300.0, 100.0],
                "top_right": [300.0, 100.0],
                "bottom_right": [300.0, 300.0],
                "bottom_left": [300.0, 300.0],
                "center": [300.0, 200.0],
            },
        }
        with self.assertRaises(ValueError):
            GazeMapper(profile_zero_delta)

    def test_display_origin_and_last_pixel_are_respected(self):
        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["monitor"].update(x=-1920, y=100)
        profile["screen_targets"] = {
            name: [point[0] - 1920, point[1] + 100]
            for name, point in profile["screen_targets"].items()
        }
        mapper = GazeMapper(profile)
        self.assertEqual(mapper.map_coordinates(100, 0), (-1920.0, 100.0))
        self.assertEqual(mapper.map_coordinates(500, 400), (-1.0, 1179.0))


class MouseControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.queue = Queue()
        self.mock_mouse = MagicMock()
        self.controller = MouseController(self.queue, SAMPLE_PROFILE, mouse_module=self.mock_mouse,
                                          clock=lambda: self.now)

    def tearDown(self):
        self.controller.stop()

    def feed(self, x=300, y=200, dt=.05, **kwargs):
        self.now += dt
        sample = CalibrationSample(x, y, kwargs.pop("pitch", 0), kwargs.pop("yaw", 0),
                                   kwargs.pop("roll", 0), timestamp=self.now, **kwargs)
        self.controller.update(sample)
        return sample

    def recover(self, **kwargs):
        for _ in range(4):
            self.feed(**kwargs)

    def wink(self, side="left"):
        for _ in range(5):
            self.feed(eyes_open=False, **{side + "_winking": True})
        for _ in range(7):
            self.feed()

    def test_valid_sample_moves_cursor(self):
        self.feed()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_continuous_mapping_feeds_dynamic_filter_and_preserves_covariance(self):
        reference = KalmanFilter2D()
        for index in range(120):
            interval = (.01, .05, .02, .12)[index % 4]
            x, y = 300 + 140 * np.sin(index / 10), 200 + 140 * np.cos(index / 10)
            screen = self.controller.mapper.map_coordinates(x, y)
            expected = reference.update(*screen, dt=None if index == 0 else interval)
            self.feed(x, y, dt=interval)
            self.mock_mouse.move.assert_called_with(
                int(max(0, min(1919, expected[0]))), int(max(0, min(1079, expected[1]))), absolute=True)
            covariance = self.controller.kalman._covariance
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-9)

    def test_moving_average_honors_window_and_reset_after_long_loss(self):
        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["smoothing"] = {"method": "moving_average", "window_size": 3}
        self.controller = MouseController(profile=profile, mouse_module=self.mock_mouse, clock=lambda: self.now)
        mapped = []
        for x in (250, 300, 350, 375):
            mapped.append(self.controller.mapper.map_coordinates(x, 200))
            self.feed(x)
            expected = np.mean(mapped[-3:], axis=0).astype(int)
            self.mock_mouse.move.assert_called_with(*expected, absolute=True)
        previous = self.controller.last_position
        self.feed(tracking_valid=False)
        self.now += 2
        self.recover(x=300)
        result = self.mock_mouse.move.call_args.args
        self.assertGreater(result[0], 960)
        self.assertLess(result[0], previous[0])

    def test_unknown_smoothing_is_rejected(self):
        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["smoothing"] = {"method": "unknown"}
        with self.assertRaises(ValueError):
            MouseController(profile=profile, mouse_module=self.mock_mouse)

    def test_kalman_filter_smooths_jitter(self):
        self.feed()
        self.feed(305)
        self.assertGreater(self.mock_mouse.move.call_args.args[0], 960)
        self.assertLess(self.mock_mouse.move.call_args.args[0], 1008)

    def test_blink_recovery_rejects_first_noisy_reopening_for_both_filters(self):
        for method in ("kalman", "moving_average"):
            with self.subTest(method=method):
                profile = copy.deepcopy(SAMPLE_PROFILE)
                profile["smoothing"] = {"method": method, "window_size": 3}
                self.controller = MouseController(profile=profile, mouse_module=self.mock_mouse, clock=lambda: self.now)
                self.feed()
                self.mock_mouse.move.reset_mock()
                self.feed(eyes_open=False)
                self.now += self.controller.freeze_duration
                self.feed(400, 100)
                self.mock_mouse.move.assert_not_called()
                self.recover()
                self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_tracking_loss_freezes_and_recovers_without_origin_jump(self):
        self.feed()
        self.mock_mouse.move.reset_mock()
        self.feed(tracking_valid=False)
        self.now += 3
        self.feed(400, 100)
        self.mock_mouse.move.assert_not_called()
        self.recover()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_stale_future_nonfinite_and_nonmonotonic_samples_never_act(self):
        self.feed()
        self.mock_mouse.reset_mock()
        for stamp in (self.now - 30, self.now + .1, float("nan"), self.now, self.now - .01):
            self.controller.update(CalibrationSample(400, 100, 0, 0, -18, timestamp=stamp,
                                                      eyes_open=False, left_winking=True))
        self.mock_mouse.move.assert_not_called()
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()
        self.recover()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_worker_consumes_newest_sample_and_cancels_pending_gestures(self):
        from unittest.mock import patch
        old = CalibrationSample(200, 100, 0, 0, -18, timestamp=self.now-.1)
        latest = CalibrationSample(400, 100, 0, 0, 0, timestamp=self.now)
        self.queue.put(old)
        self.queue.put(latest)
        with patch.object(self.controller, "update", side_effect=lambda s: self.controller.stop_event.set()) as update:
            self.controller._run()
        update.assert_called_once_with(latest)
        self.assertTrue(self.controller._paused)
        self.assertTrue(self.controller.wink_debounce_flag)

    def test_repeated_fresh_backlogs_do_not_prevent_startup_recovery(self):
        controller = self.controller
        controller._pause(cancel_gestures=True)
        owner = self

        class PairedFrames(Queue):
            def __init__(self):
                super().__init__()
                self.count = 0

            def get(self, block=True, timeout=None):
                # Each blocking read encounters two fresh frames, as can happen
                # while the cursor consumer is being scheduled at startup.
                if timeout is not None:
                    if self.count == 10:
                        controller.stop_event.set()
                        from queue import Empty
                        raise Empty
                    self.count += 1
                    owner.now += .06
                    self.put(CalibrationSample(300, 200, 0, 0, 0, timestamp=owner.now-.01))
                    self.put(CalibrationSample(300, 200, 0, 0, 0, timestamp=owner.now))
                return super().get(block=block, timeout=timeout)

        controller.queue = PairedFrames()
        controller._run()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)
        self.assertGreaterEqual(self.mock_mouse.move.call_count, 7)
        self.assertFalse(controller._paused)
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()

    def test_discarded_blink_still_pauses_until_tracking_recovers(self):
        from unittest.mock import patch
        self.feed()
        self.mock_mouse.move.reset_mock()
        self.now += .05
        self.queue.put(CalibrationSample(400, 100, 0, 0, 0, eyes_open=False,
                                         timestamp=self.now-.01))
        self.queue.put(CalibrationSample(300, 200, 0, 0, 0, timestamp=self.now))
        original_update = self.controller.update
        def update_and_stop(sample):
            original_update(sample)
            self.controller.stop_event.set()
        with patch.object(self.controller, "update", side_effect=update_and_stop):
            self.controller._run()
        self.mock_mouse.move.assert_not_called()
        self.assertTrue(self.controller._paused)
        self.recover()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_persistent_pause_reports_reason_without_per_frame_spam(self):
        from unittest.mock import patch
        with patch("core.mouse_controller.logging.warning") as warning:
            for _ in range(50):
                self.feed(yaw=25)
            warning.assert_called_once()
            self.assertIn("Head pose differs", warning.call_args.args[1])
            self.assertIn("yaw +25.0", warning.call_args.args[1])
            self.recover()
            self.assertEqual(self.controller._status_key, "moving")

    def test_no_samples_at_startup_reports_waiting_reason(self):
        from queue import Empty
        from unittest.mock import patch
        controller = self.controller
        owner = self
        class EmptyFrames(Queue):
            def get(self, block=True, timeout=None):
                owner.now += .5
                if owner.now > 104:
                    controller.stop_event.set()
                raise Empty
        controller.queue = EmptyFrames()
        with patch("core.mouse_controller.logging.warning") as warning:
            controller._run()
        warning.assert_called_once()
        self.assertIn("No fresh tracking samples", warning.call_args.args[1])

    def test_mouse_failure_uses_windows_fallback(self):
        from unittest.mock import patch
        self.mock_mouse.move.side_effect = RuntimeError("Mouse backend unavailable")
        with patch("core.mouse_controller.ctypes.windll.user32.SetCursorPos", return_value=1) as fallback:
            self.feed()
        fallback.assert_called_once_with(960, 540)
        self.assertEqual(self.controller._status_key, "moving")

    def test_mouse_failure_is_reported_if_fallback_also_fails(self):
        from unittest.mock import patch
        self.mock_mouse.move.side_effect = RuntimeError("Mouse backend unavailable")
        with patch("core.mouse_controller.ctypes.windll.user32.SetCursorPos", return_value=0), \
             patch("core.mouse_controller.logging.warning") as warning:
            for _ in range(50):
                self.feed()
        warning.assert_called_once()
        self.assertIn("Mouse movement failed", warning.call_args.args[1])
        self.assertIsNone(self.controller.last_position)

    def test_head_pose_blocks_movement_and_clicks_then_recovers(self):
        self.feed()
        self.mock_mouse.reset_mock()
        for _ in range(8):
            self.feed(400, 100, yaw=25, roll=-18)
        self.mock_mouse.move.assert_not_called()
        self.mock_mouse.click.assert_not_called()
        self.recover()
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_pitch_and_wrapped_angles_use_neutral_baseline(self):
        self.controller.neutral_head = [179, 0, 0]
        self.feed(pitch=-179)
        self.mock_mouse.move.assert_called_once()
        self.mock_mouse.reset_mock()
        self.feed(pitch=160)
        self.mock_mouse.move.assert_not_called()

    def test_head_tilt_never_triggers_clicks(self):
        self.feed()
        self.mock_mouse.reset_mock()
        # Feed extreme and held head tilt (roll=-18, roll=+18, roll=30)
        for _ in range(10):
            self.feed(400, 100, roll=-18)
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()

        for _ in range(10):
            self.feed(400, 100, roll=18)
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()

        for _ in range(6):
            self.feed(roll=30)
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()

    def test_left_and_right_wink_click_after_reopening(self):
        self.feed()
        self.wink("left")
        self.mock_mouse.click.assert_called_once_with("left")
        self.wink("right")
        self.mock_mouse.double_click.assert_called_once_with("left")

    def test_asymmetric_blink_never_clicks(self):
        self.feed()
        self.feed(eyes_open=False, left_winking=True)
        self.feed(eyes_open=False)
        self.feed(eyes_open=False, left_winking=True)
        for _ in range(8):
            self.feed()
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()

    def test_tracking_loss_cancels_qualified_wink(self):
        for _ in range(5):
            self.feed(eyes_open=False, left_winking=True)
        self.feed(tracking_valid=False)
        for _ in range(8):
            self.feed()
        self.mock_mouse.click.assert_not_called()

    def test_freeze_uses_injected_monotonic_clock_and_explicit_duration(self):
        self.controller.freeze_duration = 1.0
        self.controller.last_click_time = self.now
        self.assertTrue(self.controller.is_frozen)
        self.now += 1.01
        self.assertFalse(self.controller.is_frozen)

    def test_camera_fps_determines_nominal_dt_for_recovery(self):
        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["camera"] = {"fps": 60.0}
        controller = MouseController(profile=profile, mouse_module=self.mock_mouse, clock=lambda: self.now)
        self.assertEqual(controller.nominal_fps, 60.0)
        self.assertAlmostEqual(controller.nominal_dt, 1.0 / 60.0)

    def test_skipped_backlog_pauses_kalman_velocity(self):
        controller = self.controller
        controller.feed = lambda **kwargs: None
        old = CalibrationSample(200, 100, 0, 0, 0, timestamp=self.now - 0.05)
        latest = CalibrationSample(300, 200, 0, 0, 0, timestamp=self.now)
        controller.queue.put(old)
        controller.queue.put(latest)
        controller.kalman._state = np.array([200.0, 100.0, 500.0, 300.0])
        controller._paused = False
        from unittest.mock import patch
        with patch.object(controller, "update", side_effect=lambda s: controller.stop_event.set()):
            controller._run()
        # Velocity in Kalman filter was zeroed so it does not project across skipped frames
        np.testing.assert_allclose(controller.kalman._state[2:], [0.0, 0.0])

    def test_head_pose_compensation_in_controller(self):
        self.feed(x=300, y=200, yaw=0.0, pitch=0.0)
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

        profile = copy.deepcopy(SAMPLE_PROFILE)
        profile["head_pose_compensation"] = {"enabled": False}
        ctrl_no_comp = MouseController(profile=profile, mouse_module=self.mock_mouse, clock=lambda: self.now)
        self.assertFalse(ctrl_no_comp.head_pose_compensation)

        profile_high_gain = copy.deepcopy(SAMPLE_PROFILE)
        profile_high_gain["head_pose_compensation"] = {"enabled": True, "yaw_gain": 2.0}
        ctrl_high = MouseController(profile=profile_high_gain, mouse_module=self.mock_mouse, clock=lambda: self.now)
        sample = CalibrationSample(300, 200, 0, 5.0, 0, timestamp=self.now + 0.05)
        ctrl_high.update(sample)
        mapped = ctrl_high.mapper.map_coordinates(310, 200)
        self.mock_mouse.move.assert_called_with(int(mapped[0]), int(mapped[1]), absolute=True)

    def test_init_dpi_awareness(self):
        from core.mouse_controller import init_dpi_awareness
        # Must execute without raising an error
        init_dpi_awareness()

    def test_move_with_default_mouse_uses_setcursorpos_on_windows(self):
        from unittest.mock import patch
        import ctypes
        import sys

        # With default mouse (mouse_module=None), move should call SetCursorPos directly on Windows
        ctrl = MouseController(profile=SAMPLE_PROFILE, clock=lambda: self.now)
        if sys.platform.startswith("win"):
            with patch.object(ctypes.windll.user32, "SetCursorPos", return_value=1) as mock_setpos:
                result = ctrl.move(800, 600, absolute=True)
                self.assertTrue(result)
                mock_setpos.assert_called_once_with(800, 600)


if __name__ == "__main__":
    unittest.main()




