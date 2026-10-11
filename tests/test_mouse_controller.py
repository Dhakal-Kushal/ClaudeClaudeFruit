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
        self.queue = Queue()
        self.mock_mouse = MagicMock()
        self.controller = MouseController(self.queue, SAMPLE_PROFILE, mouse_module=self.mock_mouse)

    def tearDown(self):
        self.controller.stop()

    def test_valid_sample_moves_cursor(self):
        sample = CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time())
        self.controller.update(sample)
        self.mock_mouse.move.assert_called_with(960, 540, absolute=True)

    def test_continuous_mapping_feeds_dynamic_filter_and_preserves_covariance(self):
        reference = KalmanFilter2D()
        timestamp = 100.0
        for index in range(120):
            interval = (0.01, 0.05, 0.02, 0.12)[index % 4]
            timestamp += interval
            eye_x = 300 + 140 * np.sin(index / 10)
            eye_y = 200 + 140 * np.cos(index / 10)
            screen = self.controller.mapper.map_coordinates(eye_x, eye_y)
            expected = reference.update(*screen, dt=None if index == 0 else interval)
            self.controller.update(CalibrationSample(eye_x, eye_y, 0, 0, 0, timestamp=timestamp))
            np.testing.assert_allclose(self.controller.kalman._state, reference._state, atol=1e-8)
            expected_x = int(max(0, min(1919, expected[0])))
            expected_y = int(max(0, min(1079, expected[1])))
            self.mock_mouse.move.assert_called_with(expected_x, expected_y, absolute=True)
            covariance = self.controller.kalman._covariance
            np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-9)

    def test_left_tilt_triggers_single_click_and_freezes_movement(self):
        sample_tilt = CalibrationSample(300.0, 200.0, 0.0, 0.0, -18.0, timestamp=time.time())
        self.controller.update(sample_tilt)
        self.mock_mouse.click.assert_called_once_with("left")
        self.mock_mouse.double_click.assert_not_called()
        self.assertTrue(self.controller.debounce_flag)

        self.mock_mouse.move.reset_mock()
        sample_tilt_right = CalibrationSample(300.0, 200.0, 0.0, 0.0, 18.0, timestamp=time.time())
        self.controller.update(sample_tilt_right)
        self.mock_mouse.move.assert_not_called()


    def test_kalman_filter_smooths_jitter(self):
        t = 100.0
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=t))
        self.assertEqual(self.mock_mouse.move.call_args[0], (960, 540))

        # Sudden jump from 300 to 305 (raw mapped screen x jump from 960 to 1008)
        self.controller.update(CalibrationSample(305.0, 200.0, 0.0, 0.0, 0.0, timestamp=t + 0.033))
        smoothed_x = self.mock_mouse.move.call_args[0][0]
        self.assertGreater(smoothed_x, 960)
        self.assertLess(smoothed_x, 1008)

    def test_invalid_sample_resets_kalman_filter(self):
        t = 10.0
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=t))
        self.assertIsNotNone(self.controller.kalman._state)

        # Blink / loss of tracking
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, eyes_open=False, timestamp=t + 0.033))
        self.assertIsNone(self.controller.kalman._state)
        self.assertIsNone(self.controller.last_sample_time)

    def test_worker_thread_processes_queue(self):
        self.controller.start()
        sample = CalibrationSample(400.0, 100.0, 0.0, 0.0, 0.0, timestamp=time.time())
        self.queue.put(sample)
        time.sleep(0.1)
        self.mock_mouse.move.assert_called_with(1919, 0, absolute=True)
        self.controller.stop()

    def test_left_wink_triggers_left_click(self):
        sample = CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), left_winking=True)
        self.controller.update(sample)
        self.mock_mouse.click.assert_called_once_with("left")
        self.assertTrue(self.controller.wink_debounce_flag)

    def test_right_wink_triggers_double_left_click(self):
        sample = CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), right_winking=True)
        self.controller.update(sample)
        self.mock_mouse.double_click.assert_called_once_with("left")
        self.assertTrue(self.controller.wink_debounce_flag)

    def test_wink_debounce_resets_after_opening_both_eyes(self):
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), left_winking=True))
        self.assertEqual(self.mock_mouse.click.call_count, 1)
        self.assertTrue(self.controller.wink_debounce_flag)

        # Still winking - should not click again
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), left_winking=True))
        self.assertEqual(self.mock_mouse.click.call_count, 1)

        # Both eyes open - debounces
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), left_winking=False, right_winking=False))
        self.assertFalse(self.controller.wink_debounce_flag)

        # Next wink triggers double left click
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 0.0, timestamp=time.time(), right_winking=True))
        self.assertEqual(self.mock_mouse.double_click.call_count, 1)

