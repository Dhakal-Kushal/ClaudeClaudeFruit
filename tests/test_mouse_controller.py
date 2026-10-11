import time
from queue import Queue
import unittest
from unittest.mock import MagicMock

from core.calibration import CalibrationSample
from core.gaze_mapper import GazeMapper
from core.mouse_controller import MouseController


SAMPLE_PROFILE = {
    "monitor": {"width": 1920, "height": 1080, "x": 0, "y": 0},
    "screen_targets": {
        "top_left": [0, 0],
        "top_right": [1920, 0],
        "bottom_right": [1920, 1080],
        "bottom_left": [0, 1080],
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

    def test_corner_mappings(self):
        self.assertEqual(self.mapper.map_coordinates(200.0, 100.0), (0.0, 0.0))
        self.assertEqual(self.mapper.map_coordinates(400.0, 100.0), (1920.0, 0.0))
        self.assertEqual(self.mapper.map_coordinates(400.0, 300.0), (1920.0, 1080.0))
        self.assertEqual(self.mapper.map_coordinates(200.0, 300.0), (0.0, 1080.0))

    def test_intermediate_interpolation(self):
        sx, sy = self.mapper.map_coordinates(250.0, 150.0)
        self.assertEqual((sx, sy), (480.0, 270.0))

    def test_boundary_clamping(self):
        sx, sy = self.mapper.map_coordinates(100.0, 50.0)
        self.assertEqual((sx, sy), (0.0, 0.0))
        sx, sy = self.mapper.map_coordinates(500.0, 400.0)
        self.assertEqual((sx, sy), (1920.0, 1080.0))

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
        # Eye looking past top-right dot can reach the full monitor edge (1920)
        sx, sy = mapper.map_coordinates(450.0, 100.0)
        self.assertEqual(sx, 1920.0)

    def test_near_zero_delta_protection(self):
        profile_zero_delta = {
            "monitor": {"width": 1920, "height": 1080, "x": 0, "y": 0},
            "screen_targets": SAMPLE_PROFILE["screen_targets"],
            "eye_targets": {
                "top_left": [300.1, 100.0],
                "top_right": [300.2, 100.0],  # dx from center is only 0.2 (< MIN_EYE_DELTA)
                "bottom_right": [400.0, 300.0],
                "bottom_left": [200.0, 300.0],
                "center": [300.0, 200.0],
            },
        }
        mapper = GazeMapper(profile_zero_delta)
        # Should not raise ZeroDivisionError and should not produce infinite coordinates
        sx, sy = mapper.map_coordinates(300.5, 100.0)
        self.assertEqual(sx, 960.0)  # Safe fallback to center x


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

    def test_left_tilt_triggers_single_click_and_freezes_movement(self):
        sample_tilt = CalibrationSample(300.0, 200.0, 0.0, 0.0, -18.0, timestamp=time.time())
        self.controller.update(sample_tilt)
        self.mock_mouse.click.assert_called_once_with("left")
        self.mock_mouse.double_click.assert_not_called()
        self.assertTrue(self.controller.debounce_flag)

        self.mock_mouse.move.reset_mock()
        sample_move = CalibrationSample(250.0, 150.0, 0.0, 0.0, -18.0, timestamp=time.time())
        self.controller.update(sample_move)
        self.mock_mouse.move.assert_not_called()

    def test_right_tilt_triggers_double_click_and_freezes_movement(self):
        sample_tilt = CalibrationSample(300.0, 200.0, 0.0, 0.0, 18.0, timestamp=time.time())
        self.controller.update(sample_tilt)
        self.mock_mouse.double_click.assert_called_once_with("left")
        self.mock_mouse.click.assert_not_called()
        self.assertTrue(self.controller.debounce_flag)

        self.mock_mouse.move.reset_mock()
        sample_move = CalibrationSample(250.0, 150.0, 0.0, 0.0, 18.0, timestamp=time.time())
        self.controller.update(sample_move)
        self.mock_mouse.move.assert_not_called()

    def test_extreme_tilt_does_not_click(self):
        sample_extreme = CalibrationSample(300.0, 200.0, 0.0, 0.0, 35.0, timestamp=time.time())
        self.controller.update(sample_extreme)
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()
        self.assertTrue(self.controller.debounce_flag)

    def test_extreme_tilt_returning_past_threshold_does_not_click(self):
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 35.0, timestamp=time.time()))
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 18.0, timestamp=time.time()))
        self.mock_mouse.click.assert_not_called()
        self.mock_mouse.double_click.assert_not_called()
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 5.0, timestamp=time.time()))
        self.assertFalse(self.controller.debounce_flag)

    def test_debounce_resets_after_head_straightens(self):
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, -16.0, timestamp=time.time()))
        self.assertEqual(self.mock_mouse.click.call_count, 1)

        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, -17.0, timestamp=time.time()))
        self.assertEqual(self.mock_mouse.click.call_count, 1)

        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, -5.0, timestamp=time.time()))
        self.assertFalse(self.controller.debounce_flag)

        self.controller.last_click_time = 0
        self.controller.update(CalibrationSample(300.0, 200.0, 0.0, 0.0, 16.0, timestamp=time.time()))
        self.assertEqual(self.mock_mouse.double_click.call_count, 1)

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
        self.mock_mouse.move.assert_called_with(1920, 0, absolute=True)
        self.controller.stop()
