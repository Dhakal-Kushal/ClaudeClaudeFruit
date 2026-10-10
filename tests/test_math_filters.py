import unittest

import numpy as np

from core.math_filters import KalmanFilter2D


class KalmanFilter2DTests(unittest.TestCase):
    def test_first_sample_and_reset_do_not_pull_toward_origin(self):
        tracker = KalmanFilter2D()
        self.assertEqual(tracker.update(1200.0, 700.0), (1200.0, 700.0))
        tracker.update(1250.0, 750.0)
        tracker.reset()
        self.assertEqual(tracker.update(100.0, 200.0), (100.0, 200.0))

    def test_stationary_cursor_jitter_is_reduced(self):
        rng = np.random.default_rng(42)
        target = np.array([960.0, 540.0])
        measurements = target + rng.normal(0.0, 5.0, size=(600, 2))
        tracker = KalmanFilter2D()
        filtered = np.array([tracker.update(*point) for point in measurements])
        raw_error = np.mean((measurements[60:] - target) ** 2)
        filtered_error = np.mean((filtered[60:] - target) ** 2)
        self.assertLess(filtered_error, raw_error * 0.5)

    def test_constant_velocity_is_tracked_with_variable_frame_intervals(self):
        rng = np.random.default_rng(7)
        intervals = rng.uniform(1.0 / 60.0, 1.0 / 20.0, size=600)
        times = np.cumsum(intervals)
        positions = np.array([300.0, 200.0]) + times[:, None] * [120.0, -45.0]
        measurements = positions + rng.normal(0.0, 5.0, size=positions.shape)
        tracker = KalmanFilter2D()
        filtered = np.array(
            [
                tracker.update(*point, dt=interval)
                for point, interval in zip(measurements, intervals)
            ]
        )
        raw_error = np.mean((measurements[60:] - positions[60:]) ** 2)
        filtered_error = np.mean((filtered[60:] - positions[60:]) ** 2)
        self.assertLess(filtered_error, raw_error * 0.5)
        np.testing.assert_allclose(
            np.mean(filtered[60:] - positions[60:], axis=0), [0.0, 0.0], atol=1.0
        )

    def test_follows_a_deliberate_change_in_cursor_target(self):
        tracker = KalmanFilter2D()
        for _ in range(60):
            tracker.update(100.0, 100.0)
        for _ in range(60):
            result = tracker.update(500.0, 300.0)
        np.testing.assert_allclose(result, [500.0, 300.0], atol=1.0)

    def test_invalid_parameters_are_rejected(self):
        for name in ("dt", "process_noise", "measurement_noise"):
            for value in (0.0, -1.0, np.nan, np.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        KalmanFilter2D(**{name: value})

    def test_invalid_samples_do_not_change_tracking_history(self):
        tracker = KalmanFilter2D()
        reference = KalmanFilter2D()
        tracker.update(100.0, 200.0)
        reference.update(100.0, 200.0)
        invalid_samples = (
            (np.nan, 200.0, None),
            (100.0, np.inf, None),
            (100.0, 200.0, 0.0),
            (100.0, 200.0, -1.0),
            (100.0, 200.0, np.nan),
            (100.0, 200.0, np.inf),
        )
        for x, y, dt in invalid_samples:
            with self.subTest(x=x, y=y, dt=dt):
                with self.assertRaises(ValueError):
                    tracker.update(x, y, dt)
        np.testing.assert_allclose(
            tracker.update(110.0, 205.0), reference.update(110.0, 205.0)
        )


if __name__ == "__main__":
    unittest.main()
