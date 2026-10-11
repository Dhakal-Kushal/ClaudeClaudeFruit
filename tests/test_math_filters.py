import unittest

import numpy as np

from core.math_filters import CoordinateAverager2D, KalmanFilter2D, PiecewiseAffineMapping2D


# Recorded geometry that produced a roughly 72-pixel jump with quadrant mapping.
SOURCE_POINTS = np.array([
    [351.57215118408203, 188.35407257080078],
    [371.4020252227783, 193.04396152496338],
    [371.25958919525146, 198.2134473323822],
    [356.6120910644531, 198.84636640548706],
    [365.30489921569824, 195.8926820755005],
])
TARGET_POINTS = np.array([
    [24, 24], [2535, 24], [2535, 1575], [24, 1575], [1279.5, 799.5],
])


class PiecewiseAffineMapping2DTests(unittest.TestCase):
    def setUp(self):
        self.mapping = PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS)

    def test_all_five_calibration_anchors_are_reproduced(self):
        for source, target in zip(SOURCE_POINTS, TARGET_POINTS):
            np.testing.assert_allclose(self.mapping.map_coordinates(*source), target, atol=1e-9)

    def test_interpolation_inside_each_triangle(self):
        for index in range(4):
            following = (index + 1) % 4
            source = 0.3 * SOURCE_POINTS[4] + 0.25 * SOURCE_POINTS[index] + 0.45 * SOURCE_POINTS[following]
            expected = 0.3 * TARGET_POINTS[4] + 0.25 * TARGET_POINTS[index] + 0.45 * TARGET_POINTS[following]
            np.testing.assert_allclose(self.mapping.map_coordinates(*source), expected, atol=1e-9)

    def test_continuity_along_every_shared_edge_and_its_extension(self):
        for index in range(4):
            direction = SOURCE_POINTS[index] - SOURCE_POINTS[4]
            perpendicular = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            for fraction in (0.0, 0.3, 0.8, 1.0, 1.5):
                with self.subTest(edge=index, fraction=fraction):
                    point = SOURCE_POINTS[4] + fraction * direction
                    expected = TARGET_POINTS[4] + fraction * (TARGET_POINTS[index] - TARGET_POINTS[4])
                    np.testing.assert_allclose(self.mapping.map_coordinates(*point), expected, atol=1e-9)
                    before = np.array(self.mapping.map_coordinates(*(point - perpendicular * 1e-8)))
                    after = np.array(self.mapping.map_coordinates(*(point + perpendicular * 1e-8)))
                    self.assertLess(np.linalg.norm(after - before), 1e-4)

    def test_old_quadrant_boundary_no_longer_jumps(self):
        x, y = SOURCE_POINTS[4]
        before = np.array(self.mapping.map_coordinates(x - 1e-5, y + 1))
        after = np.array(self.mapping.map_coordinates(x + 1e-5, y + 1))
        self.assertLess(np.linalg.norm(after - before), 0.01)

    def test_mapping_is_independent_of_eye_units_and_axis_orientation(self):
        point = 0.3 * SOURCE_POINTS[4] + 0.2 * SOURCE_POINTS[0] + 0.5 * SOURCE_POINTS[1]
        expected = self.mapping.map_coordinates(*point)
        for transform in (
            np.eye(2) * 1e-5,
            np.array([[-0.01, 0], [0, 0.01]]),
            np.array([[0.02, 0.008], [-0.003, 0.015]]),
        ):
            source = (SOURCE_POINTS - SOURCE_POINTS[4]) @ transform.T
            mapped_point = (point - SOURCE_POINTS[4]) @ transform.T
            with self.subTest(transform=transform):
                mapping = PiecewiseAffineMapping2D(source, TARGET_POINTS)
                np.testing.assert_allclose(mapping.map_coordinates(*mapped_point), expected, atol=1e-8)

    def test_extrapolation_beyond_outer_edge_uses_the_same_affine_map(self):
        source = -0.6 * SOURCE_POINTS[4] + 0.8 * SOURCE_POINTS[0] + 0.8 * SOURCE_POINTS[1]
        expected = -0.6 * TARGET_POINTS[4] + 0.8 * TARGET_POINTS[0] + 0.8 * TARGET_POINTS[1]
        np.testing.assert_allclose(self.mapping.map_coordinates(*source), expected, atol=1e-9)
        self.assertLess(expected[1], 0)

    def test_invalid_source_and_target_polygons_are_rejected(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]], dtype=float)
        malformed = [np.zeros((5, 2)), square[:4], square[[0, 2, 1, 3, 4]]]
        outside, boundary, concave, nonfinite, narrow = (square.copy() for _ in range(5))
        outside[4] = [2, 0]
        boundary[4] = [1, 0]
        concave[1] = [-0.5, 0]
        nonfinite[0, 0] = np.nan
        narrow[:, 1] *= 1e-7
        malformed.extend((outside, boundary, concave, nonfinite, narrow))
        for points in malformed:
            for source, target in ((points, TARGET_POINTS), (SOURCE_POINTS, points)):
                with self.subTest(points=points, source=source), self.assertRaises(ValueError):
                    PiecewiseAffineMapping2D(source, target)

    def test_nonfinite_or_nonscalar_coordinates_do_not_change_the_mapping(self):
        expected = self.mapping.map_coordinates(*SOURCE_POINTS[4])
        for point in ((np.nan, 0), (0, np.inf), ([1, 2], [3, 4]), (1e308, 1e308)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                self.mapping.map_coordinates(*point)
        self.assertEqual(self.mapping.map_coordinates(*SOURCE_POINTS[4]), expected)

    def test_mapping_snapshots_the_calibration_arrays(self):
        source, target = SOURCE_POINTS.copy(), TARGET_POINTS.copy()
        mapping = PiecewiseAffineMapping2D(source, target)
        source[:] = 0
        target[:] = 0
        np.testing.assert_allclose(mapping.map_coordinates(*SOURCE_POINTS[0]), TARGET_POINTS[0], atol=1e-9)

    def test_blend_width_validation(self):
        # Valid values
        for valid in (0.0, 0.01, 0.05, 0.49):
            m = PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS, blend_width=valid)
            self.assertEqual(m._blend_width, valid)

        # Invalid values
        for invalid in (-0.01, 0.5, 1.0, np.nan, np.inf, True, False, "0.05"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS, blend_width=invalid)

    def test_sector_boundary_derivative_continuity(self):
        center = SOURCE_POINTS[4]
        unblended = PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS, blend_width=0.0)
        blended = PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS, blend_width=0.04)

        for corner_idx in range(4):
            corner = SOURCE_POINTS[corner_idx]
            ray_dir = corner - center
            ray_perp = np.array([-ray_dir[1], ray_dir[0]])
            ray_perp = ray_perp / np.linalg.norm(ray_perp)

            test_point_on_ray = center + 0.6 * ray_dir
            s_vals = np.linspace(-0.015, 0.015, 601)

            pts_unblended = np.array([unblended.map_coordinates(*(test_point_on_ray + s * ray_perp)) for s in s_vals])
            pts_blended = np.array([blended.map_coordinates(*(test_point_on_ray + s * ray_perp)) for s in s_vals])

            ds = s_vals[1] - s_vals[0]
            d_unblended = np.diff(pts_unblended, axis=0) / ds
            d_blended = np.diff(pts_blended, axis=0) / ds

            step_unblended = np.max(np.abs(np.diff(d_unblended, axis=0)))
            step_blended = np.max(np.abs(np.diff(d_blended, axis=0)))

            # Unblended exhibits a severe derivative jump across the sector ray
            self.assertGreater(step_unblended, 20.0)
            # Blended eliminates the derivative step, transitioning smoothly
            self.assertLess(step_blended, 1.0)



class KalmanFilter2DTests(unittest.TestCase):
    def test_step_changes_do_not_overshoot_at_multiple_frame_rates(self):
        for fps in (15, 20, 30, 60):
            tracker = KalmanFilter2D()
            for _ in range(120):
                tracker.update(500, 500, dt=1/fps)
            for target in ((1500, 900), (300, 200), (1800, 700)):
                previous = tracker._last_output.copy()
                for _ in range(fps * 2):
                    result = np.array(tracker.update(*target, dt=1/fps))
                    self.assertTrue((result >= np.minimum(previous, target)-1e-9).all())
                    self.assertTrue((result <= np.maximum(previous, target)+1e-9).all())
                    previous = result
                np.testing.assert_allclose(result, target, atol=2)

    def test_mapped_measurements_preserve_symmetric_positive_covariance_with_variable_dt(self):
        mapping = PiecewiseAffineMapping2D(SOURCE_POINTS, TARGET_POINTS)
        tracker = KalmanFilter2D()
        rng = np.random.default_rng(19)
        for interval in rng.uniform(1 / 120, 0.5, size=800):
            weights = rng.dirichlet(np.ones(5))
            gaze = weights @ SOURCE_POINTS
            screen = mapping.map_coordinates(*gaze)
            filtered = tracker.update(*screen, dt=interval)
            self.assertTrue(np.isfinite(filtered).all())
            covariance = tracker._covariance
            np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-9)
        tracker.reset()
        self.assertIsNone(tracker._covariance)
        screen = mapping.map_coordinates(*SOURCE_POINTS[4])
        self.assertEqual(tracker.update(*screen), screen)

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

    def test_internal_state_divergence_is_bounded_during_overshoot(self):
        tracker = KalmanFilter2D()
        for _ in range(60):
            tracker.update(500.0, 500.0)
        rng = np.random.default_rng(42)
        measurements = 600.0 + rng.normal(0, 5, size=(60, 2))
        max_dev = 2.0 * np.sqrt(tracker._measurement_noise)
        for m in measurements:
            out = tracker.update(*m)
            divergence = np.linalg.norm(tracker._state[:2] - out)
            self.assertLessEqual(divergence, max_dev * np.sqrt(2) + 1e-6)

    def test_stationary_fixation_jitter_reduction(self):
        tracker = KalmanFilter2D(measurement_noise=25.0)
        rng = np.random.default_rng(123)
        target = np.array([960.0, 540.0])
        # Settle filter
        for _ in range(30):
            tracker.update(*target)
        # 100 noisy samples around stationary fixation
        noise = rng.normal(0, 4.0, size=(100, 2))
        raw_samples = target + noise
        filtered = np.array([tracker.update(*sample) for sample in raw_samples])
        raw_rms = np.sqrt(np.mean(np.sum((raw_samples - target)**2, axis=1)))
        filtered_rms = np.sqrt(np.mean(np.sum((filtered - target)**2, axis=1)))
        # Filter should damp stationary measurement noise by at least 30%
        self.assertLess(filtered_rms, raw_rms * 0.70)



if __name__ == "__main__":
    unittest.main()
