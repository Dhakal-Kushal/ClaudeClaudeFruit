import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import main as app

from core.calibration import (
    CalibrationSample,
    CalibrationSession,
    TrackingAdapter,
    angle_delta,
    load_profile,
    save_profile,
    screen_targets,
    screen_targets_in_canvas,
)


MONITOR = {"width": 1920, "height": 1080, "x": 0, "y": 0}
CAMERA = {"index": 0, "width": 640, "height": 480, "backend": 700,
          "flip": True, "fps": 30.0, "fourcc": None}
EYES = {
    "neutral_head": (0.5, 0.5),
    "top_left": (0.4, 0.4),
    "top_right": (0.6, 0.4),
    "bottom_right": (0.6, 0.6),
    "bottom_left": (0.4, 0.6),
    "center": (0.5, 0.5),
}


def completed_session():
    session = CalibrationSession(MONITOR, countdown=0.2, capture_seconds=0.3, min_samples=4)
    timestamp = 0.0
    while not session.complete:
        step = session.step
        eye = EYES[step]
        for _ in range(20):
            timestamp += 0.1
            sample = CalibrationSample(*eye, 2.0, -1.0, 3.0, timestamp=timestamp)
            session.accept(sample, now=timestamp)
            if session.step != step:
                break
        else:
            raise AssertionError(f"Calibration did not advance past {step}")
    return session


def validated_profile(profile=None):
    """Generate actual synthetic held-out measurements for persistence tests."""
    from core.validation import ValidationSession
    from core.math_filters import PiecewiseAffineMapping2D
    from core.calibration import EYE_TARGETS
    profile = completed_session().build_profile() if profile is None else profile
    profile["camera"] = dict(CAMERA)
    inverse = PiecewiseAffineMapping2D([profile["screen_targets"][key] for key in EYE_TARGETS],
                                       [profile["eye_targets"][key] for key in EYE_TARGETS])
    validation = ValidationSession(profile, settle_seconds=0, capture_seconds=.2)
    timestamp = 100.0
    while not validation.complete:
        eye = inverse.map_coordinates(*validation.targets[validation.step])
        timestamp += .02
        sample = CalibrationSample(*eye, 2, -1, 3, timestamp=timestamp)
        validation.accept(sample, now=timestamp)
    profile["validation"] = validation.build_report()
    profile["validation"]["accepted"] = True
    return profile


class CalibrationTests(unittest.TestCase):
    def test_live_tracking_requires_eye_relative_profiles(self):
        session = completed_session()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            legacy = validated_profile(session.build_profile(coordinate_space="camera_pixels"))
            save_profile(path, legacy)
            self.assertIsNone(app.check_profile(path, MONITOR))
            with self.assertRaisesRegex(ValueError, "eye-relative calibration"):
                app.run_tracking(0, legacy)
            current = validated_profile(session.build_profile())
            self.assertEqual(current["mapping"]["coordinate_space"], "eye_relative_v1")
            save_profile(path, current)
            self.assertEqual(app.check_profile(path, MONITOR), current)

    def test_six_steps_produce_complete_profile(self):
        session = completed_session()
        profile = session.build_profile(camera={"index": 0}, coordinate_space="normalized_camera")
        for name, eye in EYES.items():
            if name != "neutral_head":
                np.testing.assert_allclose(profile["eye_targets"][name], eye)
        np.testing.assert_allclose(list(profile["neutral_head"].values()), [2, -1, 3])
        self.assertEqual(profile["screen_targets"]["top_left"], [24, 24])
        self.assertEqual(profile["mapping"]["method"], "piecewise_affine_v1")

    def test_old_profile_requires_validation_without_mutating_file(self):
        profile = completed_session().build_profile()
        profile["schema_version"] = 1
        profile["mapping"]["method"] = "four_quadrant_linear"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(profile))
            original = path.read_bytes()
            loaded = load_profile(path, MONITOR)
            self.assertIsNone(loaded)
            self.assertEqual(path.read_bytes(), original)
            profile["mapping"]["coordinate_space"] = "camera_pixels"
            path.write_text(json.dumps(profile))
            self.assertIsNone(load_profile(path, MONITOR))

    def test_nearly_collinear_targets_fail_before_saving(self):
        session = completed_session()
        session.eye_targets = {
            name: [point[0], (point[1] - 0.5) * 1e-7]
            for name, point in session.eye_targets.items()
        }
        with self.assertRaisesRegex(ValueError, "narrow|convex"):
            session.build_profile()

    def test_closed_eyes_and_stale_samples_reset_current_capture(self):
        session = CalibrationSession(MONITOR, countdown=0.1)
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=1.0), now=1.0)
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=1.1), now=1.1)
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, eyes_open=False, timestamp=1.2), now=1.2)
        self.assertIsNone(session.started_at)
        self.assertEqual(session.samples, [])
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=2.0), now=3.0)
        self.assertIsNone(session.started_at)
        self.assertEqual(session.step, "neutral_head")

    def test_gaps_and_duplicate_frames_do_not_complete_capture(self):
        session = CalibrationSession(MONITOR, countdown=0, capture_seconds=0.2, min_samples=4)
        sample = CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=1.0)
        session.accept(sample, now=1.0)
        for _ in range(30):
            session.accept(sample, now=1.0)
        self.assertEqual(len(session.samples), 1)
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=2.0), now=2.0)
        self.assertEqual(len(session.samples), 1)
        self.assertEqual(session.step, "neutral_head")

    def test_unstable_head_restarts_capture(self):
        session = CalibrationSession(MONITOR, countdown=0)
        session.accept(CalibrationSample(0.5, 0.5, 0, 0, 0, timestamp=1.0), now=1.0)
        session.accept(CalibrationSample(0.5, 0.5, 20, 0, 0, timestamp=1.1), now=1.1)
        self.assertEqual(session.samples, [])
        self.assertIsNone(session.started_at)

    def test_neutral_angles_handle_wraparound(self):
        session = CalibrationSession(MONITOR, countdown=0, capture_seconds=0.3, min_samples=4)
        for index in range(8):
            angle = 179.0 if index % 2 else -179.0
            sample = CalibrationSample(0.5, 0.5, angle, 0, 0, timestamp=index * 0.1)
            session.accept(sample, now=sample.timestamp)
            if session.neutral_head is not None:
                break
        self.assertIsNotNone(session.neutral_head)
        self.assertLess(abs(angle_delta(session.neutral_head[0], 180)), 1.1)

    def test_incomplete_and_degenerate_calibration_cannot_be_saved(self):
        with self.assertRaises(ValueError):
            CalibrationSession(MONITOR).build_profile()
        session = completed_session()
        session.eye_targets = {name: [0.5, 0.5] for name in session.eye_targets}
        with self.assertRaises(ValueError):
            session.build_profile()

    def test_save_load_rejects_missing_data_and_changed_monitor(self):
        profile = validated_profile()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            self.assertIsNone(load_profile(path))
            path.write_text("")
            self.assertIsNone(load_profile(path))
            save_profile(path, profile)
            self.assertEqual(load_profile(path, MONITOR), profile)
            self.assertIsNone(load_profile(path, {**MONITOR, "width": 2560}))
            incomplete = copy.deepcopy(profile)
            del incomplete["neutral_head"]["pitch"]
            path.write_text(json.dumps(incomplete))
            self.assertIsNone(load_profile(path))

    def test_failed_save_preserves_previous_profile(self):
        profile = validated_profile()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_profile(path, profile)
            previous = path.read_bytes()
            invalid = copy.deepcopy(profile)
            invalid["gesture_settings"]["trigger_degrees"] = float("nan")
            with self.assertRaises(ValueError):
                save_profile(path, invalid)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_window_target_positions_are_saved_and_reloaded(self):
        session = completed_session()
        area = {"width": 1904, "height": 1017, "x": 8, "y": 31}
        session.screen_positions = screen_targets(area)
        profile = validated_profile(session.build_profile())
        self.assertEqual(profile["screen_targets"]["top_left"], [32, 55])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_profile(path, profile)
            self.assertEqual(load_profile(path, MONITOR)["screen_targets"], screen_targets(area))

    def test_canvas_targets_map_to_monitor_coordinates_at_scaled_resolution(self):
        width, height = 1707, 950
        canvas_targets = screen_targets_in_canvas(MONITOR, width, height)
        monitor_targets = screen_targets(MONITOR)
        for name in monitor_targets:
            canvas_x, canvas_y = canvas_targets[name]
            mapped_x = canvas_x * MONITOR["width"] / width
            mapped_y = canvas_y * MONITOR["height"] / height
            np.testing.assert_allclose([mapped_x, mapped_y], monitor_targets[name])

    def test_malformed_saved_calibration_is_not_reused(self):
        profile = completed_session().build_profile()
        mutations = (
            lambda item: item["screen_targets"].update(top_left=[24]),
            lambda item: item["neutral_head"].update(pitch=[1, 2]),
            lambda item: item["eye_targets"].update(center=[0.9, 0.9]),
            lambda item: item["gesture_settings"].update(release_degrees=30),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    malformed = copy.deepcopy(profile)
                    mutate(malformed)
                    path.write_text(json.dumps(malformed))
                    self.assertIsNone(load_profile(path))

    def test_queue_adapter_drops_oldest_samples_without_blocking(self):
        adapter = TrackingAdapter()
        for index in range(250):
            adapter.publish(index, index, 0, 0, 0)
        self.assertEqual(adapter.samples.qsize(), 180)
        self.assertEqual(adapter.samples.get_nowait().eye_x, 70)


if __name__ == "__main__":
    unittest.main()
