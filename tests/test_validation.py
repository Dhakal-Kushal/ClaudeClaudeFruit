import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main as app
from core.calibration import (CalibrationSample, EYE_TARGETS, TrackingAdapter,
                              CalibrationSession,
                              camera_compatibility_errors, load_profile,
                              profile_compatibility_errors, save_profile)
from core.math_filters import PiecewiseAffineMapping2D
from core.validation import ValidationSession, validation_targets, quality_warnings
from core.vision_pipeline import VisionPipeline
from test_calibration import CAMERA, MONITOR, completed_session, validated_profile
from ui.calibration_window import CalibrationWindow


def finish_validation(session, offset=(0, 0)):
    profile = session.profile
    inverse = PiecewiseAffineMapping2D([profile["screen_targets"][key] for key in EYE_TARGETS],
                                       [profile["eye_targets"][key] for key in EYE_TARGETS])
    timestamp = 10.0
    for _ in range(2000):
        if session.complete:
            return
        point = np.asarray(session.targets[session.step]) + offset
        eye = inverse.map_coordinates(*point)
        timestamp += .02
        session.accept(CalibrationSample(*eye, 2, -1, 3, timestamp=timestamp), now=timestamp)
    raise AssertionError("Validation did not complete")


class ValidationTests(unittest.TestCase):
    def test_quality_limits_and_each_failure_reason(self):
        profile = validated_profile()
        diagonal = np.hypot(MONITOR["width"], MONITOR["height"])
        report = profile["validation"]
        self.assertEqual(quality_warnings(report, MONITOR), [])
        for key, limit in (("median_error_px", .025), ("p90_error_px", .05)):
            altered = copy.deepcopy(report)
            altered[key] = diagonal * limit
            self.assertEqual(quality_warnings(altered, MONITOR), [])
            altered[key] += 1
            self.assertTrue(quality_warnings(altered, MONITOR))
        for key, value in (("bias_px", [diagonal * .031, 0]),
                           ("rms_jitter_px", diagonal * .021)):
            altered = copy.deepcopy(report)
            altered["targets"][0][key] = value
            self.assertTrue(quality_warnings(altered, MONITOR))

    def test_high_error_report_can_be_explicitly_accepted_saved_loaded_and_started(self):
        profile = completed_session().build_profile(camera=CAMERA)
        session = ValidationSession(profile)
        finish_validation(session, (180, 100))
        profile["validation"] = session.build_report()
        self.assertTrue(quality_warnings(profile["validation"], MONITOR))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text("original")
            with self.assertRaises(ValueError):
                save_profile(path, profile)
            self.assertEqual(path.read_text(), "original")
            profile["validation"]["accepted"] = True
            original = copy.deepcopy(profile)
            save_profile(path, profile)
            self.assertEqual(load_profile(path), original)
            self.assertEqual(app.check_profile(path, MONITOR), original)
            self.assertEqual(profile_compatibility_errors(original, MONITOR, CAMERA), [])
        with patch.object(app, "get_current_monitor", return_value=MONITOR), \
             patch.object(app, "VisionPipeline") as pipeline_factory, \
             patch.object(app, "MouseController") as controller:
            pipeline = pipeline_factory.return_value
            pipeline.camera_ready.wait.return_value = True
            pipeline.get_camera_metadata.return_value = CAMERA
            pipeline.stop_event.is_set.side_effect = [False, True]
            pipeline.errors.empty.return_value = True
            app.run_tracking(0, profile)
            controller.return_value.start.assert_called_once()
            controller.return_value.stop.assert_called_once()

    def test_independent_targets_respect_monitor_origin(self):
        profile = completed_session().build_profile()
        monitor = {**MONITOR, "x": -1920, "y": 120}
        targets = validation_targets(monitor)
        self.assertEqual(len(targets), 8)
        for point in targets.values():
            self.assertTrue(-1920 < point[0] < 0)
            self.assertTrue(120 < point[1] < 120 + 1080)
        for point in validation_targets(MONITOR).values():
            self.assertNotIn(point, profile["screen_targets"].values())

    def test_known_offset_measures_error_bias_jitter_without_refitting(self):
        profile = completed_session().build_profile()
        original = copy.deepcopy(profile)
        session = ValidationSession(profile)
        finish_validation(session, (30, 40))
        report = session.build_report()
        self.assertEqual(profile, original)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["median_error_px"], 50)
        self.assertAlmostEqual(report["p90_error_px"], 50)
        self.assertAlmostEqual(report["median_error_percent_diagonal"], 5000 / np.hypot(1920, 1080))
        self.assertEqual(report["target_count"], 8)
        self.assertEqual(report["rejected_sample_rate"], 0)
        for result in report["targets"]:
            np.testing.assert_allclose(result["bias_px"], [30, 40], atol=1e-8)
            self.assertLess(result["rms_jitter_px"], 1e-8)
            self.assertGreaterEqual(result["sample_count"], 15)

    def test_incomplete_validation_cannot_be_saved_or_loaded(self):
        profile = completed_session().build_profile(camera=CAMERA)
        session = ValidationSession(profile)
        with self.assertRaises(ValueError):
            session.build_report()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text("original")
            with self.assertRaises(KeyError):
                save_profile(path, profile)
            self.assertEqual(path.read_text(), "original")
            finish_validation(session)
            profile["validation"] = session.build_report()
            with self.assertRaises(ValueError):
                save_profile(path, profile)
            path.write_text(json.dumps(profile))
            self.assertIsNone(load_profile(path))
            profile["validation"]["accepted"] = True
            save_profile(path, profile)
            self.assertEqual(load_profile(path), profile)

    def test_missing_targets_nonfinite_metrics_and_missing_camera_rejected(self):
        profile = validated_profile()
        mutations = (
            lambda p: p["validation"]["targets"].pop(),
            lambda p: p["validation"].update(median_error_px=float("nan")),
            lambda p: p["validation"]["targets"][0].update(sample_count=14),
            lambda p: p["camera"].pop("width"),
            lambda p: p.update(schema_version=1),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(profile)
            mutate(candidate)
            self.assertTrue(profile_compatibility_errors(candidate, MONITOR, CAMERA))

    def test_invalid_stale_duplicate_and_unstable_samples_cannot_complete_target(self):
        session = ValidationSession(completed_session().build_profile(), settle_seconds=0)
        def accept(timestamp, **kw):
            sample = CalibrationSample(.5, .5, kw.pop("pitch", 2), -1, 3, timestamp=timestamp, **kw)
            session.accept(sample, now=timestamp)
        accept(1)
        accept(1)
        self.assertEqual(len(session.samples), 1)
        accept(1.02, eyes_open=False)
        self.assertEqual(len(session.samples), 0)
        accept(1.04, pitch=30)
        self.assertEqual(len(session.samples), 0)
        session.accept(CalibrationSample(float("nan"), .5, 2, -1, 3, timestamp=1.06), now=1.06)
        session.accept(CalibrationSample(.5, .5, 2, -1, 3, timestamp=1.08), now=3)
        self.assertEqual(session.step_index, 0)
        self.assertEqual(session.rejected_samples, 5)
        accept(4)
        accept(4.1, pitch=10)
        self.assertEqual(len(session.samples), 0)
        accept(5)
        accept(6)
        self.assertEqual(len(session.samples), 1)

    def test_settling_is_not_included_in_accuracy_capture(self):
        session = ValidationSession(completed_session().build_profile())
        for index in range(30):
            timestamp = 1 + index * .02
            session.accept(CalibrationSample(.5, .5, 2, -1, 3, timestamp=timestamp), now=timestamp)
        self.assertEqual(session.samples, [])
        self.assertFalse(session.complete)


class CompatibilityTests(unittest.TestCase):
    def test_resolution_drift_stops_capture_before_changed_frame_is_published(self):
        camera = Mock()
        camera.__enter__ = Mock(return_value=camera)
        camera.__exit__ = Mock(return_value=False)
        camera.read.side_effect = [np.zeros((480, 640, 3)), np.zeros((720, 1280, 3))]
        camera.get_metadata.side_effect = [CAMERA, {**CAMERA, "width": 1280, "height": 720}]
        pipeline = VisionPipeline(TrackingAdapter(), camera_factory=Mock(return_value=camera))
        pipeline._capture_frames()
        self.assertTrue(pipeline.stop_event.is_set())
        self.assertEqual(pipeline.frames.qsize(), 1)
        self.assertEqual(pipeline.get_camera_metadata(), CAMERA)
        self.assertIn("width", pipeline.errors.get_nowait())

    def test_settings_changes_and_unknown_properties(self):
        self.assertEqual(camera_compatibility_errors(CAMERA, {**CAMERA, "fps": 29.97}), [])
        for key, value in (("index", 1), ("width", 1280), ("height", 720),
                           ("flip", False), ("backend", 1400), ("fps", 60), ("fourcc", 123)):
            with self.subTest(key=key):
                self.assertTrue(camera_compatibility_errors(CAMERA, {**CAMERA, key: value}))
        self.assertTrue(camera_compatibility_errors(CAMERA, {**CAMERA, "fps": None}))
        self.assertEqual(camera_compatibility_errors({**CAMERA, "fps": None}, {**CAMERA, "fps": None}), [])

    def test_pipeline_rejects_mismatch_before_publishing_frames(self):
        camera = Mock()
        camera.__enter__ = Mock(return_value=camera)
        camera.__exit__ = Mock(return_value=False)
        camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        camera.get_metadata.return_value = {**CAMERA, "flip": False}
        pipeline = VisionPipeline(TrackingAdapter(), camera_factory=Mock(return_value=camera), expected_camera=CAMERA)
        pipeline._capture_frames()
        self.assertTrue(pipeline.stop_event.is_set())
        self.assertFalse(pipeline.camera_ready.is_set())
        self.assertTrue(pipeline.frames.empty())
        self.assertIn("flip", pipeline.errors.get_nowait())
        camera.__exit__.assert_called_once()

    def test_live_tracking_never_starts_mouse_on_mismatch(self):
        profile = validated_profile()
        with patch.object(app, "get_current_monitor", return_value=MONITOR), \
             patch.object(app, "VisionPipeline") as pipeline_factory, \
             patch.object(app, "MouseController") as controller:
            pipeline = pipeline_factory.return_value
            pipeline.camera_ready.wait.return_value = True
            pipeline.get_camera_metadata.return_value = {**CAMERA, "width": 1280}
            pipeline.stop_event.is_set.return_value = False
            pipeline.errors.empty.return_value = True
            with self.assertRaisesRegex(ValueError, "compatibility"):
                app.run_tracking(0, profile)
            controller.assert_not_called()
            pipeline.stop.assert_called_once()

    def test_first_run_still_prompts_for_camera(self):
        with patch.object(app, "check_profile", return_value=None), \
             patch.object(app, "run_calibration", return_value=False) as calibrate:
            prompt = Mock(return_value=("calibrate", 3))
            app.main(prompt_fn=prompt)
            prompt.assert_called_once()
            self.assertEqual(calibrate.call_args.args[0], 3)


class CalibrationUITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        physical = patch("ui.calibration_window._physical_screen_size", return_value=(1920, 1080))
        physical.start()
        self.addCleanup(physical.stop)
        self.path = Path(self.directory.name) / "profile.json"
        self.path.write_text("original")
        self.window = CalibrationWindow(profile_path=self.path, camera_metadata=CAMERA, auto_start=False)
        original_destroy = self.window.destroy
        def destroy_with_pending_callbacks_cancelled():
            for callback in self.window.tk.call("after", "info"):
                self.window.after_cancel(callback)
            original_destroy()
        self.window.destroy = destroy_with_pending_callbacks_cancelled
        self.window.withdraw()
        self.addCleanup(lambda: self.window.close() if not self.window._closed else None)

    def test_validation_precedes_save_and_explicit_acceptance(self):
        window = self.window
        window.session = completed_session()
        window._finish_calibration()
        self.assertEqual(window.phase, "validation")
        self.assertEqual(self.path.read_text(), "original")
        finish_validation(window.session, (30, 40))
        window._finish_validation()
        self.assertEqual(window.phase, "results")
        self.assertIn("50 px", window.summary.cget("text"))
        self.assertEqual(self.path.read_text(), "original")
        window._accept_calibration()
        self.assertTrue(window._closed)
        self.assertTrue(load_profile(self.path)["validation"]["accepted"])

    def test_high_error_results_offer_use_anyway_and_save_measured_errors(self):
        window = self.window
        window.threshold.set("18\u00b0")
        window.session = completed_session()
        window._finish_calibration()
        self.assertEqual(window.candidate_profile["gesture_settings"]["trigger_degrees"], 18)
        finish_validation(window.session, (180, 100))
        window._finish_validation()
        self.assertEqual(window.start_button.cget("state"), "normal")
        self.assertEqual(window.start_button.cget("text"), "Use anyway")
        self.assertIn("suggested targets", window.message.cget("text"))
        self.assertIn("Retry calibration", window.message.cget("text"))
        self.assertEqual(self.path.read_text(), "original")
        measured = copy.deepcopy(window.candidate_profile["validation"])
        window._accept_calibration()
        saved = load_profile(self.path)
        measured["accepted"] = True
        self.assertEqual(saved["validation"], measured)
        self.assertTrue(window._closed)

    def test_high_error_results_can_be_retried_without_saving(self):
        window = self.window
        window.session = completed_session()
        window._finish_calibration()
        finish_validation(window.session, (180, 100))
        window._finish_validation()
        window.start_calibration()
        self.assertEqual(window.phase, "calibration")
        self.assertEqual(window.threshold.cget("state"), "normal")
        self.assertEqual(self.path.read_text(), "original")

    def test_queued_frames_from_previous_target_do_not_start_next_capture(self):
        from time import monotonic
        window = self.window
        window.start_calibration()
        window.after_cancel(window._fullscreen_id)
        window._enter_stage_fullscreen(window.stage)
        window.session = CalibrationSession(MONITOR, countdown=0, capture_seconds=.2, min_samples=4)
        window._drawn_step = None
        window._draw_stage()
        base = monotonic()
        for delta in (.01, .11, .21, .31, .32):
            window.adapter.publish(.5, .5, 2, -1, 3, timestamp=base + delta)
        with patch("ui.calibration_window.monotonic", return_value=base + .4):
            window._poll_tracking()
            self.assertEqual(window.session.step, "top_left")
            window._poll_tracking()
        self.assertIsNone(window.session.started_at)
        self.assertEqual(window.session.samples, [])
        window.adapter.publish(.4, .4, 2, -1, 3, timestamp=base + .42)
        with patch("ui.calibration_window.monotonic", return_value=base + .45):
            window._poll_tracking()
        self.assertEqual(len(window.session.samples), 1)

    def test_retry_discards_validation_results_without_saving(self):
        window = self.window
        window.session = completed_session()
        window._finish_calibration()
        finish_validation(window.session)
        window._finish_validation()
        window.start_calibration()
        self.assertEqual(window.phase, "calibration")
        self.assertIsNone(window.candidate_profile)
        self.assertEqual(window.session.step, "neutral_head")
        self.assertEqual(self.path.read_text(), "original")
        window.cancel_calibration()
        self.assertEqual(window.phase, "idle")
        self.assertEqual(window.start_button.cget("text"), "Start calibration")

    def test_cancellation_does_not_replace_profile(self):
        window = self.window
        window.session = completed_session()
        window._finish_calibration()
        finish_validation(window.session)
        window._finish_validation()
        window.cancel_calibration()
        self.assertTrue(window.cancelled)
        self.assertEqual(self.path.read_text(), "original")

    def test_changed_camera_blocks_acceptance_and_shows_visible_error(self):
        window = self.window
        window._check_camera()
        window.session = completed_session()
        window._finish_calibration()
        finish_validation(window.session)
        window._finish_validation()
        window.camera_metadata_provider = lambda: {**CAMERA, "width": 1280}
        window._accept_calibration()
        self.assertEqual(self.path.read_text(), "original")
        self.assertEqual(window.phase, "error")
        self.assertNotEqual(window.state(), "withdrawn")
        self.assertIn("width", window.message.cget("text"))

    def test_camera_ready_gate_and_timeout_are_visible(self):
        window = self.window
        window.camera_metadata = None
        window.camera_metadata_provider = lambda: None
        window.start_calibration()
        self.assertIsNone(window.session)
        window._waiting_since -= 11
        window._poll_tracking()
        self.assertIn("timed out", window.message.cget("text"))
        self.assertNotEqual(window.state(), "withdrawn")

    def test_save_failure_keeps_results_for_retry_and_preserves_previous_file(self):
        window = self.window
        window.session = completed_session()
        window._finish_calibration()
        finish_validation(window.session)
        window._finish_validation()
        with patch("ui.calibration_window.save_profile", side_effect=OSError("Disk full")):
            window._accept_calibration()
        self.assertFalse(window._closed)
        self.assertEqual(window.phase, "results")
        self.assertIn("Disk full", window.message.cget("text"))
        self.assertEqual(self.path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
