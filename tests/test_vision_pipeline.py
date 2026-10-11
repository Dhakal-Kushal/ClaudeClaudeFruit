import importlib
from queue import Queue
from threading import Event, Thread
from time import sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main as app
from core.calibration import CalibrationSample, TrackingAdapter
from core.vision_pipeline import VisionPipeline
from core import vision_processor


def eye_landmarks(transform=None):
    """Asymmetric eyes in pixel space, with known iris displacements."""
    pixels = np.tile([320.0, 240.0], (478, 1))
    for indices, points in (
        ((33, 160, 158, 133, 153, 144),
         ((160, 200), (180, 185), (220, 185), (240, 200), (220, 215), (180, 215))),
        ((263, 387, 385, 362, 380, 373),
         ((420, 210), (405, 198), (375, 198), (360, 210), (375, 222), (405, 222))),
    ):
        pixels[list(indices)] = points
    pixels[468] = [208, 196]  # (0.10, -0.05) eye widths from its corner midpoint.
    pixels[473] = [381, 216]  # (-0.15, 0.10) for the other eye.
    if transform is not None:
        pixels = transform(pixels)
    return [SimpleNamespace(x=x / 640, y=y / 480) for x, y in pixels]


class EyeRelativeMeasurementTests(unittest.TestCase):
    def test_averages_vectors_normalized_by_each_eyes_own_width(self):
        result = vision_processor.normalized_iris_position(eye_landmarks(), 640, 480)
        np.testing.assert_allclose(result, [-0.025, 0.025], atol=1e-12)

    def test_translation_scale_and_roll_do_not_change_eye_vectors(self):
        reference = vision_processor.normalized_iris_position(eye_landmarks(), 640, 480)
        for degrees in (-35, 0, 35):
            angle = np.deg2rad(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for scale in (0.6, 1.0, 1.4):
                with self.subTest(degrees=degrees, scale=scale):
                    landmarks = eye_landmarks(
                        lambda pixels: (pixels - [320, 240]) @ rotation.T * scale + [345, 260]
                    )
                    np.testing.assert_allclose(
                        vision_processor.normalized_iris_position(landmarks, 640, 480), reference, atol=1e-12
                    )

    def test_mirroring_preserves_eye_pairing_and_reverses_horizontal_direction(self):
        for degrees in (-25, 0, 25):
            angle = np.deg2rad(degrees)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            landmarks = eye_landmarks(
                lambda pixels: ((pixels - [320, 240]) @ rotation.T + [320, 240]) * [-1, 1] + [640, 0]
            )
            with self.subTest(degrees=degrees):
                np.testing.assert_allclose(
                    vision_processor.normalized_iris_position(landmarks, 640, 480), [0.025, 0.025], atol=1e-12
                )

    def test_iris_movement_changes_gaze_without_head_movement(self):
        landmarks = eye_landmarks()
        before = np.array(vision_processor.normalized_iris_position(landmarks, 640, 480))
        landmarks[468].x += 8 / 640
        landmarks[468].y += 4 / 480
        after = np.array(vision_processor.normalized_iris_position(landmarks, 640, 480))
        np.testing.assert_allclose(after - before, [0.05, 0.025], atol=1e-12)

    def test_eyelid_opening_is_not_used_as_the_gaze_scale(self):
        landmarks = eye_landmarks()
        before = vision_processor.normalized_iris_position(landmarks, 640, 480)
        for index in (160, 158, 153, 144):
            landmarks[index].y = 200 / 480
        np.testing.assert_allclose(vision_processor.normalized_iris_position(landmarks, 640, 480), before)

    def test_degenerate_nonfinite_and_out_of_frame_geometry_is_rejected(self):
        mutations = (
            lambda items: setattr(items[133], "x", items[33].x),
            lambda items: setattr(items[33], "x", float("nan")),
            lambda items: setattr(items[473], "y", float("inf")),
            lambda items: setattr(items[468], "x", -0.01),
            lambda items: setattr(items[263], "x", 1.0),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                landmarks = eye_landmarks()
                mutate(landmarks)
                with self.assertRaises(ValueError):
                    vision_processor.normalized_iris_position(landmarks, 640, 480)

    def test_invalid_frame_dimensions_and_incomplete_mesh_are_rejected(self):
        for width, height in ((0, 480), (640, -1), (float("nan"), 480)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                vision_processor.normalized_iris_position(eye_landmarks(), width, height)
        with self.assertRaises(ValueError):
            vision_processor.normalized_iris_position(eye_landmarks()[:468], 640, 480)


class VisionProcessorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=float)
        self.distortion = np.zeros((4, 1))
        self.mesh = Mock()
        self.processor = vision_processor.VisionProcessor(face_mesh=self.mesh)
        self.addCleanup(self.processor.close)

    def test_import_does_not_open_camera_or_start_face_mesh(self):
        with patch("core.camera.Camera") as camera:
            importlib.reload(vision_processor)
        camera.assert_not_called()

    def test_no_face_publishes_invalid_state_and_discards_previous_pose(self):
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=None)
        self.processor.prev_rvec = self.processor.prev_tvec = np.ones((3, 1))
        sample = self.processor.process(self.frame, self.matrix, self.distortion, timestamp=12.0)
        self.assertFalse(sample.tracking_valid)
        self.assertFalse(sample.eyes_open)
        self.assertEqual(sample.timestamp, 12.0)
        self.assertIsNone(self.processor.prev_rvec)

    def test_eye_relative_gaze_and_pose_preserve_closed_eye_state_and_timestamp(self):
        landmarks = eye_landmarks()
        self.mesh.process.return_value = SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
        )
        for eyes_open in (True, False):
            with self.subTest(eyes_open=eyes_open), patch.object(
                vision_processor, "estimate_head_pose", return_value=(2, -1, 3, np.zeros((3, 1)), np.ones((3, 1)))
            ), patch.object(
                vision_processor, "eyes_are_open", return_value=eyes_open
            ):
                sample = self.processor.process(self.frame, self.matrix, self.distortion, timestamp=15.0)
                np.testing.assert_allclose((sample.eye_x, sample.eye_y), [-0.025, 0.025], atol=1e-12)
                np.testing.assert_allclose(self.processor.last_pupil_midpoint, [294.5, 206.0])
                self.assertEqual((sample.pitch, sample.yaw, sample.roll), (2.0, -1.0, 3.0))
                self.assertEqual(sample.timestamp, 15.0)
                self.assertTrue(sample.tracking_valid)
                self.assertEqual(sample.eyes_open, eyes_open)

    def test_bad_eye_geometry_publishes_invalid_sample_then_recovers(self):
        self.processor.prev_rvec = self.processor.prev_tvec = np.ones((3, 1))
        self.processor.last_pupil_midpoint = (300, 200)
        self.processor.left_winking = True
        self.mesh.process.return_value = SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5) for _ in range(478)])]
        )
        with patch.object(vision_processor, "estimate_head_pose") as pose:
            sample = self.processor.process(self.frame, self.matrix, self.distortion, timestamp=16.0)
        pose.assert_not_called()
        self.assertFalse(sample.tracking_valid)
        self.assertFalse(sample.left_winking)
        self.assertEqual(sample.timestamp, 16.0)
        self.assertIsNone(self.processor.prev_rvec)
        self.assertIsNone(self.processor.prev_tvec)
        self.assertIsNone(self.processor.last_pupil_midpoint)
        self.mesh.process.return_value.multi_face_landmarks[0].landmark = eye_landmarks()
        with patch.object(vision_processor, "estimate_head_pose", return_value=(2, -1, 3, np.zeros((3, 1)), np.ones((3, 1)))):
            recovered = self.processor.process(self.frame, self.matrix, self.distortion)
        self.assertTrue(recovered.tracking_valid)

    def test_head_pose_receives_original_geometry_and_previous_pose(self):
        landmarks = eye_landmarks()
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])
        rvec, tvec = np.zeros((3, 1)), np.ones((3, 1))
        with patch.object(vision_processor, "estimate_head_pose", return_value=(2, -1, 3, rvec, tvec)) as pose:
            for _ in range(2):
                self.processor.process(self.frame, self.matrix, self.distortion)
        args = pose.call_args.args
        self.assertIs(args[0], landmarks)
        self.assertEqual(args[1:3], (640, 480))
        self.assertIs(args[3], self.matrix)
        self.assertIs(args[4], self.distortion)
        self.assertIs(args[5], rvec)
        self.assertIs(args[6], tvec)
        self.assertTrue(pose.call_args.kwargs["mirrored"])

    def test_processing_worker_preserves_normalized_gaze_pose_and_wink_fields(self):
        landmarks = eye_landmarks()
        for index in (160, 158, 153, 144):
            landmarks[index].y = 200 / 480
        self.mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])
        frames, errors = Queue(), Queue()
        frames.put((20.0, self.frame, self.matrix, self.distortion))
        adapter, stop = TrackingAdapter(), Event()
        with patch.object(vision_processor, "estimate_head_pose", return_value=(2, -1, 3, np.zeros((3, 1)), np.ones((3, 1)))):
            worker = Thread(target=vision_processor.process_frames, args=(frames, adapter, stop, errors),
                            kwargs={"processor_factory": lambda: self.processor})
            worker.start()
            try:
                sample = adapter.samples.get(timeout=2)
                np.testing.assert_allclose((sample.eye_x, sample.eye_y), [-0.025, 0.025], atol=1e-12)
                self.assertEqual((sample.pitch, sample.yaw, sample.roll, sample.timestamp), (2, -1, 3, 20.0))
                self.assertTrue(sample.tracking_valid)
                self.assertFalse(sample.eyes_open)
                self.assertTrue(sample.left_winking)
                self.assertFalse(sample.right_winking)
                self.assertTrue(errors.empty())
            finally:
                stop.set()
                worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

    def test_mirrored_model_points_used_when_mirrored(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        with patch("cv2.solvePnP", return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)))) as mock_solve:
            vision_processor.estimate_head_pose(landmarks, 640, 480, self.matrix, self.distortion, mirrored=True)
            called_model = mock_solve.call_args[0][0]
            np.testing.assert_allclose(called_model, vision_processor.MODEL_POINTS_MIRRORED)

            vision_processor.estimate_head_pose(landmarks, 640, 480, self.matrix, self.distortion, mirrored=False)
            called_model_unmirrored = mock_solve.call_args[0][0]
            np.testing.assert_allclose(called_model_unmirrored, vision_processor.MODEL_POINTS)

    def test_eye_closure_detection_uses_both_eyes(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for indices in ((33, 160, 158, 133, 153, 144), (263, 387, 385, 362, 380, 373)):
            points = ((0.2, 0.5), (0.25, 0.45), (0.35, 0.45), (0.4, 0.5), (0.35, 0.55), (0.25, 0.55))
            for index, (x, y) in zip(indices, points):
                landmarks[index] = SimpleNamespace(x=x, y=y)
        self.assertTrue(vision_processor.eyes_are_open(landmarks, 640, 480))
        for index in (160, 158, 153, 144):
            landmarks[index].y = 0.5
        self.assertFalse(vision_processor.eyes_are_open(landmarks, 640, 480))

    def test_check_winks_detects_single_eye_closure(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for indices in ((33, 160, 158, 133, 153, 144), (263, 387, 385, 362, 380, 373)):
            points = ((0.2, 0.5), (0.25, 0.45), (0.35, 0.45), (0.4, 0.5), (0.35, 0.55), (0.25, 0.55))
            for index, (x, y) in zip(indices, points):
                landmarks[index] = SimpleNamespace(x=x, y=y)
        # Both eyes open
        self.assertEqual(vision_processor.check_winks(landmarks, 640, 480), (False, False))

        # Close left eye only
        for index in (160, 158, 153, 144):
            landmarks[index].y = 0.5
        self.assertEqual(vision_processor.check_winks(landmarks, 640, 480), (True, False))

    def test_processing_worker_delivers_sample_to_the_adapter(self):
        frames = Queue()
        frames.put((20.0, self.frame, self.matrix, self.distortion))
        adapter = TrackingAdapter()
        stop = Event()
        errors = Queue()
        processor = Mock()
        processor.__enter__ = Mock(return_value=processor)
        processor.__exit__ = Mock()
        processor.process.return_value = CalibrationSample(320, 240, 2, -1, 3, timestamp=20.0)
        worker = Thread(target=vision_processor.process_frames, args=(frames, adapter, stop, errors), kwargs={"processor_factory": lambda: processor})
        worker.start()
        try:
            sample = adapter.samples.get(timeout=2)
            self.assertEqual(sample, processor.process.return_value)
            self.assertTrue(errors.empty())
        finally:
            stop.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())


class VisionPipelineTests(unittest.TestCase):
    def setUp(self):
        from test_calibration import MONITOR
        patcher = patch.object(app, "get_current_monitor", return_value=MONITOR)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_camera_and_vision_workers_deliver_data_and_release_resources(self):
        adapter = TrackingAdapter()
        camera = Mock()
        camera.camera_matrix = np.eye(3)
        camera.dist_coeffs = np.zeros((4, 1))
        from test_calibration import CAMERA
        camera.get_metadata.return_value = {**CAMERA, "index": 2, "width": 20, "height": 20}
        camera.__enter__ = Mock(return_value=camera)
        camera.__exit__ = Mock()

        def read():
            sleep(0.01)
            return np.zeros((20, 20, 3), dtype=np.uint8)

        camera.read.side_effect = read
        processor = Mock()
        processor.__enter__ = Mock(return_value=processor)
        processor.__exit__ = Mock()
        processor.process.side_effect = lambda frame, matrix, distortion, timestamp: CalibrationSample(10, 10, 0, 0, 0, timestamp=timestamp)
        camera_factory = Mock(return_value=camera)
        pipeline = VisionPipeline(adapter, camera_index=2, camera_factory=camera_factory, processor_factory=lambda: processor)
        pipeline.start()
        try:
            sample = adapter.samples.get(timeout=2)
            self.assertTrue(sample.tracking_valid)
            self.assertGreater(sample.timestamp, 0)
            self.assertTrue(pipeline.errors.empty())
        finally:
            pipeline.stop()
        camera_factory.assert_called_once_with(index=2)
        camera.__exit__.assert_called_once()
        processor.__exit__.assert_called_once()
        self.assertTrue(all(not worker.is_alive() for worker in pipeline.threads))

    def test_camera_failure_is_reported_and_stops_workers(self):
        processor = Mock()
        processor.__enter__ = Mock(return_value=processor)
        processor.__exit__ = Mock()
        pipeline = VisionPipeline(
            TrackingAdapter(), camera_factory=Mock(side_effect=RuntimeError("Camera unavailable")),
            processor_factory=lambda: processor,
        )
        pipeline.start()
        try:
            self.assertIn("Camera unavailable", pipeline.errors.get(timeout=2))
        finally:
            pipeline.stop()
        self.assertTrue(pipeline.stop_event.is_set())
        self.assertTrue(all(not worker.is_alive() for worker in pipeline.threads))

    def test_main_starts_pipeline_and_stops_it_even_when_ui_fails(self):
        with patch.object(app, "check_profile", return_value=None), patch.object(app, "CalibrationWindow") as window_factory, patch.object(app, "VisionPipeline") as pipeline_factory:
            window = window_factory.return_value
            window._closed = False
            window.mainloop.side_effect = RuntimeError("UI failure")
            with self.assertRaisesRegex(RuntimeError, "UI failure"):
                app.main(camera_index=2, prompt_fn=lambda *args, **kwargs: "calibrate")
            pipeline_factory.return_value.start.assert_called_once()
            pipeline_factory.return_value.stop.assert_called_once()
            window.close.assert_called_once()
            self.assertIs(window_factory.call_args.kwargs["adapter"], pipeline_factory.call_args.args[0])
            self.assertEqual(window_factory.call_args.kwargs["coordinate_space"], "eye_relative_v1")

    def test_main_runs_tracking_when_profile_exists_and_user_chooses_track(self):
        from test_calibration import validated_profile
        sample_profile = validated_profile()
        sample_profile["camera"]["index"] = 1
        with patch.object(app, "check_profile", return_value=sample_profile), \
             patch.object(app, "prompt_startup_choice", return_value="track"), \
             patch.object(app, "CalibrationWindow") as window_factory, \
             patch.object(app, "VisionPipeline") as pipeline_factory, \
             patch.object(app, "MouseController") as controller_factory, \
             patch.object(app.time, "sleep", side_effect=[None, KeyboardInterrupt]):
            pipeline = pipeline_factory.return_value
            pipeline.get_camera_metadata.return_value = sample_profile["camera"]
            pipeline.stop_event.is_set.side_effect = [False, False, True]
            app.main(camera_index=1)
            window_factory.assert_not_called()
            pipeline.start.assert_called_once()
            pipeline.stop.assert_called_once()
            controller_factory.return_value.start.assert_called_once()
            controller_factory.return_value.stop.assert_called_once()

    def test_main_runs_calibration_when_profile_exists_and_user_chooses_calibrate(self):
        sample_profile = {"mapping": {"coordinate_space": "eye_relative_v1"}}
        with patch.object(app, "check_profile", side_effect=[sample_profile, None]), \
             patch.object(app, "prompt_startup_choice", return_value="calibrate"), \
             patch.object(app, "CalibrationWindow") as window_factory, \
             patch.object(app, "VisionPipeline") as pipeline_factory:
            window = window_factory.return_value
            window._closed = True
            app.main(camera_index=0)
            window_factory.assert_called_once()

    def test_main_exits_cleanly_when_user_chooses_exit(self):
        sample_profile = {"mapping": {"coordinate_space": "eye_relative_v1"}}
        with patch.object(app, "check_profile", return_value=sample_profile), \
             patch.object(app, "prompt_startup_choice", return_value="exit"), \
             patch.object(app, "CalibrationWindow") as window_factory, \
             patch.object(app, "VisionPipeline") as pipeline_factory:
            app.main(camera_index=0)
            window_factory.assert_not_called()
            pipeline_factory.assert_not_called()

    def test_main_uses_camera_selected_in_prompt(self):
        from test_calibration import validated_profile
        sample_profile = validated_profile()
        sample_profile["camera"]["index"] = 3
        with patch.object(app, "check_profile", return_value=sample_profile), \
             patch.object(app, "prompt_startup_choice", return_value=("track", 3)), \
             patch.object(app, "CalibrationWindow") as window_factory, \
             patch.object(app, "VisionPipeline") as pipeline_factory, \
             patch.object(app, "MouseController") as controller_factory, \
             patch.object(app.time, "sleep", side_effect=[None, KeyboardInterrupt]):
            pipeline = pipeline_factory.return_value
            pipeline.get_camera_metadata.return_value = sample_profile["camera"]
            pipeline.stop_event.is_set.side_effect = [False, False, True]
            app.main(camera_index=0)
            self.assertEqual(pipeline_factory.call_args.kwargs["camera_index"], 3)

    def test_main_forces_calibration_when_calibrate_flag_passed(self):
        with patch.object(app, "check_profile", return_value=None), \
             patch.object(app, "CalibrationWindow") as window_factory, \
             patch.object(app, "VisionPipeline") as pipeline_factory:
            window = window_factory.return_value
            window._closed = True
            app.main(camera_index=0, calibrate=True)
            window_factory.assert_called_once()
            pipeline_factory.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
