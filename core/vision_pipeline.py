import logging
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from functools import partial

from core.camera import Camera
from core.calibration import camera_compatibility_errors, validate_camera_metadata
from core.vision_processor import VisionProcessor, process_frames


class VisionPipeline:
    def __init__(self, adapter, camera_index=0, camera_factory=None, processor_factory=None,
                 freeze_duration=0.5, *, capture_settings=None, expected_camera=None, initial_head=None):
        self.adapter = adapter
        self.camera_index = camera_index
        self.camera_factory = camera_factory if camera_factory is not None else Camera
        self.capture_settings = dict(capture_settings or {})
        self.expected_camera = dict(expected_camera) if expected_camera else None
        self.processor_factory = processor_factory or partial(
            VisionProcessor, mirrored=self.capture_settings.get("flip", True), initial_head=initial_head
        )
        self.freeze_duration = freeze_duration
        self.frames = Queue(maxsize=2)
        self.errors = Queue()
        self.stop_event = Event()
        self.threads = []
        self._frame_lock = Lock()
        self._metadata_lock = Lock()
        self._camera_metadata = None
        self.camera_ready = Event()

    def get_camera_metadata(self):
        with self._metadata_lock:
            return dict(self._camera_metadata) if self._camera_metadata is not None else None

    def start(self):
        if self.threads:
            raise RuntimeError("The vision pipeline has already been started")
        self.threads = [
            Thread(
                target=process_frames,
                args=(self.frames, self.adapter, self.stop_event, self.errors),
                kwargs={
                    "processor_factory": self.processor_factory,
                    "freeze_duration": self.freeze_duration,
                },
                name="vision-processing", daemon=True,
            ),
            Thread(target=self._capture_frames, name="camera-capture", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _capture_frames(self):
        try:
            if self.stop_event.is_set():
                return
            with self.camera_factory(index=self.camera_index, **self.capture_settings) as camera:
                while not self.stop_event.is_set():
                    frame = camera.read()
                    if frame is None:
                        raise RuntimeError("The camera stopped providing video frames")
                    metadata = camera.get_metadata()
                    validate_camera_metadata(metadata)
                    if [metadata["height"], metadata["width"]] != list(frame.shape[:2]):
                        raise ValueError("Camera metadata does not match captured frame dimensions")
                    expected = self.get_camera_metadata() or self.expected_camera
                    if expected is not None:
                        errors = camera_compatibility_errors(expected, metadata)
                        if errors:
                            raise ValueError("Camera settings changed; recalibrate. " + "; ".join(errors))
                    with self._metadata_lock:
                        if self._camera_metadata is None:
                            self._camera_metadata = dict(metadata)
                    self.camera_ready.set()
                    packet = (monotonic(), frame, camera.camera_matrix, camera.dist_coeffs)
                    with self._frame_lock:
                        try:
                            self.frames.put_nowait(packet)
                        except Full:
                            try:
                                self.frames.get_nowait()
                            except Empty:
                                pass
                            try:
                                self.frames.put_nowait(packet)
                            except Full:
                                pass
        except Exception as error:
            self.errors.put(f"Camera capture failed: {error}")
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logging.warning("%s is still finishing a camera or tracking call", thread.name)
