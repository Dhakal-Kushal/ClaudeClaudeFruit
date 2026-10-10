import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

from core.camera import Camera
from core.vision_processor import process_frames


class VisionPipeline:
    def __init__(self, adapter, camera_index=0, camera_factory=None, processor_factory=None):
        self.adapter = adapter
        self.camera_index = camera_index
        self.camera_factory = camera_factory if camera_factory is not None else Camera
        self.processor_factory = processor_factory
        self.frames = Queue(maxsize=2)
        self.errors = Queue()
        self.stop_event = Event()
        self.threads = []

    def start(self):
        if self.threads:
            raise RuntimeError("The vision pipeline has already been started")
        self.threads = [
            Thread(
                target=process_frames,
                args=(self.frames, self.adapter, self.stop_event, self.errors),
                kwargs={"processor_factory": self.processor_factory},
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
            with self.camera_factory(index=self.camera_index) as camera:
                while not self.stop_event.is_set():
                    frame = camera.read()
                    if frame is None:
                        raise RuntimeError("The camera stopped providing video frames")
                    packet = (monotonic(), frame, camera.camera_matrix, camera.dist_coeffs)
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
