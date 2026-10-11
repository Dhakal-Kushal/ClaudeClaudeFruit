import math
import time
from queue import Empty
from threading import Event, Thread

import mouse

from core.calibration import angle_delta
from core.gaze_mapper import GazeMapper
from core.math_filters import KalmanFilter2D


class MouseController:
    """Thread 3: Consumes vision tracking samples, maps eye coordinates to screen, and triggers actions."""

    def __init__(self, queue, profile, mouse_module=None):
        self.queue = queue
        self.mapper = GazeMapper(profile)
        self.mouse = mouse_module if mouse_module is not None else mouse

        settings = profile.get("gesture_settings", {})
        self.tilt_threshold = settings.get("trigger_degrees", 15.0)
        self.reset_threshold = settings.get("release_degrees", 7.5)
        self.max_roll_degrees = settings.get("max_roll_degrees", 25.0)
        self.freeze_duration = settings.get("cooldown_ms", 500) / 1000.0

        neutral = profile.get("neutral_head", {})
        self.neutral_roll = neutral.get("roll", 0.0)

        smoothing = profile.get("smoothing", {})
        process_noise = smoothing.get("process_noise", 10000.0)
        measurement_noise = smoothing.get("measurement_noise", 25.0)
        self.kalman = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)

        self.last_click_time = 0
        self.last_sample_time = None
        self.debounce_flag = False
        self.stop_event = Event()
        self.thread = None

    def start(self):
        """Starts the mouse controller consumer thread."""
        if self.thread is not None:
            raise RuntimeError("Mouse controller is already running")
        self.stop_event.clear()
        self.thread = Thread(target=self._run, name="mouse-controller", daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the mouse controller consumer thread."""
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=2.0)
            self.thread = None

    def _run(self):
        while not self.stop_event.is_set():
            try:
                sample = self.queue.get(timeout=0.05)
            except Empty:
                continue

            self.update(sample)

    def update(self, sample):
        if not sample.tracking_valid or not sample.eyes_open:
            self.kalman.reset()
            self.last_sample_time = None
            return

        current_time = time.time()
        is_frozen = (current_time - self.last_click_time) < self.freeze_duration

        screen_x, screen_y = self.mapper.map_coordinates(sample.eye_x, sample.eye_y)
        dt = (sample.timestamp - self.last_sample_time) if self.last_sample_time is not None else None
        if dt is not None and (dt <= 0 or not math.isfinite(dt) or dt > 1.0):
            dt = None
        smooth_x, smooth_y = self.kalman.update(screen_x, screen_y, dt=dt)
        self.last_sample_time = sample.timestamp

        smooth_x = max(self.mapper.min_x, min(self.mapper.max_x, smooth_x))
        smooth_y = max(self.mapper.min_y, min(self.mapper.max_y, smooth_y))

        if not is_frozen:
            self.mouse.move(int(smooth_x), int(smooth_y), absolute=True)

        d_roll = angle_delta([sample.roll], [self.neutral_roll])[0]
        self.headClick(d_roll)

    def headClick(self, d_roll):
        current_time = time.time()
        if not self.debounce_flag:
            if self.tilt_threshold < d_roll <= self.max_roll_degrees:
                self.mouse.double_click("left")
                self.last_click_time = current_time
                self.debounce_flag = True
            elif -self.max_roll_degrees <= d_roll < -self.tilt_threshold:
                self.mouse.click("left")
                self.last_click_time = current_time
                self.debounce_flag = True
        elif abs(d_roll) < self.reset_threshold:
            self.debounce_flag = False