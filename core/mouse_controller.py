import ctypes
import logging
import math
import time
from queue import Empty
from threading import Event, Thread

try:
    import mouse
except ImportError:
    mouse = None

from core.gaze_mapper import GazeMapper
from core.math_filters import KalmanFilter2D


class MouseController:
    """Thread 3: Consumes vision tracking samples, maps eye coordinates to screen, and triggers actions."""

    def __init__(self, queue=None, profile=None, mouse_module=None, freeze_duration=0.5, *args, **kwargs):
        self.queue = queue
        self.mouse = mouse_module if mouse_module is not None else mouse
        profile = profile if profile is not None else {}
        if profile:
            self.mapper = GazeMapper(profile)
        else:
            self.mapper = None

        settings = profile.get("gesture_settings", {})
        cooldown = settings.get("cooldown_ms", 500)
        self.freeze_duration = (cooldown / 1000.0) if cooldown else freeze_duration
        self.sensitivity = 30

        try:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        except Exception:
            self.screen_w = 1920
            self.screen_h = 1080

        smoothing = profile.get("smoothing", {})
        process_noise = smoothing.get("process_noise", 10000.0)
        measurement_noise = smoothing.get("measurement_noise", 25.0)
        self.kalman = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
        self.last_click_time = 0.0
        self.last_sample_time = None
        self.wink_debounce_flag = False
        self.freeze_pos = None
        self.stop_event = Event()
        self.thread = None

    @property
    def is_frozen(self):
        return (time.time() - self.last_click_time) < self.freeze_duration

    def move(self, x, y, absolute=True):
        if not self.is_frozen:
            try:
                self.mouse.move(x, y, absolute=absolute)
            except Exception:
                pass

    def _trigger_click(self, button='left'):
        """Clean press, 40ms hold, and release so Windows registers the click."""
        try:
            if button in ('double_left', 'double'):
                if hasattr(self.mouse, 'double_click'):
                    self.mouse.double_click('left')
                else:
                    mouse.double_click('left')
            elif button == 'right':
                if hasattr(self.mouse, 'click'):
                    self.mouse.click('right')
                else:
                    mouse.click('right')
            else:
                if hasattr(self.mouse, 'click'):
                    self.mouse.click(button)
                elif hasattr(self.mouse, 'press'):
                    self.mouse.press(button)
                    time.sleep(0.04)
                    self.mouse.release(button)
                else:
                    mouse.click(button)
        except Exception:
            # Direct Windows user32 mouse_event fallback
            try:
                user32 = ctypes.windll.user32
                if button == 'left':
                    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
                elif button in ('double_left', 'double'):
                    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
                    time.sleep(0.05)
                    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
                elif button == 'right':
                    user32.mouse_event(0x0008, 0, 0, 0, 0)  # RIGHTDOWN
                    time.sleep(0.04)
                    user32.mouse_event(0x0010, 0, 0, 0, 0)  # RIGHTUP
            except Exception:
                pass

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
            if self.queue is None:
                time.sleep(0.05)
                continue

            try:
                sample = self.queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                self.update(sample)
            except Exception as error:
                logging.exception("Error in mouse controller update: %s", error)

    def update(self, sample):
        if not sample.tracking_valid or not sample.eyes_open:
            self.kalman.reset()
            self.last_sample_time = None
            return

        if self.mapper is not None:
            screen_x, screen_y = self.mapper.map_coordinates(sample.eye_x, sample.eye_y)
            dt = (sample.timestamp - self.last_sample_time) if self.last_sample_time is not None else None
            if dt is not None and (dt <= 0 or not math.isfinite(dt) or dt > 1.0):
                dt = None
            smooth_x, smooth_y = self.kalman.update(screen_x, screen_y, dt=dt)
            self.last_sample_time = sample.timestamp

            smooth_x = max(self.mapper.min_x, min(self.mapper.max_x, smooth_x))
            smooth_y = max(self.mapper.min_y, min(self.mapper.max_y, smooth_y))
        else:
            smooth_x, smooth_y = sample.eye_x, sample.eye_y

        self.winkClick(getattr(sample, "left_winking", False), getattr(sample, "right_winking", False))

        if not self.is_frozen and self.mapper is not None:
            self.move(int(smooth_x), int(smooth_y), absolute=True)

    def winkClick(self, left_winking, right_winking):
        current_time = time.time()
        # 1. Left Wink -> Left Click
        if left_winking and not self.wink_debounce_flag:
            try:
                self.freeze_pos = self.mouse.get_position()
            except Exception:
                self.freeze_pos = None
            self._trigger_click("left")
            self.last_click_time = current_time
            self.wink_debounce_flag = True
            print("--- LEFT CLICK FIRED ---")

        # 2. Right Wink -> Double Left Click
        elif right_winking and not self.wink_debounce_flag:
            try:
                self.freeze_pos = self.mouse.get_position()
            except Exception:
                self.freeze_pos = None
            self._trigger_click("double_left")
            self.last_click_time = current_time
            self.wink_debounce_flag = True
            print("--- DOUBLE LEFT CLICK FIRED ---")

        # 3. Release: Both eyes are open, allow next click
        elif not left_winking and not right_winking:
            self.wink_debounce_flag = False

        if not self.is_frozen:
            self.freeze_pos = None
