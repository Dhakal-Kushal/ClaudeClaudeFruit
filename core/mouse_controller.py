import ctypes
import logging
import math
import sys
import time
from queue import Empty
from threading import Event, Thread

import numpy as np

try:
    import mouse
except ImportError:
    mouse = None

from core.calibration import angle_delta, compensate_head_pose
from core.gaze_mapper import GazeMapper
from core.math_filters import CoordinateAverager2D, KalmanFilter2D


def init_dpi_awareness():
    """Ensure Per-Monitor V2 or V1 DPI awareness on Windows."""
    if sys.platform.startswith("win"):
        try:
            try:
                ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
                return
            except (AttributeError, OSError):
                pass
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                return
            except (AttributeError, OSError):
                pass
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass
        except (AttributeError, OSError):
            pass


class MouseController:
    """Thread 3: Consumes vision tracking samples, maps eye coordinates to screen, and triggers actions."""

    def __init__(self, queue=None, profile=None, mouse_module=None, freeze_duration=None,
                 *, clock=None):
        self.clock = clock or time.monotonic
        self.queue = queue
        self.mouse = mouse_module if mouse_module is not None else mouse
        profile = profile if profile is not None else {}
        if profile:
            self.mapper = GazeMapper(profile)
        else:
            self.mapper = None

        settings = profile.get("gesture_settings", {})
        self.tilt_threshold = settings.get("trigger_degrees", 15.0)
        self.reset_threshold = settings.get("release_degrees", 7.5)
        self.max_roll_degrees = settings.get("max_roll_degrees", 25.0)
        self.cooldown = settings.get("cooldown_ms", 500) / 1000.0
        self.freeze_duration = self.cooldown if freeze_duration is None else freeze_duration
        self.hold_seconds = settings.get("hold_ms", 150) / 1000.0
        self.settle_seconds = settings.get("click_settle_ms", 250) / 1000.0
        self.closed_eye_frames = settings.get("closed_eye_frames", 3)
        self.profile = profile
        self.max_sample_age = 0.25
        self.max_head_deviation = 10.0
        fps_val = profile.get("camera", {}).get("fps") if isinstance(profile, dict) else None
        try:
            self.nominal_fps = float(fps_val) if fps_val and float(fps_val) > 0 and math.isfinite(float(fps_val)) else 30.0
        except (ValueError, TypeError):
            self.nominal_fps = 30.0
        self.nominal_dt = 1.0 / self.nominal_fps
        pose_settings = profile.get("head_pose_compensation", {}) if isinstance(profile, dict) else {}
        self.head_pose_compensation = pose_settings.get("enabled", True)
        self.yaw_gain = float(pose_settings.get("yaw_gain", 0.007))
        self.pitch_gain = float(pose_settings.get("pitch_gain", 0.007))

        init_dpi_awareness()
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
        method = smoothing.get("method", "kalman")
        if method == "kalman":
            self.smoother = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
        elif method == "moving_average":
            self.smoother = CoordinateAverager2D(smoothing.get("window_size", 5))
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        self.kalman = self.smoother if method == "kalman" else None
        self.neutral_head = [profile.get("neutral_head", {}).get(key, 0.0)
                             for key in ("pitch", "yaw", "roll")]
        self.neutral_roll = self.neutral_head[2]
        self.last_click_time = -math.inf
        self.last_blink_time = -math.inf
        self.last_sample_time = None
        self.last_seen_time = None
        self.last_position = None
        self._paused = False
        self._recovery = []
        self._tilt_candidate = None
        self._tilt_since = None
        self._wink_candidate = None
        self._wink_since = None
        self._wink_frames = 0
        self._wink_qualified = False
        self._open_since = None
        self.debounce_flag = False
        self.wink_debounce_flag = False
        self.freeze_pos = None
        self.stop_event = Event()
        self.thread = None
        self._status_key = "waiting"
        self._status_message = "Waiting for tracking samples"
        self._status_since = self.clock()
        self._last_status_log = -math.inf
        self._last_input_at = self.clock()

    def _set_status(self, key, message):
        if key != self._status_key:
            self._status_key = key
            self._status_since = self.clock()
        self._status_message = message
        self._report_status()

    def _report_status(self):
        now = self.clock()
        if (self._status_key != "moving" and now - self._status_since >= 2.0
                and now - self._last_status_log >= 5.0):
            logging.warning("Cursor paused: %s", self._status_message)
            self._last_status_log = now

    @property
    def is_frozen(self):
        return (self.clock() - max(self.last_click_time, self.last_blink_time)) < self.freeze_duration

    def move(self, x, y, absolute=True):
        if self.is_frozen:
            return False
        # If a custom mock or test mouse module is injected, route through it
        if self.mouse is not mouse:
            try:
                self.mouse.move(x, y, absolute=absolute)
                return True
            except Exception as error:
                try:
                    if absolute and sys.platform.startswith("win") and ctypes.windll.user32.SetCursorPos(int(x), int(y)):
                        return True
                except Exception:
                    pass
                self._set_status("mouse_error", f"Mouse movement failed: {error}")
                return False

        # On Windows, prioritize direct Win32 SetCursorPos for sub-millisecond,
        # non-virtualized physical pixel accuracy on high-DPI displays
        if absolute and sys.platform.startswith("win"):
            try:
                if ctypes.windll.user32.SetCursorPos(int(x), int(y)):
                    return True
            except Exception:
                pass

        try:
            self.mouse.move(x, y, absolute=absolute)
            return True
        except Exception as error:
            try:
                if absolute and sys.platform.startswith("win") and ctypes.windll.user32.SetCursorPos(int(x), int(y)):
                    return True
            except Exception:
                pass
            self._set_status("mouse_error", f"Mouse movement failed: {error}")
            return False

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
        init_dpi_awareness()
        while not self.stop_event.is_set():
            if self.queue is None:
                time.sleep(0.05)
                continue

            try:
                sample = self.queue.get(timeout=0.05)
            except Empty:
                if not self.stop_event.is_set() and self.clock() - self._last_input_at > .5:
                    self._pause(cancel_gestures=True)
                    self._set_status("waiting", "No fresh tracking samples; check the webcam/vision worker")
                self._report_status()
                continue

            try:
                skipped = False
                interrupted = False
                while True:
                    try:
                        latest = self.queue.get_nowait()
                        interrupted = interrupted or self._interrupts_movement(sample)
                        sample = latest
                        skipped = True
                    except Empty:
                        break
                if skipped:
                    # Missing frames break gesture continuity, but dropping
                    # healthy measurements must not restart cursor recovery.
                    self._cancel_gestures()
                    if interrupted:
                        self._pause()
                    elif self.kalman is not None and not self._paused:
                        self.kalman.pause()
                self.update(sample)
            except Exception as error:
                logging.exception("Error in mouse controller update: %s", error)

    def _cancel_wink(self, blocked=False):
        self._wink_candidate = None
        self._wink_since = None
        self._wink_frames = 0
        self._wink_qualified = False
        self._open_since = None
        self.wink_debounce_flag = blocked

    def _cancel_gestures(self):
        self._cancel_wink(blocked=True)
        self._tilt_candidate = self._tilt_since = None
        self.debounce_flag = True

    def _interrupts_movement(self, sample):
        values = (sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll)
        if (not sample.tracking_valid or not sample.eyes_open
                or sample.left_winking or sample.right_winking
                or not all(math.isfinite(value) for value in values)):
            return True
        return np.max(np.abs(angle_delta(values[2:], self.neutral_head))) > self.max_head_deviation

    def _pause(self, cancel_gestures=False):
        self._paused = True
        self._recovery.clear()
        if self.kalman is not None:
            self.kalman.pause()
        if cancel_gestures:
            self._cancel_gestures()

    def update(self, sample):
        now = self.clock()
        self._last_input_at = now
        values = (sample.eye_x, sample.eye_y, sample.pitch, sample.yaw,
                  sample.roll, sample.timestamp)
        if (not all(math.isfinite(value) for value in values)
                or not -0.05 <= now - sample.timestamp <= self.max_sample_age
                or (self.last_seen_time is not None and sample.timestamp <= self.last_seen_time)):
            self._pause(cancel_gestures=True)
            self._set_status("invalid_sample", "Stale, out-of-order or invalid tracking sample; waiting for fresh data")
            return
        if self.last_seen_time is not None and sample.timestamp - self.last_seen_time > self.max_sample_age:
            self._pause(cancel_gestures=True)
        self.last_seen_time = sample.timestamp
        if not sample.tracking_valid:
            if self.kalman is not None:
                self.kalman.reset()
            self.last_sample_time = None
            self._pause(cancel_gestures=True)
            self._set_status("tracking_lost", "Face/iris tracking unavailable; face the camera with both eyes visible")
            return

        pitch, yaw, roll = angle_delta((sample.pitch, sample.yaw, sample.roll), self.neutral_head)
        if max(abs(pitch), abs(yaw)) > self.max_head_deviation:
            self._pause(cancel_gestures=True)
            self._set_status("head_pose", f"Head pose differs from calibration (pitch {pitch:+.1f}, "
                             f"yaw {yaw:+.1f}, roll {roll:+.1f} degrees); return to your calibrated posture")
            return
        left, right = sample.left_winking, sample.right_winking
        is_blinking = getattr(sample, "is_blinking", False) or (not sample.eyes_open and not left and not right)
        if not sample.eyes_open or left or right:
            self._pause()
            self._set_status("eyes_closed", "Eyes closed or wink detected; waiting for both eyes to reopen")
            self._tilt_candidate = self._tilt_since = None
            if abs(roll) <= self.max_head_deviation:
                self.winkClick(left, right, eyes_open=False, timestamp=sample.timestamp, is_blinking=is_blinking)
            else:
                self._cancel_wink(blocked=True)
            return

        self.headClick(roll, timestamp=sample.timestamp)
        if abs(roll) > self.max_head_deviation:
            self._pause()
            self._cancel_wink(blocked=True)
            self._set_status("head_pose", f"Head roll differs from calibration by {roll:+.1f} degrees; return to neutral")
            return
        self.winkClick(False, False, eyes_open=True, timestamp=sample.timestamp)
        if self.is_frozen or self._wink_candidate is not None:
            self._pause()
            self._set_status("gesture_pause", "Click cooldown or wink reopening confirmation in progress")
            return
        if self.mapper is None:
            self._set_status("no_mapping", "No calibrated gaze mapping is available")
            return

        eye_x, eye_y = sample.eye_x, sample.eye_y
        if self.head_pose_compensation:
            eye_x, eye_y = compensate_head_pose(
                eye_x, eye_y, sample.pitch, sample.yaw,
                self.neutral_head, yaw_gain=self.yaw_gain, pitch_gain=self.pitch_gain
            )
        point = self.mapper.map_coordinates(eye_x, eye_y)
        dt = None if self.last_sample_time is None else sample.timestamp - self.last_sample_time
        if dt is not None and dt > self.max_sample_age:
            dt = self.max_sample_age
        if self._paused:
            self._recovery.append((sample.timestamp, point))
            # Require a short run of open eyes and use its median so one noisy
            # reopening observation cannot immediately relocate the cursor.
            if len(self._recovery) < 3 or sample.timestamp - self._recovery[0][0] < 0.1:
                self._set_status("recovering", "Reacquiring stable open-eye tracking")
                return
            point = np.median([item[1] for item in self._recovery], axis=0)
            if dt is not None and dt > 1.0:
                self.smoother.reset()
                if self.last_position is not None:
                    self.smoother.update(*self.last_position)
            dt = self.nominal_dt
            self._paused = False
            self._recovery.clear()

        if self.kalman is not None:
            smooth_x, smooth_y = self.smoother.update(*point, dt=dt)
        else:
            smooth_x, smooth_y = self.smoother.update(*point)
        self.last_sample_time = sample.timestamp
        smooth_x = max(self.mapper.min_x, min(self.mapper.max_x, smooth_x))
        smooth_y = max(self.mapper.min_y, min(self.mapper.max_y, smooth_y))
        if self.move(int(smooth_x), int(smooth_y), absolute=True):
            self.last_position = (smooth_x, smooth_y)
            self._set_status("moving", "Cursor movement active")

    def _click_if_ready(self, button):
        now = self.clock()
        if now - self.last_click_time < self.cooldown:
            return
        self._trigger_click(button)
        self.last_click_time = now
        self._pause()

    def headClick(self, d_roll, timestamp=None):
        timestamp = self.clock() if timestamp is None else timestamp
        if abs(d_roll) > self.max_roll_degrees:
            self.debounce_flag = True
            self._tilt_candidate = self._tilt_since = None
            return
        if abs(d_roll) < self.reset_threshold:
            self.debounce_flag = False
            self._tilt_candidate = self._tilt_since = None
            return
        if self.debounce_flag:
            return
        direction = 1 if d_roll > self.tilt_threshold else -1 if d_roll < -self.tilt_threshold else None
        if direction is None:
            self._tilt_candidate = self._tilt_since = None
        elif direction != self._tilt_candidate:
            self._tilt_candidate, self._tilt_since = direction, timestamp
        elif timestamp - self._tilt_since >= self.hold_seconds:
            self._click_if_ready("double_left" if direction == 1 else "left")
            self.debounce_flag = True

    def winkClick(self, left_winking=False, right_winking=False, is_blinking=False, *, eyes_open=None, timestamp=None):
        """Handle wink clicking, supporting both immediate mode and hold-and-reopen mode.

        If eyes_open is None, operates in immediate mode (fires upon wink detection with debounce,
        freezes for freeze_duration on blinks).
        If eyes_open is provided (bool), operates in hold-and-reopen mode (qualifies held wink and
        triggers click upon stable reopening).
        """
        current_time = timestamp if timestamp is not None else self.clock()

        # Immediate mode (eyes_open is not explicitly passed)
        if eyes_open is None:
            # Explicit rule: Blinking is ignored and never counted as a click.
            # Freeze the mouse for freeze_duration to prevent mouse drag.
            if is_blinking or (left_winking and right_winking):
                self.last_blink_time = current_time
                self.wink_debounce_flag = True
                try:
                    self.freeze_pos = self.mouse.get_position()
                except Exception:
                    self.freeze_pos = None
                return

            # Suppress any clicks while post-blink freeze is active
            if (current_time - self.last_blink_time) < self.freeze_duration:
                return

            # 1. Left Wink -> Single Left Click
            if left_winking and not self.wink_debounce_flag:
                try:
                    self.freeze_pos = self.mouse.get_position()
                except Exception:
                    self.freeze_pos = None
                self._trigger_click("left")
                self.last_click_time = current_time
                self.wink_debounce_flag = True

            # 2. Right Wink -> Double Left Click
            elif right_winking and not self.wink_debounce_flag:
                try:
                    self.freeze_pos = self.mouse.get_position()
                except Exception:
                    self.freeze_pos = None
                self._trigger_click("double_left")
                self.last_click_time = current_time
                self.wink_debounce_flag = True

            # 3. Release: Both eyes are open, allow next click
            elif not left_winking and not right_winking and not is_blinking:
                self.wink_debounce_flag = False

            if not self.is_frozen:
                self.freeze_pos = None
            return

        # Hold-and-reopen mode (eyes_open is True or False)
        if is_blinking or (left_winking and right_winking):
            self.last_blink_time = current_time
            self._cancel_wink(blocked=True)
            try:
                self.freeze_pos = self.mouse.get_position()
            except Exception:
                self.freeze_pos = None
            return

        if (current_time - self.last_blink_time) < self.freeze_duration:
            return

        if eyes_open and not left_winking and not right_winking and not is_blinking:
            if self._open_since is None:
                self._open_since = current_time
            if current_time - self._open_since >= self.settle_seconds:
                if self._wink_qualified and not self.wink_debounce_flag:
                    self._click_if_ready(self._wink_candidate)
                self._cancel_wink()
            return
        if left_winking == right_winking:
            self._cancel_wink(blocked=True)
            return
        if self.wink_debounce_flag:
            self._open_since = None
            return
        direction = "left" if left_winking else "double_left"
        if self._open_since is not None and self._wink_candidate is not None:
            self._cancel_wink(blocked=True)
            return
        self._open_since = None
        if self._wink_candidate is not None and direction != self._wink_candidate:
            self._cancel_wink(blocked=True)
            return
        if self._wink_candidate is None:
            self._wink_candidate, self._wink_since = direction, current_time
        self._wink_frames += 1
        self._wink_qualified = (self._wink_frames >= self.closed_eye_frames
                                and current_time - self._wink_since >= self.hold_seconds)
