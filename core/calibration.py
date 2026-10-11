import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic

import numpy as np

from core.math_filters import validate_calibration_points


EYE_TARGETS = ("top_left", "top_right", "bottom_right", "bottom_left", "center")
CALIBRATION_STEPS = ("neutral_head",) + EYE_TARGETS
EYE_RELATIVE_COORDINATE_SPACE = "eye_relative_v1"
GAZE_MAPPING_METHOD = "piecewise_affine_v1"


def validate_camera_metadata(camera):
    for key in ("index", "width", "height", "backend"):
        value = camera[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid camera {key}")
    if min(camera["width"], camera["height"]) <= 0 or type(camera["flip"]) is not bool:
        raise ValueError("Missing capture dimensions or mirror setting")
    for key in ("fps", "fourcc"):
        value = camera[key]
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
            raise ValueError(f"Invalid camera {key}")


def camera_compatibility_errors(expected, actual):
    """Compare negotiated settings, allowing small nominal frame-rate rounding."""
    try:
        validate_camera_metadata(expected)
        validate_camera_metadata(actual)
    except (KeyError, TypeError, ValueError):
        return ["Complete camera capture settings are unavailable"]
    errors = []
    for key in ("index", "width", "height", "flip", "backend", "fourcc", "fps"):
        saved, current = expected[key], actual[key]
        matches = saved == current
        if key == "fps" and saved is not None and current is not None:
            matches = abs(saved - current) <= max(0.5, saved * 0.02)
        if not matches:
            errors.append(f"{key}: calibrated {saved}, current {current}")
    return errors


def profile_compatibility_errors(profile, monitor, camera_metadata):
    try:
        validate_profile(profile, monitor, require_validation=True)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return [f"A new calibration and validation are required: {error}"]
    if profile["mapping"]["coordinate_space"] != EYE_RELATIVE_COORDINATE_SPACE:
        return ["A new eye-relative calibration is required"]
    return camera_compatibility_errors(profile["camera"], camera_metadata)


@dataclass(frozen=True)
class CalibrationSample:
    """Queue payload; the live tracker expresses eye_x/y in eye-width units."""

    eye_x: float
    eye_y: float
    pitch: float
    yaw: float
    roll: float
    eyes_open: bool = True
    tracking_valid: bool = True
    timestamp: float = field(default_factory=monotonic)
    left_winking: bool = False
    right_winking: bool = False


class TrackingAdapter:
    """A separate calibration queue, safe to publish to from a tracking thread."""

    def __init__(self):
        self.samples = Queue(maxsize=180)
        self._lock = Lock()

    def publish(self, eye_x, eye_y, pitch, yaw, roll, **kwargs):
        sample = CalibrationSample(eye_x, eye_y, pitch, yaw, roll, **kwargs)
        with self._lock:
            try:
                self.samples.put_nowait(sample)
            except Full:
                try:
                    self.samples.get_nowait()
                except Empty:
                    pass
                try:
                    self.samples.put_nowait(sample)
                except Full:
                    pass


def angle_delta(angles, baseline):
    return (np.asarray(angles) - np.asarray(baseline) + 180.0) % 360.0 - 180.0


def screen_targets(monitor, inset=24):
    width, height = monitor["width"], monitor["height"]
    left, top = monitor.get("x", 0), monitor.get("y", 0)
    return {
        "top_left": [left + inset, top + inset],
        "top_right": [left + width - 1 - inset, top + inset],
        "bottom_right": [left + width - 1 - inset, top + height - 1 - inset],
        "bottom_left": [left + inset, top + height - 1 - inset],
        "center": [left + (width - 1) / 2, top + (height - 1) / 2],
    }


def screen_targets_in_canvas(monitor, canvas_width, canvas_height):
    """Map monitor-coordinate targets into the calibration canvas dimensions."""
    targets = screen_targets(monitor)
    left, top = monitor.get("x", 0), monitor.get("y", 0)
    return {
        name: [
            (point[0] - left) * canvas_width / monitor["width"],
            (point[1] - top) * canvas_height / monitor["height"],
        ]
        for name, point in targets.items()
    }


def validate_eye_targets(targets):
    validate_calibration_points([targets[name] for name in EYE_TARGETS])


class CalibrationSession:
    def __init__(self, monitor, countdown=2.0, capture_seconds=1.5, min_samples=15, screen_positions=None):
        if countdown < 0 or capture_seconds <= 0 or min_samples < 2:
            raise ValueError("Invalid calibration timing")
        self.monitor = dict(monitor)
        self.screen_positions = screen_positions if screen_positions is not None else screen_targets(monitor)
        self.countdown = countdown
        self.capture_seconds = capture_seconds
        self.min_samples = min_samples
        self.step_index = 0
        self.eye_targets = {}
        self.neutral_head = None
        self.interrupt()

    @property
    def complete(self):
        return self.step_index == len(CALIBRATION_STEPS)

    @property
    def step(self):
        return None if self.complete else CALIBRATION_STEPS[self.step_index]

    def interrupt(self):
        self.started_at = None
        self.last_timestamp = None
        self.samples = []
        self.progress = 0.0
        self.status = "Waiting for valid tracking"

    def accept(self, sample, now=None):
        if self.complete:
            return
        now = monotonic() if now is None else now
        values = [sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll, sample.timestamp]
        if (
            not sample.tracking_valid or not sample.eyes_open
            or not np.isfinite(values).all()
            or not -0.1 <= now - sample.timestamp <= 0.5
        ):
            self.interrupt()
            return
        if self.last_timestamp is not None:
            if sample.timestamp <= self.last_timestamp:
                return
            if sample.timestamp - self.last_timestamp > 0.3:
                self.interrupt()
        if self.started_at is None:
            self.started_at = sample.timestamp
        self.last_timestamp = sample.timestamp
        head = (sample.pitch, sample.yaw, sample.roll)
        if self.neutral_head is not None and np.abs(angle_delta(head, self.neutral_head)).max() > 10:
            self.interrupt()
            self.status = "Return to your relaxed head position"
            return
        elapsed = sample.timestamp - self.started_at
        if elapsed < self.countdown:
            self.status = f"Hold your gaze · {int(np.ceil(self.countdown - elapsed))}"
            return
        self.samples.append(sample)
        heads = np.array([(item.pitch, item.yaw, item.roll) for item in self.samples])
        if np.std(angle_delta(heads, heads[0]), axis=0).max() > 2.5:
            self.interrupt()
            self.status = "Keep your head still — trying again"
            return
        duration = self.samples[-1].timestamp - self.samples[0].timestamp
        self.progress = min(duration / self.capture_seconds, 1.0)
        self.status = "Capturing — keep looking at the dot"
        if self.progress < 1 or len(self.samples) < self.min_samples:
            return
        if self.step == "neutral_head":
            radians = np.deg2rad(heads)
            self.neutral_head = np.rad2deg(
                np.arctan2(np.sin(radians).mean(axis=0), np.cos(radians).mean(axis=0))
            ).tolist()
        else:
            eyes = np.array([(item.eye_x, item.eye_y) for item in self.samples])
            self.eye_targets[self.step] = np.median(eyes, axis=0).tolist()
        self.step_index += 1
        self.interrupt()

    def build_profile(self, trigger_degrees=15.0, camera=None, coordinate_space=EYE_RELATIVE_COORDINATE_SPACE):
        if not self.complete:
            raise ValueError("Finish every calibration test before saving")
        validate_eye_targets(self.eye_targets)
        profile = {
            "schema_version": 2,
            "calibrated": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "monitor": self.monitor,
            "camera": camera or {"source": "tracking_queue"},
            "eye_targets": self.eye_targets,
            "screen_targets": self.screen_positions,
            "neutral_head": dict(zip(("pitch", "yaw", "roll"), self.neutral_head)),
            "mapping": {"method": GAZE_MAPPING_METHOD, "coordinate_space": coordinate_space},
            "gesture_settings": {
                "trigger_degrees": float(trigger_degrees),
                "release_degrees": float(trigger_degrees) / 2,
                "hold_ms": 150,
                "cooldown_ms": 500,
                "click_settle_ms": 250,
                "closed_eye_frames": 3,
                "max_roll_degrees": 25.0,
                "bindings": {},
            },
            "smoothing": {"method": "kalman", "process_noise": 10000.0, "measurement_noise": 25.0},
        }
        validate_profile(profile)
        return profile


def validate_profile(profile, monitor=None, *, require_validation=False):
    if profile["schema_version"] != 2 or profile["calibrated"] is not True:
        raise ValueError("Calibration is incomplete")
    saved_monitor = profile["monitor"]
    for key in ("width", "height", "x", "y"):
        value = saved_monitor[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Invalid monitor information")
    if min(saved_monitor["width"], saved_monitor["height"]) <= 48:
        raise ValueError("Invalid monitor resolution")
    if monitor and any(saved_monitor[key] != monitor[key] for key in ("width", "height", "x", "y")):
        raise ValueError("The monitor setup changed; recalibrate")
    validate_eye_targets(profile["eye_targets"])
    validate_eye_targets(profile["screen_targets"])
    for name in EYE_TARGETS:
        point = np.asarray(profile["screen_targets"][name], dtype=float)
        minimum = np.array([saved_monitor["x"], saved_monitor["y"]])
        maximum = minimum + [saved_monitor["width"] - 1, saved_monitor["height"] - 1]
        if point.shape != (2,) or (point < minimum).any() or (point > maximum).any():
            raise ValueError("Screen calibration targets are incomplete")
    head = np.array([profile["neutral_head"][key] for key in ("pitch", "yaw", "roll")], dtype=float)
    if head.shape != (3,) or not np.isfinite(head).all():
        raise ValueError("Neutral head calibration is incomplete")
    if not isinstance(profile["camera"], dict) or not profile["camera"]:
        raise ValueError("Camera information is missing")
    if profile["mapping"]["method"] != GAZE_MAPPING_METHOD or not profile["mapping"]["coordinate_space"]:
        raise ValueError("Mapping information is missing")
    gestures = profile["gesture_settings"]
    for key in ("trigger_degrees", "release_degrees", "hold_ms", "cooldown_ms", "click_settle_ms", "closed_eye_frames", "max_roll_degrees"):
        if not np.isfinite(gestures[key]) or gestures[key] <= 0:
            raise ValueError("Gesture settings must be finite and positive")
    if gestures["release_degrees"] >= gestures["trigger_degrees"] or not isinstance(gestures["bindings"], dict):
        raise ValueError("Gesture settings are inconsistent")
    smoothing = profile["smoothing"]
    if smoothing["method"] not in ("kalman", "moving_average"):
        raise ValueError("Smoothing settings are missing")
    if smoothing["method"] == "kalman":
        for key in ("process_noise", "measurement_noise"):
            if not np.isfinite(smoothing[key]) or smoothing[key] <= 0:
                raise ValueError("Invalid smoothing settings")
    elif (not isinstance(smoothing.get("window_size"), int)
          or isinstance(smoothing.get("window_size"), bool) or smoothing["window_size"] < 1):
        raise ValueError("Invalid moving-average window size")
    if require_validation:
        validate_camera_metadata(profile["camera"])
        # Local import avoids a cycle: validation uses the gaze mapper.
        from core.validation import validate_report
        validate_report(profile["validation"], profile)


def load_profile(path, monitor=None):
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_profile(profile, monitor, require_validation=True)
        return profile
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def save_profile(path, profile):
    validate_profile(profile, require_validation=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(profile, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
