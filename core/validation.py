"""Held-out gaze measurements. Validation never changes the fitted mapping."""
from copy import deepcopy
from datetime import datetime, timezone
from time import monotonic

import numpy as np

from core.calibration import angle_delta, compensate_head_pose, validate_profile
from core.gaze_mapper import GazeMapper


def quality_warnings(report, monitor):
    """Advisory accuracy targets, not calibrated hardware acceptance limits.

    Evaluate held-out raw mapping errors, so smoothing cannot hide systematic
    bias. Show these in the UI; complete, explicitly accepted reports remain
    usable even when a webcam cannot meet these targets.
    """
    diagonal = float(np.hypot(monitor["width"], monitor["height"]))
    errors = []
    for key, fraction, label in (("median_error_px", .025, "Median error"),
                                  ("p90_error_px", .05, "90th-percentile error")):
        if report[key] > diagonal * fraction:
            errors.append(f"{label} {report[key]:.0f} px exceeds {diagonal * fraction:.0f} px")
    for index, target in enumerate(report["targets"], 1):
        bias = float(np.linalg.norm(target["bias_px"]))
        if bias > diagonal * .03:
            errors.append(f"Target {index} bias {bias:.0f} px exceeds {diagonal * .03:.0f} px")
        if target["rms_jitter_px"] > diagonal * .02:
            errors.append(f"Target {index} jitter {target['rms_jitter_px']:.0f} px exceeds {diagonal * .02:.0f} px")
    return errors


def validation_targets(monitor):
    fractions = ((.25, .25), (.75, .25), (.75, .75), (.25, .75),
                 (.5, .1), (.9, .5), (.5, .9), (.1, .5))
    return {f"validation_{i + 1}": [monitor["x"] + x * (monitor["width"] - 1),
                                    monitor["y"] + y * (monitor["height"] - 1)]
            for i, (x, y) in enumerate(fractions)}


class ValidationSession:
    def __init__(self, profile, settle_seconds=.75, capture_seconds=1.0, min_samples=15):
        validate_profile(profile)
        if (not np.isfinite([settle_seconds, capture_seconds]).all()
                or settle_seconds < 0 or capture_seconds <= 0 or min_samples < 15):
            raise ValueError("Invalid validation timing or minimum sample count")
        self.profile = deepcopy(profile)
        self.mapper = GazeMapper(self.profile)
        self.targets = validation_targets(profile["monitor"])
        for point in self.targets.values():
            if any(np.allclose(point, anchor) for anchor in profile["screen_targets"].values()):
                raise ValueError("Validation targets must be independent of calibration targets")
        self.settle_seconds = settle_seconds
        self.capture_seconds = capture_seconds
        self.min_samples = min_samples
        self.step_index = 0
        self.results = []
        self.accepted_samples = 0
        self.rejected_samples = 0
        self._last_seen = None
        pose_settings = self.profile.get("head_pose_compensation", {}) if isinstance(self.profile, dict) else {}
        self.head_pose_compensation = pose_settings.get("enabled", True)
        self.yaw_gain = float(pose_settings.get("yaw_gain", 0.007))
        self.pitch_gain = float(pose_settings.get("pitch_gain", 0.007))
        self.interrupt()

    @property
    def complete(self):
        return self.step_index == len(self.targets)

    @property
    def step(self):
        return None if self.complete else list(self.targets)[self.step_index]

    def interrupt(self):
        self.started_at = None
        self.samples = []
        self.heads = []
        self.progress = 0.0
        self.status = "Waiting for valid tracking"

    def accept(self, sample, now=None):
        if self.complete:
            return
        now = monotonic() if now is None else now
        values = [sample.eye_x, sample.eye_y, sample.pitch, sample.yaw, sample.roll, sample.timestamp]
        if (not sample.tracking_valid or not sample.eyes_open
                or sample.left_winking or sample.right_winking
                or not np.isfinite(values).all() or not -.1 <= now - sample.timestamp <= .5):
            self.rejected_samples += 1
            self.interrupt()
            return
        if self._last_seen is not None:
            if sample.timestamp <= self._last_seen:
                self.rejected_samples += 1
                return
            if sample.timestamp - self._last_seen > .3:
                self.interrupt()
        self._last_seen = sample.timestamp
        head = np.array([sample.pitch, sample.yaw, sample.roll])
        neutral = [self.profile["neutral_head"][key] for key in ("pitch", "yaw", "roll")]
        if np.max(np.abs(angle_delta(head, neutral))) > 10:
            self.rejected_samples += 1
            self.interrupt()
            self.status = "Return to your relaxed head position"
            return
        self.accepted_samples += 1
        if self.started_at is None:
            self.started_at = sample.timestamp
        if sample.timestamp - self.started_at < self.settle_seconds:
            self.status = "Settle your gaze on the dot"
            return
        self.heads.append(head)
        if np.std(angle_delta(self.heads, self.heads[0]), axis=0).max() > 2.5:
            self.accepted_samples -= 1
            self.rejected_samples += 1
            self.interrupt()
            self.status = "Keep your head still - trying again"
            return
        eye_x, eye_y = sample.eye_x, sample.eye_y
        if self.head_pose_compensation:
            eye_x, eye_y = compensate_head_pose(
                eye_x, eye_y, sample.pitch, sample.yaw,
                self.profile["neutral_head"], yaw_gain=self.yaw_gain, pitch_gain=self.pitch_gain
            )
        point = self.mapper.map_coordinates(eye_x, eye_y)
        self.samples.append((sample.timestamp, point))
        self.progress = min((sample.timestamp - self.samples[0][0]) / self.capture_seconds, 1.0)
        self.status = "Measuring accuracy - keep looking at the dot"
        if self.progress < 1 or len(self.samples) < self.min_samples:
            return
        points = np.asarray([point for _, point in self.samples])
        target = np.asarray(self.targets[self.step])
        errors = np.linalg.norm(points - target, axis=1)
        self.results.append({
            "name": self.step, "target": target.tolist(), "sample_count": len(points),
            "capture_duration_seconds": self.samples[-1][0] - self.samples[0][0],
            "median_error_px": float(np.median(errors)),
            "p90_error_px": float(np.percentile(errors, 90)),
            "bias_px": (points.mean(axis=0) - target).tolist(),
            "rms_jitter_px": float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))),
            "errors": errors.tolist(),
        })
        self.step_index += 1
        self.interrupt()

    def build_report(self):
        if not self.complete:
            raise ValueError("Finish all independent validation targets before saving")
        errors = np.concatenate([result["errors"] for result in self.results])
        monitor = self.profile["monitor"]
        diagonal = np.hypot(monitor["width"], monitor["height"])
        report = {
            "method": "independent_targets_v1", "completed_at": datetime.now(timezone.utc).isoformat(),
            "accepted": False, "target_count": len(self.results), "sample_count": len(errors),
            "settle_seconds": self.settle_seconds, "capture_seconds": self.capture_seconds,
            "min_samples": self.min_samples,
            "median_error_px": float(np.median(errors)), "p90_error_px": float(np.percentile(errors, 90)),
            "median_error_percent_diagonal": float(np.median(errors) / diagonal * 100),
            "p90_error_percent_diagonal": float(np.percentile(errors, 90) / diagonal * 100),
            "accepted_samples": self.accepted_samples, "rejected_samples": self.rejected_samples,
            "rejected_sample_rate": self.rejected_samples / (self.accepted_samples + self.rejected_samples),
            "targets": [{key: value for key, value in result.items() if key != "errors"}
                        for result in self.results],
        }
        return report


def validate_report(report, profile):
    if report["method"] != "independent_targets_v1" or report["accepted"] is not True:
        raise ValueError("Validation results have not been accepted")
    targets = validation_targets(profile["monitor"])
    if (not np.isfinite([report["settle_seconds"], report["capture_seconds"]]).all()
            or report["settle_seconds"] < 0 or report["capture_seconds"] <= 0
            or type(report["min_samples"]) is not int or report["min_samples"] < 15):
        raise ValueError("Invalid validation capture settings")
    if report["target_count"] != len(targets) or len(report["targets"]) != len(targets):
        raise ValueError("Independent validation is incomplete")
    for result, (name, point) in zip(report["targets"], targets.items()):
        if (result["name"] != name or not np.array_equal(result["target"], point)
                or type(result["sample_count"]) is not int or result["sample_count"] < report["min_samples"]):
            raise ValueError("Missing independent validation samples")
        bias = np.asarray(result["bias_px"], dtype=float)
        if bias.shape != (2,) or not np.isfinite(bias).all():
            raise ValueError("Invalid validation bias")
        for key in ("median_error_px", "p90_error_px", "rms_jitter_px"):
            if not np.isfinite(result[key]) or result[key] < 0:
                raise ValueError("Invalid target accuracy metrics")
        if (not np.isfinite(result["capture_duration_seconds"])
                or result["capture_duration_seconds"] < report["capture_seconds"]
                or result["p90_error_px"] < result["median_error_px"]):
            raise ValueError("Incomplete validation capture or inconsistent percentiles")
        if any(np.allclose(point, anchor) for anchor in profile["screen_targets"].values()):
            raise ValueError("Validation targets overlap calibration targets")
    if report["sample_count"] != sum(result["sample_count"] for result in report["targets"]):
        raise ValueError("Inconsistent validation sample count")
    for key in ("median_error_px", "p90_error_px", "median_error_percent_diagonal",
                "p90_error_percent_diagonal", "rejected_sample_rate"):
        if not np.isfinite(report[key]) or report[key] < 0:
            raise ValueError("Invalid validation accuracy metrics")
    for key in ("accepted_samples", "rejected_samples", "sample_count"):
        if type(report[key]) is not int or report[key] < 0:
            raise ValueError("Invalid validation frame count")
    if report["accepted_samples"] < report["sample_count"]:
        raise ValueError("Inconsistent accepted sample count")
    total = report["accepted_samples"] + report["rejected_samples"]
    if not total or not np.isclose(report["rejected_sample_rate"], report["rejected_samples"] / total):
        raise ValueError("Invalid rejected sample rate")
    diagonal = np.hypot(profile["monitor"]["width"], profile["monitor"]["height"])
    for percentile in ("median", "p90"):
        if not np.isclose(report[f"{percentile}_error_percent_diagonal"],
                          report[f"{percentile}_error_px"] / diagonal * 100):
            raise ValueError("Inconsistent normalized accuracy")
    if report["p90_error_px"] < report["median_error_px"]:
        raise ValueError("Inconsistent validation percentiles")
