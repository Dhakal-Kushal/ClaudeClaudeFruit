from collections import deque

import numpy as np


def validate_calibration_points(points):
    """Validate four ordered corners followed by a strictly interior center."""
    points = np.array(points, dtype=float, copy=True)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Calibration needs five finite 2D points")
    scale = float(np.ptp(points[:4], axis=0).max())
    if scale <= 0:
        raise ValueError("Calibration points overlap; recalibrate")

    corners = (points[:4] - points[4]) / scale
    following = np.roll(corners, -1, axis=0)
    edges = following - corners
    next_edges = np.roll(edges, -1, axis=0)
    areas = corners[:, 0] * following[:, 1] - corners[:, 1] * following[:, 0]
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    tolerance = 1e-10
    if not (
        ((areas > tolerance).all() and (turns > tolerance).all())
        or ((areas < -tolerance).all() and (turns < -tolerance).all())
    ):
        raise ValueError("Calibration corners must form a convex polygon enclosing the center")
    bases = np.stack((corners, following), axis=-1)
    if (np.linalg.cond(bases) > 1e6).any():
        raise ValueError("Calibration triangles are too narrow; recalibrate")
    return points


class PiecewiseAffineMapping2D:
    """Continuous interpolation through four corners and a calibrated center.

    Each triangle joins the center to adjacent corners. The same affine maps
    extend beyond the polygon, so callers can clamp to their display bounds.
    Near sector boundaries (diagonal corner rays), cubic Hermite angular
    blending transitions smoothly between adjacent triangles to eliminate
    C^1 derivative kinks and cursor velocity jumps.
    """

    def __init__(self, source_points, target_points, blend_width: float = 0.04):
        source = validate_calibration_points(source_points)
        target = validate_calibration_points(target_points)
        if (
            not isinstance(blend_width, (int, float))
            or isinstance(blend_width, bool)
            or not np.isfinite(blend_width)
            or blend_width < 0.0
            or blend_width >= 0.5
        ):
            raise ValueError("blend_width must be a finite float in [0.0, 0.5)")
        self._blend_width = float(blend_width)
        self._source_center = source[4]
        self._target_center = target[4]
        self._source_scale = float(np.ptp(source[:4], axis=0).max())
        offsets = (source[:4] - self._source_center) / self._source_scale
        source_bases = np.stack((offsets, np.roll(offsets, -1, axis=0)), axis=-1)
        self._inverse_bases = np.linalg.inv(source_bases)
        target_offsets = target[:4] - self._target_center
        self._target_bases = np.stack(
            (target_offsets, np.roll(target_offsets, -1, axis=0)), axis=-1
        )

    def map_coordinates(self, x: float, y: float) -> tuple[float, float]:
        point = np.asarray([x, y], dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("coordinates must be finite scalars")
        with np.errstate(over="ignore", invalid="ignore"):
            offset = (point - self._source_center) / self._source_scale
            weights = self._inverse_bases @ offset
            # The containing angular sector has two nonnegative coefficients.
            # A sum above one extrapolates past its outer calibration edge.
            sector = int(np.argmax(weights.min(axis=1)))
            result = self._target_center + self._target_bases[sector] @ weights[sector]

            if self._blend_width > 0:
                w0, w1 = weights[sector]
                w_sum = w0 + w1
                if w_sum > 1e-12:
                    q0 = max(0.0, float(w0)) / w_sum
                    q1 = max(0.0, float(w1)) / w_sum
                    if q0 < self._blend_width:
                        adjacent = (sector + 1) % 4
                        t = 0.5 * (1.0 - q0 / self._blend_width)
                        alpha = t * t * (3.0 - 2.0 * t)
                        adj_result = self._target_center + self._target_bases[adjacent] @ weights[adjacent]
                        result = (1.0 - alpha) * result + alpha * adj_result
                    elif q1 < self._blend_width:
                        adjacent = (sector - 1) % 4
                        t = 0.5 * (1.0 - q1 / self._blend_width)
                        alpha = t * t * (3.0 - 2.0 * t)
                        adj_result = self._target_center + self._target_bases[adjacent] @ weights[adjacent]
                        result = (1.0 - alpha) * result + alpha * adj_result

        if not np.isfinite(result).all():
            raise ValueError("coordinates exceed the mapping's numeric range")
        return float(result[0]), float(result[1])


class CoordinateAverager2D:
    """Average the latest N mapped screen-coordinate samples."""

    def __init__(self, window_size: int = 5) -> None:
        if not isinstance(window_size, (int, np.integer)) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        self.window_size = int(window_size)
        self.reset()

    def reset(self) -> None:
        self._coordinates = deque(maxlen=self.window_size)

    def update(self, x: float, y: float) -> tuple[float, float]:
        coordinates = np.asarray([x, y], dtype=float)
        if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite scalars")
        self._coordinates.append(coordinates)
        average = np.mean(self._coordinates, axis=0)
        return float(average[0]), float(average[1])


class KalmanFilter2D:
    """Smooth pixel coordinates with a constant-velocity Kalman model."""

    # Set the baseline tuning parameters. 
    # dt = expected time between frames.
    # process_noise = how much the filter trusts the physics model.
    # measurement_noise = how much the filter trusts the webcam measurements.
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise: float = 10000.0,
        measurement_noise: float = 25.0,
    ) -> None:
        parameters = np.array([dt, process_noise, measurement_noise], dtype=float)
        if not np.isfinite(parameters).all() or (parameters <= 0).any():
            raise ValueError("dt and noise variances must be finite and positive")

        self._dt, self._process_noise, self._measurement_noise = parameters
        self._observation = np.eye(2, 4)
        self._measurement_covariance = np.eye(2) * self._measurement_noise
        self.reset()

    # Clear the filter's memory. Should be triggered when the vision thread loses tracking (eg. the user blinks or looks away)
    def reset(self) -> None:
        """Discard tracking history after recalibration or loss of tracking."""
        self._state = None
        self._covariance = None
        self._last_measurement = None
        self._last_output = None

    def pause(self) -> None:
        """Keep position but discard motion through a blink or gesture."""
        if self._state is not None:
            self._state[:2] = self._last_output
            self._state[2:] = 0

    # Update the filter with a new measurement and return the smoothed coordinates.
    def update(self, x: float, y: float, dt: float | None = None) -> tuple[float, float]:
        """Return smoothed coordinates; the first sample is returned unchanged."""
        measurement = np.array([x, y], dtype=float)
        interval = self._dt if dt is None else float(dt)
        if measurement.shape != (2,) or not np.isfinite(measurement).all():
            raise ValueError("coordinates must be finite scalars")
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("dt must be finite and positive")

        # if filter has just begun or has been reset, init the state and covariance
        if self._state is None:
            self._state = np.array([x, y, 0.0, 0.0], dtype=float)
            self._last_measurement = measurement.copy()
            self._last_output = measurement.copy()
            position_variance = self._measurement_noise
            velocity_variance = position_variance / interval**2
            self._covariance = np.diag(
                [position_variance, position_variance, velocity_variance, velocity_variance]
            )
            return float(x), float(y)

        # predict where the eye is currently looking at
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = interval
        acceleration = np.array(
            [
                [0.5 * interval**2, 0.0],
                [0.0, 0.5 * interval**2],
                [interval, 0.0],
                [0.0, interval],
            ]
        )
        process_covariance = (
            self._process_noise * acceleration @ acceleration.T
        )
        predicted_state = transition @ self._state
        predicted_covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )
        # calculate the distance between the filter's prediction and the actual measurement
        innovation = measurement - self._observation @ predicted_state
        innovation_covariance = (
            self._observation @ predicted_covariance @ self._observation.T
            + self._measurement_covariance
        )
        # calculate dynamic weighting ratio.
        # i.e. if webcam measurement is noisy, trust the prected measurement more.
        gain = np.linalg.solve(
            innovation_covariance, self._observation @ predicted_covariance
        ).T

        # multiply the innovation by the gain and add it to the predicion.
        previous_position = self._last_output
        self._state = predicted_state + gain @ innovation
        # Gaze can stop abruptly. Do not let inferred velocity carry the
        # cursor past the observation or away from it when the target changes.
        bounded = np.clip(self._state[:2], np.minimum(previous_position, measurement),
                          np.maximum(previous_position, measurement))
        constrained = bounded != self._state[:2]
        # A noisy observation can cross a moving estimate. Retain velocity
        # unless successive observations also indicate a stopped target.
        if np.linalg.norm(measurement - self._last_measurement) < 1e-6:
            self._state[2:][constrained] = 0
            self._state[:2] = bounded
        elif constrained.any():
            max_dev = 2.0 * np.sqrt(self._measurement_noise)
            self._state[:2] = np.clip(self._state[:2], bounded - max_dev, bounded + max_dev)
        self._last_measurement = measurement.copy()
        self._last_output = bounded.copy()
        residual = np.eye(4) - gain @ self._observation
        self._covariance = (
            residual @ predicted_covariance @ residual.T
            + gain @ self._measurement_covariance @ gain.T
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        return float(bounded[0]), float(bounded[1])
