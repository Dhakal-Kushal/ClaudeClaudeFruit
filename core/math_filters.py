import numpy as np


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
        self._state = predicted_state + gain @ innovation
        residual = np.eye(4) - gain @ self._observation
        self._covariance = (
            residual @ predicted_covariance @ residual.T
            + gain @ self._measurement_covariance @ gain.T
        )
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        return float(self._state[0]), float(self._state[1])