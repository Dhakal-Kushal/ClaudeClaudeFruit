from core.calibration import EYE_TARGETS
from core.math_filters import PiecewiseAffineMapping2D


class GazeMapper:
    """Continuously map calibrated gaze to display pixels before smoothing."""

    def __init__(self, profile):
        self.eye = profile["eye_targets"]
        self.screen = profile["screen_targets"]
        blend_width = profile.get("mapping", {}).get("blend_width", 0.04)
        self._mapping = PiecewiseAffineMapping2D(
            [self.eye[name] for name in EYE_TARGETS],
            [self.screen[name] for name in EYE_TARGETS],
            blend_width=blend_width,
        )
        monitor = profile.get("monitor", {})
        if "width" in monitor and "height" in monitor:
            left = monitor.get("x", 0)
            top = monitor.get("y", 0)
            self.min_x = float(left)
            self.max_x = float(left + monitor["width"] - 1)
            self.min_y = float(top)
            self.max_y = float(top + monitor["height"] - 1)
        else:
            screen_coords = list(self.screen.values())
            self.min_x = float(min(p[0] for p in screen_coords))
            self.max_x = float(max(p[0] for p in screen_coords))
            self.min_y = float(min(p[1] for p in screen_coords))
            self.max_y = float(max(p[1] for p in screen_coords))

    def map_coordinates(self, eye_x, eye_y):
        screen_x, screen_y = self._mapping.map_coordinates(eye_x, eye_y)
        screen_x = max(self.min_x, min(self.max_x, screen_x))
        screen_y = max(self.min_y, min(self.max_y, screen_y))
        return float(screen_x), float(screen_y)
