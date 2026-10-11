import math
import numpy as np


class GazeMapper:
    def __init__(self, profile):
        self.eye = profile["eye_targets"]
        self.screen = profile["screen_targets"]
        monitor = profile.get("monitor", {})
        if "width" in monitor and "height" in monitor:
            left = monitor.get("x", 0)
            top = monitor.get("y", 0)
            self.min_x = float(left)
            self.max_x = float(left + monitor["width"])
            self.min_y = float(top)
            self.max_y = float(top + monitor["height"])
        else:
            screen_coords = list(self.screen.values())
            self.min_x = float(min(p[0] for p in screen_coords))
            self.max_x = float(max(p[0] for p in screen_coords))
            self.min_y = float(min(p[1] for p in screen_coords))
            self.max_y = float(max(p[1] for p in screen_coords))

    def map_coordinates(self, eye_x, eye_y):
        c_e = self.eye["center"]
        c_s = self.screen["center"]
        is_left = eye_x < c_e[0]
        is_top = eye_y < c_e[1]

        if is_top:
            q = "top_left" if is_left else "top_right"
        else:
            q = "bottom_left" if is_left else "bottom_right"

        corner_e = self.eye[q]
        corner_s = self.screen[q]

        dx_e = corner_e[0] - c_e[0]
        dy_e = corner_e[1] - c_e[1]

        MIN_EYE_DELTA = 0.5
        ratio_x = (eye_x - c_e[0]) / dx_e if abs(dx_e) >= MIN_EYE_DELTA else 0.0
        ratio_y = (eye_y - c_e[1]) / dy_e if abs(dy_e) >= MIN_EYE_DELTA else 0.0

        screen_x = c_s[0] + ratio_x * (corner_s[0] - c_s[0])
        screen_y = c_s[1] + ratio_y * (corner_s[1] - c_s[1])

        screen_x = max(self.min_x, min(self.max_x, screen_x))
        screen_y = max(self.min_y, min(self.max_y, screen_y))
        return float(screen_x), float(screen_y)
