class GazeMapper:
    def __init__(self, profile):
        """
        Initializes the gaze mapper using the calibration profile.
        Requires 'eye_targets' and 'screen_targets' from the profile.
        """
        self.eye = profile["eye_targets"]
        self.screen = profile["screen_targets"]
        
        # Determine strict clamping boundaries from the calibrated screen targets
        screen_coords = list(self.screen.values())
        self.min_x = min(p[0] for p in screen_coords)
        self.max_x = max(p[0] for p in screen_coords)
        self.min_y = min(p[1] for p in screen_coords)
        self.max_y = max(p[1] for p in screen_coords)

    def map_coordinates(self, eye_x, eye_y):
        """
        Maps raw eye coordinates to screen coordinates using 4-quadrant linear interpolation.
        """
        c_e = self.eye["center"]
        c_s = self.screen["center"]

        # 1. Determine Quadrant relative to center
        is_left = eye_x < c_e[0]
        is_top = eye_y < c_e[1]

        if is_top:
            q = "top_left" if is_left else "top_right"
        else:
            q = "bottom_left" if is_left else "bottom_right"

        corner_e = self.eye[q]
        corner_s = self.screen[q]

        # 2. Calculate relative ratios
        dx_e = corner_e[0] - c_e[0]
        dy_e = corner_e[1] - c_e[1]

        # Protect against division by zero if calibration points exactly overlap
        ratio_x = (eye_x - c_e[0]) / dx_e if dx_e != 0 else 0
        ratio_y = (eye_y - c_e[1]) / dy_e if dy_e != 0 else 0

        # 3. Map to Screen Coordinates
        screen_x = c_s[0] + ratio_x * (corner_s[0] - c_s[0])
        screen_y = c_s[1] + ratio_y * (corner_s[1] - c_s[1])

        # 4. Clamp to monitor bounds
        screen_x = max(self.min_x, min(self.max_x, screen_x))
        screen_y = max(self.min_y, min(self.max_y, screen_y))

        return screen_x, screen_y
