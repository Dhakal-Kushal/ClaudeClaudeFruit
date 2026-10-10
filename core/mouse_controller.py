import time 
import mouse
import ctypes

class MouseController:
    def __init__(self, tilt_threshold=15.0, reset_threshold=5.0, freeze_duration=0.5):
        # Configurable settings
        self.tilt_threshold = tilt_threshold    # Degrees of roll to trigger click
        self.reset_threshold = reset_threshold  # Degrees of roll to allow next click
        self.freeze_duration = freeze_duration  # Seconds to pause movement after click
        self.sensitivity = 30                   # Multiplier for head movement distance
        
        user32 = ctypes.windll.user32               # Gives our python script access to windows 
        self.screen_w = user32.GetSystemMetrics(0) 
        self.screen_h = user32.GetSystemMetrics(1)
        print(f"Screen resolution detected: {self.screen_w}x{self.screen_h}")
        
        # Internal state tracking
        self.last_click_time = 0
        self.debounce_flag = False

    def update(self, d_pitch, d_yaw, d_roll):
        """Called every frame by the vision processor."""
        current_time = time.time()
        is_frozen = (current_time - self.last_click_time) < self.freeze_duration

        # 1. Handle Movement (only if not frozen)
        if not is_frozen:
            # Map head turn (yaw) to X, and head nod (pitch) to Y
            target_x = int((self.screen_w / 2) + (d_yaw * self.sensitivity))
            target_y = int((self.screen_h / 2) + (d_pitch * self.sensitivity))
            
            # Keep the cursor inside the screen boundaries
            target_x = max(0, min(self.screen_w, target_x))
            target_y = max(0, min(self.screen_h, target_y))
            
            mouse.move(target_x, target_y, absolute=True)

        # 2. Handle Clicking (based on head tilt / d_roll)
        if abs(d_roll) > self.tilt_threshold and not self.debounce_flag: # The difference from baseline to the future tilt of your head is greater than the threshold
            mouse.click('left')
            self.last_click_time = time.time()
            self.debounce_flag = True
            print(f"--- CLICK EXECUTED (Tilt: {d_roll:.1f}°) ---")
            
        # 3. Reset Click Lock (when head returns to upright)
        elif abs(d_roll) < self.reset_threshold:
            self.debounce_flag = False