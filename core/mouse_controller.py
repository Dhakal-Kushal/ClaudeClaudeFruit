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

    def headClick(self, d_roll):
        # Handle Clicking (based on head tilt / d_roll)
        if abs(d_roll) > self.tilt_threshold and not self.debounce_flag: # The difference from baseline to the future tilt of your head is greater than the threshold
            mouse.click('left')
            self.last_click_time = time.time()
            self.debounce_flag = True
            print(f"--- CLICK EXECUTED (Tilt: {d_roll:.1f}°) ---")
            
        # 3. Reset Click Lock (when head returns to upright)
        elif abs(d_roll) < self.reset_threshold:
            self.debounce_flag = False