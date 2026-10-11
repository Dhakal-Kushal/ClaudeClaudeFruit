import time 
import mouse
import ctypes

class MouseController:
    def __init__(self, freeze_duration=0.5):
        self.freeze_duration = freeze_duration
        self.sensitivity = 30   
        
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0) 
        self.screen_h = user32.GetSystemMetrics(1)
        print(f"Screen resolution detected: {self.screen_w}x{self.screen_h}")
        
        # Internal state tracking
        self.debounce_flag = False

    def winkClick(self, left_winking, right_winking):
        
        # 1. Left Wink -> Left Click
        if left_winking and not self.debounce_flag: 
            mouse.click('left')
            self.debounce_flag = True
            print("--- LEFT CLICK FIRED ---")
            
        # 2. Right Wink -> Right Click
        elif right_winking and not self.debounce_flag:
            mouse.click('right')
            self.debounce_flag = True
            print("--- RIGHT CLICK FIRED ---")
            
        # 3. Release: Both eyes are open, unlock the door
        elif not left_winking and not right_winking:
            self.debounce_flag = False