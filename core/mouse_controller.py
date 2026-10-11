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
        self.last_click_time = 0.0
        self.freeze_pos = None

    @property
    def is_frozen(self):
        return (time.time() - self.last_click_time) < self.freeze_duration

    def move(self, x, y, absolute=True):
        if not self.is_frozen:
            mouse.move(x, y, absolute=absolute)

    def winkClick(self, left_winking, right_winking):
        current_time = time.time()
        
        # 1. Left Wink -> Left Click
        if left_winking and not self.debounce_flag:
            try:
                self.freeze_pos = mouse.get_position()
            except Exception:
                self.freeze_pos = None
            self.last_click_time = current_time
            mouse.click('left')
            self.debounce_flag = True
            print("--- LEFT CLICK FIRED ---")
            
        # 2. Right Wink -> Right Click
        elif right_winking and not self.debounce_flag:
            try:
                self.freeze_pos = mouse.get_position()
            except Exception:
                self.freeze_pos = None
            self.last_click_time = current_time
            mouse.double_click('left')
            self.debounce_flag = True
            print("--- RIGHT CLICK FIRED ---")
            
        # 3. Release: Both eyes are open, unlock the door
        elif not left_winking and not right_winking:
            self.debounce_flag = False

        # Keep mouse frozen on target during the freeze duration to prevent dragging
        if self.is_frozen and self.freeze_pos is not None:
            try:
                mouse.move(self.freeze_pos[0], self.freeze_pos[1], absolute=True)
            except Exception:
                pass
        elif not self.is_frozen:
            self.freeze_pos = None