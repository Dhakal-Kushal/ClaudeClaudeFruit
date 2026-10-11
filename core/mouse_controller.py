import time 
import mouse
import ctypes

class MouseController:
    def __init__(self, freeze_duration=0.5, *args, **kwargs):
        self.freeze_duration = freeze_duration
        self.sensitivity = 30   
        
        try:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0) 
            self.screen_h = user32.GetSystemMetrics(1)
        except Exception:
            self.screen_w = 1920
            self.screen_h = 1080
        
        # Internal state tracking
        self.debounce_flag = False
        self.last_click_time = 0.0
        self.freeze_pos = None

    @property
    def is_frozen(self):
        return (time.time() - self.last_click_time) < self.freeze_duration

    def move(self, x, y, absolute=True):
        if not self.is_frozen:
            try:
                mouse.move(x, y, absolute=absolute)
            except Exception:
                pass

    def _trigger_click(self, button='left'):
        """Clean press, 40ms hold, and release so Windows registers the click."""
        try:
            if button in ('double_left', 'right'):
                mouse.double_click(button='left')
            else:
                mouse.press(button)
                time.sleep(0.04)
                mouse.release(button)
        except Exception:
            # Direct Windows user32 mouse_event fallback
            user32 = ctypes.windll.user32
            if button == 'left':
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                time.sleep(0.04)
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
            elif button in ('double_left', 'right'):
                # Double left click: first click
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                time.sleep(0.04)
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
                time.sleep(0.05)
                # Second click
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
                time.sleep(0.04)
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP

    def winkClick(self, left_winking, right_winking):
        current_time = time.time()
        
        # 1. Left Wink -> Left Click
        if left_winking and not self.debounce_flag:
            self.last_click_time = current_time
            self._trigger_click('left')
            self.debounce_flag = True
            print("--- LEFT CLICK FIRED ---")
            
        # 2. Right Wink -> Double Left Click
        elif right_winking and not self.debounce_flag:
            self.last_click_time = current_time
            self._trigger_click('double_left')
            self.debounce_flag = True
            print("--- DOUBLE LEFT CLICK FIRED ---")
            
        # 3. Release: Both eyes are open, allow next click
        elif not left_winking and not right_winking:
            self.debounce_flag = False

