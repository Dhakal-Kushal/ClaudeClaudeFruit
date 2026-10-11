# Implementation Plan: Fix Code Review Issues

## Step 1: Fix Critical Severity Issues
- [x] **Issue 1.1**: Added `import math` and `import numpy as np` to `core/gaze_mapper.py`.
- [x] **Issue 1.2**: Updated `core/vision_processor.py` (`VisionProcessor.process`) to use `pupils.mean(axis=0)` in camera pixel coordinates, preserve `eyes_open` state correctly in `CalibrationSample`, and retain valid tracking during blinks.
- [x] **Issue 1.3**: Updated `core/gaze_mapper.py` to use four-quadrant linear interpolation with safe division (`MIN_EYE_DELTA`) and correct display bounds `[left, left + monitor["width"]]` / `[top, top + monitor["height"]]`.
- [x] **Issue 1.4**: Updated `requirements.txt` to include `opencv-contrib-python>=4.8.0` and `mouse>=0.7.1`.

## Step 2: Fix High Severity Issues
- [x] **Issue 2.1**: Updated `main.py` (`run_tracking`) to check and report `pipeline.errors` when tracking terminates unexpectedly.
- [x] **Issue 2.2**: Updated `core/mouse_controller.py` (`MouseController._run`) to wrap `self.update(sample)` in `try...except` and log exceptions.
- [x] **Issue 2.3**: Updated `ui/calibration_window.py` (`_poll_tracking`) to return immediately after `self._finish_calibration()` and check `if not self._closed:` before calling `self.after()`.
- [x] **Issue 2.4**: Updated `core/vision_processor.py` (`run_preview`) to use `BASELINE_KEY = "neutral_head"`, `angle_delta()`, and safe temporary file profile updates.

## Step 3: Fix Medium Severity Issues
- [x] **Issue 3.1**: Updated `ui/calibration_window.py` to bind `WM_DELETE_WINDOW` to `self._cancel_and_close` instead of `self.close`.
- [x] **Issue 3.2**: Updated `core/mouse_controller.py` (`headClick`) to lock out gesture clicks when `abs(d_roll) > self.max_roll_degrees` until head returns below `reset_threshold`.
- [x] **Issue 3.3**: Updated `core/mouse_controller.py` (`update`) to process head click gestures before moving the cursor so the cursor freezes synchronously on clicks.
- [x] **Issue 3.4**: Updated `core/camera.py` (`Camera.__init__`) to read a test frame if `self.width <= 0` or `self.height <= 0`.
- [x] **Issue 3.5**: Updated `main.py` to initialize Windows DPI awareness early at startup via `init_dpi_awareness()`.

## Step 4: Fix Low Severity Issues
- [x] **Issue 4.1**: Updated `core/calibration.py` (`save_profile`) to call `path.parent.mkdir(parents=True, exist_ok=True)`.
- [x] **Issue 4.2**: Updated `main.py` (`main`) to validate that `camera_index >= 0`.
- [x] **Issue 4.3**: Added synchronization locks to `TrackingAdapter.publish` in `core/calibration.py` and `VisionPipeline._capture_frames` in `core/vision_pipeline.py`.

## Step 5: Test & Verification
- [x] Ran full test suite with `.venv/Scripts/python.exe -B -m unittest discover -s tests -v` (48/48 tests passed).
- [x] Verified `test_mouse_controller.py` (all tests passing including boundary clamping, near-zero delta protection, and extreme tilt lockout).
- [x] Verified `test_vision_pipeline.py` (all tests passing including pupil midpoint and pose in camera pixels).
