# Cursor tracking repair plan (2026-10-11)

1. Calibration accuracy: show advisory median <= 2.5% and p90 <= 5% of display
   diagonal, with per-target bias <= 3% and RMS jitter <= 2%. Offer Use anyway
   or Retry calibration. Only malformed/incomplete data, invalid geometry and
   camera/display mismatches block persistence/reuse. Accuracy targets are not
   empirically validated hardware limits and must not force repeated calibration.
   Never rewrite measured results or fit to held-out targets.
2. Smoothing selection: instantiate the requested Kalman or moving-average
   filter and honor its parameters; reject unknown methods.
3. Overshoot: constrain Kalman outputs between the previous output and current
   measurement. Retain the internal motion estimate on noisy observations; clear
   constrained velocity when successive observations show a stopped target.
   Preserve Joseph covariance updates and symmetry.
4. Blinks: freeze on closure, preserve position history, remove velocity,
   and collect at least three open samples over 100 ms before resuming from
   their median. Reinitialize long-loss history from the last cursor position.
5. Freshness: use monotonic time; reject samples older than 250 ms, future
   samples beyond 50 ms, duplicates, and out-of-order samples. Use a small
   live queue and drain backlogs with gesture/reacquisition reset.
6. Head pose: use the calibration's 10-degree limit for live pitch/yaw/roll
   movement. Tilt gestures operate while position is frozen, provided
   pitch/yaw remain in range. Resume after stable open-eye samples.
7. Gestures: honor hold duration, minimum closed frames, reopening settle
   duration, and shared cooldown. Cancel winks on bilateral closure, tracking
   loss, or skipped frames. Fire qualified winks after confirmed reopening.
8. Calibration UI: restore the tilt threshold control, preserve retry/cancel/
   save behavior, and repair the pose test's missing mock.

Verification: synthetic regressions with a mocked mouse for both filter modes,
accuracy limits and persistence, full UI flow, full unittest suite, diff checks.
Real accuracy requires a fresh webcam calibration and independent validation.

## Completion and verification

All eight software changes implemented. The full suite passes 132 tests,
including step responses at 15/20/30/60 FPS, both smoothing modes, stale input,
blink recovery, sustained gestures, pose gating, calibration quality/persistence,
and UI acceptance/retry/cancel flows. The missing pose-test mock was repaired.
The existing user-modified calibration profile was not edited; its recorded
errors exceed advisory accuracy targets but no longer prevent explicit use or reuse.

## Follow-up: repeated calibration rejection

The initial hard accuracy gates caused repeated rejection of complete webcam
calibrations. Removed those gates from persistence and runtime validation;
the results UI now offers Use anyway with measured-error warnings. Regression
tests cover explicit acceptance, unchanged measurements, save/load/startup,
retry, and continued rejection of malformed/incomplete validation.

## Follow-up: intermittent startup freeze

Reproduced startup recovery starvation when every cursor iteration skipped a
fresh queued frame: the recovery buffer was cleared on each iteration. Healthy
skips now cancel gesture continuity without resetting movement recovery. Skipped
blink, invalid, and out-of-pose samples retain their pause behavior.

Tracking pose fitting now starts from the saved neutral rotation with current
frame translation, and reuses this seed on reacquisition. Calibration is unseeded;
genuine head deviations still block movement. Persistent blocks print actionable
reasons after two seconds, with a five-second log rate limit. Mouse-library failure
tries Windows SetCursorPos, and failed input delivery does not update held position.

Verification covers the previously failing backlog reproduction, skipped blinks,
no-frame startup, pose diagnostics, mocked backend failures, seeded synthetic
pose geometry (mirrored/unmirrored, changed distance and actual head movement),
and initialization after tracking loss. Full suite: 140 tests. No live webcam
or physical cursor input was used for tests; the reported symptom needs a live
retry to confirm which of these failure paths occurred on the user's system.

# Historical implementation plan

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
