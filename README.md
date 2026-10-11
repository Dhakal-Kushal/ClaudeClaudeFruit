# Eye and head calibration

The app starts camera capture and facial landmark processing in separate worker
threads. The vision worker publishes eye-relative iris vectors and head angles to the GUI's
calibration queue. Tkinter stays on the main thread.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Use `main.py --camera 1` to select another webcam. Closing the GUI signals both
workers to stop and releases the camera and FaceMesh resources. Camera or tracking
failures appear in the GUI instead of leaving it waiting for measurements.

The app opens maximized with the standard close, minimize, and restore buttons.
Calibration runs fullscreen with the title bar hidden, whether started manually
or on startup. Press **Esc** to cancel and return to the maximized dashboard with
its standard window controls in manual dashboard mode. Automatically launched
calibration closes on cancellation. Calibration dots scale from the saved monitor
coordinate system to the fullscreen canvas, so the displayed targets and saved
screen positions stay aligned even when Windows reports different logical and
canvas sizes.
On startup, choose a camera even when no compatible profile exists. Select
**Redo Calibration** for a missing or incompatible profile. A validated profile
for the current primary display can be reused after checking actual capture
settings. The six calibration steps capture a neutral head pose, four corner eye
positions, and the center eye position. Each test has a two-second countdown and
collects at least 15 measurements over 1.5 seconds. Tracking loss, closed eyes,
and unstable head movement restart the current capture. Press **Esc** to cancel.

Calibration is followed by eight independent validation targets: four interior
quarter positions and four inset edge midpoints. Each target has a 0.75-second
settling period and collects at least 15 valid measurements over one second.
Validation does not refit the mapping. The results screen reports median and
90th-percentile cursor error in pixels and percent of screen diagonal, per-target
signed X/Y bias and RMS jitter, and the fraction of frames rejected for invalid
tracking, blinks, stale/duplicate timestamps or unstable head pose. Measurements
use the continuous mapping with display clamping, before cursor smoothing.

Review the results and select **Use this calibration** to atomically replace
`calibration_profile.json`, **Retry calibration** to repeat the full process, or
**Cancel** to preserve the previous file. Suggested accuracy targets are median
error <= 2.5% and p90 error <= 5% of the screen diagonal, with per-target bias
<= 3% and RMS jitter <= 2%. These are advisory targets: webcam measurements
may exceed them even after careful calibration. The results screen shows the
measured errors and offers **Use anyway** or **Retry calibration** when accuracy
is below these targets. A complete calibration can be explicitly accepted, saved,
and reused with its original measurements. Small controls may remain difficult
to select. Incomplete validation, nonfinite data, invalid mapping geometry, and
camera/display incompatibility still block use. Validation never refits the map
or changes measured errors. Previously accepted profiles remain usable regardless
of these advisory accuracy targets.
The profile contains the
primary monitor resolution, actual screen target positions, eye-relative coordinates,
neutral pitch/yaw/roll, mapping metadata, gesture settings, and smoothing settings.
Schema 2 also stores the validation report and the actual camera index, backend,
frame width/height, mirroring, nominal FPS and FOURCC pixel format. Tracking
requests the saved capture settings, reads back the negotiated values after a
successful frame, and blocks mouse control on a mismatch. The UI waits for actual
camera metadata before capturing calibration. Checks also detect setting drift
during capture. FPS allows a tolerance of 0.5 FPS or 2%, whichever is greater;
unsupported optional FPS/FOURCC properties are stored as `null`. Unknown settings
must remain unknown on reuse. Camera indices are not persistent hardware IDs;
recalibrate if devices are replaced or reordered, the camera is repositioned, or
focus/zoom changes. Exposure, autofocus and lighting are not compatibility checks.
The GUI does not assign
head gestures to mouse buttons; `gesture_settings.bindings` is reserved for that
separate configuration.

## Tracking integration

`main.py` connects `VisionPipeline` to `TrackingAdapter` automatically. The camera
thread uses `Camera`; the vision thread uses `VisionProcessor` and publishes on
every processed frame, including invalid tracking and closed-eye states. Captured
frame timestamps are preserved so calibration can reject stale measurements.

The live app records dimensionless `eye_relative_v1` coordinates and camera metadata
in the profile. Each iris is paired with its own eye corners. Its displacement from
the corner midpoint is projected onto axes aligned with the eye, and both components
are divided by eye width. The output is the mean of the two eye vectors: positive x
points toward image right and positive y points below the eye. Eyelid opening is not
used as a scale. Translation, uniform scaling, and moderate image-plane head roll
are reduced by this geometry; head yaw/pitch and perspective effects remain.

**Stage 3 requires recalibration and validation.** Startup and direct tracking reject older
camera-pixel profiles rather than applying them to the new measurements. The saved
profile file is replaced only after completing and accepting validation.

**Stage 2 uses continuous piecewise affine mapping.** Four triangles connect the
calibrated eye center to adjacent corners. Each maps to the corresponding screen
triangle, reproducing all five calibration anchors and agreeing along shared
edges. Gaze outside the calibration polygon is extrapolated before clamping to
valid display pixels. Position is continuous, although its slope can change at
triangle edges. Invalid, overlapping, or nearly collinear calibration geometry is
rejected rather than collapsing an axis to the center.

New profiles record `mapping.method="piecewise_affine_v1"`. Stage 1 and Stage 2
profiles lack the required capture metadata and accepted independent validation;
they require a fresh calibration. Loading never rewrites the existing file.

Live cursor movement honors `smoothing.method`: `kalman` uses process and
measurement noise parameters, while `moving_average` uses `window_size`.
New calibrations default to Kalman. The Kalman output stays between the previous
output and the current measurement on each axis to prevent predictive overshoot.
Its internal motion estimate, Joseph covariance update and symmetry enforcement
are retained. Final display bounds apply to both filters.

Live movement rejects samples older than 250 ms, more than 50 ms in the future,
or with duplicate/backward timestamps. A two-sample live queue and backlog
draining avoid replaying old movements; skipped samples cancel pending gestures.
Calibration keeps its separate 180-sample queue and half-second freshness limit.
All runtime timing uses the monotonic clock.

At startup and after tracking loss, head-pose fitting uses the saved neutral
orientation as its initial estimate, with translation estimated from the current
frame. This keeps startup fitting consistent with calibration while still fitting
actual head movement. Calibration itself continues to estimate pose independently.
Discarding healthy queued frames cancels pending gestures without restarting
cursor recovery; discarded blink, invalid, or out-of-pose frames still pause it.

If the console says live tracking is active but the cursor remains stationary,
a persistent pause reports its reason after two seconds, at most once every five
seconds: missing face/iris tracking, closed eyes, stale/invalid samples, pose
mismatch, or mouse movement failure. Live tracking means workers are running;
it does not guarantee usable gaze measurements. If the mouse library fails,
absolute movement tries Windows SetCursorPos and reports failure if that fails too.

Blinks, tracking loss, clicks, and head deviations pause movement. At least three
open-eye samples spanning 100 ms are required before resuming; their median is
fed into the retained filter. Short pauses discard Kalman velocity, and pauses
over one second rebuild filter history from the last cursor position.
Movement pauses when pitch, yaw, or roll differs by more than 10 degrees from
calibration. Return to your calibrated posture to resume. This is a posture guard;
full head-pose compensation is not implemented.

Head tilts must persist for `hold_ms` (default 150 ms). Left tilt clicks once;
right tilt double-clicks. Position stays frozen during tilts outside the movement
range. Extreme roll beyond 25 degrees blocks tilt clicks until returning neutral.
A wink must last `hold_ms` and at least `closed_eye_frames` (default 3), followed by
both eyes being open for `click_settle_ms` (default 250 ms). It clicks only after
this reopening period. Bilateral closure, switching eyes, loss of tracking, and
skipped samples cancel a pending wink. Left wink clicks once; right wink
double-clicks. Both gesture types share `cooldown_ms` (default 500 ms).
`--freeze-duration` controls the post-click movement freeze; changing it does not
shorten the gesture cooldown. The calibration dashboard includes a tilt-angle
selector (default 15 degrees).

The vision worker sends raw solvePnP head angles; neutral-head calibration remains
owned by the GUI. Importing `core.vision_processor` does not open the camera or run
the standalone diagnostic preview. That preview can be run explicitly with
`python -m core.vision_processor`.

The requirements pin MediaPipe to the installed FaceMesh-compatible runtime used
by this tracker. OpenCV is supplied by MediaPipe's `opencv-contrib-python`
dependency, which provides the `cv2` API used by `Camera`.

To connect another tracking implementation directly:

Create the window on the main thread, and publish a separate copy of measurements
to its adapter from the tracking thread. Do not pass the mouse controller's queue:
consuming from that queue would take measurements away from mouse control.

```python
from core.calibration import TrackingAdapter
from core.vision_pipeline import VisionPipeline
from ui.calibration_window import CalibrationWindow

adapter = TrackingAdapter()
pipeline = VisionPipeline(adapter, camera_index=0)
window = CalibrationWindow(
    adapter=adapter,
    camera_metadata_provider=pipeline.get_camera_metadata,
    pipeline_errors=pipeline.errors,
    coordinate_space="eye_relative_v1",
)

# In your existing tracking thread, for every processed frame:
adapter.publish(
    eye_x=relative_eye_x,
    eye_y=relative_eye_y,
    pitch=pitch_degrees,
    yaw=yaw_degrees,
    roll=roll_degrees,
    eyes_open=both_eyes_open,
    tracking_valid=face_detected,
)

# On the main thread:
pipeline.start()
try:
    window.mainloop()
finally:
    pipeline.stop()
```

Eye coordinates must be the binocular mean of the normalized eye-local vectors,
using the same calculation during calibration and tracking. Invalid or degenerate
eye geometry publishes an invalid tracking sample. The queue fields are unchanged.
Head angles must be in degrees. Invalid tracking should still publish with
`tracking_valid=False` or `eyes_open=False`. Timestamps default to
`time.monotonic()`; if supplied, use the same clock and mark the measurement's
capture time. Samples older than half a second are rejected. The queue is bounded
and discards its oldest samples if the GUI falls behind. Tkinter widgets are only
updated on the main thread using `after()`.

In the example, `pipeline` is your `VisionPipeline(adapter, camera_index=0)`
instance (import it from `core.vision_pipeline`). A custom capture implementation
must provide the same complete metadata dictionary after its first valid frame;
an index alone is insufficient. Calibration and validation never start the mouse
controller.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
