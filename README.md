# Eye and head calibration

The app starts camera capture and facial landmark processing in separate worker
threads. The vision worker publishes pupil midpoints and head angles to the GUI's
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
its standard window controls. Calibration dots scale from the saved monitor
coordinate system to the fullscreen canvas, so the displayed targets and saved
screen positions stay aligned even when Windows reports different logical and
canvas sizes.
On startup, an empty, incomplete, or incompatible profile begins calibration.
A complete profile for the current primary display is reused; **Redo calibration**
starts another run. The six tests capture a neutral head pose, four corner eye
positions, and the center eye position. Each test has a two-second countdown and
collects at least 15 measurements over 1.5 seconds. Tracking loss, closed eyes,
and unstable head movement restart the current capture. Press **Esc** to cancel.

Successful calibration atomically replaces `calibration_profile.json`. A failed
or cancelled run leaves the previous file intact. The profile contains the
primary monitor resolution, actual screen target positions, raw eye coordinates,
neutral pitch/yaw/roll, mapping metadata, gesture settings, and Kalman settings.
Camera metadata can be supplied by the tracking pipeline. The GUI does not assign
head gestures to mouse buttons; `gesture_settings.bindings` is reserved for that
separate configuration.

## Tracking integration

`main.py` connects `VisionPipeline` to `TrackingAdapter` automatically. The camera
thread uses `Camera`; the vision thread uses `VisionProcessor` and publishes on
every processed frame, including invalid tracking and closed-eye states. Captured
frame timestamps are preserved so calibration can reject stale measurements.

The live app records mirrored camera-pixel coordinates and camera metadata in the
profile. The vision worker sends raw head angles; neutral-head calibration remains
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
from ui.calibration_window import CalibrationWindow

adapter = TrackingAdapter()
window = CalibrationWindow(
    adapter=adapter,
    camera_metadata={"index": 0},
    coordinate_space="camera_pixels",  # or "normalized_camera"
)

# In your existing tracking thread, for every processed frame:
adapter.publish(
    eye_x=midpoint_x,
    eye_y=midpoint_y,
    pitch=pitch_degrees,
    yaw=yaw_degrees,
    roll=roll_degrees,
    eyes_open=both_eyes_open,
    tracking_valid=face_detected,
)

# On the main thread:
window.mainloop()
```

Eye coordinates must be the midpoint of the two pupils, in consistent raw units.
Head angles must be in degrees. Invalid tracking should still publish with
`tracking_valid=False` or `eyes_open=False`. Timestamps default to
`time.monotonic()`; if supplied, use the same clock and mark the measurement's
capture time. Samples older than half a second are rejected. The queue is bounded
and discards its oldest samples if the GUI falls behind. Tkinter widgets are only
updated on the main thread using `after()`.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
