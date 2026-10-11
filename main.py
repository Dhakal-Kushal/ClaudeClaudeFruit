import argparse
from pathlib import Path
import sys
import time

from core.calibration import (EYE_RELATIVE_COORDINATE_SPACE, TrackingAdapter, load_profile,
                              validate_profile, profile_compatibility_errors)
from core.mouse_controller import MouseController, init_dpi_awareness
from core.vision_pipeline import VisionPipeline
from ui.calibration_window import CalibrationWindow, _physical_screen_size
from ui.startup_dialog import prompt_startup_choice

def check_profile(profile_path, monitor=None):
    path = Path(profile_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    profile = load_profile(path, monitor)
    if profile is None or profile["mapping"]["coordinate_space"] != EYE_RELATIVE_COORDINATE_SPACE:
        return None
    return profile


def get_current_monitor():
    physical = _physical_screen_size()
    return {"width": physical[0], "height": physical[1], "x": 0, "y": 0} if physical else None


def run_calibration(camera_index, profile_path, freeze_duration=0.5):
    adapter = TrackingAdapter()
    pipeline = VisionPipeline(adapter, camera_index=camera_index, freeze_duration=freeze_duration)
    window = CalibrationWindow(
        adapter=adapter,
        profile_path=profile_path,
        camera_metadata_provider=pipeline.get_camera_metadata,
        coordinate_space=EYE_RELATIVE_COORDINATE_SPACE,
        pipeline_errors=pipeline.errors,
    )
    try:
        pipeline.start()
        window.mainloop()
        return not getattr(window, "cancelled", False)
    finally:
        pipeline.stop()
        if not window._closed:
            window.close()

    
def run_tracking(camera_index, profile, freeze_duration=0.5):
    if profile.get("mapping", {}).get("coordinate_space") != EYE_RELATIVE_COORDINATE_SPACE:
        raise ValueError("This tracker requires a new eye-relative calibration")
    validate_profile(profile, get_current_monitor(), require_validation=True)
    adapter = TrackingAdapter(max_samples=2)
    settings = {key: profile["camera"][key] for key in ("width", "height", "fps", "flip", "backend", "fourcc")}
    pipeline = VisionPipeline(adapter, camera_index=camera_index, freeze_duration=freeze_duration,
                              capture_settings=settings,
                              expected_camera=profile["camera"], initial_head=profile["neutral_head"])
    controller = None
    try:
        pipeline.start()
        deadline = time.monotonic() + 10
        while not pipeline.camera_ready.wait(.05):
            if pipeline.stop_event.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Camera startup failed or timed out; check the camera and recalibrate")
        errors = profile_compatibility_errors(profile, get_current_monitor(), pipeline.get_camera_metadata())
        if errors or pipeline.stop_event.is_set():
            raise ValueError("Camera compatibility check failed; recalibrate. " + "; ".join(errors))
        controller = MouseController(adapter.samples, profile, freeze_duration=freeze_duration)
        controller.start()
        print("Live tracking active. Press Ctrl+C to stop. Persistent cursor pauses will report their reason.")
        while not pipeline.stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if controller is not None:
            controller.stop()
        pipeline.stop()
        if not pipeline.errors.empty():
            error_msg = pipeline.errors.get_nowait()
            print(f"Tracking error encountered: {error_msg}", file=sys.stderr)


def main(camera_index=0, calibrate=False, profile_path="calibration_profile.json", prompt_fn=None, freeze_duration=0.5):
    if camera_index < 0:
        raise ValueError("camera_index must be >= 0")
    if freeze_duration < 0:
        raise ValueError("freeze_duration must be >= 0")
    init_dpi_awareness()
    path = Path(profile_path)
    current_monitor = get_current_monitor()
    profile = None if calibrate else check_profile(path, monitor=current_monitor)

    if calibrate:
        success = run_calibration(camera_index, path, freeze_duration=freeze_duration)
        if not success:
            return
        profile = check_profile(path, monitor=get_current_monitor())
        if profile is None:
            return
        run_tracking(camera_index, profile, freeze_duration=freeze_duration)
        return

    prompt = prompt_fn if prompt_fn is not None else prompt_startup_choice
    try:
        result = prompt(profile, default_camera=camera_index, return_camera=True)
    except TypeError:
        result = prompt(profile)

    if isinstance(result, tuple):
        choice, camera_index = result
    else:
        choice = result

    if choice == "track" and profile is not None:
        try:
            run_tracking(camera_index, profile, freeze_duration=freeze_duration)
        except (ValueError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            # Reopen the calibration UI with the selected camera's actual settings.
            if run_calibration(camera_index, path, freeze_duration=freeze_duration):
                updated = check_profile(path, monitor=get_current_monitor())
                if updated is not None:
                    run_tracking(camera_index, updated, freeze_duration=freeze_duration)
    elif choice == "calibrate":
        success = run_calibration(camera_index, path, freeze_duration=freeze_duration)
        if success:
            profile = check_profile(path, monitor=get_current_monitor())
            if profile is not None:
                run_tracking(camera_index, profile, freeze_duration=freeze_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye and head tracking control")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--freeze-duration",
        type=float,
        default=0.5,
        help="Seconds to freeze mouse cursor after a click (default: 0.5)",
    )
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration")
    parser.add_argument("--profile", type=str, default="calibration_profile.json", help="Path to profile JSON")
    parser.add_argument("--list-cameras", action="store_true", help="List available webcams and exit")
    args = parser.parse_args()

    if args.list_cameras:
        from core.camera import list_available_cameras
        print("Scanning for available webcams...")
        cameras = list_available_cameras()
        if not cameras:
            print("No webcams detected.")
        else:
            print(f"Detected {len(cameras)} webcam(s):")
            for c in cameras:
                print(f"  [{c['index']}] {c['name']} ({c['width']}x{c['height']})")
        sys.exit(0)

    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.freeze_duration < 0:
        parser.error("--freeze-duration must be zero or greater")
    main(
        camera_index=args.camera,
        calibrate=args.calibrate,
        profile_path=args.profile,
        freeze_duration=args.freeze_duration,
    )
