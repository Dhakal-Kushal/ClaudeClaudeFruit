import argparse
from pathlib import Path
import sys
import time

from core.calibration import TrackingAdapter, load_profile
from core.mouse_controller import MouseController
from core.vision_pipeline import VisionPipeline
from ui.calibration_window import CalibrationWindow, _physical_screen_size
from ui.startup_dialog import prompt_startup_choice


def init_dpi_awareness():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except (AttributeError, OSError):
                pass
        except (AttributeError, OSError):
            pass

def check_profile(profile_path, monitor=None):
    path = Path(profile_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return load_profile(path, monitor)


def get_current_monitor():
    physical = _physical_screen_size()
    return {"width": physical[0], "height": physical[1], "x": 0, "y": 0} if physical else None


def run_calibration(camera_index, profile_path, freeze_duration=0.5):
    adapter = TrackingAdapter()
    pipeline = VisionPipeline(adapter, camera_index=camera_index, freeze_duration=freeze_duration)
    window = CalibrationWindow(
        adapter=adapter,
        profile_path=profile_path,
        camera_metadata={"index": camera_index, "flip": True},
        coordinate_space="camera_pixels",
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
    adapter = TrackingAdapter()
    pipeline = VisionPipeline(adapter, camera_index=camera_index, freeze_duration=freeze_duration)
    controller = MouseController(adapter.samples, profile, freeze_duration=freeze_duration)
    print("Live tracking active. Press Ctrl+C to stop.")
    try:
        pipeline.start()
        controller.start()
        while not pipeline.stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
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

    if profile is None:
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

    if choice == "track":
        run_tracking(camera_index, profile, freeze_duration=freeze_duration)
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
