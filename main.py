import argparse
import mediapipe

from core.calibration import TrackingAdapter
from core.vision_pipeline import VisionPipeline
from core import mouse_controller
from ui.calibration_window import CalibrationWindow




def main(camera_index=0):
    adapter = TrackingAdapter()
    pipeline = VisionPipeline(adapter, camera_index=camera_index)
    window = CalibrationWindow(
        adapter=adapter,
        camera_metadata={"index": camera_index, "flip": True},
        coordinate_space="camera_pixels",
        pipeline_errors=pipeline.errors,
    )
    try:
        pipeline.start()
        window.mainloop()
    finally:
        pipeline.stop()
        if not window._closed:
            window.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate eye and head tracking")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    main(camera_index=args.camera)
