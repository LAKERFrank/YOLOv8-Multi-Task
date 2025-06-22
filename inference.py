import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Return command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 inference script")
    parser.add_argument("--weights", required=True, help="Path to trained weight file")
    parser.add_argument("--source", required=True, help="Image/video path, URL or webcam index")
    parser.add_argument("--track", action="store_true", help="Enable tracking head")
    parser.add_argument("--pose", action="store_true", help="Enable pose estimation head")
    parser.add_argument("--save", action="store_true", help="Save prediction results")
    parser.add_argument("--show", action="store_true", help="Display prediction results")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run inference using YOLOv8 model with custom weights."""
    model = YOLO(args.weights)
    model.predict(source=args.source, track=args.track, pose=args.pose, save=args.save, show=args.show)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)

