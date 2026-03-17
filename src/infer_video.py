import argparse
from pill_detector import PillDetector
import cv2
import os

def main(args):
    # Initialize the model
    pill_detector = PillDetector(model_path=args.model_path, device=args.device)
    pill_detector.track(args.input_path)


    print(f"Inference completed! Saved to ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pill Detection on Video")

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pill_detector.pt",
        help="Path to the trained YOLO model weights"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input video"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run inference on (cpu, cuda, mps)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the video while processing"
    )

    args = parser.parse_args()
    main(args)
