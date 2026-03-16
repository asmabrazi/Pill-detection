import argparse
from pill_detector import PillDetector
import cv2

def main(args):
    # Initialize the model
    pill_detector = PillDetector(model_path=args.model_path, device=args.device)

    # Run inference
    results = pill_detector.infer(args.image_path)

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    boxes, scores, classes = pill_detector.apply_nms(results)

    # Draw bounding boxes and counts using the class method
    image = cv2.imread(args.image_path)
    pill_detector.count_pills(
        input=image,
        input_path=args.image_path,
        boxes=boxes,
        scores=scores,
        classes=classes,
        show=args.show
    )

    print("Inference completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pill Detection on a Single Image")

    parser.add_argument("--model_path", type=str, default="models/pill_detector.pt",
                        help="Path to the trained YOLO model weights")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--show", action="store_true",
                        help="Show the output image in a window")

    args = parser.parse_args()
    main(args)