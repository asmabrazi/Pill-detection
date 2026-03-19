import argparse
from pill_detector import PillDetector
import cv2
import os

def main(args):
    # Initialize the model
    pill_detector = PillDetector(model_path=args.model_path, device=args.device)

    # Open video
    cap = cv2.VideoCapture(args.input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video {args.input_path}")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video path
    output_path = args.input_path.replace(".mp4", "_predicted.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_id = 0
    
    video_dir = os.path.dirname(args.input_path)
    video_name = os.path.splitext(os.path.basename(args.input_path))[0]



    print("Running video inference...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        """
        if frame_id < 200:
            frame_id += 1
            continue
        """

        # Run inference on frame
        results = pill_detector.infer(frame)

        # Apply NMS
        boxes, scores, classes = pill_detector.apply_nms(results)

        frame_name = os.path.join(video_dir, f"{video_name}_{frame_id}_predicted.png")

        # Draw detections and count pills
        annotated_frame = pill_detector.count_pills(
            input=frame,
            input_path=frame_name,
            boxes=boxes,
            scores=scores,
            classes=classes,
            show=args.show,
            save=False,
        )
        # Write frame to output video
        out.write(annotated_frame[0])

        if args.show:
            cv2.imshow("Pill Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_id += 1
        """
        if frame_id == 210:
            exit()
        """

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Inference completed! Saved to {output_path}")


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
