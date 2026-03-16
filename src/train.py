import argparse
from pill_detector import PillDetector

def main(args):
    # Initialize the pill detector with the path to the model weights
    pill_detector = PillDetector(model_path=args.model_path)

    # Train the model on the dataset
    training_results = pill_detector.train(
        data_path=args.data_path,
        epochs=args.epochs,
        device=args.device,
        enable_export=args.enable_export
    )

    print("Training completed!")
    print(f"Results saved at: {training_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pill Detection model")

    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo26n.pt",
        help="Path to YOLOv11 model weights"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/medical-pills/medical-pills.yaml",
        help="Path to your dataset YAML file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to train on (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--enable_export", 
        action="store_true",
        help="Enable exporting the trained model")

    args = parser.parse_args()
    main(args)