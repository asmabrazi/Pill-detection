import argparse
from pill_detector import PillDetector

def main(args):
    # Initialize the pill detector
    pill_detector = PillDetector(model_path=args.model_path, device=args.device)

    # Evaluate the model on the dataset
    results = pill_detector.evaluate(data_path=args.data_path)

    print("Evaluation completed!\n")

    # Access metrics as a dictionary
    metrics_dict = results.results_dict
    print("\nOverall metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Pill Detection model")

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pill_detector.pt",
        help="Path to the trained YOLOv11 model weights"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/medical-pills/medical-pills.yaml",
        help="Path to the dataset YAML file for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run evaluation on (cpu, cuda, mps)"
    )

    args = parser.parse_args()
    main(args)