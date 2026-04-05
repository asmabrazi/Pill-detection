from matplotlib import image
import torch
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import os 

class PillDetector:
    def __init__(self, model_path="yolo11n.pt", device="mps"):
        '''Pill Detection using YOLOv11.
        Parameters:
        - model_path: Path to the YOLOv11 model weights (default is "yolo11n.pt" for the nano version).
        - device: Device to use for inference (default is "mps").
        '''
        self.device = device 
        self.backend = None
        self.onnx_session = None

        # Determine the model format based on the file extension
        suffix = Path(model_path).suffix

        if suffix == ".pt":
            self.backend = "pytorch"

        elif suffix == ".onnx":
            self.backend = "onnx"

        else:
            raise ValueError("Unsupported model format")
        
        self.model = YOLO(model_path)

    def train(self, data_path="data/medical-pills.yaml", epochs=100, device="mps", enable_export=True):
        '''Train the YOLOv11 model on the provided dataset.
        Parameters:
        - data_path: Path to the dataset in YOLO format:  "data/pill_data.yaml".
        - epochs: Number of training epochs (default is 100).
        - device: Device to use for training (default is "mps").
        '''

        # Training is only supported for PyTorch models
        if self.backend != "pytorch":
            raise RuntimeError("Training only supported with .pt models")
        
        training_results = self.model.train(
            data=data_path,
            epochs=epochs,
            patience=15,        # Stop training if no improvement after 15 epochs
            freeze=10,          # Freeze the first 10 layers of the model during training
            imgsz=640,
            batch=8, 
            lr0=0.001,          # Initial learning rate
            device=device,
            augment=True,       # Enable augmentation
            flipud=0.5,         # 50% chance vertical flip
            fliplr=0.5,         # 50% chance horizontal flip
            hsv_h=0.015,        # Hue jitter
            hsv_s=0.7,          # Saturation jitter
            hsv_v=0.4,          # Value jitter
            degrees=10.0,       # random rotation ±10 degrees
            translate=0.1,      # random shift ±10%
            scale=0.1,          # random scaling ±10%
            single_cls=False,      # Multiple classes in dataset
        )

        # Export the trained model to ONNX format
        if enable_export:
            self.model.export(format="onnx", dynamic=True)

        return training_results
    
    def evaluate(self, data_path="data/medical-pills.yaml"):
        '''Evaluate the model on a test dataset.
        Parameters:
        - data_path: Path to the test dataset in YOLO format (default is "data/medical-pills.yaml").
        Returns:
        - Evaluation metrics such as precision, recall, and mAP.
        '''

        if self.backend != "pytorch":
            raise RuntimeError("Evaluation only supported with .pt models")

        results = self.model.val(data=data_path)
        return results
    
    def infer(self, image_path):
        """
        Run inference on the input image to detect pills.

        Parameters:
        - image_path: Path to the input image.
        """

        results = self.model.predict(image_path, save=True, conf=0.28, iou=0.6)
        return results
        
    def track(self, video_path):
        """
        Run inference on the input video to detect and track pills.

        Parameters:
        - video_path: Path to the input video.
        """

        results = self.model.track(video_path, save=True, conf=0.28, iou=0.6, show=True, show_labels=True, show_conf=True)
        return results
        
    def apply_nms(self, results, iou_threshold=0.45):
        """
        Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        Parameters:
        - results: The raw output from the model inference.
        - iou_threshold: IoU threshold for NMS (default is 0.45).

        Returns:
        - Filtered boxes, scores, and class indices after applying NMS.
        """
        boxes = results[0].boxes.xyxy.cpu()      
        scores = results[0].boxes.conf.cpu()     
        classes = results[0].boxes.cls.cpu()    

        # Perform NMS: remove boxes with IoU > threshold relative to the box with the highest score
        keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

        return boxes[keep_indices].numpy(), scores[keep_indices].numpy(), classes[keep_indices].numpy()

    def count_pills(self, input, input_path, boxes, scores, classes, show=False, save=True):
        """
        Draw bounding boxes and pill counts on the image.
        Returns: output image and total count

        Parameters:
        - input: The input image on which to draw the results.
        - input_path: Path to the input image (used for saving the output).
        - boxes: Filtered bounding boxes.
        - scores: Filtered confidence scores.
        - classes: Filtered class indices.
        - show: Whether to display the output image in a window (default is False).
        """
        image = input.copy()
        count = 0

        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            conf = float(scores[i])
            cls = int(classes[i])
            count += 1

            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            # Label with class and confidence
            cv2.putText(image, f"Pill {cls}: {conf:.2f}", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Add total count
        cv2.putText(image, f"Total Pills: {count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_predicted{ext}"

        if save:
            cv2.imwrite(output_path, image)
    
        if show:
            cv2.imshow("Pill Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image, count
