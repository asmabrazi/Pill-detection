# Pill-detection
Pill detection and counting system for images and videos with efficient inference.

---

## **Project Overview**

This project provides:

- Training on a custom pill dataset
- Detection and counting of pills in images and videos
- Efficient inference with PyTorch
- Exporting trained models to ONNX
- Clear code and documentation for running on new data

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Pill-detection.git
cd Pill-detection
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

---

## **Data**
The [`Medical Pills dataset`](https://github.com/ultralytics/assets/releases/download/v0.0.0/medical-pills.zip) includes 92 training images and 23 validation images. After downloading the dataset, unzip it into the `data/` folder in the root of the repository so your structure looks like this:

```bash
data
└── medical-pills
    ├── LICENSE.txt
    ├── images
    │   ├── train
    │   │   ├── Frame_0.jpg
    │   │   ├── Frame_100.jpg
    │   │   ├── ...
    │   └── val
    │       ├── Frame_144.jpg
    │       ├── Frame_172.jpg
    │       ├── ...
    ├── labels
    │   ├── train
    │   │   ├── Frame_0.txt
    │   │   ├── Frame_100.txt
    │   │   ├── ...
    │   └── val
    │       ├── Frame_144.txt
    │       ├── Frame_172.txt
    │       ├── ...
    └── medical-pills.yaml # Dataset configuration file
```

---

## **Training the Model**
Run the training script with custom arguments:

```bash
python src/train.py \
    --data_path "data/medical-pills/medical-pills.yaml" \
    --model_path "yolo26n.pt" \
    --epochs 100 \
    --device "mps" \
    --enable_export
```

---

## **Evaluation**
```bash
python src/eval.py --model_path "models/pill_detector.pt" --data_path "data/medical-pills/medical-pills.yaml" --device "mps"
```

---

## Performance Summary (Medical Pills Dataset)

The model achieved strong performance on the validation set, demonstrating reliable pill detection even with a limited dataset of **115 images** (92 Train / 23 Val).  

| Metric | Score | Key Takeaway |
| :--- | :--- | :--- |
| **mAP@50** | **92.7%** | Excellent localization precision. |
| **mAP@50-95** | **68.9%** | Overall detection performance across stricter IoU thresholds. |
| **Precision** | **83.9%** | High accuracy; relatively few false positives. |
| **Recall** | **86.0%** | Successfully detected most pills in validation images. |
| **Fitness** | **68.9%** | Combined metric used for early stopping and model selection. |

---

## **Inference and Pill Counting**
### Image inference:
You can run the pill detector on a single image, and it will detect all pills, draw bounding boxes, and display the total count.

1. Download a [test image](https://www.pexels.com/photo/set-of-small-pills-on-green-surface-5699524/) from Pexels and save it into `data/test_images`
2. Run inference and counting:
```bash
python src/infer_image.py \
    --image_path data/test_images/pexels-alex-green-5699524.jpg \
    --model_path models/pill_detector.pt \
    --show
```

---
