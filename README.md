# Aerial Human Detection using YOLOv8

End-to-end computer vision pipeline for detecting and counting small human targets in aerial drone imagery using YOLOv8. The system is optimized for high-resolution training to address small-object localization challenges and dense crowd scenarios.



## Overview

Detecting humans in aerial imagery is a challenging small-object detection problem due to:

* Very small bounding boxes (≈2–4% of image size)
* Dense clustering
* Background interference (terrain, rocks, shadows)

This project builds a complete training, evaluation, and inference pipeline while improving model performance through resolution scaling and backbone optimization.


## Key Results

| Metric    | Baseline | Final Model |
| --------- | -------- | ----------- |
| mAP50     | 0.44     | **0.76**    |
| mAP50-95  | 0.16     | **0.33**    |
| Recall    | 0.44     | **0.72**    |
| Precision | 0.60     | **0.82**    |

Performance improved significantly by upgrading the model architecture and increasing input resolution to better capture small targets.



## Approach

### 1. Baseline Model

* YOLOv8n
* 640 resolution
* Identified small-object localization bottleneck

### 2. Optimization Strategy

* Upgraded to YOLOv8s backbone
* Increased resolution to 960
* Tuned augmentation and training schedule
* Extended training to 100 epochs
* Evaluated mAP50 and mAP50-95 to assess localization quality

### 3. Final System

* Clean bounding box visualization
* Automated people-count overlay
* Separate training, evaluation, prediction, and counting modules



## Project Structure

```
aerial-human-detection/
│
├── assets/
│   ├── detection_example.png
│   └── count_overlay_example.png
│
├── dataset_sample/
│
├── train.py
├── evaluate.py
├── predict.py
├── count_people.py
├── dataset.yaml
├── requirements.txt
├── README.md
└── .gitignore
```


## Dataset

The full dataset is not included due to size limitations.
To reproduce results, organize your aerial images and YOLO-format annotations in the following structure:

```
dataset/
├── images/{train, val, test}
└── labels/{train, val, test}
```

Each image must have a corresponding `.txt` annotation file with normalized bounding box coordinates.

A small `dataset_sample/` directory is provided to demonstrate the expected structure.



## Installation

```bash
pip install -r requirements.txt
```



## Training

```bash
python train.py
```



## Evaluation

```bash
python evaluate.py
```



## Inference

```bash
python predict.py
```



## People Counting (With On-Image Overlay)

```bash
python count_people.py
```

Images with bounding boxes and total count are saved in:

```
runs/detect/count_output/
```



## Final Result


<!-- Insert screenshot below -->
![1_peoplepegia_1125](https://github.com/user-attachments/assets/2f66063e-0c0b-45e0-924a-93c2416aec22)



## Tech Stack

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Matplotlib



## Highlights

* End-to-end object detection pipeline
* Small-object optimization via resolution scaling
* Quantitative performance improvement analysis
* Modular and reproducible training workflow
* Deployment-ready inference and counting system
