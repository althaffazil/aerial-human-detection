import torch
from ultralytics import YOLO


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("Using CPU")
        return "cpu"


def main():
    device = get_device()

    # Upgrade model size
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=960,          # bigger resolution for small objects
        batch=4,            # safe for RTX 3050 4GB
        device=device,
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.0005,
        patience=30,
        workers=4,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=10,
        scale=0.5
    )


if __name__ == "__main__":
    main()
