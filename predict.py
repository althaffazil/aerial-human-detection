import torch
from ultralytics import YOLO


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    device = get_device()

    model = YOLO("runs/detect/train/weights/best.pt")  # adjust if needed

    model.predict(
        source="dataset/images/test",
        conf=0.15,
        imgsz=960,
        save=True,
        device=device,
        show_labels=False,   # 🔹 removes class names
        show_conf=False      # 🔹 removes confidence scores
    )

    print("\nPredictions saved to runs/detect/predict/")


if __name__ == "__main__":
    main()
