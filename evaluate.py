import torch
from ultralytics import YOLO


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    device = get_device()

    model = YOLO("runs/detect/train2/weights/best.pt")

    print("\nEvaluating on validation set:")
    model.val(device=device)

    print("\nEvaluating on TEST set:")
    model.val(split="test", device=device)


if __name__ == "__main__":
    main()
