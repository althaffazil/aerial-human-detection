import torch
from ultralytics import YOLO
import cv2
import os


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    device = get_device()

    model = YOLO("runs/detect/train/weights/best.pt")  # adjust if needed

    source_folder = "dataset/images/test"
    output_folder = "runs/detect/count_output"
    os.makedirs(output_folder, exist_ok=True)

    results = model.predict(
        source=source_folder,
        conf=0.15,
        imgsz=960,
        device=device,
        show_labels=False,
        show_conf=False,
        line_width=1,
        save=False
    )

    print("\nProcessing images with count overlay...\n")

    for result in results:
        img = result.plot(line_width=1, labels=False, conf=False)
        count = len(result.boxes)

        # Add count text (top-left corner)
        cv2.putText(
            img,
            f"People: {count}",
            (20, 40),  # position
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # font scale
            (255, 255, 255),  # white text
            2,  # thickness
            cv2.LINE_AA
        )

        filename = os.path.basename(result.path)
        cv2.imwrite(os.path.join(output_folder, filename), img)

        print(f"{filename} → {count} people")

    print("\nSaved images with count overlay in runs/detect/count_output/")


if __name__ == "__main__":
    main()