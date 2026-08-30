import cv2
import os
from pathlib import Path

# Class names from your dataset
class_names = [
    "military_tank",       # 0
    "military_truck",      # 1
    "military_vehicle",    # 2 
    "civilian",            # 3
    "soldier",             # 4
    "civilian_vehicle",    # 5
]

# Directories
source_base = Path(r'../../../Datasets/Military_Vehicles_Only')  # <-- your base folder
output_base = Path(r'../../../Datasets/Military_Vehicles_Only/Annotations')

# Splits to process
splits = ['train', 'val', 'test']

for split in splits:
    image_dir = source_base / split / 'images'
    label_dir = source_base / split / 'labels'
    out_dir = output_base / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_file in label_dir.glob('*.txt'):
        image_stem = label_file.stem
        image_path = None

        for ext in ['.jpg', '.jpeg', '.png']:
            possible_path = image_dir / f"{image_stem}{ext}"
            if possible_path.exists():
                image_path = possible_path
                break

        if image_path is None:
            print(f"Image not found for {label_file.name}")
            continue

        # Load image
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        # Read and draw bounding boxes
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, width, height = map(float, parts)
            cls_id = int(cls_id)

            # Convert from YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save annotated image
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")
