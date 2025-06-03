import os
import cv2

# === CONFIGURATION ===
images_dir = r'../../Datasets/Combined_Dataset/train/images'         # Folder with .jpg images
labels_dir = r'../../Datasets/Combined_Dataset/train/labels'         # Folder with .txt YOLO labels

# List of class names (ordered by class ID)
class_names = [
    "military_tank",
    "military_truck",
    "military_vehicle",
    "civilian",
    "soldier",
    "civilian_vehicle",
]

# === VISUALIZATION LOOP ===
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))

    # Skip if label doesn't exist
    if not os.path.exists(label_path):
        continue

    # Load image
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Load label(s)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, bw, bh = map(float, parts)
            cls_id = int(cls_id)

            if cls_id < 0 or cls_id >= len(class_names):
                continue  # Skip invalid class IDs

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            # Draw bounding box and class label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[cls_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show image with annotations
    cv2.imshow('YOLO Preview', image)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
