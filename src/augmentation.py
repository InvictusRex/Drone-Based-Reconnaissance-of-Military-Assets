import os
import cv2
import random
import shutil
from glob import glob
from tqdm import tqdm
import albumentations as A

# Class-specific configuration
target_classes = {
    4: "soldier",
    5: "civilian_vehicle",
    2: "military_vehicle"
}
augment_per_class = 1000

# Set dataset paths
dataset_path = r'../../Datasets/Augmented_Dataset'
splits = ["train", "val", "test"]
image_exts = [".jpg", ".jpeg", ".png"]

# Albumentations augmentation pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.GaussianBlur(p=0.2),
    A.HueSaturationValue(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_label(path):
    with open(path, "r") as f:
        return [line.strip().split() for line in f.readlines()]

def save_augmented(image, bboxes, class_ids, save_image_path, save_label_path):
    cv2.imwrite(save_image_path, image)
    with open(save_label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")

# Count how many augmentations done per class
augmented_counts = {cls: 0 for cls in target_classes}

for split in splits:
    image_dir = os.path.join(dataset_path, split, "images")
    label_dir = os.path.join(dataset_path, split, "labels")
    image_files = [f for ext in image_exts for f in glob(os.path.join(image_dir, f"*{ext}"))]

    random.shuffle(image_files)

    for image_path in tqdm(image_files, desc=f"Processing {split}"):
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        label_data = read_yolo_label(label_path)
        if not label_data:
            continue

        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        bboxes = []
        class_ids = []

        for item in label_data:
            cls_id = int(item[0])
            if cls_id in target_classes and augmented_counts[cls_id] < augment_per_class:
                bbox = list(map(float, item[1:]))
                bboxes.append(bbox)
                class_ids.append(cls_id)

        if not bboxes:
            continue

        # Perform augmentation
        try:
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_ids)
        except:
            continue

        if len(augmented["bboxes"]) == 0:
            continue

        # Save new image and label
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_name = f"{base_name}_aug_{random.randint(1000, 9999)}"
        save_img_path = os.path.join(image_dir, new_name + ".jpg")
        save_lbl_path = os.path.join(label_dir, new_name + ".txt")

        save_augmented(augmented["image"], augmented["bboxes"], augmented["class_labels"], save_img_path, save_lbl_path)

        for cid in augmented["class_labels"]:
            augmented_counts[cid] += 1

        # Check if done
        if all(augmented_counts[c] >= augment_per_class for c in target_classes):
            break
    if all(augmented_counts[c] >= augment_per_class for c in target_classes):
        break

print("Augmentation completed. Instance increases per class:")
for cid, count in augmented_counts.items():
    print(f"{target_classes[cid]} ({cid}): {count} new instances")