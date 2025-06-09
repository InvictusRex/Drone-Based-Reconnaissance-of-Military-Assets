import os
import cv2
import random
from glob import glob
from tqdm import tqdm
import albumentations as A

# Configuration
target_class_id = 5  # military_vehicle
augment_per_class = 200

# Dataset paths
dataset_path = r'../../../Datasets/Augmented Downsampled Dataset'  # change if needed
splits = ["train", "val", "test"]
image_exts = [".jpg", ".jpeg", ".png"]

# Albumentations pipeline
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.GaussianBlur(p=0.2),
    A.HueSaturationValue(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_label(path):
    with open(path, "r") as f:
        return [line.strip().split() for line in f if line.strip()]

def save_augmented(image, bboxes, class_ids, save_image_path, save_label_path):
    cv2.imwrite(save_image_path, image)
    with open(save_label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")

augmented_count = 0

for split in splits:
    image_dir = os.path.join(dataset_path, split, "images")
    label_dir = os.path.join(dataset_path, split, "labels")

    # Collect all images with allowed extensions
    image_files = []
    for ext in image_exts:
        image_files.extend(glob(os.path.join(image_dir, f"*{ext}")))
    random.shuffle(image_files)

    for image_path in tqdm(image_files, desc=f"Augmenting {split}"):
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        label_data = read_yolo_label(label_path)
        if not label_data:
            continue

        # Extract only boxes with target_class_id
        bboxes = []
        class_ids = []

        for item in label_data:
            if len(item) < 5:
                continue
            try:
                cls_id = int(item[0])
                bbox = list(map(float, item[1:5]))
            except:
                continue

            if cls_id == target_class_id:
                bboxes.append(bbox)
                class_ids.append(cls_id)

        if not bboxes:
            continue  # No military_vehicle in this image, skip

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Apply augmentation
        try:
            augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_ids)
        except Exception as e:
            print(f"Augmentation failed for {image_path}: {e}")
            continue

        if len(augmented["bboxes"]) == 0:
            continue  # no boxes after augmentation, skip

        # Save augmented image and labels
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_name = f"{base_name}_aug_{random.randint(1000, 9999)}"
        save_img_path = os.path.join(image_dir, new_name + ".jpg")
        save_lbl_path = os.path.join(label_dir, new_name + ".txt")

        save_augmented(augmented["image"], augmented["bboxes"], augmented["class_labels"], save_img_path, save_lbl_path)

        augmented_count += 1

        if augmented_count >= augment_per_class:
            print(f"Reached augmentation limit of {augment_per_class} for class {target_class_id}")
            break
    if augmented_count >= augment_per_class:
        break

print(f"Augmentation completed: {augmented_count} new instances of class {target_class_id}")
