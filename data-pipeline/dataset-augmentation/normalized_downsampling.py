import os
import random

# Configuration
target_class = 5
target_count = 4000
dataset_path = r'../../Datasets/Augmented Downsampled Dataset'
splits = ["train", "val", "test"]
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# Count existing target_class instances and collect deletable files
target_class_instance_count = 0
deletable_files = []

for split in splits:
    labels_dir = os.path.join(dataset_path, split, "labels")
    images_dir = os.path.join(dataset_path, split, "images")

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()]
        target_class_count = class_ids.count(target_class)

        target_class_instance_count += target_class_count

        # If all annotations are target_class, mark for deletion
        if all(cls == target_class for cls in class_ids):
            deletable_files.append((label_path, images_dir, label_file, target_class_count))

# Shuffle to randomly remove
random.shuffle(deletable_files)

# Delete until we bring count down to target_count
to_delete = target_class_instance_count - target_count
deleted_instances = 0
deleted_files = 0

for label_path, images_dir, label_file, count in deletable_files:
    if deleted_instances >= to_delete:
        break

    # Delete label file
    os.remove(label_path)

    # Delete corresponding image
    base_name = os.path.splitext(label_file)[0]
    for ext in image_extensions:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            os.remove(img_path)
            break

    deleted_instances += count
    deleted_files += 1

print(f"Deleted {deleted_files} image-label pairs containing only class {target_class} to reduce count to ~{target_count}.")
print(f"New class {target_class} estimate: {target_class_instance_count - deleted_instances}")
