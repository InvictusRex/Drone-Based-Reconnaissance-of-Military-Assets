import os
import random
from collections import defaultdict

# === CONFIGURATION ===
root_dir = r'../../Datasets/Dataset_1'
target_total_images = 25000  # Target dataset size

# === STEP 1: Map class occurrences to file paths ===
class_to_files = defaultdict(set)
file_to_classes = defaultdict(set)
all_label_files = []

splits = ["train", "val", "test"]
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

for split in splits:
    labels_dir = os.path.join(root_dir, split, "labels")
    if not os.path.exists(labels_dir):
        continue

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)
        all_label_files.append((split, label_file))

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_to_files[class_id].add((split, label_file))
                file_to_classes[(split, label_file)].add(class_id)

print(f"Initial total image-label pairs: {len(all_label_files)}")

# === STEP 2: Compute reduction ratio per class ===
total_files = len(set(file_to_classes.keys()))
if total_files <= target_total_images:
    print("No reduction needed.")
    exit()

reduction_needed = total_files - target_total_images
print(f"Reducing {reduction_needed} image-label pairs...")

# === STEP 3: Select files to remove ===
# We'll randomly drop files per class until target is met, avoiding duplicates
removed_files = set()
class_files = list(file_to_classes.keys())
random.shuffle(class_files)

for split_file in class_files:
    if total_files - len(removed_files) <= target_total_images:
        break
    removed_files.add(split_file)

# === STEP 4: Delete selected files ===
for split, label_file in removed_files:
    labels_dir = os.path.join(root_dir, split, "labels")
    images_dir = os.path.join(root_dir, split, "images")

    label_path = os.path.join(labels_dir, label_file)
    if os.path.exists(label_path):
        os.remove(label_path)

    base_name = os.path.splitext(label_file)[0]
    for ext in image_extensions:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            os.remove(img_path)
            break

print(f"Deleted {len(removed_files)} image-label pairs.")
print(f"Remaining: {total_files - len(removed_files)} images.")
