import os
import random
import shutil
from glob import glob

# Paths - change these!
src_dataset_path = r'../../../Datasets/Stanford_Cars/split'  # source dataset root
dst_dataset_path = r'../../../Datasets/Stanford_Cars/random 1-4'  # destination root for subset

splits = ["train", "val", "test"]
image_exts = [".jpg", ".jpeg", ".png"]

fraction_to_copy = 0.25  # 1/4th

for split in splits:
    src_img_dir = os.path.join(src_dataset_path, split, "images")
    src_lbl_dir = os.path.join(src_dataset_path, split, "labels")

    dst_img_dir = os.path.join(dst_dataset_path, split, "images")
    dst_lbl_dir = os.path.join(dst_dataset_path, split, "labels")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # Gather all images of allowed extensions
    all_images = []
    for ext in image_exts:
        all_images.extend(glob(os.path.join(src_img_dir, f"*{ext}")))

    # Randomly select fraction_to_copy images
    num_to_select = int(len(all_images) * fraction_to_copy)
    selected_images = random.sample(all_images, num_to_select)

    print(f"{split}: copying {num_to_select} images and labels...")

    for img_path in selected_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Source label path
        label_path = os.path.join(src_lbl_dir, base_name + ".txt")

        # Destination paths
        dst_img_path = os.path.join(dst_img_dir, os.path.basename(img_path))
        dst_lbl_path = os.path.join(dst_lbl_dir, base_name + ".txt")

        # Copy image
        shutil.copy2(img_path, dst_img_path)

        # Copy label if exists
        if os.path.exists(label_path):
            shutil.copy2(label_path, dst_lbl_path)
        else:
            print(f"Warning: Label not found for image {img_path}")

print("Subset creation complete.")
