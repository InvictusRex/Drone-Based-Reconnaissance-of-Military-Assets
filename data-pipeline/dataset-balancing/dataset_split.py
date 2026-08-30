import os
import shutil
import random

# === CONFIGURATION ===
images_dir = r'../../../Datasets/mil_vehicle/train/images'         # Original folder with .jpg images
labels_dir = r'../../../Datasets/mil_vehicle/train/labels'       # Original folder with .txt labels

output_base = r'../../../Datasets/mil_vehicle/split'     # Where the split folders will be created
splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.15, 0.15]   # 70% train, 15% val, 15% test

# === CREATE SPLIT FOLDERS ===
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

# === GATHER AND SHUFFLE DATA ===
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

# === SPLIT ===
total = len(image_files)
train_end = int(split_ratio[0] * total)
val_end = train_end + int(split_ratio[1] * total)

split_indices = {
    'train': image_files[:train_end],
    'val': image_files[train_end:val_end],
    'test': image_files[val_end:]
}

# === COPY FILES ===
for split, files in split_indices.items():
    for img_file in files:
        label_file = img_file.replace('.jpg', '.txt')

        src_img_path = os.path.join(images_dir, img_file)
        src_lbl_path = os.path.join(labels_dir, label_file)

        dst_img_path = os.path.join(output_base, split, 'images', img_file)
        dst_lbl_path = os.path.join(output_base, split, 'labels', label_file)

        shutil.copy(src_img_path, dst_img_path)

        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, dst_lbl_path)
        else:
            print(f"[Warning] Label not found for: {img_file}")

print("✅ Dataset split into train/val/test with corresponding labels.")
