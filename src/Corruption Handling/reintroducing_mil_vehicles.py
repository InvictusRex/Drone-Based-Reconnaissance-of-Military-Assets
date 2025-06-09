import os
import shutil
from pathlib import Path

# Set the base source and destination directories
source_base = Path(r'../../../Datasets/Military_Object_Dataset')
destination_base = Path(r'../../../Datasets/Military_Vehicles_Only')
# Class index for military_vehicle
target_class_index = '2'

# Dataset splits
splits = ['train', 'val', 'test']

for split in splits:
    src_image_dir = source_base / split / 'images'
    src_label_dir = source_base / split / 'labels'

    dst_image_dir = destination_base / split / 'images'
    dst_label_dir = destination_base / split / 'labels'

    # Create destination folders if they don't exist
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for label_file in src_label_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # If any annotation line starts with the target class
            if any(line.strip().startswith(target_class_index) for line in lines):
                # Copy label file
                shutil.copy(label_file, dst_label_dir / label_file.name)

                # Copy corresponding image
                image_stem = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = src_image_dir / f"{image_stem}{ext}"
                    if image_path.exists():
                        shutil.copy(image_path, dst_image_dir / image_path.name)
                        break

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
