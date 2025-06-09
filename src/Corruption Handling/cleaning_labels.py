import os
from pathlib import Path

# Path to your dataset (adjust this!)
base_path = Path(r'../../../Datasets/Military_Vehicles_Only/Annotations')  # <-- change this

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png']

# Dataset splits
splits = ['train', 'val', 'test']

for split in splits:
    image_dir = base_path / split / 'images'
    label_dir = base_path / split / 'labels'

    for label_file in label_dir.glob('*.txt'):
        label_stem = label_file.stem
        matching_image = None

        for ext in image_extensions:
            if (image_dir / f"{label_stem}{ext}").exists():
                matching_image = image_dir / f"{label_stem}{ext}"
                break

        if matching_image is None:
            print(f"Deleting label with no image: {label_file.name}")
            os.remove(label_file)
