import os
from pathlib import Path

# Base dataset directory
base_path = Path(r'../../../Datasets/Augmented Downsampled Dataset')

# Class index to remove (civilian_vehicle = index 5)
target_class_index = '2'

# Dataset splits
splits = ['train', 'val', 'test']

# Process each split
for split in splits:
    image_dir = base_path / split / 'images'
    label_dir = base_path / split / 'labels'

    for label_file in label_dir.glob('*.txt'):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            # If any annotation line starts with the target class index
            if any(line.strip().startswith(target_class_index) for line in lines):
                # Delete the label file
                os.remove(label_file)
                print(f"Deleted label: {label_file}")

                # Get the image file name (same stem)
                image_stem = label_file.stem
                # Try different common image extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = image_dir / f"{image_stem}{ext}"
                    if image_path.exists():
                        os.remove(image_path)
                        print(f"Deleted image: {image_path}")
                        break

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
