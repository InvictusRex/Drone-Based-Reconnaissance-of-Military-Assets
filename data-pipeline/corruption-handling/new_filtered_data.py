import shutil
from pathlib import Path

# Use relative paths (no .resolve() or __file__)
annotated_base = Path('../../../Datasets/Military_Vehicles_Only/Annotations')
original_base = Path('../../../Datasets/Military_Object_Dataset')
final_base = Path('../../../Datasets/Military_Vehicles_Only/Filtered')

splits = ['train', 'val', 'test']

for split in splits:
    annotated_dir = annotated_base / split / 'images'  # <- FIXED
    orig_image_dir = original_base / split / 'images'
    orig_label_dir = original_base / split / 'labels'
    final_image_dir = final_base / split / 'images'
    final_label_dir = final_base / split / 'labels'

    final_image_dir.mkdir(parents=True, exist_ok=True)
    final_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split} set...")
    if not annotated_dir.exists():
        print(f"⚠️  Annotated directory not found: {annotated_dir}")
        continue

    annotated_images = list(annotated_dir.glob('*.*'))
    print(f"Found {len(annotated_images)} annotated images in {annotated_dir}")

    for annotated_image in annotated_images:
        image_stem = annotated_image.stem

        # Try to find and copy the corresponding original image
        copied = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            orig_img = orig_image_dir / f"{image_stem}{ext}"
            if orig_img.exists():
                shutil.copy(orig_img, final_image_dir / orig_img.name)
                copied = True
                break
        if not copied:
            print(f"❌ Image not found in original set: {image_stem}")

        # Copy label
        orig_label = orig_label_dir / f"{image_stem}.txt"
        if orig_label.exists():
            shutil.copy(orig_label, final_label_dir / orig_label.name)
        else:
            print(f"⚠️  Label not found for: {image_stem}")
