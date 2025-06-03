import os

dataset_path = r'../Datasets/Combined_Dataset'

splits = ["train", "val", "test"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

unique_images = set()

for split in splits:
    images_dir = os.path.join(dataset_path, split, "images")
    if not os.path.isdir(images_dir):
        print(f"Skipping {split}: images folder not found.")
        continue

    for filename in os.listdir(images_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            unique_images.add(os.path.join(split, "images", filename))

print(f"Total unique images across splits: {len(unique_images)}")