import os

# Path to the new dataset root folder (containing train/val/test)
dataset_path = r'../Datasets/Person detection.v16i.yolov8'

# Classes info
persona_class_id = 0       # old class label in new dataset
civilian_class_id = 3      # new class label you want to assign

splits = ["train", "val", "test"]
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

total_deleted = 0

for split in splits:
    labels_dir = os.path.join(dataset_path, split, "labels")
    images_dir = os.path.join(dataset_path, split, "images")

    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        print(f"Skipping {split} - missing labels or images folder.")
        continue

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    deleted_count = 0

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Filter lines that have persona class, and relabel them as civilian (3)
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == persona_class_id:
                parts[0] = str(civilian_class_id)
                new_lines.append(" ".join(parts) + "\n")

        if len(new_lines) == 0:
            # No persona class found, delete label and image
            os.remove(label_path)
            base_name = os.path.splitext(label_file)[0]
            for ext in image_extensions:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    break
            deleted_count += 1
            print(f"Deleted {label_file} and its image because no persona class found.")
        else:
            # Overwrite label file with relabeled lines only
            with open(label_path, "w") as f:
                f.writelines(new_lines)

    print(f"{split}: Deleted {deleted_count} image-label pairs with no persona class.")
    total_deleted += deleted_count

print(f"Total deleted image-label pairs: {total_deleted}")
