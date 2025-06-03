import os

keep_classes = {
    2: 0,  # military_tank
    3: 1,  # military_truck
    4: 2,  # military_vehicle
    5: 3,  # civilian
    6: 4,  # soldier
    7: 5   # civilian_vehicle
}

dataset_path = r"E:\1_Work_Files\D_Research Paper - Military Assets Detection\Drone-Based-Reconnaissance-of-Military-Assets\Datasets\Military_Assets\military_object_dataset"

image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

splits = ["train", "val", "test"]

total_deleted = 0

for split in splits:
    labels_dir = os.path.join(dataset_path, split, "labels")
    images_dir = os.path.join(dataset_path, split, "images")

    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        print(f"Skipping {split} - labels or images folder missing.")
        continue

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    deleted_files_count = 0

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        original_classes = set()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            original_classes.add(class_id)
            if class_id in keep_classes:
                parts[0] = str(keep_classes[class_id])
                new_lines.append(" ".join(parts) + "\n")

        if len(new_lines) == 0:
            # Delete label file
            os.remove(label_path)
            # Delete corresponding image file
            base_name = os.path.splitext(label_file)[0]
            deleted_img_name = None
            for ext in image_extensions:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    deleted_img_name = base_name + ext
                    break
            deleted_files_count += 1
            # Print deleted image and classes it had
            print(f"Deleted image: {deleted_img_name}, classes present: {sorted(original_classes)}")
        else:
            # Overwrite label file with filtered lines
            with open(label_path, "w") as f:
                f.writelines(new_lines)

    print(f"{split}: Deleted {deleted_files_count} image-label pairs with no valid classes.")
    total_deleted += deleted_files_count

print(f"Total deleted image-label pairs across splits: {total_deleted}")