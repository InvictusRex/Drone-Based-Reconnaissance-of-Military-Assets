import os

# Path to YOLO dataset
dataset_path = r'../../Datasets/mil_vehicle/train'

# Define folders
images_folder = os.path.join(dataset_path, "images")
labels_folder = os.path.join(dataset_path, "labels")

# Loop through all label files
for label_file in os.listdir(labels_folder):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_folder, label_file)

    # Read annotations
    with open(label_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue

        class_id = int(tokens[0])

        # Skip aircraft (0) and AFV (1)
        if class_id in [0, 1]:
            continue

        # Map all other classes (2, 3, 4, 5) to class 2
        new_line = "2 " + " ".join(tokens[1:]) + "\n"
        updated_lines.append(new_line)

    # If no valid labels remain, delete label and image
    if not updated_lines:
        os.remove(label_path)
        for ext in [".jpg", ".png", ".jpeg"]:
            image_path = os.path.join(images_folder, label_file.replace(".txt", ext))
            if os.path.exists(image_path):
                os.remove(image_path)
                break
    else:
        # Save updated label file
        with open(label_path, "w") as f:
            f.writelines(updated_lines)

print("Classes 0 and 1 removed. Remaining classes mapped to class 2 ('military_vehicle').")
