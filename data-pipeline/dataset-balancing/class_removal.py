import os

# Set your dataset path
dataset_path = r'../../Datasets/mil_vehicle/train'  # Update this

# Define YOLO folders
images_folder = os.path.join(dataset_path, "images")
labels_folder = os.path.join(dataset_path, "labels")

# Loop through all label files
for label_file in os.listdir(labels_folder):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_folder, label_file)

    # Read and filter annotations (skip class 1 only)
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_lines = [line for line in lines if line.strip() and line.strip().split()[0] != "1"]

    # If all lines are removed, delete label and corresponding image
    if not new_lines:
        os.remove(label_path)
        for ext in [".jpg", ".png", ".jpeg"]:
            image_path = os.path.join(images_folder, label_file.replace(".txt", ext))
            if os.path.exists(image_path):
                os.remove(image_path)
                break
    else:
        # Save updated label file
        with open(label_path, "w") as f:
            f.writelines(new_lines)

print("AFV class (class 1) removal completed.")