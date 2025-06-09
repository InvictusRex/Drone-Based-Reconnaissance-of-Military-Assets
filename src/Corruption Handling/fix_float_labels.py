import os
from glob import glob

dataset_path = r'../../../Datasets/Augmented Downsampled Dataset'  # Change as needed
splits = ["train", "val", "test"]
label_subfolder = "labels"

def fix_label_file(label_path):
    changed = False
    new_lines = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            # Not a valid YOLO line, keep as is
            new_lines.append(line)
            continue

        cls_id_str = parts[0]
        try:
            # Convert class id to float first, then to int (to handle '2.0', etc.)
            cls_id_int = int(float(cls_id_str))
            if cls_id_str != str(cls_id_int):
                changed = True
                parts[0] = str(cls_id_int)
        except ValueError:
            # If cannot convert, keep as is
            new_lines.append(line)
            continue

        new_line = " ".join(parts) + "\n"
        new_lines.append(new_line)

    if changed:
        with open(label_path, "w") as f:
            f.writelines(new_lines)
        print(f"Fixed class IDs in: {label_path}")

for split in splits:
    label_dir = os.path.join(dataset_path, split, label_subfolder)
    label_files = glob(os.path.join(label_dir, "*.txt"))

    for label_file in label_files:
        fix_label_file(label_file)

print("Label files checked and fixed if needed.")
