import os

# Class remapping: TriNet original class ID -> Merged class ID
remap_classes = {
    0: 4,  # Military -> soldier
    1: 4,  # Para-military -> soldier
    2: 3   # Non-military -> civilian
}

# Root of your TriNet dataset
dataset_root = r"E:\1_Work_Files\D_Research Paper - Military Assets Detection\Drone-Based-Reconnaissance-of-Military-Assets\Datasets\Military_Assets\TriNet"

splits = ["train", "val", "test"]

for split in splits:
    labels_dir = os.path.join(dataset_root, split, "labels")

    if not os.path.isdir(labels_dir):
        print(f"[!] Skipping '{split}' - no labels folder found at {labels_dir}")
        continue

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    print(f"\nProcessing {len(label_files)} label files in '{split}/labels'...")

    for file in label_files:
        label_path = os.path.join(labels_dir, file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in remap_classes:
                original_class = class_id
                parts[0] = str(remap_classes[class_id])
                new_lines.append(" ".join(parts) + "\n")
            else:
                print(f"[!] Unknown class ID {class_id} in file: {file}")

        # Overwrite label file with remapped classes
        with open(label_path, "w") as f:
            f.writelines(new_lines)

    print(f"[✓] Finished relabeling: {split}/labels")

print("\n✅ All class IDs in TriNet dataset relabeled successfully.")
