import os
import shutil

# === CONFIGURATION ===
dataset_base = r"E:\1_Work_Files\Internship - Garuda Aerospace\MVD\Dataset\Filtered_MVD"      # Original dataset folder with train/val/test subfolders
output_base = r'../../Datasets/Filtered Dataset'    # Folder to save filtered images + labels

# Original classes
class_name_to_id = {
    'tank': 0,
    'armored_vehicle': 1,
    'truck': 2,
    'hummer': 3,
    'rocket_artillery': 4
}

# Target class map (your class_map)
class_map = {
    0: 'military_tank',
    1: 'military_truck',
    2: 'military_vehicle',
    3: 'civilian',
    4: 'soldier',
    5: 'civilian_vehicle'
}

# Classes to filter and their remap (old -> new)
remap_dict = {
    class_name_to_id['truck']: 1,          # truck (2) -> military_truck (1)
    class_name_to_id['armored_vehicle']: 2 # armored_vehicle (1) -> military_vehicle (2)
}

# We only keep images/labels that contain these class IDs
class_ids_to_keep = set(remap_dict.keys())

# Dataset splits
splits = ['train', 'val', 'test']

# Create output folders
for split in splits:
    os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

for split in splits:
    images_dir = os.path.join(dataset_base, split, 'images')
    labels_dir = os.path.join(dataset_base, split, 'labels')

    filtered_images_dir = os.path.join(output_base, split, 'images')
    filtered_labels_dir = os.path.join(output_base, split, 'labels')

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Check if label file has any bounding box of classes to keep
        keep = any(int(line.strip().split()[0]) in class_ids_to_keep for line in lines)

        if keep:
            img_file = label_file.replace('.txt', '.jpg')
            src_img_path = os.path.join(images_dir, img_file)
            src_label_path = label_path

            dst_img_path = os.path.join(filtered_images_dir, img_file)
            dst_label_path = os.path.join(filtered_labels_dir, label_file)

            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)

                # Remap classes in labels and save
                with open(src_label_path, 'r') as f:
                    old_lines = f.readlines()
                new_lines = []
                for line in old_lines:
                    parts = line.strip().split()
                    old_cls_id = int(parts[0])
                    if old_cls_id in remap_dict:
                        new_cls_id = remap_dict[old_cls_id]
                        new_line = f"{new_cls_id} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)
                    # skip other classes
                
                with open(dst_label_path, 'w') as f:
                    f.writelines(new_lines)
            else:
                print(f"[Warning] Image missing: {src_img_path}")

print("✅ Filtered dataset created with remapped classes.")
