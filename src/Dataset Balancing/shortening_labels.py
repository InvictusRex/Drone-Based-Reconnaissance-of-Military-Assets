import os
import shutil

def rename_dataset_sequentially(dataset_root):
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(dataset_root, split, 'images')
        labels_dir = os.path.join(dataset_root, split, 'labels')

        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        count = 1
        for image_file in image_files:
            image_base, image_ext = os.path.splitext(image_file)
            label_file = image_base + '.txt'
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, label_file)

            # New names
            new_image_name = f"{count}{image_ext.lower()}"
            new_label_name = f"{count}.txt"

            # Rename image
            os.rename(image_path, os.path.join(images_dir, new_image_name))

            # Rename label (if it exists)
            if os.path.exists(label_path):
                os.rename(label_path, os.path.join(labels_dir, new_label_name))
            else:
                print(f"⚠️ Warning: No label found for image {image_file}")

            count += 1

        print(f"✅ Renamed {count - 1} files in '{split}' set.")

#Example usage:
rename_dataset_sequentially(r'../Augmented Downsampled Dataset')
