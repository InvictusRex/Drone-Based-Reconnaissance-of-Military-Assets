import os
from scipy.io import loadmat
from PIL import Image

# === CONFIGURATION ===
mat_anno_path = r"C:\Users\TheKi\Downloads\Stanford_Car\car_devkit\devkit\cars_train_annos.mat"  # Path to annotations .mat
mat_class_path = r"C:\Users\TheKi\Downloads\Stanford_Car\car_devkit\devkit\cars_meta.mat"        # Path to class names .mat
images_dir = r"C:\Users\TheKi\Downloads\Stanford_Car\cars_train\cars_train"                      # Folder containing training images
labels_dir = r"C:\Users\TheKi\Downloads\Stanford_Car\labels"                   # Output folder for YOLO labels

# Create output label directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# === LOAD DATA ===
print("Loading .mat files...")
anno_data = loadmat(mat_anno_path)['annotations'][0]
class_names = loadmat(mat_class_path)['class_names'][0]

# Class index is 1-based, YOLO uses 0-based
label_map = {i + 1: i for i in range(len(class_names))}

# === CONVERT TO YOLO FORMAT ===
print("Converting annotations...")
for entry in anno_data:
    x1 = int(entry['bbox_x1'][0][0])
    y1 = int(entry['bbox_y1'][0][0])
    x2 = int(entry['bbox_x2'][0][0])
    y2 = int(entry['bbox_y2'][0][0])
    class_id = int(entry['class'][0][0])  # 1-based
    fname = str(entry['fname'][0])

    image_path = os.path.join(images_dir, fname)
    label_path = os.path.join(labels_dir, fname.replace('.jpg', '.txt'))

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except FileNotFoundError:
        print(f"[Warning] Image not found: {image_path}")
        continue

    # Normalize bbox to YOLO format
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    # Save label file
    with open(label_path, 'w') as f:
        f.write(f"{label_map[class_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("✅ Conversion complete. YOLO labels saved to:", labels_dir)