from scipy.io import loadmat

# Path to cars_meta.mat
mat_class_path = r"C:\Users\TheKi\Downloads\Stanford_Car\car_devkit\devkit\cars_meta.mat"
output_txt = r"C:\Users\TheKi\Downloads\Stanford_Car\classes.txt"  # or 'names.txt' for YOLOv5/8

# Load the .mat file
meta = loadmat(mat_class_path)
class_names = meta['class_names'][0]  # Each item is a MATLAB cell (1-element array)

# Write to .txt file
with open(output_txt, 'w', encoding='utf-8') as f:
    for cls in class_names:
        f.write(cls[0] + '\n')

print(f"✅ Saved {len(class_names)} class names to {output_txt}")
