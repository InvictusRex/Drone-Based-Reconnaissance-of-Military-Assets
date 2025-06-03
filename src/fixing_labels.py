import os

def fix_label_class_ids(labels_dir):
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(labels_dir, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = str(int(float(parts[0])))  # Fix float to int
                    fixed_line = " ".join([class_id] + parts[1:])
                    fixed_lines.append(fixed_line)
            
            with open(file_path, 'w') as f:
                f.write("\n".join(fixed_lines) + "\n")

# Example usage:
fix_label_class_ids(r'../../Datasets/Augmented_Dataset/val/labels')