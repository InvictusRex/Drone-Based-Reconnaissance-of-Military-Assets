import os

# === CONFIGURATION ===
labels_dir = r'../Datasets/Stanford_Cars/labels'    # Folder containing YOLO .txt label files
target_class_id = 5            # New class ID for all boxes

# === PROCESS ALL LABEL FILES ===
for fname in os.listdir(labels_dir):
    if not fname.endswith('.txt'):
        continue

    label_path = os.path.join(labels_dir, fname)

    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines
            # Replace class ID with target
            parts[0] = str(target_class_id)
            new_lines.append(' '.join(parts))

    # Overwrite with updated class ID
    with open(label_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

print("✅ All label files updated to class ID 5 (civilian_vehicle).")
