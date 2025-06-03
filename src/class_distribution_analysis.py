import os
from collections import defaultdict

# Class names in order of class IDs
class_names = [
    "military_tank",
    "military_truck",
    "military_vehicle",
    "civilian",
    "soldier",
    "civilian_vehicle",
  ]

def check_yolo_class_distribution(labels_dir):
    class_counts = defaultdict(int)

    # Walk through all label files in the directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    # Sort results by frequency
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nClass Distribution:")
    for class_id, count in sorted_counts:
        name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"{name}: {count} instances")

    if len(sorted_counts) > 1:
        most = sorted_counts[0][1]
        least = sorted_counts[-1][1]
        imbalance_ratio = most / least if least > 0 else float('inf')
        print(f"\nImbalance ratio (most/least): {imbalance_ratio:.2f}")
    else:
        print("Only one class found.")

# Example usage
check_yolo_class_distribution(labels_dir=r'../../Datasets/Combined_Dataset/train/labels')  # Change this to your label path
