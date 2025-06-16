# Drone-Based Reconnaissance of Military Assets using YOLOv8 and YOLOv11

This project delivers a comprehensive pipeline for real-time detection and classification of military vehicles from UAV footage using advanced YOLOv8 and YOLOv11 deep learning models. The repository encompasses dataset preparation (balancing, augmentation, downsampling), model training and benchmarking, and evaluation of drone-suitable object detectors for mission-critical edge deployment.

---

## 🛰️ Project Overview

- **Goal:** Enable robust, real-time identification of military assets (e.g., tanks, trucks, humvees, armored vehicles) from aerial video.
- **Approach:** Benchmark six YOLO variants (YOLOv8n/s/m, YOLOv11n/s/m) for optimal trade-off between detection accuracy and computational resource usage.
- **Application:** Designed for deployment on UAV platforms requiring fast, energy-efficient inference.

---

## ⚙️ System Architecture

### 1. Dataset Preparation
- **Manual annotation, class balancing, and cleaning**
    - Scripts for label correction, class remapping, and removal of corrupted/duplicate images.
    - Handles multi-class labeling (e.g., tank, truck, armored_vehicle, hummer, soldier, civilian, etc.).
- **Augmentation**
    - Color (HSV), blur, cutout, resizing for improved model generalization.
    - Downsampling and class balancing for fair benchmarking.
- **Utilities**
    - Class distribution visualization, unique image counting, sequential renaming, and more.

### 2. Model Training and Evaluation
- **YOLO Model Variants:** YOLOv8n, YOLOv8s, YOLOv8m, YOLOv11n, YOLOv11s, YOLOv11m.
- **Training Details:**
    - Optimizer: AdamW
    - Input size: 1024×1024
    - Epochs: 100
    - IoU Threshold: 0.7
    - Augmented P2 layer enabled for small object detection
- **Output:** Per-model results and metrics stored in dedicated folders.

### 3. Inference and Deployment
- **Real-time bounding box detection at up to 30 FPS**
- **Scripts for video inference and model evaluation**
- **Suitability for real-world deployment on Jetson/Coral or similar edge hardware**

---

## 📂 Directory Structure (Partial)

```
.
├── Augmented Downsampled Dataset/
├── Full Custom Dataset/
├── Preliminary Datasets/
├── Testing Videos/
├── YOLOv11[msn] Results & Metrics/
├── YOLOv8[msn] Results & Metrics/
├── src/
│   ├── Corruption Handling/
│   ├── Dataset Balancing/
│   ├── annotation_visualization.py
│   ├── augmentation.py
│   ├── class_distribution_analysis.py
│   ├── confident_detection.py
│   ├── detection.py
│   ├── downsampling.py
│   ├── normalized_downsampling.py
│   └── unique_images_count.py
├── Dataset Class Distribution Analysis.png
├── mas-v[8|11][msn](-continued).ipynb
├── README.md
└── .gitattributes
```
*[See the full structure in the [GitHub UI](https://github.com/InvictusRex/Drone-Based-Reconnaissance-of-Military-Assets/tree/main)]*

---

## 🐍 Python Dependencies

To run preprocessing scripts and model inference, install the following:

```bash
pip install ultralytics torch opencv-python scipy matplotlib numpy
```
**(The above covers all imports found in the codebase. Some scripts may require `shutil`, `pathlib`, or other standard libraries.)**

---

## 🚀 Getting Started

### 1. Prepare Datasets
- Place your raw datasets in the respective folders.
- Use scripts in `src/Dataset Balancing/` and `src/Corruption Handling/` to clean, remap, and balance the dataset.

### 2. Augment & Downsample
- Use `src/augmentation.py` for image augmentations.
- Use `src/downsampling.py` and `src/normalized_downsampling.py` for dataset balancing.

### 3. Train Models
- Launch training notebooks: `mas-v8m.ipynb`, `mas-v11n.ipynb`, etc.
- Edit paths and parameters as needed for your environment.

### 4. Run Inference
- Use `src/detection.py` for real-time video inference with a trained YOLO model.
- Example usage (edit paths in script before running):
    ```bash
    python src/detection.py
    ```

---

## 🔢 Model Configurations

| Variant    | Architecture               | GFLOPs | Deployment Suitability         |
|------------|--------------------------- |--------|------------------------------- |
| YOLOv8n    | CSPDarknet (C2f + PAN-FPN) | 8.1    | Ultra-lightweight drones       |
| YOLOv8s    | Medium-scale encoder       | 28.4   | Jetson Nano / Xavier NX        |
| YOLOv8m    | Extended feature backbone  | 78.7   | High-performance edge          |
| YOLOv11n   | Custom PAN+Backbone        | 6.3    | Low-power UAVs                 |
| YOLOv11s   | Intermediate architecture  | 21.3   | Balanced accuracy/efficiency   |
| YOLOv11m   | Deeper spatial encoder     | 67.7   | Edge GPU (Jetson AGX, Xavier)  |

---

## 📊 Evaluation Summary

| Model      | GFLOPs | Precision | Recall | mAP50 | mAP50-95 | F1 Score | Accuracy |
|------------|--------|-----------|--------|-------|----------|----------|----------|
| YOLOv8n    | 8.1    | 0.758     | 0.717  | 0.768 | 0.569    | 0.737    | 0.690    |
| YOLOv8s    | 28.4   | 0.762     | 0.755  | 0.785 | 0.589    | 0.758    | 0.708    |
| YOLOv8m    | 78.7   | 0.810     | 0.771  | 0.827 | 0.633    | 0.790    | 0.752    |
| YOLOv11n   | 6.3    | 0.790     | 0.743  | 0.805 | 0.600    | 0.766    | 0.726    |
| YOLOv11s   | 21.3   | 0.787     | 0.764  | 0.805 | 0.606    | 0.775    | 0.731    |
| YOLOv11m   | 67.7   | 0.816     | 0.792  | 0.832 | 0.640    | 0.803    | 0.758    |

---

## ✅ Model Selection Guidance

- **Best Overall:** YOLOv11m (highest accuracy, for high-end edge GPUs)
- **Best Lightweight:** YOLOv11n (lowest GFLOPs, high mAP, for resource-constrained UAVs)
- **Balanced:** YOLOv8s or YOLOv11s (good accuracy, moderate compute, suitable for real-time deployment)

---

## 🛠️ Core Scripts and Utilities

- `src/annotation_visualization.py`: Visualize dataset annotations.
- `src/class_distribution_analysis.py`: Analyze and plot class distribution.
- `src/augmentation.py`: Perform dataset augmentations.
- `src/detection.py`: Run video inference with trained YOLO models.
- `src/dataset_balancing/`: Fix, remap, filter, and balance datasets.
- `src/corruption_handling/`: Remove corrupted or unpaired labels/images.

---

## 🚧 Known Limitations

- Reduced performance in occlusion, background clutter, low-light.
- Dataset may lack diversity in certain classes or conditions.

---

## 🛠️ Future Work

- Extend to thermal/night-vision data.
- Integrate object tracking (e.g., DeepSORT).
- Real-world UAV deployment and field testing.
- Quantization/pruning for even faster inference.

---

## 📌 Conclusion

This repository establishes a robust, efficient, and modular pipeline for military asset detection via drone imagery, supporting both research and real-world edge deployment. Use the provided scripts and benchmarks to tailor the solution to your hardware and mission needs.

---

**For questions or contributions, please open an issue or pull request.**
