import cv2
from ultralytics import YOLO
import torch
import time

# === CONFIG ===
MODEL_PATH = r'../YOLOv11s Results & Metrics/weights/best.pt'  # <-- change if needed
VIDEO_PATH = r'../Testing Videos/Test5.mp4'                    # <-- your input video
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.4

# === LOAD MODEL ===
print(f'Using device: {DEVICE}')
model = YOLO(MODEL_PATH).to(DEVICE)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, device=DEVICE)
    r = results[0]

    # Filter detections by confidence
    if r.boxes is not None and r.boxes.conf is not None:
        conf_mask = r.boxes.conf > CONFIDENCE_THRESHOLD
        r.boxes = r.boxes[conf_mask]

    # Draw filtered results
    annotated_frame = r.plot()

    # Show frame
    cv2.imshow("YOLOv8/YOLOv11 Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
