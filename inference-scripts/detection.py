import cv2
from ultralytics import YOLO
import torch
import time

# === CONFIG ===
MODEL_PATH = r'../YOLOv11m Results & Metrics/weights/best.pt'                  # <-- change if needed
VIDEO_PATH = r'../Testing Videos/Test5.mp4'        # <-- your input video
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Draw results
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
