import time
import cv2
import torch

from model import load_model
from utils import read_classes

# ----------------- CONFIG -----------------
IMG_SIZE = 224            # training size
CAM_INDEX = 1             # we discovered index 1 works
CLASSIFY_EVERY = 1        # set to 2 or 3 to skip frames if FPS is too low
# ------------------------------------------

# ----- Load classes -----
CLASSES = read_classes("docs/classes.txt")
NUM_CLASSES = len(CLASSES)

device = "cpu"

print("Loading model...")
model = load_model("models/mobilenet_v2_office.pth", NUM_CLASSES, device)
model.eval()

# mean/std used during training (ImageNet / MobileNet defaults)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

# ----- Open camera -----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX}")
    raise SystemExit

# Smaller resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera opened. Press ESC to quit.")

t0 = time.time()
frames = 0
frame_idx = 0
last_label = "..."

with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        frames += 1
        frame_idx += 1

        # Only run the network every N frames (optional speed boost)
        if frame_idx % CLASSIFY_EVERY == 0:
            # Resize and preprocess
            frame_small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            img = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device).float() / 255.0
            img = (img - mean) / std         # normalisation
            img = img.unsqueeze(0)           # add batch dim

            # Inference
            logits = model(img)
            _, pred = logits.max(1)
            last_label = CLASSES[pred]

        # FPS from total frames processed
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0

        # Draw label & FPS on original (320x240) frame
        cv2.putText(frame, f"{last_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("DOFBOT Live Classification", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()

