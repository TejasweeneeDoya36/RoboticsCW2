import cv2
import torch
from model import load_model
from utils import read_classes
import time

CLASSES = read_classes("classes.txt")
NUM_CLASSES = len(CLASSES)

device = "cpu"

# Load trained model
model = load_model("dofbot_model/mobilenet_v2_office.pth", NUM_CLASSES, device)

# Open Pi camera (0 for USB cam, 0/1 for PiCam depending on setup)
cap = cv2.VideoCapture(0)

t0 = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to match training image size
    img = cv2.resize(frame, (224, 224))
    img_t = torch.tensor(img).permute(2, 0, 1).float() / 255
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_t)
        _, pred_class = preds.max(1)
        label = CLASSES[pred_class]

    # Draw label
    cv2.putText(frame, f"{label}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # FPS counter
    frames += 1
    fps = frames / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("DOFBOT Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
