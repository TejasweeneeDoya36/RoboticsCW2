# main.py  (run_both)
import time
import cv2
import torch

from Arm_Lib import Arm_Device          # DOFBOT servo library
from model import load_model
from utils import read_classes

# -------------- CONFIG --------------
IMG_SIZE = 160        # smaller than 224 → faster but still fine
CAM_INDEX = 1         # your working camera index
CLASSIFY_EVERY = 4    # run model every N frames
STABLE_N = 3          # need same label N inferences in a row
COOLDOWN_SEC = 3.0    # minimum time between arm actions
# ------------------------------------

CLASSES = read_classes("docs/classes.txt")
NUM_CLASSES = len(CLASSES)
device = "cpu"

print("Loading model...")
model = load_model("models/mobilenet_v2_office.pth", NUM_CLASSES, device)
model.eval()
torch.set_num_threads(2)

# ImageNet / MobileNet normalisation
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

# ----- Setup DOFBOT arm -----
arm = Arm_Device()

SAFE_POSE     = [90, 100, 90, 90, 90, 90]
PENDRIVE_POSE = [60, 110, 80, 90, 90, 40]
MOUSE_POSE    = [120, 110, 80, 90, 90, 40]
PEN_POSE      = [90, 120, 70, 90, 90, 20]
ADAPTER_POSE  = [90, 90, 110, 90, 90, 40]

# Make sure this covers ALL labels from docs/classes.txt
CLASS_POSES = {
    "mouse": MOUSE_POSE,
    "pen": PEN_POSE,
    "pendrive": PENDRIVE_POSE,
    "Eraser": SAFE_POSE,      # temporarily use SAFE pose
    "stapler": SAFE_POSE,     # temporarily use SAFE pose
    "adapter": ADAPTER_POSE,
}

def move_pose(pose, duration=800):
    """Move all 6 DOF to given pose in 'duration' ms."""
    for idx, angle in enumerate(pose, start=1):
        arm.Arm_serial_servo_write(idx, angle, duration)
    time.sleep(duration / 1000.0)

print("Moving to safe pose...")
move_pose(SAFE_POSE, 800)

# ----- Open camera -----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX}")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Camera opened. Press ESC to quit.")

t0 = time.perf_counter()
frame_count = 0
inf_count = 0
frame_idx = 0
last_label = "..."
stable_label = None
stable_count = 0
last_action_time = 0.0

with torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        frame_count += 1
        frame_idx += 1
        do_infer = (frame_idx % CLASSIFY_EVERY == 0)

        if do_infer:
            # Preprocess
            small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            img = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device).float() / 255.0
            img = (img - mean) / std
            img = img.unsqueeze(0)

            logits = model(img)
            _, pred = logits.max(1)
            pred_idx = int(pred.item())
            label = CLASSES[pred_idx]

            # Debug: see exactly what model predicts
            print("Prediction index:", pred_idx, "| Label:", label)

            # Stability check
            if label == stable_label:
                stable_count += 1
            else:
                stable_label = label
                stable_count = 1

            last_label = label
            inf_count += 1

            # Trigger arm action when:
            # - label is known in CLASS_POSES
            # - stable for STABLE_N inferences
            # - cooldown interval passed
            now = time.perf_counter()
            if (
                label in CLASS_POSES
                and stable_count >= STABLE_N
                and (now - last_action_time) >= COOLDOWN_SEC
            ):
                print(f"[ARM] Detected stable '{label}' – moving arm.")
                move_pose(CLASS_POSES[label], 600)
                move_pose(SAFE_POSE, 800)
                last_action_time = now

        # FPS estimations
        now = time.perf_counter()
        dt = now - t0
        cam_fps = frame_count / dt if dt > 0 else 0.0
        inf_fps = inf_count / dt if dt > 0 else 0.0

        # Draw info
        cv2.putText(frame, f"{last_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"camFPS: {cam_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"infFPS: {inf_fps:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Stable: {stable_label} ({stable_count})", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("DOFBOT Live Classification + Arm", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

print("Returning to safe pose...")
move_pose(SAFE_POSE, 800)
print("Done.")