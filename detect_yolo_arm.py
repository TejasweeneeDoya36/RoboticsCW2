# detect_yolo_arm.py
import time
import cv2
from ultralytics import YOLO

from Arm_Lib import Arm_Device
from utils import read_classes

# ---------- CONFIG ----------
CAM_INDEX = 1
CONF_THRES = 0.6
MODEL_PATH = "models/office_yolo.pt"
COOLDOWN_SEC = 3.0
# ----------------------------

CLASSES = read_classes("docs/classes.txt")

print("Loading YOLO detection model...")
model = YOLO(MODEL_PATH)

# ----- Setup DOFBOT arm -----
arm = Arm_Device()

SAFE_POSE     = [90, 100, 90, 90, 90, 90]
PENDRIVE_POSE = [60, 110, 80, 90, 90, 40]
MOUSE_POSE    = [120, 110, 80, 90, 90, 40]
PEN_POSE      = [90, 120, 70, 90, 90, 20]
ADAPTER_POSE  = [90, 90, 110, 90, 90, 40]

CLASS_POSES = {
    "mouse": MOUSE_POSE,
    "pen": PEN_POSE,
    "pendrive": PENDRIVE_POSE,
    "Eraser": SAFE_POSE,      # adjust later
    "stapler": SAFE_POSE,     # adjust later
    "adapter": ADAPTER_POSE,
}

def move_pose(pose, duration=800):
    for idx, angle in enumerate(pose, start=1):
        arm.Arm_serial_servo_write(idx, angle, duration)
    time.sleep(duration / 1000.0)

print("Moving arm to SAFE pose...")
move_pose(SAFE_POSE, 800)

# ----- Open camera -----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_INDEX}")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("YOLO detection + arm. Press ESC to quit.")

t0 = time.perf_counter()
frame_count = 0
last_action_time = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to grab frame")
        break

    frame_count += 1

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf < CONF_THRES:
            continue

        if hasattr(results, "names") and cls_id in results.names:
            label = results.names[cls_id]
        else:
            label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        detections.append((conf, label, x1_i, y1_i, x2_i, y2_i))

    # Draw all detections
    for conf, label, x1_i, y1_i, x2_i, y2_i in detections:
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1_i, max(15, y1_i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- Arm logic: pick the highest-confidence detection ----
    if detections:
        # sort by confidence descending
        detections.sort(key=lambda d: d[0], reverse=True)
        best_conf, best_label, bx1, by1, bx2, by2 = detections[0]

        now = time.perf_counter()
        if (best_label in CLASS_POSES) and (now - last_action_time >= COOLDOWN_SEC):
            print(f"[ARM] Best detection: {best_label} ({best_conf:.2f}) → moving arm")
            move_pose(CLASS_POSES[best_label], 600)
            move_pose(SAFE_POSE, 800)
            last_action_time = now

    # FPS
    now = time.perf_counter()
    dt = now - t0
    fps = frame_count / dt if dt > 0 else 0.0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("YOLO – Detection + Arm", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Returning arm to SAFE pose...")
move_pose(SAFE_POSE, 800)
print("Done.")
