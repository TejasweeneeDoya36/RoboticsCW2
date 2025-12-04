# detect_yolo_arm.py
#
# YOLO detection + DOFBOT arm behaviour:
# - SEARCH mode: sweep base servo left/right while no objects detected
# - When an object is detected:
#     * pick highest-confidence detection
#     * slightly rotate base to center the object
#     * move to class-specific pose (pre-calibrated)
#     * return to SAFE pose and resume search
#
# Camera index is 1 (as per your scan_cams.py result).

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from Arm_Lib import Arm_Device
from utils import read_classes

# ---------------- CONFIG ----------------

CAM_INDEX = 1
CONF_THRES = 0.6

# Use your trained YOLO model here once it's ready:
MODEL_PATH = "models/office_yolo.pt"

# How often the arm is allowed to perform a pick action (seconds)
ACTION_COOLDOWN = 5.0

# Search sweep config (base servo)
SEARCH_MIN_ANGLE = 50
SEARCH_MAX_ANGLE = 130
SEARCH_STEP_DEG = 5
SEARCH_MOVE_INTERVAL = 1.0  # seconds between sweep moves

# SAFE & class poses (tune if needed)
SAFE_POSE     = [90, 100, 90, 90, 90, 90]
PENDRIVE_POSE = [60, 110, 80, 90, 90, 40]
MOUSE_POSE    = [120, 110, 80, 90, 90, 40]
PEN_POSE      = [90, 120, 70, 90, 90, 20]
ERASER_POSE   = [85, 110, 85, 90, 90, 30]
STAPLER_POSE  = [95, 105, 80, 90, 90, 35]
ADAPTER_POSE  = [90, 90, 110, 90, 90, 40]

CLASS_POSES = {
    "mouse":    MOUSE_POSE,
    "pen":      PEN_POSE,
    "pendrive": PENDRIVE_POSE,
    "Eraser":   ERASER_POSE,
    "stapler":  STAPLER_POSE,
    "adapter":  ADAPTER_POSE,
}

# ----------------------------------------


def move_pose(arm, pose, duration=800):
    """Move all 6 servos to 'pose' over 'duration' ms."""
    for idx, angle in enumerate(pose, start=1):
        arm.Arm_serial_servo_write(idx, angle, duration)
    time.sleep(duration / 1000.0)


def set_base_angle(arm, angle, duration=500):
    """Move only the base (servo 1)."""
    angle = max(0, min(180, int(angle)))
    arm.Arm_serial_servo_write(1, angle, duration)
    time.sleep(duration / 1000.0)


def main():
    # ---------- Load classes ----------
    classes_path = Path("docs") / "classes.txt"
    class_names = read_classes(str(classes_path))

    # ---------- Load YOLO model ----------
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    # ---------- Setup arm ----------
    arm = Arm_Device()
    print("[INFO] Moving arm to SAFE pose...")
    move_pose(arm, SAFE_POSE, duration=800)

    # Base servo starts in center
    base_angle = SAFE_POSE[0]
    search_direction = 1  # 1 = going right, -1 = going left
    last_search_move_time = time.perf_counter()
    last_action_time = 0.0

    # ---------- Setup camera ----------
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] YOLO detection + arm control running. Press ESC to quit.")

    t0 = time.perf_counter()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame from camera")
            break

        frame_count += 1
        h, w, _ = frame.shape
        frame_center_x = w / 2.0

        # ---------- YOLO inference ----------
        results_list = model(frame, verbose=False)
        if not results_list:
            results = None
        else:
            results = results_list[0]

        detections = []
        if results is not None and hasattr(results, "boxes"):
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < CONF_THRES:
                    continue

                # Prefer YOLO's names if available
                if hasattr(results, "names") and cls_id in results.names:
                    label = results.names[cls_id]
                else:
                    label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                detections.append((conf, label, x1_i, y1_i, x2_i, y2_i))

        # ---------- Draw detections ----------
        for conf, label, x1_i, y1_i, x2_i, y2_i in detections:
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1_i, max(15, y1_i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = time.perf_counter()

        # ---------- Behaviour logic ----------
        if detections:
            # We have at least one detection → TRACK + ACTION mode
            detections.sort(key=lambda d: d[0], reverse=True)
            best_conf, best_label, bx1, by1, bx2, by2 = detections[0]

            # Centering: adjust base to reduce horizontal offset
            cx = 0.5 * (bx1 + bx2)
            offset_norm = (cx - frame_center_x) / frame_center_x  # -1 .. 1

            if abs(offset_norm) > 0.1:  # only adjust if significantly off-center
                delta_angle = -offset_norm * 10.0  # 10 degrees per 0.1 offset
                base_angle += delta_angle
                # Clamp to safe sweep range
                if base_angle > SEARCH_MAX_ANGLE:
                    base_angle = SEARCH_MAX_ANGLE
                if base_angle < SEARCH_MIN_ANGLE:
                    base_angle = SEARCH_MIN_ANGLE
                print(f"[TRACK] Adjusting base to {base_angle:.1f}° for centering {best_label}")
                set_base_angle(arm, base_angle, duration=400)

            # Trigger action (pick pose) with cooldown
            if (best_label in CLASS_POSES) and (now - last_action_time >= ACTION_COOLDOWN):
                print(f"[ACTION] Detected {best_label} ({best_conf:.2f}) → moving to its pose")
                move_pose(arm, CLASS_POSES[best_label], duration=800)
                move_pose(arm, SAFE_POSE, duration=800)
                base_angle = SAFE_POSE[0]
                last_action_time = now

        else:
            # No detections → SEARCH mode (sweep base left/right)
            if now - last_search_move_time >= SEARCH_MOVE_INTERVAL:
                # Reverse direction at limits
                if base_angle >= SEARCH_MAX_ANGLE:
                    search_direction = -1
                elif base_angle <= SEARCH_MIN_ANGLE:
                    search_direction = 1

                base_angle += SEARCH_STEP_DEG * search_direction
                print(f"[SEARCH] Sweeping base to {base_angle}°")
                set_base_angle(arm, base_angle, duration=500)
                last_search_move_time = now

        # ---------- FPS display ----------
        dt = now - t0
        fps = frame_count / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("YOLO – Detect + DOFBOT Arm", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    # ---------- Cleanup ----------
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Returning arm to SAFE pose and exiting...")
    move_pose(arm, SAFE_POSE, duration=800)


if __name__ == "__main__":
    main()
