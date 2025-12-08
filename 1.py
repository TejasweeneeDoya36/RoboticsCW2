#!/usr/bin/env python3
# detect_webcam_dofbot.py
#
# Use YOLO + webcam to inspect the desk while DOFBOT stands
# in a fixed "scan pose" and only the BASE (servo 1) rotates
# through a hard-coded range of angles.
#
# When YOLO sees a known object in the camera view, we call:
#   move_object_named(<label>)
# which moves it from <label>_wrong to <label>_correct
# using your existing slot_move_demo.py logic.

import time
import cv2
from ultralytics import YOLO

# ---- Import your DOFBOT motion logic & Arm instance ----
from slot_move_demo import move_object_named, PAIR_MAP, go_safe_open, Arm

# ===================== CONFIG =====================

# Camera / model config
CAM_INDEX   = 1
MODEL_PATH  = "models/office_yolo.pt"
CONF_THRES  = 0.5

# Base scan angles (servo 1) – EDIT as you like
SCAN_BASE_ANGLES = [30, 90]   # degrees
SCAN_STEP_TIME   = 2.0             # seconds between steps

# Map YOLO class indices → logical names used in PAIR_MAP / SLOTS_GRIP
# Make sure this matches your training order in Roboflow!
CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

# ==================================================

def read_all_servos():
    """Read current angles of all 6 servos from DOFBOT."""
    return [Arm.Arm_serial_servo_read(i + 1) for i in range(6)]

def move_base_with_same_pose(base_angle, template_pose, move_time=1000):
    """
    Move only the base (servo 1) to base_angle,
    keep servos 2–6 equal to template_pose.
    """
    pose = template_pose.copy()
    pose[0] = base_angle
    print(f"[SCAN] Rotating base to {base_angle}° with pose {pose}")
    Arm.Arm_serial_servo_write6(*pose, move_time)
    time.sleep(move_time / 1000.0)

def main():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    # Put DOFBOT in a safe scan pose (arm up, gripper open)
    print("[INFO] Moving DOFBOT to safe scan pose (go_safe_open)...")
    go_safe_open()
    time.sleep(1.0)

    # Read that pose as our scan template (we'll only change servo 1)
    scan_template = read_all_servos()
    print(f"[INFO] Scan template pose (servos 2–6 fixed): {scan_template}")

    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    # Optional: configure resolution & FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Press ESC in the window to quit.")
    print("[INFO] Place an out-of-place object somewhere in the workspace.")

    frame_count     = 0
    t0              = time.perf_counter()
    scan_index      = 0
    last_scan_time  = time.time()
    moving_robot    = False  # True while executing pick/place

    # Keep track of which objects we already moved
    already_moved = set()

    while True:
        # ====== BASE SCAN CONTROL (SERVO 1 ONLY) ======
        now = time.time()
        if not moving_robot and (now - last_scan_time) > SCAN_STEP_TIME:
            # Move to next base angle in the scan list
            base_angle = SCAN_BASE_ANGLES[scan_index]
            move_base_with_same_pose(base_angle, scan_template)
            scan_index = (scan_index + 1) % len(SCAN_BASE_ANGLES)
            last_scan_time = now

        # ====== YOLO INFERENCE ======
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame")
            break

        frame_count += 1

        results_list = model(frame, verbose=False)
        results = results_list[0] if results_list else None

        detected_objects = []  # (label, conf, bbox)

        if results is not None and hasattr(results, "boxes"):
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < CONF_THRES:
                    continue

                label = CUSTOM_NAMES.get(cls_id, f"class_{cls_id}")
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

                detected_objects.append((label, conf, (x1_i, y1_i, x2_i, y2_i)))

                # Draw bounding box for debugging
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1_i, max(15, y1_i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # ====== DECIDE ROBOT ACTION ======
        if not moving_robot:
            for label, conf, _bbox in detected_objects:
                # Only move objects we know how to handle
                if label in PAIR_MAP and label not in already_moved:
                    print(f"[DETECT] Found '{label}' with conf={conf:.2f}.")
                    print(f"[ACTION] Calling move_object_named('{label}')...")
                    moving_robot = True

                    # Run full pick/place sequence (blocking)
                    move_object_named(label)
                    already_moved.add(label)

                    # After motion, re-enter safe scan pose & update template
                    go_safe_open()
                    time.sleep(1.0)
                    scan_template = read_all_servos()
                    print(f"[INFO] Updated scan template pose: {scan_template}")

                    moving_robot = False
                    # Small delay so the camera sees the new state
                    time.sleep(0.5)
                    break  # exit the detection loop for this frame

        # ====== FPS OVERLAY ======
        dt = time.perf_counter() - t0
        fps = frame_count / dt if dt > 0 else 0.0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.imshow("YOLO + DOFBOT (Base Scan)", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Closed webcam and window. Bye!")


if __name__ == "__main__":
    main()
