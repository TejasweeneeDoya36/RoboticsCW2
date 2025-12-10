#!/usr/bin/env python3
# detect_webcam_dofbot.py (OPTIMISED)
#
# YOLO + DOFBOT integration with better FPS:
# - Camera runs as fast as possible.
# - YOLO runs only every INFER_INTERVAL seconds.
# - Lower imgsz for faster inference.
# - Robot motion logic kept the same.

import time
import cv2
from ultralytics import YOLO

# ---- Import your DOFBOT motion logic & Arm instance & SAFE_OPEN pose ----
from slot_move_demo import move_object_named, PAIR_MAP, go_safe_open, Arm, SAFE_OPEN

# ===================== CONFIG =====================

# Camera / model config
CAM_INDEX   = 1
MODEL_PATH  = "models/office_yolo.pt"
CONF_THRES  = 0.5

# How often to run YOLO (seconds)
#   Smaller = more detections, lower FPS.
#   Bigger  = fewer detections, higher FPS.
INFER_INTERVAL = 0.25   # 4 inferences per second is enough for your use-case

# Size of image given to YOLO (smaller = faster)
YOLO_IMGSZ = 224

# Radar sweep configuration for BASE (servo 1)
SCAN_BASE_MIN   = 50    # minimum base angle
SCAN_BASE_MAX   = 115   # maximum base angle
SCAN_BASE_STEP  = 2     # step size in degrees per move
SCAN_MOVE_TIME  = 100   # ms time for each small base movement
SCAN_INTERVAL   = 0.25  # seconds between base updates

# Map YOLO class indices → logical names used in PAIR_MAP / SLOTS_GRIP
CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

# ==================================================

# Use SAFE_OPEN from slot_move_demo as the fixed scan pose
SCAN_POSE = SAFE_OPEN.copy()  # [base, s2, s3, s4, s5, s6]

def move_base_with_same_pose(base_angle, move_time=SCAN_MOVE_TIME):
    """
    Move only the base (servo 1) to base_angle,
    keep servos 2–6 equal to SCAN_POSE.
    """
    pose = SCAN_POSE.copy()
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

    # Start radar sweep from SCAN_POSE[0], clamped
    base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
    base_dir   = +1  # +1 = increasing angle, -1 = decreasing
    last_scan_update  = time.time()

    # Open camera
    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    # Optional: configure resolution & FPS (low-resolution → faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Optional: try MJPG for better performance on some cameras
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    print("[INFO] Press ESC in the window to quit.")
    print("[INFO] Place an out-of-place object somewhere in the workspace.")

    frame_count       = 0
    t0                = time.perf_counter()
    moving_robot      = False  # True while executing pick/place

    # Track which objects we already moved in this session
    already_moved = set()

    # For decoupling YOLO inference from camera FPS
    last_infer_time   = 0.0
    last_detections   = []  # cache labels from last YOLO run

    while True:
        # ====== RADAR SWEEP CONTROL (SERVO 1 ONLY) ======
        now = time.time()
        if not moving_robot and (now - last_scan_update) > SCAN_INTERVAL:
            # Compute next base angle
            base_angle += base_dir * SCAN_BASE_STEP

            # If we hit bounds, reverse direction
            if base_angle >= SCAN_BASE_MAX:
                base_angle = SCAN_BASE_MAX
                base_dir   = -1
            elif base_angle <= SCAN_BASE_MIN:
                base_angle = SCAN_BASE_MIN
                base_dir   = +1

            move_base_with_same_pose(base_angle)
            last_scan_update = now

        # ====== READ CAMERA FRAME ======
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame")
            break

        frame_count += 1

        # ====== DECIDE IF WE RUN YOLO ON THIS FRAME ======
        detected_objects = []  # (label, conf, bbox)

        do_infer = (
            (time.perf_counter() - last_infer_time) >= INFER_INTERVAL
            and not moving_robot
        )

        if do_infer:
            # ---- YOLO INFERENCE (not every frame) ----
            last_infer_time = time.perf_counter()

            results_list = model.predict(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRES,
                verbose=False
            )

            results = results_list[0] if results_list else None

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

            # store detections so we can still draw them until next YOLO call
            last_detections = detected_objects.copy()

        else:
            # use cached detections for drawing / decision
            detected_objects = last_detections

        # ====== DECIDE ROBOT ACTION (only when not moving) ======
        if not moving_robot and detected_objects:
            for label, conf, _bbox in detected_objects:
                # Only move objects we know how to handle
                if label in PAIR_MAP and label not in already_moved:
                    print(f"[DETECT] Found '{label}' with conf={conf:.2f}.")
                    print(f"[ACTION] Calling move_object_named('{label}')...")
                    moving_robot = True

                    # Run full pick/place sequence (blocking)
                    move_object_named(label)
                    already_moved.add(label)

                    # After motion, re-enter safe scan pose
                    go_safe_open()
                    time.sleep(1.0)

                    # Reset sweep to start from SAFE_OPEN again
                    base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
                    base_dir   = +1
                    moving_robot = False
                    # small delay so the camera sees the new state
                    time.sleep(0.5)
                    break  # exit detection loop for this frame

        # ====== DRAW BOUNDING BOXES (can disable if needed) ======
        for label, conf, (x1, y1, x2, y2) in detected_objects:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

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

        cv2.imshow("YOLO + DOFBOT (Radar Base Scan, Optimised)", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Closed webcam and window. Bye!")


if __name__ == "__main__":
    main()
