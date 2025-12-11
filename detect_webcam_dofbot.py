#!/usr/bin/env python3
# detect_webcam_dofbot.py (THREAD OPTIMISED)
#
# YOLO + DOFBOT integration with better *display* FPS:
# - Camera + GUI run in the main thread as fast as possible.
# - YOLO inference runs in a background worker thread.
# - Robot logic uses the latest YOLO detections.
# - Each object is moved only once; automation stops when all are sorted.

import time
import threading
import queue
import cv2
from ultralytics import YOLO

from slot_move_demo import move_object_named, PAIR_MAP, go_safe_open, Arm, SAFE_OPEN

# ===================== CONFIG =====================

CAM_INDEX   = 1
MODEL_PATH  = "models/office_yolo.pt"
CONF_THRES  = 0.5

# How often to run YOLO (seconds) on the worker thread
INFER_INTERVAL = 0.25

# YOLO input size (smaller = faster)
YOLO_IMGSZ = 224

# Radar base sweeping
SCAN_BASE_MIN   = 50
SCAN_BASE_MAX   = 110
SCAN_BASE_STEP  = 2
SCAN_MOVE_TIME  = 100
SCAN_INTERVAL   = 0.25

CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

SCAN_POSE = SAFE_OPEN.copy()  # [base, s2, s3, s4, s5, s6]


def move_base_with_same_pose(base_angle, move_time=SCAN_MOVE_TIME):
    """Move only base servo while keeping 2–6 at SCAN_POSE."""
    pose = SCAN_POSE.copy()
    pose[0] = base_angle
    print(f"[SCAN] Rotating base to {base_angle}° with pose {pose}")
    Arm.Arm_serial_servo_write6(*pose, move_time)
    time.sleep(move_time / 1000.0)


def main():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    print("[INFO] Moving DOFBOT to safe scan pose (go_safe_open)...")
    go_safe_open()
    time.sleep(1.0)

    base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
    base_dir   = +1
    last_scan_update = time.time()

    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    # Low resolution = faster capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Press ESC in the window to quit.")
    print("[INFO] Place an out-of-place object somewhere in the workspace.")

    frame_count  = 0
    t0           = time.perf_counter()
    moving_robot = False

    already_moved        = set()
    total_known_objects  = len(PAIR_MAP)
    all_done             = False

    # Shared state with YOLO worker
    frame_queue = queue.Queue(maxsize=1)
    last_detections = []       # list[(label, conf, (x1,y1,x2,y2))]
    detections_lock = threading.Lock()
    stop_event = threading.Event()

    # ---------- YOLO WORKER THREAD ----------

    def yolo_worker():
        nonlocal last_detections
        last_infer_time = 0.0

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()
            if now - last_infer_time < INFER_INTERVAL:
                # Skip if we're still inside interval
                continue

            last_infer_time = now

            # Run YOLO on the frame (this may be slow)
            results_list = model.predict(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRES,
                verbose=False
            )
            results = results_list[0] if results_list else None

            detected_objects = []
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

            # Update shared detections
            with detections_lock:
                last_detections = detected_objects

    worker_thread = threading.Thread(target=yolo_worker, daemon=True)
    worker_thread.start()

    try:
        while True:
            now = time.time()

            # ====== RADAR SWEEP (BASE ONLY) ======
            if not moving_robot and (now - last_scan_update) > SCAN_INTERVAL:
                base_angle += base_dir * SCAN_BASE_STEP
                if base_angle >= SCAN_BASE_MAX:
                    base_angle = SCAN_BASE_MAX
                    base_dir   = -1
                elif base_angle <= SCAN_BASE_MIN:
                    base_angle = SCAN_BASE_MIN
                    base_dir   = +1

                move_base_with_same_pose(base_angle)
                last_scan_update = now

            # ====== CAMERA FRAME ======
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to grab frame")
                break

            frame_count += 1

            # Give frame to YOLO worker (non-blocking)
            if not moving_robot:
                if frame_queue.empty():
                    # pass a copy so main can still draw safely
                    try:
                        frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass

            # Get latest detections snapshot
            with detections_lock:
                detected_objects = list(last_detections)

            # ====== ROBOT ACTION (once per object) ======
            if not moving_robot and detected_objects and not all_done:
                for label, conf, _bbox in detected_objects:
                    if label in PAIR_MAP and label not in already_moved:
                        print(f"[DETECT] Found '{label}' with conf={conf:.2f}.")
                        print(f"[ACTION] Calling move_object_named('{label}')...")
                        moving_robot = True

                        move_object_named(label)  # blocking

                        already_moved.add(label)
                        print(f"[STATE] Objects fixed so far: {len(already_moved)}/{total_known_objects}")

                        go_safe_open()
                        time.sleep(1.0)

                        base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
                        base_dir   = +1
                        moving_robot = False
                        time.sleep(0.5)

                        if len(already_moved) == total_known_objects:
                            print("[INFO] All objects have been picked and placed. Automation complete.")
                            all_done = True

                        break  # only handle one object per loop

            # ====== DRAW BOUNDING BOXES ======
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

            # FPS overlay (camera/display FPS)
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

            # Finished alert
            if all_done:
                h, w = frame.shape[:2]
                msg = "ALL OBJECTS SORTED - AUTOMATION COMPLETE"
                cv2.putText(
                    frame,
                    msg,
                    (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("YOLO + DOFBOT (Threaded Radar Scan)", frame)

            # ESC to quit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # Clean up
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Closed webcam and window. Bye!")


if __name__ == "__main__":
    main()
