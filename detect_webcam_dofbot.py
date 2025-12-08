# detect_webcam_dofbot.py
#
# Use YOLO + webcam to inspect the scene and command DOFBOT
# to move any object that is NOT in its correct area.
#
# Requires:
#   pip install ultralytics opencv-python
#   slot_move_demo.py in the same folder (with move_object_named)

import time
import cv2
from ultralytics import YOLO

# ---- DOFBOT control imports (your movement code) ----
from slot_move_demo import move_object_named, PAIR_MAP

# ---- Camera / model config ----
CAM_INDEX = 1
MODEL_PATH = "models/office_yolo.pt"
CONF_THRES = 0.5

# Map YOLO class indices → logical names used in PAIR_MAP / SLOTS_GRIP
CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

# --------------------------------------------------------------------
# IMAGE ZONES: where is "correct" position in the camera image?
#
# You must TUNE these rectangles from your camera view.
# Format per object:
#    "object_name": {
#        "correct": ((x1, y1), (x2, y2))
#    }
#
# Coordinates are in pixels (from top-left of the image).
# You can start with guesses and refine them.
# --------------------------------------------------------------------
IMAGE_ZONES = {
    "mouse":    {"correct": ((50,  50),  (150, 150))},
    "pen":      {"correct": ((200, 50),  (300, 150))},
    "pendrive": {"correct": ((350, 50),  (450, 150))},
    "eraser":   {"correct": ((50,  200), (150, 300))},
    "stapler":  {"correct": ((200, 200), (300, 300))},
    "adapter":  {"correct": ((350, 200), (450, 300))},
}

def point_in_rect(cx, cy, rect):
    """Check if (cx, cy) is inside rect=((x1,y1),(x2,y2))."""
    (x1, y1), (x2, y2) = rect
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def classify_position(label, cx, cy):
    """
    Decide if object with given label at center (cx, cy)
    is in 'correct' or 'wrong' area.
    """
    if label not in IMAGE_ZONES:
        return None  # we don't know this label's correct region

    rect = IMAGE_ZONES[label]["correct"]
    if point_in_rect(cx, cy, rect):
        return "correct"
    else:
        return "wrong"

def main():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Press ESC in the window to quit.")

    frame_count = 0
    t0 = time.perf_counter()

    # Track state: is each object currently seen as correct or wrong?
    last_state = {name: None for name in IMAGE_ZONES.keys()}
    # Track which objects have already been corrected (so we don't repeat)
    already_corrected = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame")
            break

        frame_count += 1
        h, w, _ = frame.shape

        # -------- YOLO inference --------
        results_list = model(frame, verbose=False)
        results = results_list[0] if results_list else None

        # Reset per-frame view of state
        frame_state = {name: None for name in IMAGE_ZONES.keys()}

        if results is not None and hasattr(results, "boxes"):
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf < CONF_THRES:
                    continue

                label = CUSTOM_NAMES.get(cls_id, f"class_{cls_id}")
                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                cx = (x1_i + x2_i) // 2
                cy = (y1_i + y2_i) // 2

                # Decide if this object is in correct/wrong place
                pos_status = classify_position(label, cx, cy)

                if pos_status is not None:
                    frame_state[label] = pos_status

                # Draw bounding boxes
                color = (0, 255, 0) if pos_status == "correct" else (0, 0, 255)
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                text = f"{label} {conf:.2f} ({pos_status or 'unk'})"
                cv2.putText(
                    frame,
                    text,
                    (x1_i, max(15, y1_i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # ---- Draw zones (optional helper) ----
        for lbl, cfg in IMAGE_ZONES.items():
            (zx1, zy1), (zx2, zy2) = cfg["correct"]
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 0), 1)
            cv2.putText(frame, f"{lbl}_correct", (zx1, zy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # -------- Decide actions based on state change --------
        # For each object, if now seen as 'wrong' and not yet corrected -> command robot
        for obj_name, status in frame_state.items():
            if status is None:
                continue  # not seen this frame

            # update last_state tracking
            if last_state[obj_name] != status:
                print(f"[STATE] {obj_name} now appears '{status}'")
                last_state[obj_name] = status

            # Only act if it's wrong and not already corrected
            if status == "wrong" and obj_name not in already_corrected:
                # Only move objects that have a defined pair in PAIR_MAP
                if obj_name in PAIR_MAP:
                    print(f"[ACTION] {obj_name} seems misplaced -> moving with DOFBOT...")
                    # This will BLOCK until movement is done
                    move_object_named(obj_name)
                    already_corrected.add(obj_name)
                else:
                    print(f"[WARN] {obj_name} is not in PAIR_MAP, cannot auto-move.")

        # -------- FPS overlay --------
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

        cv2.imshow("YOLO + DOFBOT Inspector", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Closed webcam and window. Bye!")


if __name__ == "__main__":
    main()
