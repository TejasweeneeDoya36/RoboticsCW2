# detect_webcam.py
#
# Test your trained YOLO model using the laptop webcam.
# Shows live video with bounding boxes, labels, confidences and FPS.
#
# Requirements:
#   pip install ultralytics opencv-python

import time
import cv2
from ultralytics import YOLO

CAM_INDEX = 0
MODEL_PATH = "models/office_yolo.pt"
CONF_THRES = 0.5

# <<< ADD THIS BLOCK AFTER THE CONSTANTS >>>
# Map YOLO class indices to your desired names
# Adjust indices if your inspect_names.py printout is different.
CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

def main():
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")

    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    # Optional: set reasonable resolution / FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Press ESC in the window to quit.")

    frame_count = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame")
            break

        frame_count += 1
        h, w, _ = frame.shape

        # -------- YOLO inference --------
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

                # Use YOLO's baked-in names
                if hasattr(results, "names") and cls_id in results.names:
                    label = CUSTOM_NAMES.get(cls_id, f"class_{cls_id}")
                else:
                    label = f"class_{cls_id}"

                x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                detections.append((conf, label, x1_i, y1_i, x2_i, y2_i))

        # -------- Draw detections --------
        for conf, label, x1_i, y1_i, x2_i, y2_i in detections:
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1_i, max(15, y1_i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

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

        cv2.imshow("YOLO Webcam Test", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Closed webcam and window. Bye!")


if __name__ == "__main__":
    main()
