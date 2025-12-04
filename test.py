
import cv2
import time
import numpy as np
from ultralytics import YOLO

import armlite                                   # <-- New control module

# ------------------------------------------------------------
# INITIALIZE ROBOT
# ------------------------------------------------------------
arm = armlite.ArmLite()
gripper = armlite.Gripper()

# ------------------------------------------------------------
# LOAD YOLOv8 MODEL
# ------------------------------------------------------------
model = YOLO("best.pt")   # Your custom model

# ------------------------------------------------------------
# ROBOT POSITIONS (edit when calibrating)
# ------------------------------------------------------------
HOME = (0, 10, 18)       # straight up
DROP_LEFT = (-10, 5, 10)
DROP_RIGHT = (10, 5, 10)

H_HOVER = 8
H_PICK  = 5

# ------------------------------------------------------------
# PIXEL → ROBOT COORDINATE MAPPING
# ------------------------------------------------------------
def pixel_to_robot(px, py):
    """
    Convert bounding box center -> robot coordinates.
    You will give me calibration values later.
    """

    # Temporary linear mapping
    x = (px - 320) / 25.0   # shift left-right
    y = 15                  # forward fixed distance
    z = H_PICK

    return (x, y, z)

# ------------------------------------------------------------
# MOVEMENT HELPERS
# ------------------------------------------------------------
def move_to(pos, delay=0.7):
    arm.move_to(pos[0], pos[1], pos[2])
    time.sleep(delay)

def pick(pos):
    # Hover
    move_to((pos[0], pos[1], pos[2] + H_HOVER))
    # Lower
    move_to(pos)
    # Grip
    gripper.close()
    time.sleep(0.4)
    # Lift
    move_to((pos[0], pos[1], pos[2] + H_HOVER))

def place(pos):
    move_to(pos)
    time.sleep(0.3)
    gripper.open()
    time.sleep(0.3)
    move_to((pos[0], pos[1], pos[2] + H_HOVER))

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    print("Starting YOLO + ArmLite pick-and-place…")

    gripper.open()
    move_to(HOME)

    cam = cv2.VideoCapture(0)
    picked = 0
    place_left_side = True

    while picked < 6:

        ret, frame = cam.read()
        if not ret:
            continue

        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        if len(detections) == 0:
            cv2.imshow("AI View", frame)
            if cv2.waitKey(1) == 27: break
            continue

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det.astype(int)

            # Compute center of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Convert pixel → robot coordinate
            pos = pixel_to_robot(cx, cy)

            print(f"Detected object at pixel ({cx},{cy}) → robot {pos}")

            # Pick object
            pick(pos)

            # Drop left or right
            if place_left_side:
                place(DROP_LEFT)
            else:
                place(DROP_RIGHT)

            place_left_side = not place_left_side
            picked += 1

            # Return home
            move_to(HOME)

            if picked >= 6:
                break

        cv2.imshow("AI View", frame)
        if cv2.waitKey(1) == 27:
            break

    print("All 6 objects picked!")
    cam.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
