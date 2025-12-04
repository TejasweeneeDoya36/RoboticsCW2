import time
import cv2
import torch
import numpy as np

from ArmPi.ArmIK import ArmIK
from ArmPi.Motion import Motion
from ArmPi.Gripper import Gripper

# ---------------------------------------------
# INITIALIZE ROBOT
# ---------------------------------------------
ik = ArmIK()
motion = Motion()
gripper = Gripper()

# ---------------------------------------------
# LOAD YOLOv8 MODEL
# ---------------------------------------------
model = torch.hub.load("ultralytics/yolov8", "custom", path="models/office_yolo.pt")

# ---------------------------------------------
# ROBOT POSITIONS
# ---------------------------------------------
HOME_POS = (0, 10, 18)         # Straight up
DROP_LEFT = (-10, 5, 10)
DROP_RIGHT = (10, 5, 10)

# Height offsets
H_OVER = 8   # Hover above object
H_PICK = 5   # Touching object surface

# ---------------------------------------------
# CONVERT PIXEL → ROBOT COORDINATE
# (TEMPORARY — YOU WILL CALIBRATE THIS)
# ---------------------------------------------
def pixel_to_robot(px, py):
    """
    Convert the bounding box center pixel to robot coordinates.
    Modify when you provide calibration values.
    """

    # Example mapping (rough):
    X = (px - 320) / 25       # center shift
    Y = 15                    # objects are at constant forward distance
    Z = 5                     # picking height

    return (X, Y, Z)

# ---------------------------------------------
# ROBOT MOVEMENT FUNCTIONS
# ---------------------------------------------
def move_to(pos, d=0.8):
    ik.setPitchRangeMoving(pos, -90, -90, 1500)
    time.sleep(d)

def pick(pos):
    # Hover
    move_to((pos[0], pos[1], pos[2] + H_OVER))
    # Go down
    move_to(pos)
    gripper.close()
    time.sleep(0.4)
    # Lift
    move_to((pos[0], pos[1], pos[2] + H_OVER))

def place(pos):
    move_to(pos)
    time.sleep(0.3)
    gripper.open()
    time.sleep(0.3)
    move_to((pos[0], pos[1], pos[2] + H_OVER))

# ---------------------------------------------
# MAIN LOOP
# ---------------------------------------------
def main():

    # 1. Start at home
    gripper.open()
    move_to(HOME_POS)

    cam = cv2.VideoCapture(0)
    picked_objects = 0
    placed_left = True

    print("Starting YOLO detection loop...")

    while picked_objects < 6:
        ret, frame = cam.read()
        if not ret:
            continue

        # Run YOLO
        results = model(frame)
        detections = results.xyxy[0]   # YOLO bbox output

        if len(detections) == 0:
            cv2.imshow("AI View", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # 2. For each detected object
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)

            # Compute bounding-box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Convert pixel to robot coordinate
            rob_x, rob_y, rob_z = pixel_to_robot(cx, cy)
            obj_pos = (rob_x, rob_y, rob_z)

            print(f"Object detected at pixel ({cx}, {cy}) → robot {obj_pos}")

            # 3. Pick the object
            pick(obj_pos)

            # 4. Drop left/right
            if placed_left:
                place(DROP_LEFT)
            else:
                place(DROP_RIGHT)

            placed_left = not placed_left
            picked_objects += 1

            # Return home
            move_to(HOME_POS)

            if picked_objects >= 6:
                break

        cv2.imshow("AI View", frame)
        if cv2.waitKey(1) == 27:
            break

    print("All 6 objects picked successfully!")

    cam.release()
    cv2.destroyAllWindows()


# ---------------------------------------------
# RUN
# ---------------------------------------------
if __name__ == "__main__":
    main()
