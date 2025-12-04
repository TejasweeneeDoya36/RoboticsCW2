import time
import cv2
from ultralytics import YOLO
from armlib.armlib import ArmLib


# ------------------------------------------------
# INITIALIZE ROBOT + AI
# ------------------------------------------------
arm = ArmLib()
arm.home()                 # move to home position
arm.gripper_open()

model = YOLO("models/office_yolo.pt")    # load your YOLOv8 model

# ------------------------------------------------
# POSITIONS (ADJUST IF NEEDED)
# ------------------------------------------------
HOME = (0, 12, 18)
DROP_LEFT = (-12, 6, 10)
DROP_RIGHT = (12, 6, 10)

H_HOVER = 8
H_PICK = 4


# ------------------------------------------------
# PIXEL -> ROBOT COORDINATE MAPPING
# (TEMPORARY – YOU WILL GIVE ME CALIBRATION VALUES)
# ------------------------------------------------
def pixel_to_robot(px, py):
    """
    Convert bounding box pixel center to robot coordinates.
    Update after your calibration.
    """

    X = (px - 320) / 25     # rough linear mapping
    Y = 14                  # constant forward distance
    Z = H_PICK              # picking height

    return (X, Y, Z)


# ------------------------------------------------
# ROBOT CONTROL FUNCTIONS
# ------------------------------------------------
def move_xyz(pos, delay=0.5):
    x, y, z = pos
    arm.move_to(x, y, z, pitch=-90, yaw=0)
    time.sleep(delay)

def pick(pos):
    # hover
    move_xyz((pos[0], pos[1], pos[2] + H_HOVER))
    # go down
    move_xyz(pos)
    # grip
    arm.gripper_close()
    time.sleep(0.4)
    # lift
    move_xyz((pos[0], pos[1], pos[2] + H_HOVER))

def place(pos):
    move_xyz(pos)
    arm.gripper_open()
    time.sleep(0.3)
    move_xyz((pos[0], pos[1], pos[2] + H_HOVER))


# ------------------------------------------------
# MAIN LOOP
# ------------------------------------------------
def main():
    cam = cv2.VideoCapture(0)

    picked = 0
    place_left = True

    print("Starting YOLOv8 detection...")

    while picked < 6:
        ret, frame = cam.read()
        if not ret:
            continue

        # YOLO DETECTION
        results = model(frame)[0]
        detections = results.boxes.data

        if len(detections) == 0:
            cv2.imshow("AI View", frame)
            cv2.waitKey(1)
            continue

        # Handle each detection
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()

            # center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # convert to robot coordinates
            rob_x, rob_y, rob_z = pixel_to_robot(cx, cy)
            obj_pos = (rob_x, rob_y, rob_z)

            print(f"[INFO] Object at pixels ({cx},{cy}) -> robot {obj_pos}")

            # pick object
            pick(obj_pos)

            # place left or right
            if place_left:
                place(DROP_LEFT)
            else:
                place(DROP_RIGHT)

            place_left = not place_left
            picked += 1

            # return home
            move_xyz(HOME)

            if picked >= 6:
                break

        cv2.imshow("AI View", frame)
        cv2.waitKey(1)

    print("All 6 objects picked successfully!")
    arm.home()
    cam.release()
    cv2.destroyAllWindows()


# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":
    main()

