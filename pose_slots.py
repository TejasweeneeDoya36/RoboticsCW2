# pose_slots.py
# DOFBOT Pose Tester using Yahboom Arm_Lib (Servo-Angle Control)

from Arm_Lib import Arm_Device
import time

# --------------------------------
# Initialize DOFBOT
# --------------------------------
Arm = Arm_Device()
time.sleep(0.1)

# --------------------------------
# EDIT THESE ANGLES
# Angles are [base, shoulder, elbow, wrist, wrist_rot, gripper]
# --------------------------------

SLOTS = {
    "home":         [90, 90, 90, 90, 90, 155],

    # Replace these with angle values you record
    "mouse_slot":   [90, 22, 55, 45, 90, 155],
    "pen_slot":     [125, 36, 51, 9, 159, 160],
    "pendrive_slot":[48, 13, 84, 7, 264, 155],
    "eraser_slot":  [75, 19, 83, None, 143, 154],
    "stapler_slot": [107, 49, 31, 7, 117, 130],
    "adapter_slot": [150, 130, 120, 90, 90, 90],
}
MOVE_TIME = 1600   # ms


# --------------------------------
# Move DOFBOT to a pose (angle set)
# --------------------------------
def move_to(name):
    if name not in SLOTS:
        print("Unknown slot.")
        return

    angles = SLOTS[name]
    print(f"\n[MOVE] {name} -> {angles}")

    Arm.Arm_serial_servo_write6(
        angles[0], angles[1], angles[2], angles[3], angles[4], angles[5],
        MOVE_TIME
    )
    time.sleep(MOVE_TIME / 1000)
    print("[OK] Done.\n")


# --------------------------------
# Interactive tester
# --------------------------------
if __name__ == "__main__":
    print("========================================")
    print("     DOFBOT Slot Tester (Arm_Lib)       ")
    print("========================================")
    print("Type a slot name to move arm to angles")
    print("Commands: list, home, q")
    print()

    while True:
        cmd = input("Enter slot name: ").strip()

        if cmd in ("q", "quit", "exit"):
            move_to("home")
            break

        elif cmd == "list":
            print("Slots:", ", ".join(SLOTS.keys()))

        elif cmd == "home":
            move_to("home")

        elif cmd in SLOTS:
            move_to(cmd)

        else:
            print("Unknown slot. Type 'list' to see names.\n")
