#!/usr/bin/env python3
# slot_grip_demo.py
# Uses Arm_Lib to move DOFBOT to each slot with the gripper open,
# then closes the gripper when in position.

import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(0.1)

MOVE_TIME = 1500  # ms for each movement

# Adjust these to match your robot:
GRIP_OPEN_ANGLE   = 90   # angle when gripper is open
# (closed angle will come from your calibrated slot poses)

# --------------------------------------------------
# 1) YOUR CALIBRATED "GRIP" POSES (INCLUDING GRIPPER)
#    Paste your measured angles here (6 values each)
#    Format: [base, shoulder, elbow, wrist, wrist_rot, gripper_closed]
# --------------------------------------------------
SLOTS_GRIP = {
    "home":         [90, 90, 90, 90, 90, GRIP_OPEN_ANGLE],  # home with open gripper

    # EXAMPLES – replace with your real values:
    "mouse_slot":   [90, 22, 55, 45, 90, 155],
    "pen_slot":     [125, 36, 51, 9, 159, 160],
    "pendrive_slot":[48, 13, 84, 7, 264, 155],
    "eraser_slot":  [75, 19, 83, 1, 143, 154],
    "stapler_slot": [107, 49, 31, 7, 117, 130],
    "adapter_slot": [150, 130, 120, 90, 90, 90],
}

# --------------------------------------------------
# Helper: send 6 servo angles
# --------------------------------------------------
def move_angles(angles, move_time=MOVE_TIME):
    b, s, e, w, wr, g = angles
    Arm.Arm_serial_servo_write6(b, s, e, w, wr, g, move_time)
    time.sleep(move_time / 1000.0)

# --------------------------------------------------
# Helper: create "open-grip" version of a slot pose
#         -> same angles but gripper = GRIP_OPEN_ANGLE
# --------------------------------------------------
def get_open_pose(slot_name):
    base_angles = SLOTS_GRIP[slot_name]
    open_angles = base_angles.copy()
    open_angles[5] = GRIP_OPEN_ANGLE   # index 5 = gripper
    return open_angles

# --------------------------------------------------
# API 1: move to slot with gripper open (no gripping)
# --------------------------------------------------
def move_to_slot_open(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    open_pose = get_open_pose(slot_name)
    print(f"\n[MOVE] To {slot_name} with gripper OPEN:", open_pose)
    move_angles(open_pose)

# --------------------------------------------------
# API 2: grip at slot
#        (assumes arm is already at that slot, just open vs close)
# --------------------------------------------------
def grip_at_slot(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    grip_pose = SLOTS_GRIP[slot_name]
    print(f"[GRIP] Closing gripper at {slot_name}:", grip_pose)
    move_angles(grip_pose)

# --------------------------------------------------
# High-level: pick from slot
#  - Move there with open gripper
#  - Then close gripper
# --------------------------------------------------
def pick_from_slot(slot_name):
    print("=== PICK from", slot_name, "===")
    move_to_slot_open(slot_name)  # always move with open grip
    time.sleep(0.5)
    grip_at_slot(slot_name)       # close at location
    print("=== Done PICK", slot_name, "===\n")

# --------------------------------------------------
# High-level: place to slot
#  (if you’re carrying something: move there with CLOSED,
#   then open to release)
# --------------------------------------------------
def place_to_slot(slot_name):
    print("=== PLACE to", slot_name, "===")
    # Move with whatever angles you already have in SLOTS_GRIP
    # (assuming gripper is closed holding an object)
    grip_pose = SLOTS_GRIP[slot_name]
    move_angles(grip_pose)
    time.sleep(0.5)

    # Now open gripper to release
    open_pose = get_open_pose(slot_name)
    print("[RELEASE] Opening gripper at slot.")
    move_angles(open_pose)
    print("=== Done PLACE", slot_name, "===\n")

# --------------------------------------------------
# Simple CLI menu
# --------------------------------------------------
def main():
    print("==============================================")
    print(" DOFBOT Slot Grip Demo (Arm_Lib / Servo Angles)")
    print("==============================================")
    print("Available slots:", ", ".join(SLOTS_GRIP.keys()))
    print("Commands:")
    print("  open  <slot>   - move to slot with gripper open")
    print("  grip  <slot>   - close gripper at slot")
    print("  pick  <slot>   - open → move → close (pick object)")
    print("  place <slot>   - move (closed) → open (place object)")
    print("  home           - move to home with open gripper")
    print("  q              - quit\n")

    while True:
        cmd = input("Command: ").strip().split()
        if not cmd:
            continue

        if cmd[0] in ("q", "quit", "exit"):
            print("Going home & exiting...")
            move_to_slot_open("home")
            break

        elif cmd[0] == "home":
            move_to_slot_open("home")

        elif cmd[0] == "open" and len(cmd) == 2:
            move_to_slot_open(cmd[1])

        elif cmd[0] == "grip" and len(cmd) == 2:
            grip_at_slot(cmd[1])

        elif cmd[0] == "pick" and len(cmd) == 2:
            pick_from_slot(cmd[1])

        elif cmd[0] == "place" and len(cmd) == 2:
            place_to_slot(cmd[1])

        else:
            print("Invalid command or wrong arguments.")


if __name__ == "__main__":
    main()
