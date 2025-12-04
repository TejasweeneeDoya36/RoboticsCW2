#!/usr/bin/env python3
# slot_move_demo.py
# DOFBOT safe pick & place using Arm_Lib (servo angles)

import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(0.1)

MOVE_TIME = 1500  # ms

# Tune these two based on your robot
GRIP_OPEN_ANGLE     = 90   # gripper open
DEFAULT_CLOSED_ANGLE = 30  # typical gripper closed angle (adjust if needed)

# High, safe pose above all objects (you must tune this!)
SAFE_OPEN = [90, 60, 60, 90, 90, GRIP_OPEN_ANGLE]       # safe travel with open gripper
SAFE_CARRY = [90, 60, 60, 90, 90, DEFAULT_CLOSED_ANGLE] # safe travel when carrying

# ---------------------------------------------
#  YOUR CALIBRATED LOW "GRIP" POSES (object in grip)
#  Replace these with the angles you recorded with read_servos.py
#  Format: [base, shoulder, elbow, wrist, wrist_rot, gripper_closed]
# ---------------------------------------------
SLOTS_GRIP = {
    "home":         [90, 90, 90, 90, 90, GRIP_OPEN_ANGLE],  # home with open gripper

    # EXAMPLES – replace with your real values:
    "mouse_slot":   [155, 26, 62, 0, 0, 85],
    "pen_slot":     [121, 32, 58, 5, 0, 175],
    "pendrive_slot":[48, 13, 84, 9, 268, 155],
    "eraser_slot":  [75, 16, 88, 0, 0, 153],
    "stapler_slot": [102, 39, 47, 3, 245, 155],
    "adapter_slot": [20, 37, 25, 29, 90, 13],
}

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def move_angles(angles, move_time=MOVE_TIME):
    b, s, e, w, wr, g = angles
    Arm.Arm_serial_servo_write6(b, s, e, w, wr, g, move_time)
    time.sleep(move_time / 1000.0)

def go_safe_open():
    print("\n[MOVE] SAFE_OPEN (high, open)")
    move_angles(SAFE_OPEN)

def go_safe_carry():
    print("\n[MOVE] SAFE_CARRY (high, carrying)")
    move_angles(SAFE_CARRY)

def get_open_pose_from_grip(slot_name):
    """Take the grip pose but with gripper open."""
    grip = SLOTS_GRIP[slot_name]
    open_pose = grip.copy()
    open_pose[5] = GRIP_OPEN_ANGLE
    return open_pose

# ---------------------------------------------
# Basic actions per slot
# ---------------------------------------------
def move_to_slot_open(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = get_open_pose_from_grip(slot_name)
    print(f"[MOVE] To {slot_name} with gripper OPEN: {pose}")
    move_angles(pose)

def grip_at_slot(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = SLOTS_GRIP[slot_name]
    print(f"[GRIP] Closing gripper at {slot_name}: {pose}")
    move_angles(pose)

def release_at_slot(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = get_open_pose_from_grip(slot_name)
    print(f"[RELEASE] Opening gripper at {slot_name}: {pose}")
    move_angles(pose)

# ---------------------------------------------
# High-level sequences
# ---------------------------------------------
def pick_from_slot(slot_name):
    """Safe pick: go high → go above slot open → go down & close."""
    print(f"\n=== PICK from {slot_name} ===")
    go_safe_open()              # 1) go high with open grip
    move_to_slot_open(slot_name)# 2) move down to slot with open gripper
    grip_at_slot(slot_name)     # 3) close gripper (now holding)
    print(f"=== DONE PICK {slot_name} ===\n")

def place_to_slot(slot_name):
    """Safe place: go high carrying → go down closed → open to drop."""
    print(f"\n=== PLACE to {slot_name} ===")
    go_safe_carry()             # 1) go high while carrying
    # 2) move to slot with closed gripper (use grip pose)
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    print(f"[MOVE] To {slot_name} (carrying): {SLOTS_GRIP[slot_name]}")
    move_angles(SLOTS_GRIP[slot_name])
    # 3) open to release
    release_at_slot(slot_name)
    print(f"=== DONE PLACE {slot_name} ===\n")

def move_object(src_slot, dst_slot):
    """Pick from src, travel high, place at dst."""
    print(f"\n##### MOVE OBJECT: {src_slot} -> {dst_slot} #####")
    pick_from_slot(src_slot)
    place_to_slot(dst_slot)
    go_safe_open()
    print("##### DONE TRANSFER #####\n")

# ---------------------------------------------
# CLI interface
# ---------------------------------------------
def main():
    print("==============================================")
    print(" DOFBOT Safe Pick & Place (Arm_Lib / Angles)  ")
    print("==============================================")
    print("Slots:", ", ".join(SLOTS_GRIP.keys()))
    print("\nCommands:")
    print("  open  <slot>          - go to slot with gripper OPEN")
    print("  grip  <slot>          - close gripper at slot")
    print("  pick  <slot>          - safe pick from slot")
    print("  place <slot>          - safe place to slot")
    print("  move  <src> <dst>     - pick from src, place at dst (safe path)")
    print("  home                  - go SAFE_OPEN (high, open)")
    print("  q                     - quit\n")

    while True:
        parts = input("Command: ").strip().split()
        if not parts:
            continue

        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            go_safe_open()
            break

        elif cmd == "home":
            go_safe_open()

        elif cmd == "open" and len(parts) == 2:
            move_to_slot_open(parts[1])

        elif cmd == "grip" and len(parts) == 2:
            grip_at_slot(parts[1])

        elif cmd == "pick" and len(parts) == 2:
            pick_from_slot(parts[1])

        elif cmd == "place" and len(parts) == 2:
            place_to_slot(parts[1])

        elif cmd == "move" and len(parts) == 3:
            move_object(parts[1], parts[2])

        else:
            print("Invalid command or wrong arguments.")

if __name__ == "__main__":
    main()
