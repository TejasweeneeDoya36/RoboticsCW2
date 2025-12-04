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
    # read current grip angle (servo 6)
    grip = Arm.Arm_serial_servo_read(6)
    pose = SAFE_CARRY.copy()
    pose[5] = grip          # keep whatever grip we currently have
    move_angles(pose)

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

def move_to_slot_keep_grip(slot_name):
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    # target pose for joints 1–5 (base, shoulder, elbow, wrist, wrist_rot)
    target = SLOTS_GRIP[slot_name].copy()

    # read current gripper angle (servo 6)
    grip = Arm.Arm_serial_servo_read(6)
    target[5] = grip        # keep current grip

    print(f"[MOVE] To {slot_name} (carrying, keep grip={grip}): {target}")
    move_angles(target)

# ---------------------------------------------
# High-level sequences
# ---------------------------------------------
def pick_from_slot(slot_name):
    """Safe pick: go high → rotate above slot → go down & close."""
    print(f"\n=== PICK from {slot_name} ===")

    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    # 1) Go to SAFE_OPEN (high, gripper open)
    go_safe_open()

    # 2) Build the low open pose for this slot (your calibrated pose, but with open gripper)
    low_open = get_open_pose_from_grip(slot_name)  # [b, s, e, w, wr, g_open]

    # 3) Phase 1: rotate base (and wrist_rot) while staying high
    #    -> shoulder & elbow stay as SAFE_OPEN[1], SAFE_OPEN[2]
    phase1 = SAFE_OPEN.copy()
    phase1[0] = low_open[0]   # base angle from slot
    phase1[4] = low_open[4]   # wrist_rot from slot (if you need it)
    # gripper stays open (already in SAFE_OPEN[5] = GRIP_OPEN_ANGLE)

    print(f"[PHASE 1] Rotate above {slot_name} (high): {phase1}")
    move_angles(phase1)

    # 4) Phase 2: go down to the low open pose (now change shoulder & elbow)
    print(f"[PHASE 2] Go down to {slot_name} with gripper OPEN: {low_open}")
    move_angles(low_open)

    # 5) Close gripper at the slot
    grip_at_slot(slot_name)

    print(f"=== DONE PICK {slot_name} ===\n")


def place_to_slot(slot_name):
    """Safe place: go high carrying → go down closed → open to drop."""
    print(f"\n=== PLACE to {slot_name} ===")
    go_safe_carry()             # 1) go high while carrying (keep grip)

    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    # 2) move down to destination, but KEEP current grip value
    move_to_slot_keep_grip(slot_name)

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
