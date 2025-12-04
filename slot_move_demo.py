#!/usr/bin/env python3
# slot_move_demo.py
# DOFBOT safe pick & place using Arm_Lib (servo angles)
#
# Requirements:
#   from Arm_Lib import Arm_Device
#
# Behaviour:
#   - Pick:
#       SAFE_OPEN (high, open)
#       -> rotate above slot (base + wrist_rot only, 2&3 stay high)
#       -> drop (2&3) to low open pose
#       -> close gripper
#       -> lift straight up by only moving 2&3 to SAFE_OPEN
#
#   - Place:
#       lift_23_to_safe() (if not already)
#       -> move above destination with 2&3 high
#       -> drop (2&3) to low closed pose
#       -> open gripper
#       -> lift_23_to_safe() again
#
#   - Between slots: move src -> dst with `move src_slot dst_slot`

import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(0.1)

MOVE_TIME = 1500  # ms for each movement

# Tune these based on your robot:
GRIP_OPEN_ANGLE = 90           # gripper open angle
SAFE_SHOULDER   = 60           # safe high shoulder angle (servo 2)
SAFE_ELBOW      = 60           # safe high elbow angle (servo 3)
SAFE_WRIST      = 90           # safe "neutral" wrist angle (servo 4)

# High, safe pose above all objects (only used as a target pattern)
SAFE_OPEN = [90, SAFE_SHOULDER, SAFE_ELBOW, SAFE_WRIST, 90, GRIP_OPEN_ANGLE]

# --------------------------------------------------------------------
# YOUR CALIBRATED LOW "GRIP" POSES (object in grip)
# Replace these with the angles you recorded with read_servos.py.
# Format: [base, shoulder, elbow, wrist, wrist_rot, gripper_closed]
# --------------------------------------------------------------------
SLOTS_GRIP = {
    "home":         [90, 90, 90, 90, 90, GRIP_OPEN_ANGLE],  # home, open

    # EXAMPLES — REPLACE with your real values for each object slot
    "mouse_slot":   [48, 13, 84, 9, 268, 155],
    "pen_slot":     [60, 20, 80, 20, 250, 150],
    "pendrive_slot":[48, 13, 84, 9, 268, 155],  # example; adjust
    "eraser_slot":  [80, 25, 78, 18, 260, 150],
    "stapler_slot": [100, 30, 72, 25, 260, 150],
    "adapter_slot": [20, 37, 25, 29, 90, 13],
}

# --------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------
def move_angles(angles, move_time=MOVE_TIME):
    """Send 6-servo move command."""
    b, s, e, w, wr, g = angles
    Arm.Arm_serial_servo_write6(b, s, e, w, wr, g, move_time)
    time.sleep(move_time / 1000.0)

def get_open_pose_from_grip(slot_name):
    """Take the grip pose for slot and replace gripper with open angle."""
    grip = SLOTS_GRIP[slot_name]
    open_pose = grip.copy()
    open_pose[5] = GRIP_OPEN_ANGLE
    return open_pose

def go_safe_open():
    """Move to a generic safe high pose with gripper open."""
    print("\n[MOVE] SAFE_OPEN (high, open)")
    move_angles(SAFE_OPEN)

def lift_23_to_safe():
    """
    Lift arm straight up by moving only servo 2 & 3 to SAFE_SHOULDER / SAFE_ELBOW.
    Keeps base, wrist, wrist_rot, and current gripper angle.
    """
    cur = [Arm.Arm_serial_servo_read(i + 1) for i in range(6)]
    cur[1] = SAFE_SHOULDER  # shoulder up
    cur[2] = SAFE_ELBOW     # elbow up
    print(f"[LIFT] Raising servos 2 & 3 to safe: {cur}")
    move_angles(cur)

def move_above_slot_keep_23(slot_name):
    """
    Rotate/move in air to be above a slot:
    - base (0) and wrist_rot (4) from target slot
    - shoulder (1) and elbow (2) stay at SAFE_SHOULDER / SAFE_ELBOW
    - wrist (3) from SAFE_OPEN (neutral)
    - gripper (5) = current grip angle (carry whatever you're holding)
    """
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    target = SLOTS_GRIP[slot_name]
    cur_grip = Arm.Arm_serial_servo_read(6)

    pose = [
        target[0],         # base -> slot base
        SAFE_SHOULDER,     # shoulder high
        SAFE_ELBOW,        # elbow high
        SAFE_WRIST,        # wrist neutral high
        target[4],         # wrist_rot -> slot rotation
        cur_grip           # keep grip angle
    ]

    print(f"[MOVE] Above {slot_name} (keep 2&3 high): {pose}")
    move_angles(pose)

# --------------------------------------------------------------------
# Basic per-slot actions
# --------------------------------------------------------------------
def move_to_slot_open(slot_name):
    """Move down to slot low pose but with gripper open."""
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = get_open_pose_from_grip(slot_name)
    print(f"[MOVE] To {slot_name} with gripper OPEN: {pose}")
    move_angles(pose)

def grip_at_slot(slot_name):
    """Close gripper at slot low pose (full 6-servo move)."""
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = SLOTS_GRIP[slot_name]
    print(f"[GRIP] Closing gripper at {slot_name}: {pose}")
    move_angles(pose)

def release_at_slot(slot_name):
    """Open gripper at slot low pose."""
    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return
    pose = get_open_pose_from_grip(slot_name)
    print(f"[RELEASE] Opening gripper at {slot_name}: {pose}")
    move_angles(pose)

# --------------------------------------------------------------------
# High-level sequences
# --------------------------------------------------------------------
def pick_from_slot(slot_name):
    """
    Safe pick:
      - high & open
      - rotate over slot in the air (2&3 high)
      - drop (2&3) to low open pose
      - close gripper
      - lift 2&3 back to safe
    """
    print(f"\n=== PICK from {slot_name} ===")

    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    low_open = get_open_pose_from_grip(slot_name)

    # 1) SAFE_OPEN
    go_safe_open()

    # 2) Phase 1: rotate over slot in air
    phase1 = SAFE_OPEN.copy()
    phase1[0] = low_open[0]  # base
    phase1[4] = low_open[4]  # wrist_rot
    print(f"[PHASE 1] Rotate above {slot_name}: {phase1}")
    move_angles(phase1)

    # 3) Phase 2: drop (2&3) to low_open
    print(f"[PHASE 2] Go down to {slot_name} with gripper OPEN: {low_open}")
    move_angles(low_open)

    # 4) Close gripper
    grip_at_slot(slot_name)

    # 5) Lift straight up using only 2&3
    lift_23_to_safe()

    print(f"=== DONE PICK {slot_name} ===\n")

def place_to_slot(slot_name):
    """
    Safe place:
      - from carrying: lift 2&3 to safe if needed
      - rotate in air above destination
      - drop (2&3) to low closed pose
      - open gripper
      - lift 2&3 back to safe
    """
    print(f"\n=== PLACE to {slot_name} ===")

    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    # 1) Make sure shoulder & elbow are high
    lift_23_to_safe()

    # 2) Rotate / move above destination in air
    move_above_slot_keep_23(slot_name)

    # 3) Drop down to low closed pose (holding object)
    dest_grip_pose = SLOTS_GRIP[slot_name].copy()
    print(f"[DROP] Down to {slot_name} (carrying): {dest_grip_pose}")
    move_angles(dest_grip_pose)

    # 4) Open gripper to release
    release_at_slot(slot_name)

    # 5) Lift up again
    lift_23_to_safe()

    print(f"=== DONE PLACE {slot_name} ===\n")

def move_object(src_slot, dst_slot):
    """Pick from src, then place to dst using safe vertical/horizontal sequence."""
    print(f"\n##### MOVE OBJECT: {src_slot} -> {dst_slot} #####")
    pick_from_slot(src_slot)
    place_to_slot(dst_slot)
    go_safe_open()
    print("##### DONE TRANSFER #####\n")

# --------------------------------------------------------------------
# Simple CLI
# --------------------------------------------------------------------
def main():
    print("================================================")
    print(" DOFBOT Safe Pick & Place (Arm_Lib / Angles)    ")
    print("================================================")
    print("Slots:", ", ".join(SLOTS_GRIP.keys()))
    print("\nCommands:")
    print("  open  <slot>          - move down to slot with gripper OPEN")
    print("  grip  <slot>          - close gripper at slot")
    print("  pick  <slot>          - safe pick from slot")
    print("  place <slot>          - safe place to slot")
    print("  move  <src> <dst>     - pick from src, place to dst")
    print("  home                  - go to SAFE_OPEN (high, open)")
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
            print("Invalid command or wrong arguments.\n")

if __name__ == "__main__":
    main()
