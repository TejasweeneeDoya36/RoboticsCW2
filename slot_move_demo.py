#!/usr/bin/env python3
# slot_move_demo.py
# DOFBOT safe pick & place using Arm_Lib (servo angles)
#
# Features:
#   - Safe pick (2-phase with vertical moves of servos 2 & 3)
#   - Safe place (2-phase with vertical moves of servos 2 & 3)
#   - Extra grip tightening after picking so objects don't fall
#   - Each object has TWO positions:
#         <name>_wrong   -> when not in correct place
#         <name>_correct -> target/correct place
#
#   - Commands:
#       open  <slot>          - move down to that slot with gripper open
#       grip  <slot>          - close gripper at that slot
#       pick  <slot>          - safe pick from any slot key
#       place <slot>          - safe place to any slot key
#       move  <src> <dst>     - safe transfer from src slot to dst slot
#       moveobj <name>        - safe transfer from <name>_wrong -> <name>_correct
#       home                  - go to SAFE_OPEN (high, open)
#       q                     - quit

import time
from Arm_Lib import Arm_Device

Arm = Arm_Device()
time.sleep(0.1)

MOVE_TIME = 1500  # ms for each movement

# Tunable safety posture:
GRIP_OPEN_ANGLE = 60           # gripper open angle
SAFE_SHOULDER   = 121           # safe high shoulder angle (servo 2)
SAFE_ELBOW      = 41           # safe high elbow angle (servo 3)
SAFE_WRIST      = -45           # safe neutral wrist angle (servo 4)

# High, safe pose above all objects (pattern)
SAFE_OPEN = [90, SAFE_SHOULDER, SAFE_ELBOW, SAFE_WRIST, 90, GRIP_OPEN_ANGLE]

# Track the last commanded pose (so we don't need to read from hardware)
CURRENT_POSE = SAFE_OPEN.copy()

# --------------------------------------------------------------------
# YOUR CALIBRATED LOW "GRIP" POSES (object in grip)
# Each object has two positions: *_wrong and *_correct
# Format: [base, shoulder, elbow, wrist, wrist_rot, gripper_closed]
# --------------------------------------------------------------------
SLOTS_GRIP = {
    "home":           [90, 90, 90, 90, 90, GRIP_OPEN_ANGLE],  # home with open gripper

    # MOUSE
    "mouse_wrong":    [34, 14, 62, 36, 89, 80],   # example: mouse starting (wrong) place
    "mouse_correct":  [146, 10, 81, 6,   140,   80],   # example: mouse final correct place

    # PEN
    "pen_wrong":      [54, 34, 40, 39,   90,   180],
    "pen_correct":    [92, 81, -7, 15,   1,   180],

    # PENDRIVE
    "pendrive_wrong": [ 98, 42, 33, 33, 90, 158],
    "pendrive_correct":[4, 25, 70, -2,   50, 155],

    # ERASER
    "eraser_wrong":   [85,  16, 88, 0,   90,   153],
    "eraser_correct": [169,  29, 38, 56,   89,   158],

    # STAPLER
    "stapler_wrong": [69, 38 , 46, 18,   89, 155],
    "stapler_correct":[-20, 18, 80, -5,   35, 155],

    # ADAPTER
    "adapter_wrong":  [109,  42, 23, 52,  89,  102],
    "adapter_correct":[180,  44, 34, 18,  184,  102],
}

# Map each object name -> (wrong_slot_key, correct_slot_key)
PAIR_MAP = {
    "mouse":    ("mouse_wrong",    "mouse_correct"),
    "pen":      ("pen_wrong",      "pen_correct"),
    "pendrive": ("pendrive_wrong", "pendrive_correct"),
    "eraser":   ("eraser_wrong",   "eraser_correct"),
    "stapler":  ("stapler_wrong",  "stapler_correct"),
    "adapter":  ("adapter_wrong",  "adapter_correct"),
}

# --------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------
def move_angles(angles, move_time=MOVE_TIME):
    """Send 6-servo move command and remember this pose."""
    global CURRENT_POSE
    b, s, e, w, wr, g = angles
    Arm.Arm_serial_servo_write6(b, s, e, w, wr, g, move_time)
    CURRENT_POSE = angles.copy()
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
    Uses CURRENT_POSE instead of reading from servos, so we keep
    base, wrist, wrist_rot, and GRIP exactly as they are.
    """
    global CURRENT_POSE
    pose = CURRENT_POSE.copy()
    pose[1] = SAFE_SHOULDER  # shoulder up
    pose[2] = SAFE_ELBOW     # elbow up
    print(f"[LIFT] Raising servos 2 & 3 to safe: {pose}")
    move_angles(pose)


def move_above_slot_keep_23(slot_name):
    """
    Rotate/move in air to be above a slot:
    - base (0) and wrist_rot (4) from target slot
    - shoulder (1) and elbow (2) stay at SAFE_SHOULDER / SAFE_ELBOW
    - wrist (3) = SAFE_WRIST
    - gripper (5) = CURRENT_POSE[5] (keep holding object)
    """
    global CURRENT_POSE

    if slot_name not in SLOTS_GRIP:
        print("[ERR] Unknown slot:", slot_name)
        return

    target = SLOTS_GRIP[slot_name]
    cur_grip = CURRENT_POSE[5]

    pose = [
        target[0],        # base -> slot base
        SAFE_SHOULDER,    # shoulder high
        SAFE_ELBOW,       # elbow high
        SAFE_WRIST,       # wrist neutral high
        target[4],        # wrist_rot -> slot rotation
        cur_grip          # keep grip
    ]

    print(f"[MOVE] Above {slot_name} (keep 2&3 high): {pose}")
    move_angles(pose)


def tighten_grip(extra=10):
    """
    Increase grip tightness by closing the gripper extra degrees.
    On your robot, BIGGER angle = more closed.
    We use CURRENT_POSE instead of reading from servos.
    """
    global CURRENT_POSE
    cur_grip = CURRENT_POSE[5]
    new_grip = min(180, cur_grip + extra)
    print(f"[TIGHTEN] Increasing grip from {cur_grip} to {new_grip} (+{extra}°)")

    # Move only servo 6
    Arm.Arm_serial_servo_write(6, new_grip, MOVE_TIME)
    CURRENT_POSE[5] = new_grip
    time.sleep(MOVE_TIME / 1000.0)



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
      - tighten grip a bit more
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

    # 4b) Extra tighten so the object does not fall while travelling
    tighten_grip(extra=10)

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

def move_object_named(obj_name):
    """
    Move an object by its logical name, using its *_wrong and *_correct slots.
    Example:
        moveobj pendrive
      -> pick from pendrive_wrong, place to pendrive_correct
    """
    if obj_name not in PAIR_MAP:
        print("[ERR] Unknown object name:", obj_name)
        print("Known objects:", ", ".join(PAIR_MAP.keys()))
        return

    src_slot, dst_slot = PAIR_MAP[obj_name]
    move_object(src_slot, dst_slot)

# --------------------------------------------------------------------
# Simple CLI
# --------------------------------------------------------------------
def main():
    print("================================================")
    print(" DOFBOT Safe Pick & Place (Arm_Lib / Angles)    ")
    print("================================================")
    print("Slots:", ", ".join(SLOTS_GRIP.keys()))
    print("Objects (for moveobj):", ", ".join(PAIR_MAP.keys()))
    print("\nCommands:")
    print("  open  <slot>          - move down to slot with gripper OPEN")
    print("  grip  <slot>          - close gripper at slot")
    print("  pick  <slot>          - safe pick from any slot")
    print("  place <slot>          - safe place to any slot")
    print("  move  <src> <dst>     - safe transfer from src slot to dst slot")
    print("  moveobj <name>        - move <name>_wrong -> <name>_correct")
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

        elif cmd == "moveobj" and len(parts) == 2:
            move_object_named(parts[1])

        else:
            print("Invalid command or wrong arguments.\n")

if __name__ == "__main__":
    main()