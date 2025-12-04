# pose_slots.py
# Run this directly on the DOFBOT Raspberry Pi.
# Simple tester: edit coordinates in SLOTS, run the script,
# type a slot name, and the arm will move there.

import time
import ArmIK  # Yahboom DOFBOT IK library

# --------------------------------
# Setup
# --------------------------------
# Create the IK object
ak = ArmIK.ArmIK()

# movement time in milliseconds (bigger = slower, smoother)
MOVE_TIME_MS = 1500

# --------------------------------
# EDIT THESE COORDINATES
# Fill them using the values you read from your keyboard script.
# Positions are (x, y, z) in mm.
# Orientation is kept fixed via the pitch range args.
# --------------------------------
SLOTS = {
    "home":         (0,   150, 150),
    "mouse_slot":   (80,  120,  30),
    "pen_slot":     (120, 100,  30),
    "pendrive_slot":(150,  50,  30),
    "eraser_slot":  (150, -10,  30),
    "stapler_slot": (120, -80,  30),
    "adapter_slot": (80, -120,  30),
}

# --------------------------------
# Movement helper
# --------------------------------
def move_to_slot(name: str):
    if name not in SLOTS:
        print(f"[ERR] Slot '{name}' not defined in SLOTS.")
        return

    x, y, z = SLOTS[name]
    print(f"\n[MOVE] {name} -> (x={x}, y={y}, z={z})")

    # setPitchRangeMoving((x, y, z), roll, pitch, yaw, time_ms)
    # Typical Yahboom usage: roll/pitch/yaw fixed at -90/-90/0
    ok = ak.setPitchRangeMoving((x, y, z), -90, -90, 0, MOVE_TIME_MS)

    if ok:
        print("[OK] Reached target (or close enough).")
    else:
        print("[WARN] IK could not reach that pose.")
    time.sleep(1)


def main():
    print("======================================")
    print("       DOFBOT Slot Pose Tester        ")
    print("======================================")
    print("Edit SLOTS in this file to change coordinates.")
    print("Then run again and test.\n")
    print("Commands:")
    print("  list  - show all slot names")
    print("  home  - move to home pose (if defined)")
    print("  q     - quit\n")

    while True:
        cmd = input("Slot name (or 'list', 'home', 'q'): ").strip()

        if cmd in ("q", "quit", "exit"):
            if "home" in SLOTS:
                move_to_slot("home")
            print("Bye.")
            break

        elif cmd == "list":
            print("Available slots:", ", ".join(SLOTS.keys()))

        elif cmd == "home":
            if "home" in SLOTS:
                move_to_slot("home")
            else:
                print("No 'home' defined in SLOTS.")

        elif cmd in SLOTS:
            move_to_slot(cmd)

        else:
            print("Unknown name. Type 'list' to see all slots.\n")


if __name__ == "__main__":
    main()
