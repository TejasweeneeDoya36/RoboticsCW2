# pose_slots.py
# DOFBOT pose tester using the official dofbot.py library.
# Edit the slot coordinates and run the script directly on DOFBOT.

from dofbot import Dofbot
import time

# --------------------------------
# Setup DOFBOT
# --------------------------------
bot = Dofbot()

# Default orientation (usually OK for placement)
DEFAULT_POSE_RPY = [0, 0, 0]  # roll, pitch, yaw

# --------------------------------
# EDIT THESE COORDINATES
# You will fill them with coordinates printed from your keyboard-control script.
# Format: (x, y, z)
# --------------------------------
SLOTS = {
    "home":         (0, 180, 120),
    "mouse_slot":   (80, 120, 30),
    "pen_slot":     (120, 100, 30),
    "pendrive_slot":(150, 50, 30),
    "eraser_slot":  (150, -10, 30),
    "stapler_slot": (120, -80, 30),
    "adapter_slot": (80, -120, 30),
}

# --------------------------------
# Movement function
# --------------------------------
def move_to(name):
    if name not in SLOTS:
        print("Unknown slot.")
        return
    x, y, z = SLOTS[name]
    rx, ry, rz = DEFAULT_POSE_RPY

    print(f"\n[MOVE] {name} -> {x, y, z}")
    bot.set_tool_pose([x, y, z, rx, ry, rz])
    time.sleep(1.5)
    print("[OK] Moved.\n")


# --------------------------------
# Main loop
# --------------------------------
if __name__ == "__main__":
    print("======================================")
    print("      DOFBOT Slot Pose Tester         ")
    print("======================================\n")
    print("Commands:")
    print("  list  - show available slots")
    print("  home  - return to home pose")
    print("  q     - quit\n")

    while True:
        cmd = input("Enter slot name: ").strip()

        if cmd.lower() in ["q", "quit", "exit"]:
            move_to("home")
            break

        elif cmd == "list":
            print("Slots:", ", ".join(SLOTS.keys()))

        elif cmd in SLOTS:
            move_to(cmd)

        else:
            print("Unknown name. Type 'list'.")
