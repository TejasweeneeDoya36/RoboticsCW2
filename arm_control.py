# arm_only.py
import time
from Arm_Lib import Arm_Device

# Poses are [s1, s2, s3, s4, s5, s6] in degrees.
SAFE_POSE     = [90, 100, 90, 90, 90, 90]   # neutral viewing pose
PENDRIVE_POSE = [60, 110, 80, 90, 90, 40]   # reach-out + open gripper
MOUSE_POSE    = [120, 110, 80, 90, 90, 40]  # reach other side
PEN_POSE      = [90, 120, 70, 90, 90, 20]   # point down
ADAPTER_POSE  = [90, 90, 110, 90, 90, 40]   # slightly lower

POSES = {
    "0": ("SAFE", SAFE_POSE),
    "1": ("PENDRIVE", PENDRIVE_POSE),
    "2": ("MOUSE", MOUSE_POSE),
    "3": ("PEN", PEN_POSE),
    "4": ("ADAPTER", ADAPTER_POSE),
}

def move_pose(arm, pose, duration=800):
    """Move all 6 DOF to given pose in 'duration' ms."""
    for idx, angle in enumerate(pose, start=1):
        arm.Arm_serial_servo_write(idx, angle, duration)
    time.sleep(duration / 1000.0)

def main():
    print("Connecting to DOFBOT arm...")
    arm = Arm_Device()

    print("Moving to SAFE pose...")
    move_pose(arm, SAFE_POSE, 800)

    print("Arm-only control:")
    print("  0 = SAFE pose")
    print("  1 = PENDRIVE pose")
    print("  2 = MOUSE pose")
    print("  3 = PEN pose")
    print("  4 = ADAPTER pose")
    print("  q = quit")

    try:
        while True:
            key = input("Enter command (0-4, q): ").strip()
            if key == "q":
                break
            if key in POSES:
                name, pose = POSES[key]
                print(f"Moving to {name} pose...")
                move_pose(arm, pose, 800)
            else:
                print("Unknown command.")
    finally:
        print("Returning to SAFE pose...")
        move_pose(arm, SAFE_POSE, 800)
        print("Done.")

if __name__ == "__main__":
    main()
