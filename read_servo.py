#!/usr/bin/env python3
# coding=utf-8
#
# read_servos.py
# Move the arm manually, then use this script to read the current servo angles.

import time
from Arm_Lib import Arm_Device

# Create robot arm object
Arm = Arm_Device()
time.sleep(0.1)

def read_all_servos():
    """Read angles of all 6 servos and return as a list."""
    angles = []
    for i in range(6):
        ang = Arm.Arm_serial_servo_read(i + 1)  # servos are 1..6
        angles.append(ang)
    return angles

def main():
    print("======================================")
    print("      DOFBOT Servo Angle Reader       ")
    print("======================================")
    print("Move the arm to a pose (manually or via keys).")
    print("Then:")
    print("  - Press ENTER to read all servo angles")
    print("  - Type 'q' and ENTER to quit")
    print("======================================\n")

    while True:
        cmd = input("Press ENTER to read, or 'q' + ENTER to quit: ").strip().lower()
        if cmd == "q":
            print("Exiting...")
            break

        angles = read_all_servos()
        print("\nCurrent servo angles:", angles)
        print("Copy this list into your SLOTS dict, e.g.:")
        print(f'    "some_slot": {angles},\n')

    # Optional: release object
    del Arm

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        del Arm
