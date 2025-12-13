DOFBOT Object Sorting System

Robotics Coursework 2

Project Overview

This project implements an autonomous object sorting system using a DOFBOT robotic arm, YOLOv8 object detection, and a Raspberry Pi.
The robot detects misplaced office objects using a trained YOLO model and performs safe pick-and-place operations to move each object to its correct location.
A GUI is provided for live monitoring, manual control, and full automation.

Key Objectives
Detect 6 different object classes (coursework requirement)
Automatically sort objects using DOFBOT
Ensure safe and repeatable robot movements
Provide a clear live demonstration interface

Object Classes
The system detects and sorts the following objects:
Adapter
Eraser
Mouse
Pen
Pendrive
Stapler

Each object has:
a wrong position
a correct target position
Positions are defined using calibrated servo angles.

Main Files
robot_gui.py            → Main GUI application (recommended)
detect_webcam_dofbot.py → Automated detection + robot control (no GUI)
slot_move_demo.py       → Safe pick & place logic (core robot motion)

train_yolo.py           → YOLOv8 training script
eval_yolo.py            → Model evaluation (mAP, precision, recall)

read_servo.py           → Read servo angles for calibration
pose_slots.py           → Test and tune slot poses

Requirements
Hardware
DOFBOT robotic arm
Raspberry Pi
USB / Pi Camera
Software
Python 3.9+
Raspberry Pi OS

Install dependencies:
pip install -r requirements.txt

Arm_Lib is included with the DOFBOT SDK and does not need pip installation.

How to Run
Recommended (GUI Mode)
python3 robot_gui.py


Features:
Live camera feed with YOLO bounding boxes
FPS display
Manual fix buttons (Fix Pen, Fix Mouse, etc.)
Start / Stop automation
Status bar

Automation Only (No GUI)
python3 detect_webcam_dofbot.py

Runs YOLO + robot automation
Press ESC to exit

Robot Motion Logic
All robot movements are handled in slot_move_demo.py.

Key features:
Safe SAFE_OPEN pose
Vertical lifting using shoulder & elbow only
Extra grip tightening to avoid object drop
Each object is moved only once
High-level command used by automation:
move_object_named("pen")

YOLO Model
YOLOv8 trained on 6 office objects
Runs fully on CPU (Raspberry Pi)
Threaded inference to improve FPS
Stable detection (object must appear in consecutive frames)

Training & Evaluation
python3 train_yolo.py
python3 eval_yolo.py

Used to generate metrics and plots for the coursework report
Not required for live demo if model is already trained

Calibration Tools
read_servo.py → Read current servo angles
pose_slots.py → Test slot positions during setup

Github repository: https://github.com/TejasweeneeDoya36/RoboticsCW2.git  

Author
Vashistha Ittoo
Tejasweenee Doya
BSc (Hons) Computer Science
Middlesex University Mauritius
