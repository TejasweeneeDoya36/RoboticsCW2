#!/usr/bin/env python3
"""
robot_gui.py

Tkinter GUI to control DOFBOT and run the fully automated
YOLO + webcam sorting pipeline INSIDE the GUI window.

Features:
- Embedded video feed (webcam + YOLO boxes + FPS) on the left
- Manual controls on the right:
    - Home (SAFE_OPEN)
    - One button per object: Fix <object> (uses move_object_named)
- Start Automation / Stop Automation buttons
- Status bar

Requirements:
    pip install ultralytics opencv-python pillow
Run on Raspberry Pi:
    python3 robot_gui.py
"""

import threading
import queue
import time

import os
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Import motion logic and robot definitions
from slot_move_demo import (
    move_object_named,
    go_safe_open,
    PAIR_MAP,
    Arm,
    SAFE_OPEN,
)

# ---------------- YOLO + CAMERA CONFIG ----------------

CAM_INDEX = 1
MODEL_PATH = "models/office_yolo.pt"
CONF_THRES = 0.5

INFER_INTERVAL = 0.25     # seconds between YOLO inferences
YOLO_IMGSZ = 224          # YOLO input size

# How many consecutive frames a label must appear
# before the robot is allowed to act
STABLE_DETECTIONS = 2

# Radar base sweep configuration
SCAN_BASE_MIN = 50
SCAN_BASE_MAX = 110
SCAN_BASE_STEP = 2
SCAN_MOVE_TIME = 100
SCAN_INTERVAL = 0.25

# YOLO class index → name
CUSTOM_NAMES = {
    0: "adapter",
    1: "eraser",
    2: "mouse",
    3: "pen",
    4: "pendrive",
    5: "stapler",
}

SCAN_POSE = SAFE_OPEN.copy()  # [base, s2, s3, s4, s5, s6]


def move_base_with_same_pose(base_angle, move_time=SCAN_MOVE_TIME):
    """
    Move only the base (servo 1) to base_angle,
    keep servos 2–6 equal to SCAN_POSE.
    """
    pose = SCAN_POSE.copy()
    pose[0] = base_angle
    print(f"[SCAN] Rotating base to {base_angle}° with pose {pose}")
    Arm.Arm_serial_servo_write6(*pose, move_time)
    time.sleep(move_time / 1000.0)


# --------------- GLOBAL RUNTIME STATE (for GUI) ---------------

automation_thread = None
automation_running = False
stop_event = threading.Event()

# latest annotated frame from automation (BGR)
latest_frame_bgr = None
latest_frame_lock = threading.Lock()


# ---------- Manual control helpers ----------

def run_move_object(obj_name, status_var):
    """Worker to move one object without freezing the GUI."""
    status_var.set(f"Moving {obj_name}...")
    try:
        move_object_named(obj_name)
        status_var.set(f"Done moving {obj_name}.")
    except Exception as e:
        status_var.set(f"Error moving {obj_name}: {e}")


def on_manual_fix_clicked(obj_name, status_var):
    """Start a background thread for manual fix."""
    t = threading.Thread(target=run_move_object, args=(obj_name, status_var), daemon=True)
    t.start()


def on_home_clicked(status_var):
    """Move to SAFE_OPEN in background thread."""
    def worker():
        status_var.set("Moving to home (SAFE_OPEN)...")
        try:
            go_safe_open()
            status_var.set("At home (SAFE_OPEN).")
        except Exception as e:
            status_var.set(f"Error going home: {e}")

    threading.Thread(target=worker, daemon=True).start()


# --------------- AUTOMATION LOOP (YOLO + ROBOT) ---------------

def automation_loop(status_var):
    """
    Runs in a background thread:
    - Opens camera
    - Spawns YOLO worker thread
    - Performs radar sweep, detection & robot actions
    - Updates latest_frame_bgr for Tkinter to display
    """

    global latest_frame_bgr

    status_var.set("Loading YOLO model...")
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        status_var.set(f"Error loading YOLO model: {e}")
        return

    status_var.set("Moving DOFBOT to safe scan pose...")
    go_safe_open()
    time.sleep(1.0)

    base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
    base_dir = +1
    last_scan_update = time.time()

    print(f"[INFO] Opening webcam at index {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        status_var.set(f"Could not open camera index {CAM_INDEX}")
        print(f"[ERROR] Could not open camera index {CAM_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    status_var.set("Automation running... (ESC in terminal to force stop)")
    print("[INFO] Automation started. Press GUI Stop button to end.")

    frame_count = 0
    t0 = time.perf_counter()
    moving_robot = False
    already_moved = set()
    total_known_objects = len(PAIR_MAP)
    all_done = False

    frame_queue = queue.Queue(maxsize=1)
    last_detections = []
    detections_lock = threading.Lock()
    last_infer_time = 0.0

    # track how long each label has been seen continuously
    detection_streak = {name: 0 for name in PAIR_MAP.keys()}


    def yolo_worker():
        nonlocal last_detections, last_infer_time
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.perf_counter()
            if now - last_infer_time < INFER_INTERVAL:
                continue

            last_infer_time = now

            results_list = model.predict(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=CONF_THRES,
                verbose=False
            )
            results = results_list[0] if results_list else None

            detected = []
            if results is not None and hasattr(results, "boxes"):
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if conf < CONF_THRES:
                        continue
                    label = CUSTOM_NAMES.get(cls_id, f"class_{cls_id}")
                    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                    detected.append((label, conf, (x1_i, y1_i, x2_i, y2_i)))

            with detections_lock:
                last_detections = detected

    worker = threading.Thread(target=yolo_worker, daemon=True)
    worker.start()

    try:
        while not stop_event.is_set():
            now = time.time()

            # Radar sweep
            if not moving_robot and (now - last_scan_update) > SCAN_INTERVAL:
                base_angle += SCAN_BASE_STEP * base_dir
                if base_angle >= SCAN_BASE_MAX:
                    base_angle = SCAN_BASE_MAX
                    base_dir = -1
                elif base_angle <= SCAN_BASE_MIN:
                    base_angle = SCAN_BASE_MIN
                    base_dir = +1
                move_base_with_same_pose(base_angle)
                last_scan_update = now

            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to grab frame")
                break

            frame_count += 1

            if not moving_robot and frame_queue.empty():
                try:
                    frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass

            with detections_lock:
                detected_objects = list(last_detections)

            # Update detection streaks for each known label
            current_labels = {label for (label, _, _) in detected_objects}
            for name in PAIR_MAP.keys():
                if name in current_labels:
                    detection_streak[name] = detection_streak.get(name, 0) + 1
                else:
                    detection_streak[name] = 0


            # Robot logic
            if not moving_robot and detected_objects and not all_done:
                for label, conf, _bbox in detected_objects:
                    # must be a known object, not already moved,
                    # and visible for at least STABLE_DETECTIONS frames
                    if (
                        label in PAIR_MAP
                        and label not in already_moved
                        and detection_streak.get(label, 0) >= STABLE_DETECTIONS
                    ):
                        print(f"[DETECT] Stable '{label}' conf={conf:.2f}")
                        status_var.set(f"Picking {label}...")
                        moving_robot = True

                        # blocking pick & place
                        move_object_named(label)
                        already_moved.add(label)
                        print(f"[STATE] Objects fixed: {len(already_moved)}/{total_known_objects}")

                        go_safe_open()
                        time.sleep(1.0)

                        base_angle = max(SCAN_BASE_MIN, min(SCAN_BASE_MAX, SCAN_POSE[0]))
                        base_dir = +1
                        moving_robot = False
                        time.sleep(0.5)

                        if len(already_moved) == total_known_objects:
                            print("[INFO] All objects sorted.")
                            status_var.set("All objects sorted. Automation complete.")
                            all_done = True
                        else:
                            status_var.set("Automation running...")
                        break

            # Draw detections + overlay text
            for label, conf, (x1, y1, x2, y2) in detected_objects:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            dt = time.perf_counter() - t0
            fps = frame_count / dt if dt > 0 else 0.0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if all_done:
                h, w = frame.shape[:2]
                msg = "ALL OBJECTS SORTED - AUTOMATION COMPLETE"
                cv2.putText(frame, msg, (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Store frame for GUI
            with latest_frame_lock:
                latest_frame_bgr = frame.copy()

        print("[INFO] Automation loop exiting...")

    finally:
        cap.release()
        status_var.set("Automation stopped.")
        with latest_frame_lock:
            latest_frame_bgr = None


# ---------- Automation control helpers for GUI ----------

def start_automation(status_var):
    """Start the automation thread if not already running."""
    global automation_thread, automation_running, stop_event

    if automation_running and automation_thread and automation_thread.is_alive():
        status_var.set("Automation already running.")
        return

    stop_event.clear()
    automation_running = True
    status_var.set("Starting automation...")

    automation_thread = threading.Thread(
        target=automation_loop, args=(status_var,), daemon=True
    )
    automation_thread.start()


def stop_automation(status_var):
    """Signal the automation thread to stop."""
    global automation_thread, automation_running, stop_event

    if not automation_running:
        status_var.set("Automation is not running.")
        return

    status_var.set("Stopping automation...")
    stop_event.set()
    automation_running = False


def on_close(root, status_var):
    """Make sure automation is stopped when GUI is closed."""
    stop_automation(status_var)
    root.after(500, root.destroy)  # small delay for thread cleanup


# ---------- Video display in Tkinter ----------

def update_video_label(root, video_label):
    """Periodically update the video label with the latest frame."""
    with latest_frame_lock:
        frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()

    if frame is not None:
        # Convert BGR → RGB → PIL → ImageTk
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        # Optional: resize to fit nicely
        img = img.resize((480, 360))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # keep reference
        video_label.configure(image=imgtk)

    # Schedule next update
    root.after(40, update_video_label, root, video_label)  # ~25 FPS GUI refresh


# ---------- Build GUI (layout + aesthetics) ----------

def build_gui():
    root = tk.Tk()
    root.title("DOFBOT Sorting Control Panel")
    root.geometry("900x500")
    root.resizable(False, False)

    # Simple dark theme
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#1e1e1e")
    style.configure("TLabel", background="#1e1e1e", foreground="#ffffff")
    style.configure("Title.TLabel", font=("Helvetica", 18, "bold"))
    style.configure("Section.TLabelframe", background="#252526", foreground="#ffffff")
    style.configure("Section.TLabelframe.Label", background="#252526", foreground="#ffffff")
    style.configure("TButton", font=("Helvetica", 10))

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)

    # Top title
    title_label = ttk.Label(
        main_frame,
        text="DOFBOT Object Sorting System",
        style="Title.TLabel"
    )
    title_label.pack(pady=(0, 10))

    # Split: left video, right controls
    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill="both", expand=True)

    # Left: video
    video_frame = ttk.LabelFrame(
        content_frame,
        text="Camera View (YOLO + FPS)",
        style="Section.TLabelframe",
        padding=5
    )
    video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

    video_label = ttk.Label(video_frame, text="Starting camera...")
    video_label.pack(fill="both", expand=True)

    # Right: controls
    control_frame = ttk.Frame(content_frame)
    control_frame.pack(side="right", fill="y")

    status_var = tk.StringVar(value="Idle")

    home_btn = ttk.Button(
        control_frame,
        text="Home (SAFE_OPEN)",
        command=lambda: on_home_clicked(status_var)
    )
    home_btn.pack(fill="x", pady=(0, 10))

    # Manual fix section
    manual_frame = ttk.LabelFrame(
        control_frame,
        text="Manual Fix Objects",
        style="Section.TLabelframe",
        padding=5
    )
    manual_frame.pack(fill="x", pady=(0, 10))

    for obj_name in sorted(PAIR_MAP.keys()):
        btn_text = f"Fix {obj_name.capitalize()}"
        b = ttk.Button(
            manual_frame,
            text=btn_text,
            command=lambda name=obj_name: on_manual_fix_clicked(name, status_var)
        )
        b.pack(fill="x", pady=2)

    # Automation section
    auto_frame = ttk.LabelFrame(
        control_frame,
        text="Automation (YOLO + Webcam)",
        style="Section.TLabelframe",
        padding=5
    )
    auto_frame.pack(fill="x", pady=(0, 10))

    start_btn = ttk.Button(
        auto_frame,
        text="Start Automation",
        command=lambda: start_automation(status_var)
    )
    start_btn.pack(fill="x", pady=(0, 5))

    stop_btn = ttk.Button(
        auto_frame,
        text="Stop Automation",
        command=lambda: stop_automation(status_var)
    )
    stop_btn.pack(fill="x")

    # Status bar
    status_label = ttk.Label(
        main_frame,
        textvariable=status_var,
        anchor="w"
    )
    status_label.pack(fill="x", pady=(10, 0))

    # Close event
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, status_var))

    # Start periodic video refresh
    root.after(100, update_video_label, root, video_label)

    return root


if __name__ == "__main__":
    app = build_gui()
    app.mainloop()
