#!/usr/bin/env python3
"""
robot_gui.py

Tkinter GUI to control DOFBOT and launch the fully automated
YOLO+webcam sorting script (detect_webcam_dofbot.py).

Features:
- Home (SAFE_OPEN) button
- One button per object: Fix <object> (uses move_object_named)
- Start Automation: runs detect_webcam_dofbot.py as a separate process
- Stop Automation: terminates the automation process

Run on Raspberry Pi:
    python3 robot_gui.py
"""

import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

# Import your motion logic
from slot_move_demo import move_object_named, go_safe_open, PAIR_MAP

# Path to your automation script
AUTOMATION_SCRIPT = "detect_webcam_dofbot.py"

# Global handle for the automation process
automation_proc = None


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


# ---------- Automation control helpers ----------

def start_automation(status_var):
    """Launch detect_webcam_dofbot.py as a separate process."""
    global automation_proc

    # If already running, do nothing
    if automation_proc is not None and automation_proc.poll() is None:
        status_var.set("Automation already running.")
        return

    status_var.set("Starting automation...")

    # Start the script without blocking the GUI
    try:
        # Use python3 and the script path; it will open its own window
        automation_proc = subprocess.Popen(
            ["python3", AUTOMATION_SCRIPT],
            cwd=os.path.dirname(os.path.abspath(__file__)) or None
        )
        status_var.set("Automation started.")
    except Exception as e:
        automation_proc = None
        status_var.set(f"Error starting automation: {e}")


def stop_automation(status_var):
    """Terminate the automation process if running."""
    global automation_proc

    if automation_proc is None or automation_proc.poll() is not None:
        status_var.set("Automation is not running.")
        automation_proc = None
        return

    status_var.set("Stopping automation...")
    try:
        automation_proc.terminate()
    except Exception as e:
        status_var.set(f"Error stopping automation: {e}")
    else:
        status_var.set("Automation stopped.")
    finally:
        automation_proc = None


def on_close(root, status_var):
    """Make sure automation is stopped when GUI is closed."""
    stop_automation(status_var)
    root.destroy()


# ---------- Build GUI ----------

def build_gui():
    root = tk.Tk()
    root.title("DOFBOT Control Panel")
    root.geometry("400x400")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(
        main_frame,
        text="DOFBOT Sorting GUI",
        font=("Helvetica", 14, "bold")
    )
    title_label.pack(pady=(0, 10))

    # Status
    status_var = tk.StringVar(value="Idle")

    # Home button
    home_btn = ttk.Button(
        main_frame,
        text="Home (SAFE_OPEN)",
        command=lambda: on_home_clicked(status_var)
    )
    home_btn.pack(fill="x", pady=(0, 10))

    # Manual fix section
    manual_frame = ttk.LabelFrame(main_frame, text="Manual Fix Objects", padding=5)
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
    auto_frame = ttk.LabelFrame(main_frame, text="Automation (YOLO + Webcam)", padding=5)
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

    # Status label
    status_label = ttk.Label(
        main_frame,
        textvariable=status_var,
        foreground="blue"
    )
    status_label.pack(fill="x", pady=(10, 0))

    # Hook close event
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, status_var))

    return root


if __name__ == "__main__":
    app = build_gui()
    app.mainloop()
