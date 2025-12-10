#!/usr/bin/env python3
"""
robot_gui.py

Simple Tkinter GUI to control DOFBOT using functions from slot_move_demo.py.

Features:
- 'Home' button: move arm to SAFE_OPEN pose.
- One button per object: calls move_object_named('<object>'),
  which moves the object from <object>_wrong to <object>_correct.
- Status label shows what the robot is doing.

Run on the Raspberry Pi:
    python3 robot_gui.py
"""

import threading
import tkinter as tk
from tkinter import ttk

# Import your existing motion logic
from slot_move_demo import move_object_named, go_safe_open, PAIR_MAP

# ------------- GUI callbacks -------------

def run_move_object(obj_name):
    """Worker function executed in a background thread."""
    status_var.set(f"Moving {obj_name}...")
    try:
        move_object_named(obj_name)
        status_var.set(f"Done moving {obj_name}.")
    except Exception as e:
        status_var.set(f"Error while moving {obj_name}: {e}")

def on_move_button(obj_name):
    """Start a background thread so GUI does not block."""
    t = threading.Thread(target=run_move_object, args=(obj_name,), daemon=True)
    t.start()

def run_home():
    """Move to safe open pose in a background thread."""
    status_var.set("Moving to home (SAFE_OPEN)...")
    def worker():
        try:
            go_safe_open()
            status_var.set("At home (SAFE_OPEN).")
        except Exception as e:
            status_var.set(f"Error going home: {e}")
    threading.Thread(target=worker, daemon=True).start()

# ------------- Build the window -------------

root = tk.Tk()
root.title("DOFBOT Control Panel")

# Make window a bit nicer
root.geometry("400x300")
root.resizable(False, False)

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill="both", expand=True)

title_label = ttk.Label(
    main_frame,
    text="DOFBOT Object Sorting GUI",
    font=("Helvetica", 14, "bold")
)
title_label.pack(pady=(0, 10))

# Home button
home_btn = ttk.Button(main_frame, text="Home (SAFE_OPEN)", command=run_home)
home_btn.pack(pady=(0, 10), fill="x")

# Frame for object buttons
objects_frame = ttk.LabelFrame(main_frame, text="Fix objects", padding=10)
objects_frame.pack(fill="both", expand=True)

# Create one button per object in PAIR_MAP
# PAIR_MAP keys should be: "mouse", "pen", "pendrive", "eraser", "stapler", "adapter"
row = 0
col = 0
for obj_name in sorted(PAIR_MAP.keys()):
    btn_text = f"Fix {obj_name.capitalize()}"
    btn = ttk.Button(
        objects_frame,
        text=btn_text,
        command=lambda name=obj_name: on_move_button(name)
    )
    btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

    col += 1
    if col >= 2:
        col = 0
        row += 1

for i in range(2):
    objects_frame.columnconfigure(i, weight=1)

# Status label
status_var = tk.StringVar(value="Idle")
status_label = ttk.Label(
    main_frame,
    textvariable=status_var,
    font=("Helvetica", 10),
    foreground="blue"
)
status_label.pack(pady=(10, 0), fill="x")

# ------------- Start GUI loop -------------

if __name__ == "__main__":
    root.mainloop()
