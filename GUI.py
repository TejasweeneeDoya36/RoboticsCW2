import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
from PIL import Image, ImageTk


class CameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DOFBOT Camera Feed")
        self.root.configure(bg="#0c1b33")  # Deep blue background

        # Set window size and center it
        window_width = 900
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # Header with gradient effect simulation
        self.header_frame = Frame(root, bg="#0c1b33", height=100)
        self.header_frame.pack(fill="x")
        self.header_frame.pack_propagate(False)

        self.title_label = Label(
            self.header_frame,
            text="DOFBOT CAMERA DASHBOARD",
            font=("Segoe UI", 24, "bold"),
            bg="#0c1b33",
            fg="#4fc3f7",
            pady=20
        )
        self.title_label.pack()

        # Subtitle
        self.subtitle_label = Label(
            self.header_frame,
            text="Robotic Arm Camera Feed - Index 1",
            font=("Segoe UI", 12),
            bg="#0c1b33",
            fg="#81d4fa"
        )
        self.subtitle_label.pack()

        # Outer container with subtle glow effect
        self.outer_frame = Frame(root, bg="#1a237e", bd=0)
        self.outer_frame.pack(padx=30, pady=20)

        # Inner frame for rounded look simulation
        self.mid_frame = Frame(self.outer_frame, bg="#283593", padx=15, pady=15)
        self.mid_frame.pack()

        # Video frame with blue border
        self.video_frame = Frame(
            self.mid_frame,
            bg="#3949ab",
            bd=3,
            relief="ridge",
            highlightbackground="#1a237e",
            highlightthickness=2
        )
        self.video_frame.pack()

        # Label to hold video feed
        self.video_label = Label(self.video_frame, bg="#1a237e")
        self.video_label.pack(padx=2, pady=2)

        # Status indicator
        self.status_frame = Frame(root, bg="#0c1b33")
        self.status_frame.pack(pady=10)

        self.status_indicator = Label(
            self.status_frame,
            text="●",
            font=("Arial", 12),
            fg="#00e676",
            bg="#0c1b33"
        )
        self.status_indicator.pack(side="left", padx=5)

        self.status_text = Label(
            self.status_frame,
            text="DOFBOT Camera Active (Index 1)",
            font=("Segoe UI", 10),
            fg="#81d4fa",
            bg="#0c1b33"
        )
        self.status_text.pack(side="left")

        # Stylish button with hover effect
        self.button = Button(
            root,
            text="START STREAM",
            font=("Segoe UI", 16, "bold"),
            bg="#2962ff",
            fg="white",
            activebackground="#2979ff",
            activeforeground="white",
            padx=40,
            pady=15,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2"
        )
        self.button.pack(pady=25)

        # Add button hover effects
        self.button.bind("<Enter>", self.on_button_hover)
        self.button.bind("<Leave>", self.on_button_leave)

        # Footer
        self.footer = Label(
            root,
            text="DOFBOT Robotic Arm Interface v1.0",
            font=("Segoe UI", 9),
            bg="#0c1b33",
            fg="#546e7a"
        )
        self.footer.pack(side="bottom", pady=10)

        # Start DOFBOT camera (camera index 1)
        self.cap = cv2.VideoCapture(1)  # Changed from 0 to 1 for DOFBOT

        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.status_indicator.config(fg="#ff5252")
            self.status_text.config(text="DOFBOT Camera Not Detected (Index 1)")
            print("Warning: Could not open DOFBOT camera at index 1")
        else:
            print("DOFBOT camera at index 1 opened successfully")

        # Begin fetching frames
        self.update_frame()

    def on_button_hover(self, event):
        self.button.config(bg="#2979ff")

    def on_button_leave(self, event):
        self.button.config(bg="#2962ff")

    def update_frame(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame))

                self.video_label.imgtk = img
                self.video_label.configure(image=img)
            else:
                # If frame capture fails, show error
                self.status_indicator.config(fg="#ff5252")
                self.status_text.config(text="DOFBOT Camera Error")

        # Refresh frame every 10 ms
        self.video_label.after(10, self.update_frame)


# Main
if __name__ == "__main__":
    root = tk.Tk()
    gui = CameraGUI(root)
    root.mainloop()