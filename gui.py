
from datetime import datetime

# import tkinter as tk
# from tkinter import filedialog, messagebox
# import os
# import cv2
# from PIL import Image, ImageTk
# from main import recognize_faces, capture_stable_image_from_camera
# import pandas as pd
# import threading

# class FaceRecognitionGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Face Recognition System")
#         self.root.geometry("800x600")
#         self.root.minsize(800, 600)

#         # Attributes
#         self.input_image_path = None
#         self.processed_image = None
#         self.results = None
#         self.process_status = tk.StringVar(value="To be processed")  # Track processing status

#         # Load student data
#         self.students_df = pd.read_csv("student.csv")  # Ensure this file exists

#         # Create GUI layout
#         self.create_widgets()

#     def create_widgets(self):
#         # Create frames for a 2x2 grid layout
#         frame1 = tk.LabelFrame(self.root, text="1. Take Input Image or Use Camera", padx=10, pady=10)
#         frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

#         frame2 = tk.LabelFrame(self.root, text="2. Process Image", padx=10, pady=10)
#         frame2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

#         frame3 = tk.LabelFrame(self.root, text="3. Open Attendance Sheet", padx=10, pady=10)
#         frame3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

#         frame4 = tk.LabelFrame(self.root, text="4. See Processed Image", padx=10, pady=10)
#         frame4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

#         # Section 1: Take Input Image or Use Camera
#         btn_choose_image = tk.Button(frame1, text="Choose Image", command=self.choose_image)
#         btn_choose_image.pack(pady=5)

#         btn_use_camera = tk.Button(frame1, text="Use Camera", command=self.capture_and_process)
#         btn_use_camera.pack(pady=5)

#         # Section 2: Process Image (Status)
#         self.status_label = tk.Label(frame2, textvariable=self.process_status, font=("Arial", 14), fg="blue")
#         self.status_label.pack(pady=10)

#         # Section 3: Open Attendance Sheet
#         btn_open_attendance = tk.Button(frame3, text="Open Attendance Sheet", command=self.open_attendance_sheet)
#         btn_open_attendance.pack(pady=5)

#         # Section 4: See Processed Image
#         btn_see_image = tk.Button(frame4, text="See Processed Image", command=self.display_processed_image)
#         btn_see_image.pack(pady=5)

#         # Configure the grid to resize properly
#         self.root.grid_rowconfigure(0, weight=1)
#         self.root.grid_rowconfigure(1, weight=1)
#         self.root.grid_columnconfigure(0, weight=1)
#         self.root.grid_columnconfigure(1, weight=1)

#     def choose_image(self):
#         """Choose an input image from the file dialog and process it."""
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
#         if file_path:
#             self.input_image_path = file_path
#             self.process_status.set("Processing...")
#             threading.Thread(target=self.process_selected_image).start()
#         else:
#             messagebox.showerror("No Image Selected", "Please select an image.")

#     def capture_and_process(self):
#         """Capture an image from the camera and process it."""
#         self.input_image_path = None  # Reset input image path
#         self.process_status.set("Processing...")

#         # Run the capture and processing in a separate thread to avoid freezing the GUI
#         threading.Thread(target=self._capture_and_process_image).start()

#     def _capture_and_process_image(self):
#         """Internal function to handle capture and processing."""
#         captured_image = capture_stable_image_from_camera(timeout=7, stability_threshold=3)
#         if captured_image is not None:
#             self.processed_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
#             self.process_image(self.processed_image)
#         else:
#             self.process_status.set("To be processed")
#             messagebox.showerror("Capture Failed", "Could not capture an image from the camera.")

#     def process_selected_image(self):
#         """Process the selected image from file dialog."""
#         if self.input_image_path:
#             image = cv2.imread(self.input_image_path)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             self.process_image(image_rgb)

#     def process_image(self, image):
#         """Process the given image."""
#         try:
#             self.process_status.set("Processing...")  # Update status to processing
#             processed_image, results = recognize_faces(image)
#             self.processed_image = processed_image
#             self.results = results

#             # Generate attendance sheet
#             if self.results:
#                 self.generate_attendance()
#                 self.process_status.set("Processed")
#                 messagebox.showinfo("Processing Complete", "Image processed, and attendance sheet generated.")
#             else:
#                 self.process_status.set("To be processed")
#                 messagebox.showerror("Processing Failed", "No faces detected in the image.")
#         except Exception as e:
#             self.process_status.set("To be processed")
#             messagebox.showerror("Error", f"An error occurred during processing: {e}")

#     def generate_attendance(self):
#         """Generate attendance sheet and mark absent students."""
#         try:

#             current_date = datetime.now().date()

#             # Initialize attendance
#             attendance = {row["roll_number"]: "A" for _, row in self.students_df.iterrows()}

#             # Mark students present
#             for _, _, _, _, roll_number in self.results:
#                 if roll_number in attendance:
#                     attendance[roll_number] = "P"

#             # Create a DataFrame
#             attendance_df = pd.DataFrame(
#                 [(roll, self.students_df[self.students_df["roll_number"] == roll]["name"].values[0], status)
#                  for roll, status in attendance.items()],
#                 columns=["Roll Number", "Name", str(current_date)]
#             )

#             # Save to CSV
#             attendance_df.to_csv("attendance.csv", index=False)
#             messagebox.showinfo("Attendance Saved", "Attendance has been saved to 'attendance.csv'.")
#         except PermissionError:
#             messagebox.showerror("Permission Denied", "Permission denied: Unable to write to 'attendance.csv'. Please close the file if it is open and try again.")
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred while generating attendance: {e}")

    
#     def open_attendance_sheet(self):
#         """Open the generated attendance sheet and display absent students."""

#         current_date = datetime.now().date()

#         attendance_file = "attendance.csv"
#         if os.path.exists(attendance_file):
#             # Open attendance sheet
#             os.startfile(attendance_file)  # Works on Windows

#             # Display absent students
#             attendance_df = pd.read_csv(attendance_file)
#             absent_students = attendance_df[attendance_df["Attendance"]] == "A"]
#             if not absent_students.empty:
#                 absent_list = "\n".join(absent_students["Name"].tolist())
#                 messagebox.showinfo("Absent Students", f"Absent Students:\n\n{absent_list}")
#             else:
#                 messagebox.showinfo("Absent Students", "All students are present.")
#         else:
#             messagebox.showerror("File Not Found", "Attendance sheet not found. Please process an image first.")

#     def display_processed_image(self):
#         """Display the processed image with recognized faces."""
#         if self.processed_image is None:
#             messagebox.showerror("No Image", "Please process an image first.")
#             return

#         # Convert image to PIL format
#         self.img_pil = Image.fromarray(self.processed_image)
#         self.img_width, self.img_height = self.img_pil.size

#         # Initialize the scale to fit the image within the window
#         self.scale = min(760 / self.img_width, 560 / self.img_height)

#         if hasattr(self, 'window') and self.window.winfo_exists():
#             self.window.destroy()

#         # Create a new window to display the image
#         self.window = tk.Toplevel(self.root)
#         self.window.title("Processed Image")
#         self.window.geometry("800x600")

#         # Create a canvas and add scrollbars
#         self.canvas = tk.Canvas(self.window, width=760, height=560)
#         self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

#         self.hbar = tk.Scrollbar(self.window, orient=tk.HORIZONTAL, command=self.canvas.xview)
#         self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
#         self.vbar = tk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.canvas.yview)
#         self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

#         self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

#         # Display the image on the canvas
#         self.update_image()

#         # Add zoom in and zoom out buttons
#         zoom_in_button = tk.Button(self.window, text="Zoom In", command=self.zoom_in)
#         zoom_in_button.pack(side=tk.LEFT, padx=10, pady=10)

#         zoom_out_button = tk.Button(self.window, text="Zoom Out", command=self.zoom_out)
#         zoom_out_button.pack(side=tk.LEFT, padx=10, pady=10)

#         # Add a close button
#         btn_close = tk.Button(self.window, text="Close", command=self.window.destroy)
#         btn_close.pack(side=tk.RIGHT, padx=10, pady=10)

#     def update_image(self):
#         """Update the image display with the current scale."""
#         new_width = int(self.img_width * self.scale)
#         new_height = int(self.img_height * self.scale)
#         img_resized = self.img_pil.resize((new_width, new_height), Image.LANCZOS)
#         self.img_tk = ImageTk.PhotoImage(img_resized)
#         self.canvas.config(scrollregion=(0, 0, new_width, new_height))
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

#     def zoom_in(self):
#         """Zoom in the image."""
#         self.scale *= 1.1  # Increase the scale by 10%
#         self.update_image()

#     def zoom_out(self):
#         """Zoom out the image."""
#         self.scale /= 1.1  # Decrease the scale by 10%
#         self.update_image()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = FaceRecognitionGUI(root)
#     root.mainloop()


import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
from main import recognize_faces, capture_stable_image_from_camera
import pandas as pd
import threading

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Attributes
        self.input_image_path = None
        self.processed_image = None
        self.results = None
        self.process_status = tk.StringVar(value="To be processed")  # Track processing status

        # Load student data
        self.students_df = pd.read_csv("student.csv")  # Ensure this file exists

        # Create GUI layout
        self.create_widgets()

    def create_widgets(self):
        # Create frames for a 2x2 grid layout
        frame1 = tk.LabelFrame(self.root, text="1. Take Input Image or Use Camera", padx=10, pady=10)
        frame1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        frame2 = tk.LabelFrame(self.root, text="2. Process Image", padx=10, pady=10)
        frame2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        frame3 = tk.LabelFrame(self.root, text="3. Open Attendance Sheet", padx=10, pady=10)
        frame3.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        frame4 = tk.LabelFrame(self.root, text="4. See Processed Image", padx=10, pady=10)
        frame4.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Section 1: Take Input Image or Use Camera
        btn_choose_image = tk.Button(frame1, text="Choose Image", command=self.choose_image)
        btn_choose_image.pack(pady=5)

        btn_use_camera = tk.Button(frame1, text="Use Camera", command=self.capture_and_process)
        btn_use_camera.pack(pady=5)

        # Section 2: Process Image (Status)
        self.status_label = tk.Label(frame2, textvariable=self.process_status, font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=10)

        # Section 3: Open Attendance Sheet
        btn_open_attendance = tk.Button(frame3, text="Open Attendance Sheet", command=self.open_attendance_sheet)
        btn_open_attendance.pack(pady=5)

        # Section 4: See Processed Image
        btn_see_image = tk.Button(frame4, text="See Processed Image", command=self.display_processed_image)
        btn_see_image.pack(pady=5)

        # Configure the grid to resize properly
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def choose_image(self):
        """Choose an input image from the file dialog and process it."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.input_image_path = file_path
            self.process_status.set("Processing...")
            threading.Thread(target=self.process_selected_image).start()
        else:
            messagebox.showerror("No Image Selected", "Please select an image.")

    def capture_and_process(self):
        """Capture an image from the camera and process it."""
        self.input_image_path = None  # Reset input image path
        self.process_status.set("Processing...")

        # Run the capture and processing in a separate thread to avoid freezing the GUI
        threading.Thread(target=self._capture_and_process_image).start()

    def _capture_and_process_image(self):
        """Internal function to handle capture and processing."""
        captured_image = capture_stable_image_from_camera(timeout=7, stability_threshold=3)
        if captured_image is not None:
            self.processed_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            self.process_image(self.processed_image)
        else:
            self.process_status.set("To be processed")
            messagebox.showerror("Capture Failed", "Could not capture an image from the camera.")

    def process_selected_image(self):
        """Process the selected image from file dialog."""
        if self.input_image_path:
            image = cv2.imread(self.input_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.process_image(image_rgb)

    def process_image(self, image):
        """Process the given image."""
        try:
            self.process_status.set("Processing...")  # Update status to processing
            processed_image, results = recognize_faces(image)
            self.processed_image = processed_image
            self.results = results

            # Generate attendance sheet
            if self.results:
                self.generate_attendance()
                self.process_status.set("Processed")
                messagebox.showinfo("Processing Complete", "Image processed, and attendance sheet generated.")
            else:
                self.process_status.set("To be processed")
                messagebox.showerror("Processing Failed", "No faces detected in the image.")
        except Exception as e:
            self.process_status.set("To be processed")
            messagebox.showerror("Error", f"An error occurred during processing: {e}")

    def generate_attendance(self):
        """Generate attendance sheet and mark absent students."""
        try:

            current_date = datetime.now().date()

            # Initialize attendance
            attendance = {row["roll_number"]: "A" for _, row in self.students_df.iterrows()}

            # Mark students present
            for _, _, _, _, roll_number in self.results:
                if roll_number in attendance:
                    attendance[roll_number] = "P"

            # Create a DataFrame
            attendance_df = pd.DataFrame(
                [(roll, self.students_df[self.students_df["roll_number"] == roll]["name"].values[0], status)
                 for roll, status in attendance.items()],
                columns=["Roll Number", "Name", str(current_date)]
            )

            # Save to CSV
            attendance_df.to_csv("attendance.csv", index=False)
            messagebox.showinfo("Attendance Saved", "Attendance has been saved to 'attendance.csv'.")
        except PermissionError:
            messagebox.showerror("Permission Denied", "Permission denied: Unable to write to 'attendance.csv'. Please close the file if it is open and try again.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating attendance: {e}")

    def open_attendance_sheet(self):
        """Open the generated attendance sheet and display absent students."""

        current_date = datetime.now().date()

        attendance_file = "attendance.csv"
        if os.path.exists(attendance_file):
            # Open attendance sheet
            os.startfile(attendance_file)  # Works on Windows

            # Display absent students
            attendance_df = pd.read_csv(attendance_file)
            absent_students = attendance_df[attendance_df[str(current_date)] == "A"]
            if not absent_students.empty:
                absent_list = "\n".join(absent_students["Name"].tolist())
                messagebox.showinfo("Absent Students", f"Absent Students:\n\n{absent_list}")
            else:
                messagebox.showinfo("Absent Students", "All students are present.")
        else:
            messagebox.showerror("File Not Found", "Attendance sheet not found. Please process an image first.")

    def display_processed_image(self):
        """Display the processed image with recognized faces."""
        if self.processed_image is None:
            messagebox.showerror("No Image", "Please process an image first.")
            return

        # Convert image to PIL format
        self.img_pil = Image.fromarray(self.processed_image)
        self.img_width, self.img_height = self.img_pil.size

        # Initialize the scale to fit the image within the window
        self.scale = min(760 / self.img_width, 560 / self.img_height)

        if hasattr(self, 'window') and self.window.winfo_exists():
            self.window.destroy()

        # Create a new window to display the image
        self.window = tk.Toplevel(self.root)
        self.window.title("Processed Image")
        self.window.geometry("800x600")

        # Create a canvas and add scrollbars
        self.canvas = tk.Canvas(self.window, width=760, height=560)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.hbar = tk.Scrollbar(self.window, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar = tk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.canvas.yview)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        # Display the image on the canvas
        self.update_image()

        # Add zoom in and zoom out buttons
        zoom_in_button = tk.Button(self.window, text="Zoom In", command=self.zoom_in)
        zoom_in_button.pack(side=tk.LEFT, padx=10, pady=10)

        zoom_out_button = tk.Button(self.window, text="Zoom Out", command=self.zoom_out)
        zoom_out_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Add a close button
        btn_close = tk.Button(self.window, text="Close", command=self.window.destroy)
        btn_close.pack(side=tk.RIGHT, padx=10, pady=10)

    def update_image(self):
        """Update the image display with the current scale."""
        new_width = int(self.img_width * self.scale)
        new_height = int(self.img_height * self.scale)
        img_resized = self.img_pil.resize((new_width, new_height), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img_resized)
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def zoom_in(self):
        """Zoom in the image."""
        self.scale *= 1.1  # Increase the scale by 10%
        self.update_image()

    def zoom_out(self):
        """Zoom out the image."""
        self.scale /= 1.1  # Decrease the scale by 10%
        self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()
