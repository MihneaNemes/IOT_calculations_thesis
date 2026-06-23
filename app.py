import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import time
import os
import uuid

from predict_body_fat import predict_body_fat
from hardware import CameraManager, SensorManager
from aws_manager import AWSManager
from calculations import BodyFatCalculator
from data_manager import DataManager

MODEL_PATH = "/home/mihnea/Desktop/bodyfat_preictor/bodyfat_model_BEST.pth"
NORM_PATH = "/home/mihnea/Desktop/bodyfat_preictor/bodyfat_norm_params.npz"

class BodyFatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Body Fat Predictor")
        self.camera_lock = threading.Lock()

        # Initialize managers
        self.camera_manager = CameraManager()
        self.sensor_manager = SensorManager()
        self.aws_manager = AWSManager()
        self.calculator = BodyFatCalculator()
        self.data_manager = DataManager()
        
        self.names = self.data_manager.load_names()

        # UI Setup
        self._setup_ui()

        # State variables
        self.captured_images = {}
        self.image_paths = {"front": None, "side": None}
        self.impedance_ohms = None
        self.prediction_result = None
        self.prediction_id = None
        self.is_live_preview = True

        # Temp directory for images
        self.temp_dir = "/tmp/bodyfat_predictor"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

        self.update_preview()

    def _setup_ui(self):
        """Setup all UI components."""
        tk.Label(self.root, text="Name:").grid(row=0, column=0, sticky="e")
        tk.Label(self.root, text="Height (cm):").grid(row=1, column=0, sticky="e")
        tk.Label(self.root, text="Weight (kg):").grid(row=2, column=0, sticky="e")
        tk.Label(self.root, text="Age:").grid(row=3, column=0, sticky="e")
        tk.Label(self.root, text="Sex:").grid(row=4, column=0, sticky="e")

        self.name_var = tk.StringVar()
        self.name_dropdown = ttk.Combobox(self.root, textvariable=self.name_var, values=self.names)
        self.name_dropdown.grid(row=0, column=1, sticky="w")
        self.name_dropdown.set("Select a name")

        self.height_entry = tk.Entry(self.root)
        self.weight_entry = tk.Entry(self.root)
        self.age_entry = tk.Entry(self.root)
        self.height_entry.grid(row=1, column=1, sticky="w")
        self.weight_entry.grid(row=2, column=1, sticky="w")
        self.age_entry.grid(row=3, column=1, sticky="w")

        self.sex_var = tk.StringVar(value="Male")
        sex_frame = tk.Frame(self.root)
        sex_frame.grid(row=4, column=1, sticky="w")
        tk.Radiobutton(sex_frame, text="Male", variable=self.sex_var, value="Male").pack(side="left")
        tk.Radiobutton(sex_frame, text="Female", variable=self.sex_var, value="Female").pack(side="left")

        self.preview_label = tk.Label(self.root, text="Camera Loading...", width=40, height=15, bg="black", fg="white")
        self.preview_label.grid(row=5, column=0, columnspan=2, pady=10)

        self.capture_front_btn = tk.Button(self.root, text="Capture Front View", command=lambda: self.capture("front"))
        self.capture_front_btn.grid(row=6, column=0, columnspan=2, pady=5)
        
        self.capture_side_btn = tk.Button(self.root, text="Capture Side View", command=lambda: self.capture("side"))
        self.capture_side_btn.grid(row=7, column=0, columnspan=2, pady=5)

        self.imp_btn = tk.Button(self.root, text="Measure Impedance (Hold Clips)", command=self.measure_impedance)
        self.imp_btn.grid(row=8, column=0, columnspan=2, pady=5)

        self.predict_btn = tk.Button(self.root, text="Run Hybrid Prediction", command=self.run_prediction)
        self.predict_btn.grid(row=9, column=0, columnspan=2, pady=10)

        self.save_btn = tk.Button(self.root, text="Save to AWS Dashboard", command=self.save_to_aws, state="disabled")
        self.save_btn.grid(row=10, column=0, columnspan=2, pady=5)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.grid(row=11, column=0, columnspan=2)

    def update_preview(self):
        if self.is_live_preview and self.camera_manager.is_initialized:
            try:
                image_array = self.camera_manager.capture_array()
                pil_image = Image.fromarray(image_array)
                img_tk = ImageTk.PhotoImage(pil_image.resize((320, 240)))
                
                self.preview_label.configure(image=img_tk, width=320, height=240)
                self.preview_label.image = img_tk
            except Exception:
                pass
        
        self.root.after(100, self.update_preview)

    def capture(self, view):
        threading.Thread(target=self._delayed_capture, args=(view,), daemon=True).start()

    def _delayed_capture(self, view):
        with self.camera_lock:
            self.result_label.config(text=f"{view.capitalize()} capture in 3 seconds... Step back!")
            time.sleep(5)
            try:
                if not self.camera_manager.is_initialized:
                    raise RuntimeError("Camera not initialized")
                
                self.is_live_preview = False
                
                image_array = self.camera_manager.capture_array()
                pil_image = Image.fromarray(image_array)
                self.captured_images[view] = pil_image
                
                image_path = os.path.join(self.temp_dir, f"{view}_{uuid.uuid4()}.jpg")
                pil_image.save(image_path)
                self.image_paths[view] = image_path
                
                img_tk = ImageTk.PhotoImage(pil_image.resize((320, 240)))
                self.preview_label.configure(image=img_tk, width=320, height=240)
                self.preview_label.image = img_tk
                self.result_label.config(text=f"{view.capitalize()} image captured! (Resuming live feed...)")
                
                time.sleep(2)
                self.is_live_preview = True
                self.result_label.config(text="Camera live. Ready for next step.")

            except Exception as e:
                self.result_label.config(text=f"Capture failed: {str(e)}")
                self.is_live_preview = True

    def measure_impedance(self):
        if not self.sensor_manager.is_ready:
            messagebox.showerror("Error", "Sensor not detected!")
            return

        self.result_label.config(text="Measuring... Hold still...")
        self.root.update()
        
        try:
            impedance = self.sensor_manager.get_impedance(num_readings=10, delay=0.2)
            if impedance:
                self.impedance_ohms = impedance
                self.result_label.config(text=f"Impedance Measured: {int(impedance)} Ohms")
                self.imp_btn.config(bg="#aaffaa")
            else:
                self.result_label.config(text="Measurement Failed. Check clips.")
        except Exception as e:
            self.result_label.config(text=f"Measurement error: {str(e)}")

    def run_prediction(self):
        name = self.name_var.get().strip()
        if not name or name == "Select a name":
            messagebox.showwarning("Input Error", "Please enter or select a name.")
            return

        try:
            h = float(self.height_entry.get())
            w = float(self.weight_entry.get())
            age = float(self.age_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Height, weight, and age must be numbers.")
            return

        if "front" not in self.captured_images or "side" not in self.captured_images:
            messagebox.showwarning("Missing Images", "Capture both front and side images first.")
            return
            
        if not self.impedance_ohms:
            messagebox.showwarning("Missing Data", "Please measure impedance first.")
            return

        self.prediction_id = str(uuid.uuid4())
        try:
            self.is_live_preview = False 
            self.result_label.config(text="Processing AI Math...")
            self.root.update()

            # Get AI prediction
            ai_bf, cat = predict_body_fat(
                self.image_paths["front"],
                self.image_paths["side"],
                h, w, MODEL_PATH, NORM_PATH
            )

            # Get BIA prediction (Now catching 3 variables!)
            sex = self.sex_var.get()
            ffm, fat_mass, bia_bf = self.calculator.calculate_bia_body_fat(h, w, age, self.impedance_ohms, sex)
            
            # Combine predictions
            final_bf = self.calculator.hybrid_prediction(ai_bf, bia_bf)
            final_cat = self.calculator.get_category(final_bf, sex)

            # Save 8 variables to state
            self.prediction_result = (final_bf, final_cat, name, h, w, sex, ffm, fat_mass) 

            result_text = (
                f" AI Vision: {ai_bf:.1f}%  |   Bio-Sensor: {bia_bf:.1f}%\n"
                f" Hybrid Body Fat for {name}: {final_bf:.1f}% ({final_cat})"
            )
            self.result_label.config(text=result_text)
            self.save_btn.config(state="normal")
            
            # Clean up temp images
            for path in self.image_paths.values():
                if path and os.path.exists(path):
                    os.remove(path)
            self.image_paths = {"front": None, "side": None} 
            
            self.is_live_preview = True

        except Exception as e:
            self.result_label.config(text=f"Prediction failed: {str(e)}")
            self.save_btn.config(state="disabled")
            self.is_live_preview = True

    def save_to_aws(self):
        if not self.prediction_result:
            return

        # Unpack all 8 variables
        final_bf, category, name, height, weight, sex, ffm, fat_mass = self.prediction_result

        try:
            # Upload images to S3
            front_key, side_key = self.aws_manager.upload_images(
                self.prediction_id,
                self.captured_images["front"],
                self.captured_images["side"]
            )

            # Save prediction to DynamoDB (Passing ffm and fat_mass)
            self.aws_manager.save_prediction(
                self.prediction_id,
                name, sex, height, weight,
                final_bf, ffm, fat_mass, category,
                front_key, side_key
            )

            # Update saved names
            self.names = self.data_manager.save_name(name, self.names)
            self.name_dropdown['values'] = self.names

            self.captured_images.clear()
            self.result_label.config(text="Data and Images saved to AWS!")
            self.save_btn.config(state="disabled")
        except Exception as e:
            self.result_label.config(text=f"Failed to save to AWS: {str(e)}")

    def close_app(self):
        self.is_live_preview = False
        self.camera_manager.shutdown()
        for path in self.image_paths.values():
            if path and os.path.exists(path):
                os.remove(path)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BodyFatApp(root)
    root.mainloop()