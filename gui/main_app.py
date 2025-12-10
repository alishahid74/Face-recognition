import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

"""
Main GUI Application for AI Vision Framework
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import os
import threading
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification import ClassificationModelFactory
from detection import DetectionModelFactory
from face_recognition import FaceModelFactory
from gesture_recognition import GestureModelFactory, MEDIAPIPE_AVAILABLE
from image_utils import ImagePreprocessor
from config import MODELS_CONFIG



class AIVisionGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vision Framework - Deep Learning Models")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize variables
        self.current_image_path = None
        self.current_model = None
        self.current_task = None
        self.preprocessor = ImagePreprocessor()
        self.video_running = False
        
        # Setup UI
        self.setup_styles()
        self.create_menu()
        self.create_widgets()
        
    def setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#4a9eff')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#ffffff')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', padding=[20, 10])
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Open Video", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        
    def create_widgets(self):
        """Create main widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 AI Vision Framework", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Right panel - Display
        self.create_display_panel(main_frame)
        
    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.Frame(parent, padding="10", relief='solid', borderwidth=1)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Task selection
        task_label = ttk.Label(control_frame, text="Select Task:", style='Header.TLabel')
        task_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        self.task_var = tk.StringVar(value="image_classification")
        tasks = [
            ("Image Classification", "image_classification"),
            ("Object Detection", "object_detection"),
            ("Face Recognition", "face_recognition"),
            ("Gesture Recognition", "gesture_recognition")
        ]
        
        for i, (text, value) in enumerate(tasks):
            rb = ttk.Radiobutton(control_frame, text=text, variable=self.task_var, 
                                value=value, command=self.on_task_change)
            rb.grid(row=i+1, column=0, sticky=tk.W, pady=2)
        
        # Model selection
        model_label = ttk.Label(control_frame, text="Select Model:", style='Header.TLabel')
        model_label.grid(row=6, column=0, sticky=tk.W, pady=(20, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                       state='readonly', width=30)
        self.model_combo.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Model info
        self.model_info_text = scrolledtext.ScrolledText(control_frame, height=6, width=35,
                                                         font=('Courier', 9), bg='#1e1e1e', 
                                                         fg='#00ff00', wrap=tk.WORD)
        self.model_info_text.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.load_btn = tk.Button(button_frame, text="📁 Load Image", command=self.load_image,
                                  bg='#4a9eff', fg='white', font=('Arial', 10, 'bold'),
                                  relief='flat', padx=20, pady=10)
        self.load_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.run_btn = tk.Button(button_frame, text="▶️ Run Inference", command=self.run_inference,
                                bg='#00cc66', fg='white', font=('Arial', 10, 'bold'),
                                relief='flat', padx=20, pady=10, state='disabled')
        self.run_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.webcam_btn = tk.Button(button_frame, text="📹 Start Webcam", command=self.toggle_webcam,
                                   bg='#ff9900', fg='white', font=('Arial', 10, 'bold'),
                                   relief='flat', padx=20, pady=10)
        self.webcam_btn.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Settings frame
        settings_label = ttk.Label(control_frame, text="Settings:", style='Header.TLabel')
        settings_label.grid(row=10, column=0, sticky=tk.W, pady=(20, 10))
        
        # Confidence threshold
        conf_frame = ttk.Frame(control_frame)
        conf_frame.grid(row=11, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(conf_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                         variable=self.confidence_var, orient='horizontal')
        self.confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.50")
        self.conf_label.grid(row=0, column=2, sticky=tk.W)
        self.confidence_var.trace('w', self.update_confidence_label)
        
        conf_frame.columnconfigure(1, weight=1)
        
        # Initialize models list
        self.on_task_change()
        
    def create_display_panel(self, parent):
        """Create display panel"""
        display_frame = ttk.Frame(parent, padding="10", relief='solid', borderwidth=1)
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Display label
        display_label = ttk.Label(display_frame, text="Output Display", style='Header.TLabel')
        display_label.grid(row=0, column=0, pady=(0, 10))
        
        # Image canvas
        canvas_frame = ttk.Frame(display_frame, relief='solid', borderwidth=2)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Results text
        results_label = ttk.Label(display_frame, text="Results", style='Header.TLabel')
        results_label.grid(row=2, column=0, pady=(20, 10))
        
        self.results_text = scrolledtext.ScrolledText(display_frame, height=10, 
                                                     font=('Courier', 10), bg='#1e1e1e', 
                                                     fg='#ffffff', wrap=tk.WORD)
        self.results_text.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
    def on_task_change(self):
        """Handle task selection change"""
        task = self.task_var.get()
        self.current_task = task
        
        # Update model list
        if task == 'image_classification':
            models = ClassificationModelFactory.get_available_models()
        elif task == 'object_detection':
            models = DetectionModelFactory.get_available_models()
        elif task == 'face_recognition':
            models = FaceModelFactory.get_available_models()
        elif task == 'gesture_recognition':
            models = GestureModelFactory.get_available_models() if MEDIAPIPE_AVAILABLE else []
        else:
            models = []
        
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(0)
            self.on_model_change()
        
    def on_model_change(self, event=None):
        """Handle model selection change"""
        model_name = self.model_var.get()
        if not model_name:
            return
        
        # Display model info
        task = self.current_task
        if task in MODELS_CONFIG:
            model_info = MODELS_CONFIG[task].get(model_name, {})
            info_text = f"Model: {model_name}\n"
            info_text += f"Framework: {model_info.get('framework', 'N/A')}\n"
            info_text += f"Input Size: {model_info.get('input_size', 'N/A')}\n"
            info_text += f"\nDescription:\n{model_info.get('description', 'N/A')}"
            
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(1.0, info_text)
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.run_btn.config(state='normal')
            self.log_result(f"Loaded image: {os.path.basename(file_path)}")
        
    def display_image(self, image_path=None, image_array=None):
        """Display image on canvas"""
        try:
            if image_path:
                img = Image.open(image_path)
            elif image_array is not None:
                img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            else:
                return
            
            # Resize to fit canvas
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width//2, canvas_height//2, 
                                          image=photo, anchor=tk.CENTER)
            self.image_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def run_inference(self):
        """Run inference on loaded image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model")
            return
        
        # Run in separate thread to avoid GUI freezing
        thread = threading.Thread(target=self._run_inference_thread)
        thread.daemon = True
        thread.start()
    
    def _run_inference_thread(self):
        """Run inference in separate thread"""
        try:
            self.run_btn.config(state='disabled', text="Processing...")
            self.log_result(f"\n{'='*50}\nRunning inference...\n")
            
            task = self.current_task
            model_name = self.model_var.get()
            
            if task == 'image_classification':
                self._run_classification()
            elif task == 'object_detection':
                self._run_detection()
            elif task == 'face_recognition':
                self._run_face_recognition()
            elif task == 'gesture_recognition':
                self._run_gesture_recognition()
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
            self.log_result(f"ERROR: {str(e)}")
        finally:
            self.run_btn.config(state='normal', text="▶️ Run Inference")
    
    def _run_classification(self):
        """Run image classification"""
        model_name = self.model_var.get()
        
        # Load model if not already loaded
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = ClassificationModelFactory.create_model(model_name)
        
        # Preprocess image
        config = MODELS_CONFIG['image_classification'][model_name]
        framework = config['framework']
        input_size = config['input_size']
        
        img_tensor = self.preprocessor.preprocess_for_classification(
            self.current_image_path, input_size, framework
        )
        
        # Run prediction
        results = self.current_model.predict(img_tensor, top_k=5)
        
        # Display results
        self.log_result("\nTop 5 Predictions:")
        for i, (label, confidence) in enumerate(results, 1):
            self.log_result(f"{i}. {label}: {confidence*100:.2f}%")
    
    def _run_detection(self):
        """Run object detection"""
        model_name = self.model_var.get()
        
        # Load model if not already loaded
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = DetectionModelFactory.create_model(model_name)
        
        # Run detection
        confidence = self.confidence_var.get()
        results = self.current_model.detect(self.current_image_path, confidence=confidence)
        
        # Display annotated image
        self.display_image(image_array=results['image'])
        
        # Display results
        self.log_result(f"\nDetected {results['count']} objects:")
        for i, (label, score) in enumerate(zip(results['labels'], results['scores']), 1):
            self.log_result(f"{i}. {label}: {score*100:.2f}%")
    
    def _run_face_recognition(self):
        """Run face recognition"""
        model_name = self.model_var.get()
        
        # Load model if not already loaded
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = FaceModelFactory.create_model(model_name)
        
        # Run detection
        if model_name == 'MTCNN':
            results = self.current_model.detect_faces(self.current_image_path)
        else:
            results = self.current_model.recognize_faces(self.current_image_path)
        
        # Display annotated image
        self.display_image(image_array=results['image'])
        
        # Display results
        self.log_result(f"\nDetected {results['count']} faces")
    
    def _run_gesture_recognition(self):
        """Run gesture recognition"""
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe is not installed")
            return
        
        model_name = self.model_var.get()
        
        # Load model if not already loaded
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = GestureModelFactory.create_model(model_name)
        
        # Run detection
        results = self.current_model.detect_hands(self.current_image_path)
        
        # Display annotated image
        self.display_image(image_array=results['image'])
        
        # Display results
        self.log_result(f"\nDetected {results['count']} hands:")
        for i, hand in enumerate(results['hands'], 1):
            self.log_result(f"{i}. {hand['handedness']} hand - Gesture: {hand['gesture']}")
    
    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.video_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Start webcam processing"""
        self.video_running = True
        self.webcam_btn.config(text="⏸️ Stop Webcam", bg='#cc0000')
        
        # Run in separate thread
        thread = threading.Thread(target=self._webcam_thread)
        thread.daemon = True
        thread.start()
    
    def stop_webcam(self):
        """Stop webcam processing"""
        self.video_running = False
        self.webcam_btn.config(text="📹 Start Webcam", bg='#ff9900')
    
    def _webcam_thread(self):
        """Webcam processing thread"""
        messagebox.showinfo("Info", "Webcam feature requires model loading. Press 'q' to quit.")
        # Implementation would go here
        self.stop_webcam()
    
    def load_video(self):
        """Load video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            messagebox.showinfo("Info", f"Video loaded: {os.path.basename(file_path)}")
    
    def log_result(self, text):
        """Log result to results text widget"""
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def update_confidence_label(self, *args):
        """Update confidence label"""
        value = self.confidence_var.get()
        self.conf_label.config(text=f"{value:.2f}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
AI Vision Framework v1.0

A comprehensive deep learning framework for:
• Image Classification
• Object Detection
• Face Recognition
• Gesture Recognition

Supported Models:
• ResNet, MobileNet, VGG, EfficientNet
• YOLOv8 (all variants)
• MTCNN, Face Recognition
• MediaPipe Hands

Frameworks: PyTorch, TensorFlow, Ultralytics
        """
        messagebox.showinfo("About", about_text)
    
    def show_documentation(self):
        """Show documentation"""
        doc_text = """
Quick Start Guide:

1. Select a task (Classification, Detection, etc.)
2. Choose a model from the dropdown
3. Load an image using 'Load Image' button
4. Click 'Run Inference' to process
5. View results in the output panel

Tips:
• Adjust confidence threshold for detection tasks
• Use webcam for real-time processing
• Supported formats: JPG, PNG, BMP
        """
        messagebox.showinfo("Documentation", doc_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AIVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
