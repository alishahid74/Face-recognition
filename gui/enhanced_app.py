import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

"""
Enhanced GUI Application with Advanced Features
Includes: Analytics dashboard, theme toggle, plugin support, batch processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import os
import threading
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classification import ClassificationModelFactory
from models.detection import DetectionModelFactory
from models.face_recognition import FaceModelFactory
from models.gesture_recognition import GestureModelFactory, MEDIAPIPE_AVAILABLE
from utils.image_utils import ImagePreprocessor
from utils.video_processing import AdvancedVideoProcessor, ObjectTracker
from config import MODELS_CONFIG
from plugins.plugin_system import PluginManager


class EnhancedAIVisionGUI:
    """Enhanced GUI Application with Advanced Features"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vision Framework Pro - Advanced Deep Learning")
        self.root.geometry("1600x1000")
        
        # Theme
        self.dark_mode = True
        self.themes = {
            'dark': {
                'bg': '#2b2b2b',
                'fg': '#ffffff',
                'button': '#4a9eff',
                'success': '#00cc66',
                'warning': '#ff9900',
                'error': '#cc0000'
            },
            'light': {
                'bg': '#f0f0f0',
                'fg': '#000000',
                'button': '#2196F3',
                'success': '#4CAF50',
                'warning': '#FF9800',
                'error': '#F44336'
            }
        }
        
        # Initialize variables
        self.current_image_path = None
        self.current_model = None
        self.current_task = None
        self.preprocessor = ImagePreprocessor()
        self.video_running = False
        self.video_processor = None
        
        # Plugin manager
        self.plugin_manager = PluginManager()
        
        # Analytics data
        self.inference_times = []
        self.class_distribution = defaultdict(int)
        self.fps_history = []
        
        # Setup UI
        self.apply_theme()
        self.create_menu()
        self.create_widgets()
        
    def apply_theme(self):
        """Apply current theme"""
        theme = self.themes['dark' if self.dark_mode else 'light']
        self.root.configure(bg=theme['bg'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=theme['bg'])
        style.configure('TLabel', background=theme['bg'], foreground=theme['fg'], font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground=theme['button'])
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground=theme['fg'])
        style.configure('TNotebook', background=theme['bg'])
        style.configure('TNotebook.Tab', padding=[20, 10])
        
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        messagebox.showinfo("Theme Changed", 
                           f"Theme switched to {'Dark' if self.dark_mode else 'Light'} mode")
        
    def create_menu(self):
        """Create enhanced menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Video", command=self.load_video)
        file_menu.add_command(label="Batch Process", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme, accelerator="Ctrl+T")
        view_menu.add_command(label="Show Analytics", command=self.show_analytics)
        view_menu.add_command(label="Performance Monitor", command=self.show_performance)
        
        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Plugin Manager", command=self.open_plugin_manager)
        models_menu.add_command(label="Model Comparison", command=self.compare_models)
        models_menu.add_command(label="Clear Cache", command=self.clear_model_cache)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Face Database", command=self.open_face_database)
        tools_menu.add_command(label="Privacy Mode", command=self.toggle_privacy_mode)
        tools_menu.add_command(label="Start API Server", command=self.start_api_server)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Check Updates", command=self.check_updates)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-t>', lambda e: self.toggle_theme())
        
    def create_widgets(self):
        """Create enhanced main widgets"""
        # Main container with notebook
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title with status
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="🚀 AI Vision Framework Pro", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(title_frame, text="Ready", font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Inference
        self.inference_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.inference_tab, text="Inference")
        self.create_inference_tab(self.inference_tab)
        
        # Tab 2: Analytics
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text="Analytics")
        self.create_analytics_tab(self.analytics_tab)
        
        # Tab 3: Batch Processing
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.create_batch_tab(self.batch_tab)
        
        # Tab 4: Plugins
        self.plugins_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plugins_tab, text="Plugins")
        self.create_plugins_tab(self.plugins_tab)
        
    def create_inference_tab(self, parent):
        """Create inference tab (similar to original)"""
        # Left panel - Controls
        control_frame = ttk.Frame(parent, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
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
        
        # Model selection with search
        model_label = ttk.Label(control_frame, text="Select Model:", style='Header.TLabel')
        model_label.grid(row=6, column=0, sticky=tk.W, pady=(20, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                       state='readonly', width=30)
        self.model_combo.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Model info
        self.model_info_text = scrolledtext.ScrolledText(control_frame, height=6, width=35,
                                                         font=('Courier', 9), 
                                                         bg='#1e1e1e' if self.dark_mode else '#ffffff',
                                                         fg='#00ff00' if self.dark_mode else '#000000',
                                                         wrap=tk.WORD)
        self.model_info_text.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Enhanced buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)
        
        theme = self.themes['dark' if self.dark_mode else 'light']
        
        self.load_btn = tk.Button(button_frame, text="📁 Load Image", command=self.load_image,
                                  bg=theme['button'], fg='white', font=('Arial', 10, 'bold'),
                                  relief='flat', padx=20, pady=10)
        self.load_btn.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.run_btn = tk.Button(button_frame, text="▶️ Run Inference", command=self.run_inference,
                                bg=theme['success'], fg='white', font=('Arial', 10, 'bold'),
                                relief='flat', padx=20, pady=10, state='disabled')
        self.run_btn.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.webcam_btn = tk.Button(button_frame, text="📹 Start Webcam", command=self.toggle_webcam,
                                   bg=theme['warning'], fg='white', font=('Arial', 10, 'bold'),
                                   relief='flat', padx=20, pady=10)
        self.webcam_btn.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Advanced settings
        settings_label = ttk.Label(control_frame, text="Advanced Settings:", style='Header.TLabel')
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
        
        # Tracking toggle
        self.tracking_var = tk.BooleanVar(value=True)
        tracking_check = ttk.Checkbutton(control_frame, text="Enable Object Tracking", 
                                        variable=self.tracking_var)
        tracking_check.grid(row=12, column=0, sticky=tk.W, pady=5)
        
        # Logging toggle
        self.logging_var = tk.BooleanVar(value=False)
        logging_check = ttk.Checkbutton(control_frame, text="Enable Frame Logging", 
                                       variable=self.logging_var)
        logging_check.grid(row=13, column=0, sticky=tk.W, pady=5)
        
        # Right panel - Display
        display_frame = ttk.Frame(parent, padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Display label
        display_label = ttk.Label(display_frame, text="Output Display", style='Header.TLabel')
        display_label.grid(row=0, column=0, pady=(0, 10))
        
        # Image canvas with drag-and-drop hint
        canvas_frame = ttk.Frame(display_frame, relief='solid', borderwidth=2)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='#1e1e1e', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add drag-and-drop hint text
        self.image_canvas.create_text(
            400, 300,
            text="Drag & Drop Image Here\nor\nClick 'Load Image' Button",
            font=('Arial', 16),
            fill='#666666',
            tags='hint'
        )
        
        # Results text with enhanced formatting
        results_label = ttk.Label(display_frame, text="Results", style='Header.TLabel')
        results_label.grid(row=2, column=0, pady=(20, 10))
        
        self.results_text = scrolledtext.ScrolledText(display_frame, height=12, 
                                                     font=('Courier', 10),
                                                     bg='#1e1e1e' if self.dark_mode else '#ffffff',
                                                     fg='#ffffff' if self.dark_mode else '#000000',
                                                     wrap=tk.WORD)
        self.results_text.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Initialize models list
        self.on_task_change()
    
    def create_analytics_tab(self, parent):
        """Create analytics dashboard"""
        # Title
        title = ttk.Label(parent, text="Analytics Dashboard", style='Header.TLabel')
        title.pack(pady=10)
        
        # Create matplotlib figure
        self.analytics_figure = Figure(figsize=(12, 8), facecolor='#2b2b2b' if self.dark_mode else 'white')
        
        # Create canvas
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_figure, parent)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Refresh button
        refresh_btn = tk.Button(parent, text="🔄 Refresh Analytics", command=self.update_analytics,
                               bg=self.themes['dark' if self.dark_mode else 'light']['button'],
                               fg='white', font=('Arial', 10, 'bold'), relief='flat', padx=20, pady=10)
        refresh_btn.pack(pady=10)
        
        self.update_analytics()
    
    def create_batch_tab(self, parent):
        """Create batch processing tab"""
        title = ttk.Label(parent, text="Batch Processing", style='Header.TLabel')
        title.pack(pady=10)
        
        # Batch file list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Files to Process:").pack(anchor=tk.W)
        
        self.batch_listbox = tk.Listbox(list_frame, height=15)
        self.batch_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Batch controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        theme = self.themes['dark' if self.dark_mode else 'light']
        
        add_files_btn = tk.Button(control_frame, text="➕ Add Files", command=self.add_batch_files,
                                 bg=theme['button'], fg='white', font=('Arial', 10, 'bold'),
                                 relief='flat', padx=20, pady=10)
        add_files_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(control_frame, text="🗑️ Clear List", command=self.clear_batch_list,
                             bg=theme['error'], fg='white', font=('Arial', 10, 'bold'),
                             relief='flat', padx=20, pady=10)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        process_btn = tk.Button(control_frame, text="⚡ Process Batch", command=self.process_batch,
                               bg=theme['success'], fg='white', font=('Arial', 10, 'bold'),
                               relief='flat', padx=20, pady=10)
        process_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.batch_progress = ttk.Progressbar(parent, mode='determinate')
        self.batch_progress.pack(fill=tk.X, padx=10, pady=10)
        
        self.batch_status_label = ttk.Label(parent, text="Ready for batch processing")
        self.batch_status_label.pack(pady=5)
    
    def create_plugins_tab(self, parent):
        """Create plugins management tab"""
        title = ttk.Label(parent, text="Plugin Manager", style='Header.TLabel')
        title.pack(pady=10)
        
        # Plugin list
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Available Plugins:").pack(anchor=tk.W)
        
        self.plugin_listbox = tk.Listbox(list_frame, height=15)
        self.plugin_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Load plugins
        self.refresh_plugin_list()
        
        # Plugin controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        theme = self.themes['dark' if self.dark_mode else 'light']
        
        refresh_btn = tk.Button(control_frame, text="🔄 Refresh", command=self.refresh_plugin_list,
                               bg=theme['button'], fg='white', font=('Arial', 10, 'bold'),
                               relief='flat', padx=20, pady=10)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        create_btn = tk.Button(control_frame, text="➕ Create Plugin", command=self.create_new_plugin,
                              bg=theme['success'], fg='white', font=('Arial', 10, 'bold'),
                              relief='flat', padx=20, pady=10)
        create_btn.pack(side=tk.LEFT, padx=5)
    
    # Continue with remaining methods... (keeping original functionality plus new features)
    
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
    
    # Additional methods for new features
    def show_analytics(self):
        """Show analytics window"""
        self.notebook.select(self.analytics_tab)
        self.update_analytics()
    
    def update_analytics(self):
        """Update analytics dashboard"""
        self.analytics_figure.clear()
        
        # Create subplots
        gs = self.analytics_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Inference time plot
        ax1 = self.analytics_figure.add_subplot(gs[0, 0])
        if self.inference_times:
            ax1.plot(self.inference_times, color='#00ff00')
            ax1.set_title('Inference Time History', color='white' if self.dark_mode else 'black')
            ax1.set_xlabel('Frame', color='white' if self.dark_mode else 'black')
            ax1.set_ylabel('Time (s)', color='white' if self.dark_mode else 'black')
            ax1.tick_params(colors='white' if self.dark_mode else 'black')
        
        # Class distribution
        ax2 = self.analytics_figure.add_subplot(gs[0, 1])
        if self.class_distribution:
            classes = list(self.class_distribution.keys())
            counts = list(self.class_distribution.values())
            ax2.barh(classes, counts, color='#4a9eff')
            ax2.set_title('Class Distribution', color='white' if self.dark_mode else 'black')
            ax2.set_xlabel('Count', color='white' if self.dark_mode else 'black')
            ax2.tick_params(colors='white' if self.dark_mode else 'black')
        
        # FPS history
        ax3 = self.analytics_figure.add_subplot(gs[1, :])
        if self.fps_history:
            ax3.plot(self.fps_history, color='#ff9900')
            ax3.set_title('FPS History', color='white' if self.dark_mode else 'black')
            ax3.set_xlabel('Frame', color='white' if self.dark_mode else 'black')
            ax3.set_ylabel('FPS', color='white' if self.dark_mode else 'black')
            ax3.tick_params(colors='white' if self.dark_mode else 'black')
        
        self.analytics_canvas.draw()
    
    def refresh_plugin_list(self):
        """Refresh plugin list"""
        self.plugin_listbox.delete(0, tk.END)
        
        plugins = self.plugin_manager.list_plugins()
        configs = self.plugin_manager.list_configs()
        
        for plugin in plugins:
            self.plugin_listbox.insert(tk.END, f"📦 {plugin}")
        
        for config in configs:
            self.plugin_listbox.insert(tk.END, f"⚙️  {config}")
    
    def create_new_plugin(self):
        """Create new plugin template"""
        name = tk.simpledialog.askstring("New Plugin", "Enter plugin name:")
        if name:
            self.plugin_manager.create_plugin_template(name)
            self.plugin_manager.create_config_template(name)
            messagebox.showinfo("Success", f"Plugin template created: {name}")
            self.refresh_plugin_list()
    
    # ... (rest of the methods from original GUI)
    
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
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
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
            
            # Remove hint text
            self.image_canvas.delete('hint')
            
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
            self.image_canvas.image = photo
            
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
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_inference_thread)
        thread.daemon = True
        thread.start()
    
    def _run_inference_thread(self):
        """Run inference in separate thread"""
        try:
            start_time = time.time()
            
            self.run_btn.config(state='disabled', text="Processing...")
            self.status_label.config(text="Running inference...")
            self.log_result(f"\n{'='*50}\nRunning inference...\n")
            
            task = self.current_task
            
            if task == 'image_classification':
                self._run_classification()
            elif task == 'object_detection':
                self._run_detection()
            elif task == 'face_recognition':
                self._run_face_recognition()
            elif task == 'gesture_recognition':
                self._run_gesture_recognition()
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            self.status_label.config(text=f"Complete ({inference_time:.2f}s)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
            self.log_result(f"ERROR: {str(e)}")
            self.status_label.config(text="Error occurred")
        finally:
            self.run_btn.config(state='normal', text="▶️ Run Inference")
    
    def _run_classification(self):
        """Run image classification"""
        model_name = self.model_var.get()
        
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = ClassificationModelFactory.create_model(model_name)
        
        config = MODELS_CONFIG['image_classification'][model_name]
        framework = config['framework']
        input_size = config['input_size']
        
        img_tensor = self.preprocessor.preprocess_for_classification(
            self.current_image_path, input_size, framework
        )
        
        results = self.current_model.predict(img_tensor, top_k=5)
        
        self.log_result("\nTop 5 Predictions:")
        for i, (label, confidence) in enumerate(results, 1):
            self.log_result(f"{i}. {label}: {confidence*100:.2f}%")
            self.class_distribution[label] += 1
    
    def _run_detection(self):
        """Run object detection"""
        model_name = self.model_var.get()
        
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = DetectionModelFactory.create_model(model_name)
        
        confidence = self.confidence_var.get()
        results = self.current_model.detect(self.current_image_path, confidence=confidence)
        
        self.display_image(image_array=results['image'])
        
        self.log_result(f"\nDetected {results['count']} objects:")
        for i, (label, score) in enumerate(zip(results['labels'], results['scores']), 1):
            self.log_result(f"{i}. {label}: {score*100:.2f}%")
            self.class_distribution[label] += 1
    
    def _run_face_recognition(self):
        """Run face recognition"""
        model_name = self.model_var.get()
        
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = FaceModelFactory.create_model(model_name)
        
        if model_name == 'MTCNN':
            results = self.current_model.detect_faces(self.current_image_path)
        else:
            results = self.current_model.recognize_faces(self.current_image_path)
        
        self.display_image(image_array=results['image'])
        self.log_result(f"\nDetected {results['count']} faces")
    
    def _run_gesture_recognition(self):
        """Run gesture recognition"""
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe is not installed")
            return
        
        model_name = self.model_var.get()
        
        if self.current_model is None or self.current_model.model_name != model_name:
            self.log_result("Loading model...")
            self.current_model = GestureModelFactory.create_model(model_name)
        
        results = self.current_model.detect_hands(self.current_image_path)
        
        self.display_image(image_array=results['image'])
        
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
        self.webcam_btn.config(text="⏸️ Stop Webcam")
        
        thread = threading.Thread(target=self._webcam_thread)
        thread.daemon = True
        thread.start()
    
    def stop_webcam(self):
        """Stop webcam processing"""
        self.video_running = False
        self.webcam_btn.config(text="📹 Start Webcam")
    
    def _webcam_thread(self):
        """Webcam processing thread"""
        # Placeholder - implement with AdvancedVideoProcessor
        messagebox.showinfo("Info", "Webcam feature - press 'q' to quit")
        self.stop_webcam()
    
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
AI Vision Framework Pro v2.0

Advanced deep learning framework with:
• 15+ Pre-trained Models
• Real-time Object Tracking
• Analytics Dashboard
• Plugin System
• REST API Server
• Batch Processing

Frameworks: PyTorch, TensorFlow, Ultralytics, MediaPipe
        """
        messagebox.showinfo("About", about_text)
    
    # Placeholder methods for new features
    def load_video(self): pass
    def batch_process(self): pass
    def export_results(self): pass
    def show_performance(self): pass
    def open_plugin_manager(self): pass
    def compare_models(self): pass
    def clear_model_cache(self): pass
    def open_face_database(self): pass
    def toggle_privacy_mode(self): pass
    def start_api_server(self): pass
    def show_documentation(self): pass
    def check_updates(self): pass
    def add_batch_files(self): pass
    def clear_batch_list(self): pass
    def process_batch(self): pass


def main():
    """Main entry point"""
    root = tk.Tk()
    app = EnhancedAIVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
