import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import time

from database import init_db
from gesture_processor import GestureProcessor
from model_manager import ModelManager
from training_manager import TrainingManager
from recognition_manager import RecognitionManager
from gui_components import SettingsDialog, AccessLoggerDialog, RegistrationControlPanel

class GestureRecognitionApp:
    def __init__(self, mode="training"):
        self.mode = mode
        self.root = tk.Tk()
        if mode == "training":
            self.root.title("Hand Gesture Recognition - Training Mode")
        else:
            self.root.title("Hand Gesture Recognition - Recognition Mode")
        
        # Initialize database connection
        self.conn = init_db()
        
        # Initialize components
        self.gesture_processor = GestureProcessor()
        self.model_manager = ModelManager()
        
        if mode == "training":
            self.training_manager = TrainingManager(self.conn, self.gesture_processor, self.model_manager)
            self.control_panel = RegistrationControlPanel(self.root, self.training_manager)
        else:
            self.recognition_manager = RecognitionManager(self.conn, self.gesture_processor, self.model_manager)
        
        # Camera
        self.cap = None
        
        # GUI setup
        self.setup_gui()
        self.setup_camera()

    def setup_gui(self):
        """Setup the GUI based on mode"""
        # Camera canvas
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
    
        # Status label
        if self.mode == "training":
            self.status_label = tk.Label(self.root, text="Ready - Place hand in blue detection box", font=("Arial", 14))
        else:
            self.status_label = tk.Label(self.root, text="Ready for authentication", font=("Arial", 14))
        self.status_label.pack(pady=10)
    
        if self.mode == "training":
            # Training mode buttons
            button_frame = tk.Frame(self.root)
            button_frame.pack(pady=10)
            
            self.register_button = tk.Button(button_frame, text="Register", command=self.register_user, 
                                             font=("Arial", 12), bg="#4CAF50", fg="white", padx=20)
            self.register_button.pack(side=tk.LEFT, padx=10)
        
            self.recognize_button = tk.Button(button_frame, text="Recognize", command=self.recognize_gesture,
                                              font=("Arial", 12), bg="#2196F3", fg="white", padx=20)
            self.recognize_button.pack(side=tk.LEFT, padx=10)
            
            # Settings button
            self.settings_button = tk.Button(button_frame, text="Settings", command=self.open_settings,
                                             font=("Arial", 12), bg="#FF9800", fg="white", padx=20)
            self.settings_button.pack(side=tk.LEFT, padx=10)

            # Access logger button
            self.logger_button = tk.Button(button_frame, text="Access Logger", command=self.open_access_logger,
                                       font=("Arial", 12), bg="#9C27B0", fg="white", padx=20)
            self.logger_button.pack(side=tk.LEFT, padx=10)
            
            # Add the control panel at the bottom
            control_panel = self.control_panel.create()
            control_panel.pack(side=tk.BOTTOM, fill=tk.X)
            
        else:
            # Recognition mode - minimal interface
            info_frame = tk.Frame(self.root)
            info_frame.pack(pady=10)
            
            info_label = tk.Label(info_frame, text="🔐 Continuous Authentication System", 
                                 font=("Arial", 16, "bold"), fg="blue")
            info_label.pack()
            
            instruction_label = tk.Label(info_frame, text="Place your hand in the detection box to authenticate", 
                                        font=("Arial", 12))
            instruction_label.pack(pady=5)

    def setup_camera(self):
        """Initialize camera and start video feed"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            self.root.quit()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_video_feed()
        
        # Update controls periodically (only for training mode)
        if self.mode == "training":
            self.update_controls_periodically()

    def update_video_feed(self):
        """Update video feed with gesture processing"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Process frame for hand detection
            hand_results, landmarks = self.gesture_processor.process_frame(frame)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw detection box
            frame_with_box = self.gesture_processor.draw_detection_box(frame_rgb.copy(), self.mode)
            
            # Draw hand landmarks
            frame_with_landmarks = self.gesture_processor.draw_hand_landmarks(
                frame_with_box, hand_results.multi_hand_landmarks
            )
            
            # Add overlay information
            self.add_overlay_info(frame_with_landmarks, landmarks, hand_results.multi_hand_landmarks)
            
            # Process based on mode
            if self.mode == "training":
                self.process_training_mode(landmarks)
            else:
                self.process_recognition_mode(landmarks)

            # Display frame
            img = Image.fromarray(frame_with_landmarks)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.root.after(10, self.update_video_feed)

    def add_overlay_info(self, frame, landmarks, all_hand_landmarks):
        """Add overlay information to the frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Hand detection status
        hands_in_box = 0
        if all_hand_landmarks:
            hands_in_box = len(self.gesture_processor.filter_hands_in_box(all_hand_landmarks))
        
        if hands_in_box > 0:
            cv2.putText(frame, f"Hands detected: {hands_in_box}", (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No hands in detection box", (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def process_training_mode(self, landmarks):
        """Process training mode specific logic"""
        if self.training_manager.recording:
            # Update countdown
            self.training_manager.update_countdown()
            
            # Process registration
            status_message = self.training_manager.process_registration(landmarks)
            if status_message:
                self.status_label.config(text=status_message)

    def process_recognition_mode(self, landmarks):
        """Process recognition mode specific logic"""
        # Process continuous recognition
        self.recognition_manager.process_continuous_recognition(landmarks)
        
        # Update status message
        recognition_status = self.recognition_manager.get_recognition_status()
        if recognition_status:
            self.status_label.config(text=recognition_status)
        else:
            current_message = self.recognition_manager.get_current_message()
            self.status_label.config(text=current_message)

    def update_controls_periodically(self):
        """Update training controls every 500ms"""
        if self.mode == "training":
            self.control_panel.update()
            self.root.after(500, self.update_controls_periodically)

    # Training mode methods
    def register_user(self):
        """Start user registration"""
        if not self.training_manager.recording:
            username = simpledialog.askstring("Input", "Enter username:")
            if username:
                status_message = self.training_manager.start_registration(username)
                self.status_label.config(text=status_message)

    def recognize_gesture(self):
        """Manual gesture recognition for training mode"""
        if not self.training_manager.recording:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count < 2:
                messagebox.showinfo("Recognition", "Please register at least two different users before recognition.")
                return
                
            if not self.model_manager.model:
                messagebox.showerror("Error", "No trained model available. Please register users first.")
                return
                
            messagebox.showinfo("Recognition", "Perform the gesture inside the blue box to authenticate")
            # This would trigger a one-time recognition similar to the original implementation
            self.manual_recognition_mode = True
            self.manual_recognition_timer = None

    def open_settings(self):
        """Open settings dialog"""
        settings_dialog = SettingsDialog(self.root, self.model_manager, self.gesture_processor)
        settings_dialog.show()

    def open_access_logger(self):
        """Open access logger dialog"""
        logger_dialog = AccessLoggerDialog(self.root, self.conn)
        logger_dialog.show()

    def run(self):
        """Start the application"""
        self.root.mainloop()
        
        # Cleanup
        if self.cap:
            self.cap.release()
        if self.conn:
            self.conn.close()