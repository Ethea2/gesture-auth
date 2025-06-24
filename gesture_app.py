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
            # Add recognition manager for training mode recognition
            self.recognition_manager = RecognitionManager(self.conn, self.gesture_processor, self.model_manager)
        else:
            self.recognition_manager = RecognitionManager(self.conn, self.gesture_processor, self.model_manager)
        
        # Camera
        self.cap = None
        
        # Manual recognition state for training mode
        self.manual_recognition_mode = False
        self.manual_recognition_timer = None
        self.manual_recognition_start_time = None
        self.manual_recognition_duration = 3.0  # 3 seconds to perform gesture
        
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
        
        # Add manual recognition countdown overlay
        if self.manual_recognition_mode and self.manual_recognition_start_time:
            elapsed = time.time() - self.manual_recognition_start_time
            remaining = max(0, self.manual_recognition_duration - elapsed)
            
            if remaining > 0:
                countdown_text = f"Recognition in progress... {remaining:.1f}s"
                cv2.putText(frame, countdown_text, (10, 60), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Processing recognition...", (10, 60), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def process_training_mode(self, landmarks):
        """Process training mode specific logic"""
        if self.training_manager.recording:
            # Update countdown
            self.training_manager.update_countdown()
            
            # Process registration
            status_message = self.training_manager.process_registration(landmarks)
            if status_message:
                self.status_label.config(text=status_message)
        
        # Handle manual recognition mode
        elif self.manual_recognition_mode:
            self.process_manual_recognition(landmarks)

    def process_manual_recognition(self, landmarks):
        """Process manual recognition in training mode"""
        current_time = time.time()
        
        # Initialize timer if not started
        if not self.manual_recognition_start_time:
            self.manual_recognition_start_time = current_time
            self.status_label.config(text=f"Place hand in box and hold for {self.manual_recognition_duration} seconds...")
            print("Manual recognition started")
            return
        
        # Check if recognition period is over
        elapsed = current_time - self.manual_recognition_start_time
        
        if elapsed < self.manual_recognition_duration:
            # Still in recognition period
            remaining = self.manual_recognition_duration - elapsed
            if landmarks:  # Hand detected
                self.status_label.config(text=f"Hold steady... {remaining:.1f}s remaining")
                print(f"Hand detected, {remaining:.1f}s remaining")
            else:
                self.status_label.config(text=f"Place hand in box... {remaining:.1f}s remaining")
                print(f"No hand detected, {remaining:.1f}s remaining")
        else:
            # Recognition period completed
            print(f"Recognition period completed. Landmarks available: {landmarks is not None and len(landmarks) > 0}")
            if landmarks and len(landmarks) > 0:
                # Perform recognition
                print("Performing recognition...")
                self.perform_manual_recognition(landmarks)
            else:
                # No hand detected during recognition period
                print("No hand detected during recognition period")
                messagebox.showwarning("Recognition Failed", "No hand detected during recognition period. Please try again.")
                self.reset_manual_recognition()

    def perform_manual_recognition(self, landmarks):
        """Perform the actual recognition process"""
        try:
            # Extract features from landmarks
            features = self.gesture_processor.extract_features(landmarks)
            
            # Use model manager directly for prediction
            result = self.model_manager.predict(features)
            
            if result is not None:
                # Check if result is a tuple (user_id, confidence) or just user_id
                if isinstance(result, tuple):
                    user_id, confidence = result
                else:
                    user_id = result
                    confidence = None
                
                # Get user info from database
                cursor = self.conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                db_result = cursor.fetchone()
                
                if db_result:
                    username = db_result[0]
                    
                    # Check confidence if available
                    if confidence is not None:
                        if self.model_manager.is_confident_prediction(confidence):
                            message = f"✅ Recognized: {username} (Confidence: {confidence:.3f})"
                            messagebox.showinfo("Recognition Successful", message)
                            self.status_label.config(text=f"Recognized: {username}")
                            
                            # Log the recognition attempt
                            cursor.execute(
                                "INSERT INTO access_logs (user_id, username, confidence) VALUES (?, ?, ?)",
                                (user_id, username, confidence)
                            )
                            self.conn.commit()
                        else:
                            message = f"⚠️ Low confidence recognition: {username} (Confidence: {confidence:.3f})"
                            messagebox.showwarning("Low Confidence", message)
                            self.status_label.config(text="Low confidence - try again")
                    else:
                        # No confidence score available
                        message = f"✅ Recognized: {username}"
                        messagebox.showinfo("Recognition Successful", message)
                        self.status_label.config(text=f"Recognized: {username}")
                        
                        # Log without confidence
                        cursor.execute(
                            "INSERT INTO access_logs (user_id, username, confidence) VALUES (?, ?, ?)",
                            (user_id, username, 1.0)  # Default confidence
                        )
                        self.conn.commit()
                else:
                    messagebox.showwarning("Recognition Failed", "User not found in database.")
                    self.status_label.config(text="Recognition failed - User not found")
            else:
                messagebox.showwarning("Recognition Failed", "Could not recognize gesture. Try again or register more samples.")
                self.status_label.config(text="Recognition failed - Unknown gesture")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"An error occurred during recognition: {str(e)}"
            messagebox.showerror("Recognition Error", error_msg)
            self.status_label.config(text="Recognition error occurred")
            print(f"Recognition error details: {error_msg}")
        
        finally:
            # Reset manual recognition mode
            self.reset_manual_recognition()

    def reset_manual_recognition(self):
        """Reset manual recognition state"""
        self.manual_recognition_mode = False
        self.manual_recognition_start_time = None
        if self.manual_recognition_timer:
            self.root.after_cancel(self.manual_recognition_timer)
            self.manual_recognition_timer = None
        
        # Reset status after a delay
        self.root.after(2000, lambda: self.status_label.config(text="Ready - Place hand in blue detection box"))

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
        if not self.training_manager.recording and not self.manual_recognition_mode:
            username = simpledialog.askstring("Input", "Enter username:")
            if username:
                status_message = self.training_manager.start_registration(username)
                self.status_label.config(text=status_message)

    def recognize_gesture(self):
        """Manual gesture recognition for training mode"""
        if not self.training_manager.recording and not self.manual_recognition_mode:
            print("Starting recognition process...")
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"Users in database: {user_count}")
            
            if user_count < 1:
                messagebox.showinfo("Recognition", "Please register at least one user before recognition.")
                return
                
            if not self.model_manager.model:
                print("No model found, attempting to load/create...")
                # Try to load or retrain the model
                cursor.execute("SELECT user_id FROM users LIMIT 1")
                if cursor.fetchone():
                    messagebox.showinfo("Model Loading", "Loading/training model from existing data...")
                    try:
                        # This should trigger model loading/training in your model manager
                        if hasattr(self.model_manager, 'load_or_create_model'):
                            self.model_manager.load_or_create_model(self.conn)
                        elif hasattr(self.model_manager, 'train_model'):
                            self.model_manager.train_model(self.conn)
                        elif hasattr(self.model_manager, 'load_model'):
                            self.model_manager.load_model()
                        
                        if not self.model_manager.model:
                            messagebox.showerror("Error", "Could not load or create model. Please register more users.")
                            return
                        else:
                            print("Model loaded successfully")
                    except Exception as e:
                        print(f"Error loading model: {str(e)}")
                        messagebox.showerror("Error", f"Error loading model: {str(e)}")
                        return
                else:
                    messagebox.showerror("Error", "No trained model available. Please register users first.")
                    return
            else:
                print("Model is available")
            
            # Start manual recognition mode
            self.manual_recognition_mode = True
            self.manual_recognition_start_time = None
            print(f"Manual recognition mode activated. Duration: {self.manual_recognition_duration}s")
            messagebox.showinfo("Recognition", f"Recognition mode activated. Place your hand in the blue box and hold steady for {self.manual_recognition_duration} seconds.")
        else:
            if self.training_manager.recording:
                print("Cannot start recognition while recording")
                messagebox.showwarning("Busy", "Cannot start recognition while recording. Please wait for registration to complete.")
            elif self.manual_recognition_mode:
                print("Recognition already in progress")
                messagebox.showwarning("Busy", "Recognition already in progress. Please wait.")

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