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
        
        # Learning feedback variables
        self.current_recognition_features = None
        self.current_recognition_result = None
        
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
        
            self.recognize_button = tk.Button(button_frame, text="🧠 Learn Recognition", command=self.recognize_gesture,
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
                countdown_text = f"🧠 Learning Recognition... {remaining:.1f}s"
                cv2.putText(frame, countdown_text, (10, 60), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Processing learning recognition...", (10, 60), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

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
            self.status_label.config(text=f"🧠 Place hand in box and hold for {self.manual_recognition_duration} seconds...")
            print("Learning recognition started")
            return
        
        # Check if recognition period is over
        elapsed = current_time - self.manual_recognition_start_time
        
        if elapsed < self.manual_recognition_duration:
            # Still in recognition period
            remaining = self.manual_recognition_duration - elapsed
            if landmarks:  # Hand detected
                self.status_label.config(text=f"🧠 Hold steady... {remaining:.1f}s remaining")
                print(f"Hand detected, {remaining:.1f}s remaining")
            else:
                self.status_label.config(text=f"🧠 Place hand in box... {remaining:.1f}s remaining")
                print(f"No hand detected, {remaining:.1f}s remaining")
        else:
            # Recognition period completed
            print(f"Recognition period completed. Landmarks available: {landmarks is not None and len(landmarks) > 0}")
            if landmarks and len(landmarks) > 0:
                # Perform learning recognition
                print("Performing learning recognition...")
                self.perform_learning_recognition(landmarks)
            else:
                # No hand detected during recognition period
                print("No hand detected during recognition period")
                messagebox.showwarning("Recognition Failed", "No hand detected during recognition period. Please try again.")
                self.reset_manual_recognition()

    def perform_learning_recognition(self, landmarks):
        """Perform recognition with learning feedback"""
        try:
            # Extract features from landmarks
            features = self.gesture_processor.extract_features(landmarks)
            self.current_recognition_features = features  # Store for learning
            
            # Use model manager directly for prediction
            result = self.model_manager.predict(features)
            
            if result is not None:
                # Check if result is a tuple (user_id, confidence) or just user_id
                if isinstance(result, tuple):
                    user_id, confidence = result
                else:
                    user_id = result
                    confidence = None
                
                # Store current recognition result for learning
                self.current_recognition_result = (user_id, confidence)
                
                # Get user info from database
                cursor = self.conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                db_result = cursor.fetchone()
                
                if db_result:
                    username = db_result[0]
                    
                    # Show recognition result and ask for feedback
                    if confidence is not None:
                        if self.model_manager.is_confident_prediction(confidence):
                            message = f"✅ Recognized: {username}\nConfidence: {confidence:.3f}\n\nIs this recognition CORRECT?"
                            is_correct = messagebox.askyesno("🧠 Learning Recognition - Verify Result", message)
                        else:
                            message = f"⚠️ Low confidence: {username}\nConfidence: {confidence:.3f}\n\nIs this recognition CORRECT?"
                            is_correct = messagebox.askyesno("🧠 Low Confidence - Verify Result", message)
                    else:
                        # No confidence score available
                        message = f"✅ Recognized: {username}\n\nIs this recognition CORRECT?"
                        is_correct = messagebox.askyesno("🧠 Learning Recognition - Verify Result", message)
                    
                    # Handle feedback for learning
                    if is_correct:
                        # Correct recognition - add positive learning sample
                        self.add_learning_sample(user_id, username, True)
                        self.status_label.config(text=f"✅ Correct: {username}")
                        messagebox.showinfo("🧠 Learning", "Thank you! This helps improve the model accuracy.")
                        
                        # Log successful recognition
                        cursor.execute(
                            "INSERT INTO access_logs (user_id, username, confidence, notes) VALUES (?, ?, ?, ?)",
                            (user_id, username, confidence if confidence else 1.0, "Learning sample (positive)")
                        )
                        self.conn.commit()
                    else:
                        # Incorrect recognition - ask for correct identity or if unauthorized
                        self.status_label.config(text="❌ Incorrect recognition - Learning...")
                        self.ask_for_correct_identity_or_unauthorized()
                        
                else:
                    messagebox.showwarning("Recognition Failed", "User not found in database.")
                    self.status_label.config(text="Recognition failed - User not found")
            else:
                # No recognition made - ask if this should be unauthorized or if they want to identify
                result = messagebox.askyesno("🧠 No Recognition", 
                                           "Could not recognize gesture.\n\nWould you like to help improve the system by identifying this gesture?")
                if result:
                    self.ask_for_correct_identity_or_unauthorized()
                else:
                    self.status_label.config(text="Recognition failed - Unknown gesture")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"An error occurred during learning recognition: {str(e)}"
            messagebox.showerror("Recognition Error", error_msg)
            self.status_label.config(text="Recognition error occurred")
            print(f"Recognition error details: {error_msg}")
        
        finally:
            # Reset manual recognition mode
            self.reset_manual_recognition()

    def ask_for_correct_identity_or_unauthorized(self):
        """Ask user for correct identity and add learning sample"""
        # Get all usernames for selection
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id, username FROM users ORDER BY username")
        users = cursor.fetchall()
        
        if not users:
            messagebox.showwarning("No Users", "No registered users found.")
            return
        
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("🧠 Learning - Correct Identity")
        selection_window.geometry("400x550")
        selection_window.grab_set()  # Make it modal
        
        # Center the window
        selection_window.geometry("+%d+%d" % (
            (self.root.winfo_screenwidth() // 2) - 200,
            (self.root.winfo_screenheight() // 2) - 275
        ))
        
        tk.Label(selection_window, text="🧠 Who performed this gesture?", 
                font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Label(selection_window, text="This helps the system learn and improve!", 
                font=("Arial", 10), fg="gray").pack(pady=5)
        
        # Create listbox with usernames
        listbox_frame = tk.Frame(selection_window)
        listbox_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Add usernames to listbox
        for user_id, username in users:
            listbox.insert(tk.END, username)
        
        # Add separator
        listbox.insert(tk.END, "────────────────────────")
        
        # Add "Unauthorized Gesture" option
        listbox.insert(tk.END, "🚫 UNAUTHORIZED GESTURE")
        
        # Add "New User" option
        listbox.insert(tk.END, "--- Register New User ---")
        
        # Add explanation for unauthorized option
        explanation_frame = tk.Frame(selection_window)
        explanation_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(explanation_frame, text="💡 Select 'UNAUTHORIZED GESTURE' if:", 
                font=("Arial", 10, "bold"), fg="blue").pack(anchor="w")
        tk.Label(explanation_frame, text="   • You're testing with random gestures", 
                font=("Arial", 9), fg="gray").pack(anchor="w")
        tk.Label(explanation_frame, text="   • This gesture shouldn't match any user", 
                font=("Arial", 9), fg="gray").pack(anchor="w")
        tk.Label(explanation_frame, text="   • You want to train rejection of this gesture", 
                font=("Arial", 9), fg="gray").pack(anchor="w")
        
        def on_confirm():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                if index < len(users):
                    # Existing user selected
                    user_id, username = users[index]
                    self.add_learning_sample(user_id, username, False)  # False = corrected wrong prediction
                    messagebox.showinfo("🧠 Learning", f"Thank you! Added corrective learning sample for {username}.")
                    self.status_label.config(text=f"🔄 Learned from: {username}")
                elif index == len(users) + 1:  # Skip separator, unauthorized gesture option
                    # Unauthorized gesture selected
                    self.add_negative_learning_sample()
                    # Don't show additional message here, it's shown in add_negative_learning_sample
                elif index == len(users) + 2:  # Register new user option
                    # "Register New User" selected
                    messagebox.showinfo("New User", "Please use the Register button to add a new user first, then try learning recognition again.")
                
                selection_window.destroy()
        
        def on_cancel():
            selection_window.destroy()
        
        # Buttons
        button_frame = tk.Frame(selection_window)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="✅ Confirm", command=on_confirm, 
                 bg="#4CAF50", fg="white", padx=20, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="❌ Cancel", command=on_cancel, 
                 bg="#f44336", fg="white", padx=20, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    def add_learning_sample(self, correct_user_id, correct_username, is_positive):
        """Add the current recognition as a learning sample"""
        if self.current_recognition_features is None:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Add the features as a new sample for the correct user
            cursor.execute(
                "INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", 
                (correct_user_id, self.current_recognition_features.tobytes())
            )
            
            # Log the learning event
            sample_type = "positive" if is_positive else "corrective"
            cursor.execute(
                "INSERT INTO access_logs (user_id, username, confidence, notes) VALUES (?, ?, ?, ?)",
                (correct_user_id, correct_username, 0.0, f"Learning sample ({sample_type})")
            )
            
            self.conn.commit()
            
            # Retrain the model with new data
            print(f"Added {sample_type} learning sample for {correct_username}")
            messagebox.showinfo("🧠 Retraining", "Model is being retrained with new learning data...")
            
            # Use the model manager's retrain functionality if available
            if hasattr(self.model_manager, 'retrain_model'):
                self.model_manager.retrain_model(self.conn)
            elif hasattr(self.model_manager, 'train_model'):
                self.model_manager.train_model(self.conn)
            
            messagebox.showinfo("✅ Learning Complete", f"Model successfully learned from {correct_username}'s gesture!")
            
        except Exception as e:
            print(f"Error adding learning sample: {e}")
            messagebox.showerror("Learning Error", f"Could not add learning sample: {str(e)}")

    def add_negative_learning_sample(self):
        """Add the current recognition as a negative sample (unauthorized gesture)"""
        if self.current_recognition_features is None:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Create a special "negative" user entry if it doesn't exist
            cursor.execute("SELECT user_id FROM users WHERE username = 'UNAUTHORIZED_SAMPLES'")
            negative_user = cursor.fetchone()
            
            if not negative_user:
                # Create special user for unauthorized samples
                cursor.execute(
                    "INSERT INTO users (username) VALUES (?)", 
                    ("UNAUTHORIZED_SAMPLES",)
                )
                cursor.execute("SELECT last_insert_rowid()")
                negative_user_id = cursor.fetchone()[0]
                print("Created UNAUTHORIZED_SAMPLES user")
            else:
                negative_user_id = negative_user[0]
            
            # Ensure is_negative column exists
            try:
                cursor.execute("SELECT is_negative FROM gesture_samples LIMIT 1")
            except:
                # Column doesn't exist, add it
                cursor.execute("ALTER TABLE gesture_samples ADD COLUMN is_negative INTEGER DEFAULT 0")
                self.conn.commit()
                print("Added is_negative column to gesture_samples")
            
            # Add the features as a negative sample
            cursor.execute(
                "INSERT INTO gesture_samples (user_id, feature_data, is_negative) VALUES (?, ?, ?)", 
                (negative_user_id, self.current_recognition_features.tobytes(), 1)
            )
            
            # Log the negative learning event
            cursor.execute(
                "INSERT INTO access_logs (user_id, username, confidence, notes) VALUES (?, ?, ?, ?)",
                (negative_user_id, "UNAUTHORIZED", 0.0, "Learning sample (negative/unauthorized)")
            )
            
            self.conn.commit()
            print("Successfully added negative learning sample")
            
            # Update status immediately
            self.status_label.config(text="🚫 Learning unauthorized gesture...")
            
            # Retrain the model with new negative data
            messagebox.showinfo("🧠 Negative Learning", "Model is learning to reject this unauthorized gesture...")
            
            # Use the model manager's retrain functionality
            if hasattr(self.model_manager, 'retrain_with_negatives'):
                success = self.model_manager.retrain_with_negatives(self.conn)
            elif hasattr(self.model_manager, 'retrain_model'):
                success = self.model_manager.retrain_model(self.conn)
            elif hasattr(self.model_manager, 'train_model'):
                success = self.model_manager.train_model(self.conn)
            else:
                success = False
            
            if success:
                messagebox.showinfo("✅ Negative Learning Complete", "Model successfully learned to reject this unauthorized gesture!")
                self.status_label.config(text="🚫 Learned: Unauthorized gesture")
            else:
                messagebox.showwarning("⚠️ Learning Issue", "Negative sample added but model retraining failed. Please check your ModelManager.")
                self.status_label.config(text="🚫 Sample added, retraining failed")
            
        except Exception as e:
            print(f"Error adding negative learning sample: {e}")
            import traceback
            traceback.print_exc()
            
            messagebox.showerror("Negative Learning Error", f"Could not add negative learning sample: {str(e)}")
            self.status_label.config(text="❌ Negative learning failed")

    def reset_manual_recognition(self):
        """Reset manual recognition state"""
        self.manual_recognition_mode = False
        self.manual_recognition_start_time = None
        self.current_recognition_features = None
        self.current_recognition_result = None
        
        if self.manual_recognition_timer:
            self.root.after_cancel(self.manual_recognition_timer)
            self.manual_recognition_timer = None
        
        # Reset status after a delay
        self.root.after(3000, lambda: self.status_label.config(text="Ready - Place hand in blue detection box"))

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
        """Manual gesture recognition with learning for training mode"""
        if not self.training_manager.recording and not self.manual_recognition_mode:
            print("Starting learning recognition process...")
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"Users in database: {user_count}")
            
            if user_count < 1:
                messagebox.showinfo("Learning Recognition", "Please register at least one user before starting learning recognition.")
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
            
            # Show learning mode explanation
            message = ("🧠 LEARNING MODE RECOGNITION 🧠\n\n"
                      "This recognition will help improve the model!\n"
                      "• You'll verify if the recognition is correct\n"
                      "• Incorrect predictions help the system learn\n"
                      "• The model will retrain with new data\n\n"
                      "Ready to help the system learn?")
            
            if messagebox.askyesno("🧠 Learning Mode", message):
                # Start manual recognition mode
                self.manual_recognition_mode = True
                self.manual_recognition_start_time = None
                self.current_recognition_features = None
                self.current_recognition_result = None
                print(f"Learning recognition mode activated. Duration: {self.manual_recognition_duration}s")
        else:
            if self.training_manager.recording:
                print("Cannot start recognition while recording")
                messagebox.showwarning("Busy", "Cannot start learning recognition while recording. Please wait for registration to complete.")
            elif self.manual_recognition_mode:
                print("Recognition already in progress")
                messagebox.showwarning("Busy", "Learning recognition already in progress. Please wait.")

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