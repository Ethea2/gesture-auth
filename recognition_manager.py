import time
from tkinter import messagebox

class RecognitionManager:
    def __init__(self, conn, gesture_processor, model_manager):
        self.conn = conn
        self.gesture_processor = gesture_processor
        self.model_manager = model_manager
        
        # Recognition state
        self.recognition_mode = False
        self.recognition_timer = None
        self.last_recognition_time = 0
        self.recognition_cooldown = 2.0  # Seconds between recognitions
        self.last_message_time = 0
        self.message_display_duration = 3.0  # Display result for 3 seconds
        self.current_message = ""

    def process_continuous_recognition(self, landmarks):
        """Process continuous recognition for recognition mode"""
        current_time = time.time()
        
        if landmarks:  # Hand detected in box
            if not self.recognition_mode:
                # Start recognition if not already running and cooldown has passed
                if current_time - self.last_recognition_time > self.recognition_cooldown:
                    self.recognition_mode = True
                    self.recognition_timer = current_time
                    self.last_recognition_time = current_time
        else:  # No hand in box
            if self.recognition_mode:
                # Cancel recognition smoothly if hand leaves box
                self.recognition_mode = False
                self.recognition_timer = None
        
        # Process recognition if active and time elapsed
        if self.recognition_mode and self.recognition_timer:
            elapsed = current_time - self.recognition_timer
            if elapsed >= 5:  # 5 seconds completed
                if landmarks:  # Hand still present
                    self.perform_recognition(landmarks)
                self.recognition_mode = False
                self.recognition_timer = None

    def perform_recognition(self, landmarks):
        """Perform the actual recognition and log results"""
        if self.model_manager.model is None:
            print("No model available for recognition")
            return
            
        try:
            # Get features from current hand position
            features = self.gesture_processor.extract_features(landmarks)
            
            # Make prediction
            user_id, confidence = self.model_manager.predict(features)
            
            if user_id is None:
                print("Recognition failed")
                return
            
            print(f"Recognition confidence: {confidence:.4f}, threshold: {self.model_manager.confidence_threshold}")
            
            current_time = time.time()
            self.last_message_time = current_time
                
            if self.model_manager.is_confident_prediction(confidence):
                cursor = self.conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if result:
                    username = result[0]
                    # Log the successful access
                    cursor.execute(
                        "INSERT INTO access_logs (user_id, username, confidence) VALUES (?, ?, ?)",
                        (user_id, username, confidence)
                    )
                    self.conn.commit()
                    print(f"Welcome back {username}!")
                    
                    # Show alert box for successful recognition
                    messagebox.showinfo("Access Granted", f"Welcome back {username}!")
                    self.current_message = f"Welcome back {username}!"
                else:
                    print("User not found in database")
                    # Show alert box for unrecognized user
                    messagebox.showwarning("Access Denied", "Unrecognized user tried to access.")
                    self.current_message = "Unrecognized user tried to access."
            else:
                print("Unrecognized user tried to access")
                # Show alert box for unrecognized user
                messagebox.showwarning("Access Denied", "Unrecognized user tried to access.")
                self.current_message = "Unrecognized user tried to access."
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Recognition error: {str(e)}")
            # Show alert box for recognition error
            messagebox.showerror("Recognition Error", "An error occurred during recognition.")
            self.current_message = "Recognition error occurred."

    def get_recognition_status(self):
        """Get current recognition status for display"""
        if self.recognition_mode and self.recognition_timer:
            elapsed = time.time() - self.recognition_timer
            remaining = 5 - elapsed
            return f"RECOGNIZING... Time remaining: {remaining:.1f}s"
        return ""

    def get_current_message(self):
        """Get current message if within display duration"""
        current_time = time.time()
        if hasattr(self, 'last_message_time') and current_time - self.last_message_time <= self.message_display_duration:
            return self.current_message
        return "Ready for authentication"

    def is_recognizing(self):
        """Check if currently in recognition mode"""
        return self.recognition_mode

    def get_remaining_time(self):
        """Get remaining recognition time"""
        if self.recognition_mode and self.recognition_timer:
            elapsed = time.time() - self.recognition_timer
            return max(0, 5 - elapsed)
        return 0