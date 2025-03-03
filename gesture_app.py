import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
from database import init_db
import time

class GestureRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Gesture Recognition")
        self.samples_count = 0
        self.recording = False
        self.current_samples = []
        self.conn = init_db()
        self.model = None
        self.scaler = StandardScaler()
        self.last_sample_time = 0
        self.sample_delay = 1
        self.remaining_time = 0
        self.recognition_mode = False
        self.recognition_timer = None
        self.countdown_seconds = 5  # 5 seconds per repetition
        self.min_samples_per_user = 15  # Minimum samples required per user
        self.confidence_threshold = 0.7  # Adjust based on testing

        try:
            self.model = joblib.load('gesture_model.joblib')
            self.scaler = joblib.load('feature_scaler.joblib')
        except:
            print("No existing model found, will create a new one")

        self.setup_gui()
        self.setup_camera()

    def extract_features(self, landmarks):
        """Extract more robust features from landmarks"""
        # Basic position features
        basic_features = []
        for landmark in landmarks:
            basic_features.extend([landmark.x, landmark.y, landmark.z])
        
        # Add relative position features (distances between key landmarks)
        relative_features = []
        if len(landmarks) >= 21:  # Full hand landmarks
            # Compute distances between fingertips and wrist
            wrist = landmarks[0]
            fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            
            for tip in fingertips:
                # Distance from wrist to fingertip
                dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                relative_features.append(dist)
                
            # Distances between adjacent fingertips
            for i in range(len(fingertips)-1):
                tip1 = fingertips[i]
                tip2 = fingertips[i+1]
                dist = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2 + (tip1.z - tip2.z)**2)
                relative_features.append(dist)
            
            # Compute angles between fingers
            for i in range(1, 5):  # For each finger
                base = landmarks[i*4+1]  # MCP joint
                mid = landmarks[i*4+2]   # PIP joint
                tip = landmarks[i*4+3]   # DIP joint
                
                # Calculate vectors
                v1 = np.array([mid.x - base.x, mid.y - base.y, mid.z - base.z])
                v2 = np.array([tip.x - mid.x, tip.y - mid.y, tip.z - mid.z])
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    relative_features.append(angle)
        
        # Combine all features
        all_features = np.concatenate([np.array(basic_features, dtype=np.float32), 
                                      np.array(relative_features, dtype=np.float32)])
        
        return all_features

    def combine_landmarks(self, hand_landmarks):
        combined_landmarks = []
        if hand_landmarks:
            for hand in hand_landmarks:
                combined_landmarks.extend(hand.landmark)
        return combined_landmarks

    def register_user(self):
        if not self.recording:
            self.username = simpledialog.askstring("Input", "Enter username:")
            if self.username:
                self.recording = True
                self.samples_count = 0 
                self.current_samples = []
                self.repetition_timer = time.time()
                self.countdown_seconds = 5  
                self.status_label.config(text=f"Repetition 1/5: Starting in {self.countdown_seconds} seconds...")
                self.update_countdown()

    def update_countdown(self):
        if self.recording:
            current_time = time.time()
            elapsed_time = current_time - self.repetition_timer
            self.countdown_seconds = max(0, 5 - int(elapsed_time))
            
            if self.countdown_seconds > 0:
                self.status_label.config(text=f"Repetition {self.samples_count+1}/5: {self.countdown_seconds} seconds remaining...")
                self.root.after(1000, self.update_countdown)  

    def process_registration(self, landmarks):
        current_time = time.time()
        
        if self.recording and self.samples_count < 5:
            elapsed_time = current_time - self.repetition_timer
            
            # Collect multiple samples during each repetition window
            if elapsed_time < 5 and time.time() - self.last_sample_time > 0.2:  # Collect a sample every 0.2 seconds
                features = self.extract_features(landmarks)
                self.current_samples.append(features)
                self.last_sample_time = time.time()
                # Display sample count
                samples_in_current_repetition = len([s for s in self.current_samples if len(s) > 0])
                self.status_label.config(text=f"Repetition {self.samples_count+1}/5: {self.countdown_seconds}s - Samples: {samples_in_current_repetition}")

            elif elapsed_time >= 5:
                self.samples_count += 1
                self.repetition_timer = time.time()
                
                if self.samples_count < 5:
                    self.countdown_seconds = 5  # Reset countdown to 5 seconds
                    self.status_label.config(text=f"Repetition {self.samples_count+1}/5: {self.countdown_seconds} seconds remaining...")
                    self.update_countdown()
                    # Ask the user to vary their gesture slightly
                    if self.samples_count == 1:
                        messagebox.showinfo("Vary Gesture", "Please vary your hand position slightly for each repetition to improve recognition.")
                else:
                    if len(self.current_samples) < self.min_samples_per_user:
                        messagebox.showwarning("Warning", f"Only {len(self.current_samples)} samples collected. Registration might not be reliable.")
                    
                    self.save_user_data()
                    self.recording = False
                    self.status_label.config(text="Registration Complete!")
                    messagebox.showinfo("Registration", f"Registration complete with {len(self.current_samples)} samples!")

    def save_user_data(self):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO users (username) VALUES (?)", (self.username,))
        user_id = cursor.lastrowid

        for sample in self.current_samples:
            cursor.execute(
                "INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", 
                (user_id, sample.tobytes())
            )

        self.conn.commit()
        self.retrain_model()

    def retrain_model(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT users.user_id, gesture_samples.feature_data 
            FROM users 
            JOIN gesture_samples ON users.user_id = gesture_samples.user_id
            """
        )
        data = cursor.fetchall()

        if not data:
            print("No training data available")
            return
        
        users_samples = {}
        for user_id, feature_data in data:
            if user_id not in users_samples:
                users_samples[user_id] = 0
            users_samples[user_id] += 1
        
        # Check if we have enough samples per user
        min_samples = min(users_samples.values()) if users_samples else 0
        if min_samples < self.min_samples_per_user:
            print(f"Warning: Some users have only {min_samples} samples, which may be insufficient")
        
        X = []
        y = []
        
        # Process each sample
        for sample in data:
            user_id, feature_data = sample
            try:
                features = np.frombuffer(feature_data, dtype=np.float32)
                if len(features) > 0:  # Make sure we have valid features
                    X.append(features)
                    y.append(user_id)
            except Exception as e:
                print(f"Error processing sample: {e}")
        
        if len(X) == 0 or len(set(y)) < 1:
            print("Not enough valid samples to train model")
            return
            
        # Find the most common feature length
        feature_lengths = [len(features) for features in X]
        common_length = max(set(feature_lengths), key=feature_lengths.count)
        
        # Pad or truncate features to common length
        X_processed = []
        for features in X:
            if len(features) < common_length:
                features = np.pad(features, (0, common_length - len(features)), mode='constant')
            elif len(features) > common_length:
                features = features[:common_length]
            X_processed.append(features)
        
        X_processed = np.array(X_processed)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Print diagnostics
        print(f"Training model with {len(X_scaled)} samples from {len(set(y))} users")
        print(f"Feature length: {common_length}")
        
        # Evaluate model with cross-validation 
        model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
        if len(set(y)) > 1 and len(X_scaled) >= 10:  # Only do cross-validation if we have enough data
            try:
                scores = cross_val_score(model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                print(f"Cross-validation accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
                
                if np.mean(scores) < 0.7:
                    print("Warning: Model accuracy is low. Consider collecting more varied samples.")
                    messagebox.showwarning("Warning", "Recognition accuracy may be low. Try adding more varied samples for each gesture.")
            except Exception as e:
                print(f"Cross-validation error: {e}")
        
        # Create and train the final model
        self.model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
        self.model.fit(X_scaled, y)
        
        # Save the model and scaler
        joblib.dump(self.model, 'gesture_model.joblib')
        joblib.dump(self.scaler, 'feature_scaler.joblib')
        print("Model training complete")

    def draw_hand_landmarks(self, frame, hand_landmarks):
        if hand_landmarks:
            for hand_lms in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                wrist = hand_lms.landmark[0]
                wrist_point = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                
                middle_finger_mcp = hand_lms.landmark[9]  
                direction_x = wrist.x - middle_finger_mcp.x
                direction_y = wrist.y - middle_finger_mcp.y
                
                forearm_length = 0.2  
                forearm_end_x = int((wrist.x + direction_x * forearm_length) * frame.shape[1])
                forearm_end_y = int((wrist.y + direction_y * forearm_length) * frame.shape[0])
                
                cv2.line(frame, wrist_point, (forearm_end_x, forearm_end_y), (0, 255, 0), 5)
                
        return frame

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            hand_results = self.hands.process(frame_rgb)
            
            frame_with_landmarks = self.draw_hand_landmarks(
                frame_rgb.copy(), 
                hand_results.multi_hand_landmarks
            )
            
            landmarks = self.combine_landmarks(hand_results.multi_hand_landmarks)

            font = cv2.FONT_HERSHEY_SIMPLEX
            
            if self.recording:
                countdown_text = f"Time: {self.countdown_seconds}s"
                cv2.putText(frame_with_landmarks, countdown_text, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_with_landmarks, f"Repetition: {self.samples_count+1}/5", (50, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                if landmarks:
                    self.process_registration(landmarks)
                
            elif self.recognition_mode:
                cv2.putText(frame_with_landmarks, "Recognition Mode", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if not self.recognition_timer:
                    self.recognition_timer = time.time()
                elif time.time() - self.recognition_timer >= 2:  # Reduced from 3 to 2 seconds
                    if landmarks:
                        self.process_recognition(landmarks)
                    self.recognition_timer = None
                else:
                    # Show countdown during recognition
                    remaining = 2 - (time.time() - self.recognition_timer)
                    cv2.putText(frame_with_landmarks, f"Analyzing in {remaining:.1f}s", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            img = Image.fromarray(frame_with_landmarks)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.root.after(10, self.update_video_feed)

    def process_recognition(self, landmarks):
        if self.recognition_mode and self.model is not None:
            try:
                # Get features from current hand position
                features = self.extract_features(landmarks)
                
                # Process features to match expected format
                if hasattr(self.model, 'n_features_in_'):
                    expected_length = self.model.n_features_in_
                elif hasattr(self.model, 'support_vectors_') and len(self.model.support_vectors_) > 0:
                    expected_length = len(self.model.support_vectors_[0])
                else:
                    print("Cannot determine expected feature length from model")
                    messagebox.showerror("Error", "Model not properly trained.")
                    self.recognition_mode = False
                    return
                    
                # Pad or truncate features to match expected length
                if len(features) < expected_length:
                    features = np.pad(features, (0, expected_length - len(features)), mode='constant')
                elif len(features) > expected_length:
                    features = features[:expected_length]
                
                # Scale features
                features_scaled = self.scaler.transform([features])
                    
                # Make prediction with probability
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    max_prob = np.max(probabilities)
                    user_id = int(self.model.classes_[np.argmax(probabilities)])
                    confidence = max_prob
                else:
                    # Fallback if probabilities not available
                    user_id = int(self.model.predict(features_scaled)[0])
                    confidence = 1.0
                    try:
                        decision_values = self.model.decision_function(features_scaled)
                        confidence = np.max(np.abs(decision_values)) / 10  # Scale to 0-1 range approximately
                    except:
                        pass
                
                print(f"Recognition confidence: {confidence:.4f}, threshold: {self.confidence_threshold}")
                    
                if confidence >= self.confidence_threshold:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                    result = cursor.fetchone()
                    if result:
                        messagebox.showinfo("Recognition Result", f"Recognized user: {result[0]}\nConfidence: {confidence:.2f}")
                    else:
                        messagebox.showerror("Error", f"User not found (ID: {user_id}).")
                else:
                    messagebox.showinfo("Recognition Result", f"Gesture not recognized with sufficient confidence. (Score: {confidence:.2f})")
                    
            except Exception as e:
                import traceback
                traceback.print_exc()  # Print the full traceback for debugging
                print(f"Recognition error: {str(e)}")
                messagebox.showerror("Error", f"Could not recognize gesture. Error: {str(e)}")
            finally:
                self.recognition_mode = False

    def setup_gui(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
    
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 14))
        self.status_label.pack(pady=10)
    
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.register_button = tk.Button(button_frame, text="Register", command=self.register_user, 
                                         font=("Arial", 12), bg="#4CAF50", fg="white", padx=20)
        self.register_button.pack(side=tk.LEFT, padx=10)
    
        self.recognize_button = tk.Button(button_frame, text="Recognize", command=self.recognize_gesture,
                                          font=("Arial", 12), bg="#2196F3", fg="white", padx=20)
        self.recognize_button.pack(side=tk.LEFT, padx=10)
        
        # Add a settings button
        self.settings_button = tk.Button(button_frame, text="Settings", command=self.open_settings,
                                         font=("Arial", 12), bg="#FF9800", fg="white", padx=20)
        self.settings_button.pack(side=tk.LEFT, padx=10)
                
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,  
            min_tracking_confidence=0.6   
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            self.root.quit()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_video_feed()

    def recognize_gesture(self):
        if not self.recording:
            if not self.model:
                messagebox.showerror("Error", "No trained model available. Please register users first.")
                return
                
            messagebox.showinfo("Recognition", "Perform the gesture to authenticate")
            self.recognition_mode = True
            self.recognition_timer = None
            
    def open_settings(self):
        """Open settings dialog to adjust parameters"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Recognition Settings")
        settings_window.geometry("400x300")
        
        # Confidence threshold setting
        tk.Label(settings_window, text="Confidence Threshold (0.0-1.0):", font=("Arial", 12)).pack(pady=(20, 5))
        confidence_slider = tk.Scale(settings_window, from_=0.0, to=1.0, resolution=0.05, 
                                    orient=tk.HORIZONTAL, length=300)
        confidence_slider.set(self.confidence_threshold)
        confidence_slider.pack()
        
        # Minimum samples setting
        tk.Label(settings_window, text="Minimum Samples per User:", font=("Arial", 12)).pack(pady=(20, 5))
        samples_slider = tk.Scale(settings_window, from_=5, to=50, resolution=5, 
                                orient=tk.HORIZONTAL, length=300)
        samples_slider.set(self.min_samples_per_user)
        samples_slider.pack()
        
        # Save button
        def save_settings():
            self.confidence_threshold = confidence_slider.get()
            self.min_samples_per_user = samples_slider.get()
            messagebox.showinfo("Settings", "Settings saved successfully")
            settings_window.destroy()
            
        tk.Button(settings_window, text="Save", command=save_settings, 
                font=("Arial", 12), bg="#4CAF50", fg="white", padx=20).pack(pady=20)