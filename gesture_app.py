import tkinter as tk
from tkinter import messagebox, simpledialog
import tkinter.ttk as ttk
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
from sklearn.ensemble import RandomForestClassifier

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
        self.access_logs = [] # to store the motherfucking access logs

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
            # Compute hand size for normalization
            wrist = landmarks[0]
            middle_tip = landmarks[12]
            hand_size = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2 + (middle_tip.z - wrist.z)**2)
            hand_size = max(hand_size, 0.001)  # Avoid division by zero
            
            # Compute distances between fingertips and wrist
            fingertips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            
            for tip in fingertips:
                # Distance from wrist to fingertip (normalized by hand size)
                dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                relative_features.append(dist / hand_size)  # Normalized distance
                
            # Distances between adjacent fingertips (normalized)
            for i in range(len(fingertips)-1):
                tip1 = fingertips[i]
                tip2 = fingertips[i+1]
                dist = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2 + (tip1.z - tip2.z)**2)
                relative_features.append(dist / hand_size)  # Normalized distance
            
            # Compute angles between fingers (these are naturally scale-invariant)
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
                else:
                    relative_features.append(0.0)
                    
            # Add angles between fingers (more view-invariant features)
            for i in range(1, 4):  # First three fingers
                # Vector for current finger
                base_current = landmarks[i*4+1]
                tip_current = landmarks[i*4+3]
                v_current = np.array([tip_current.x - base_current.x, 
                                    tip_current.y - base_current.y,
                                    tip_current.z - base_current.z])
                
                # Vector for next finger
                base_next = landmarks[(i+1)*4+1]
                tip_next = landmarks[(i+1)*4+3]
                v_next = np.array([tip_next.x - base_next.x,
                                tip_next.y - base_next.y,
                                tip_next.z - base_next.z])
                
                # Calculate angle between fingers
                v_current_norm = np.linalg.norm(v_current)
                v_next_norm = np.linalg.norm(v_next)
                
                if v_current_norm > 0 and v_next_norm > 0:
                    cos_angle = np.dot(v_current, v_next) / (v_current_norm * v_next_norm)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    relative_features.append(angle)
        
        # Combine all features
        all_features = np.concatenate([np.array(basic_features, dtype=np.float32), 
                                    np.array(relative_features, dtype=np.float32)])
        
        return all_features
    
    def augment_samples(self, features):
        """Create slightly modified versions of samples to improve robustness"""
        augmented_samples = []
        augmented_samples.append(features)  # Original sample
        
        # Add small random noise to features
        noise_scale = 0.01
        noisy = features + np.random.normal(0, noise_scale, len(features))
        augmented_samples.append(noisy)
        
        # Scale features slightly
        scale_factor = np.random.uniform(0.95, 1.05, len(features))
        scaled = features * scale_factor
        augmented_samples.append(scaled)
        
        # Add small rotational variation (for angle features)
        # Only modify the angle features which are in the latter part of the feature vector
        if len(features) > 63:  # Assuming the first 63 features are position-based
            rotated = features.copy()
            angle_indices = range(63, len(features))
            for idx in angle_indices:
                if idx < len(features):
                    rotated[idx] = features[idx] + np.random.normal(0, 0.05)
            augmented_samples.append(rotated)
        
        return augmented_samples

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
                # Show instructions before recording starts
                messagebox.showinfo("Registration Instructions", 
                                "For best recognition results:\n\n" +
                                "1. Vary your hand position slightly for each repetition\n" +
                                "2. Change the distance from the camera\n" +
                                "3. Rotate your hand subtly between repetitions\n" +
                                "4. Maintain the core gesture shape")
                
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
                
                # Add original sample
                self.current_samples.append(features)
                
                # Add augmented samples
                augmented = self.augment_samples(features)
                for aug_sample in augmented[1:]:  # Skip the first one as it's the original
                    self.current_samples.append(aug_sample)
                    
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
                        messagebox.showinfo("Vary Gesture", "Please vary your hand position, angle, and distance from camera for each repetition to improve recognition.")
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
        
        if len(X) == 0:
            print("Not enough valid samples to train model")
            return
            
        # Check if we have at least two different classes
        if len(set(y)) < 2:
            print("Need at least two different users to train the model")
            # Save the scaled features but skip training until we have 2+ users
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
            self.scaler.fit(X_processed)
            joblib.dump(self.scaler, 'feature_scaler.joblib')
            return
                
        # Continue with model training as before
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
        X_processed = np.nan_to_num(X_processed, nan=0.0)  # Replace NaN values with 0
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Print diagnostics
        print(f"Training model with {len(X_scaled)} samples from {len(set(y))} users")
        print(f"Feature length: {common_length}")
        
        # Evaluate model with cross-validation 
        
        if len(set(y)) > 1 and len(X_scaled) >= 10:
            try:
                # Use both SVC and RandomForest for comparison
                svc_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
                
                if np.isnan(X_scaled).any():
                    print("Warning: NaN values detected in scaled data. Using only RandomForest.")
                    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                    print(f"RandomForest cross-validation accuracy: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")
                    self.model = rf_model
                    model_accuracy = np.mean(rf_scores)
                else:
                    # Try both models if no NaN values
                    svc_scores = cross_val_score(svc_model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                
                print(f"SVC cross-validation accuracy: {np.mean(svc_scores):.2f} ± {np.std(svc_scores):.2f}")
                print(f"RandomForest cross-validation accuracy: {np.mean(rf_scores):.2f} ± {np.std(rf_scores):.2f}")
                
                # Choose the better model
                if np.mean(rf_scores) > np.mean(svc_scores):
                    self.model = rf_model
                    print("Using RandomForest classifier based on cross-validation")
                    model_accuracy = np.mean(rf_scores)
                else:
                    self.model = svc_model
                    print("Using SVC classifier based on cross-validation")
                    model_accuracy = np.mean(svc_scores)
                    
                # Set confidence threshold based on model accuracy
                self.confidence_threshold = min(0.9, max(0.6, 1.0 - 1.5 * (1.0 - model_accuracy)))
                print(f"Setting confidence threshold to {self.confidence_threshold:.2f}")
                
                if model_accuracy < 0.7:
                    print("Warning: Model accuracy is low. Consider collecting more varied samples.")
                    messagebox.showwarning("Warning", "Recognition accuracy may be low. Try adding more varied samples for each gesture.")
                    
            except Exception as e:
                print(f"Cross-validation error: {e}")
                # Fallback to RandomForest if cross-validation fails
                self.model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
        else:
            # If not enough data for cross-validation, use RandomForest as default
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
        
        # Train the model
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
        if self.recognition_mode:
            if self.model is None:
                messagebox.showinfo("Recognition", "Model not yet trained. Please register at least two users first.")
                self.recognition_mode = False
                return
                
            try:
                # Collect multiple samples for improved recognition
                collected_features = []
                
                # Get features from current hand position
                features = self.extract_features(landmarks)
                collected_features.append(features)
                
                # Process each feature sample
                predictions = []
                confidences = []
                
                for features in collected_features:
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
                    
                    predictions.append(user_id)
                    confidences.append(confidence)
                
                # Use the prediction with highest confidence
                best_idx = np.argmax(confidences)
                user_id = predictions[best_idx]
                confidence = confidences[best_idx]
                
                print(f"Recognition confidence: {confidence:.4f}, threshold: {self.confidence_threshold}")
                    
                if confidence >= self.confidence_threshold:
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
                        messagebox.showinfo("Recognition Result", f"Recognized user: {username}\nConfidence: {confidence:.2f}")
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

    def recognize_gesture(self):
        if not self.recording:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count < 2:
                messagebox.showinfo("Recognition", "Please register at least two different users before recognition.")
                return
                
            if not self.model:
                messagebox.showerror("Error", "No trained model available. Please register users first.")
                return
                
            messagebox.showinfo("Recognition", "Perform the gesture to authenticate")
            self.recognition_mode = True
            self.recognition_timer = None

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

#access logger button
        self.logger_button = tk.Button(button_frame, text="Access Logger", command=self.open_access_logger,
                                   font=("Arial", 12), bg="#9C27B0", fg="white", padx=20)
        self.logger_button.pack(side=tk.LEFT, padx=10)
                
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
        confidence_slider = tk.Scale(settings_window, from_=0.0, to=1.0, resolution=0.01, 
                                    orient=tk.HORIZONTAL, length=300)
        confidence_slider.set(self.confidence_threshold)
        confidence_slider.pack()
        
        # Minimum samples setting
        tk.Label(settings_window, text="Minimum Samples per User:", font=("Arial", 12)).pack(pady=(20, 5))
        samples_slider = tk.Scale(settings_window, from_=5, to=70, resolution=5, 
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
        
    # Add the new method for the access logger window
    def open_access_logger(self):
        """Open access logger window to display recognition history"""
        logger_window = tk.Toplevel(self.root)
        logger_window.title("Access Logger")
        logger_window.geometry("800x500")
        
        # Create treeview
        columns = ("timestamp", "username", "confidence")
        tree = ttk.Treeview(logger_window, columns=columns, show="headings")
        
        # Define headings
        tree.heading("timestamp", text="Timestamp")
        tree.heading("username", text="Username")
        tree.heading("confidence", text="Confidence")
        
        # Set column widths
        tree.column("timestamp", width=200)
        tree.column("username", width=200)
        tree.column("confidence", width=100)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(logger_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Add refresh button
        def refresh_logs():
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Get logs from database
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, username, confidence FROM access_logs ORDER BY timestamp DESC"
            )
            logs = cursor.fetchall()
            
            # Insert logs into treeview
            for log in logs:
                timestamp, username, confidence = log
                tree.insert("", tk.END, values=(timestamp, username, f"{confidence:.2f}"))
        
        # Add filter options
        filter_frame = tk.Frame(logger_window)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter by username:").pack(side=tk.LEFT, padx=5)
        username_var = tk.StringVar()
        username_entry = tk.Entry(filter_frame, textvariable=username_var)
        username_entry.pack(side=tk.LEFT, padx=5)
        
        # Date filtering
        tk.Label(filter_frame, text="Date range:").pack(side=tk.LEFT, padx=5)
        from_date_var = tk.StringVar()
        from_date_entry = tk.Entry(filter_frame, textvariable=from_date_var, width=10)
        from_date_entry.pack(side=tk.LEFT, padx=5)
        from_date_entry.insert(0, "YYYY-MM-DD")
        
        tk.Label(filter_frame, text="to").pack(side=tk.LEFT)
        to_date_var = tk.StringVar()
        to_date_entry = tk.Entry(filter_frame, textvariable=to_date_var, width=10)
        to_date_entry.pack(side=tk.LEFT, padx=5)
        to_date_entry.insert(0, "YYYY-MM-DD")
        
        def apply_filters():
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Construct query with filters
            query = "SELECT timestamp, username, confidence FROM access_logs WHERE 1=1"
            params = []
            
            if username_var.get():
                query += " AND username LIKE ?"
                params.append(f"%{username_var.get()}%")
            
            from_date = from_date_var.get()
            if from_date and from_date != "YYYY-MM-DD":
                query += " AND date(timestamp) >= ?"
                params.append(from_date)
                
            to_date = to_date_var.get()
            if to_date and to_date != "YYYY-MM-DD":
                query += " AND date(timestamp) <= ?"
                params.append(to_date)
                
            query += " ORDER BY timestamp DESC"
            
            # Execute query
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            logs = cursor.fetchall()
            
            # Insert logs into treeview
            for log in logs:
                timestamp, username, confidence = log
                tree.insert("", tk.END, values=(timestamp, username, f"{confidence:.2f}"))
            
            # Save the current filter for export
            self.current_filter_query = query
            self.current_filter_params = params
        
        button_frame = tk.Frame(logger_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        filter_button = tk.Button(button_frame, text="Apply Filters", command=apply_filters,
                                bg="#2196F3", fg="white", padx=10)
        filter_button.pack(side=tk.LEFT, padx=5)
        
        refresh_button = tk.Button(button_frame, text="Refresh", command=refresh_logs,
                                bg="#4CAF50", fg="white", padx=10)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        export_button = tk.Button(button_frame, text="Export to CSV", 
                                command=lambda: self.export_logs_to_csv(query=self.current_filter_query if hasattr(self, 'current_filter_query') else None, 
                                                                params=self.current_filter_params if hasattr(self, 'current_filter_params') else None),
                                bg="#FF9800", fg="white", padx=10)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize filter variables
        self.current_filter_query = None
        self.current_filter_params = None
        
        # Load logs initially
        refresh_logs()

    def export_logs_to_csv(self, query=None, params=None):
        """Export access logs to a CSV file"""
        import csv
        from tkinter import filedialog
        import datetime
        
        # Ask for file location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"access_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filename:
            return
        
        # Get logs from database using current filter if provided
        cursor = self.conn.cursor()
        
        if query and params:
            cursor.execute(query, params)
        else:
            cursor.execute(
                "SELECT timestamp, username, confidence FROM access_logs ORDER BY timestamp DESC"
            )
        
        logs = cursor.fetchall()
        
        # Write to CSV
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Username", "Confidence"])
                for log in logs:
                    writer.writerow(log)
                    
            messagebox.showinfo("Export Successful", f"Logs exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting logs: {str(e)}")