import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import numpy as np
from sklearn.svm import SVC
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
        self.model = SVC(kernel='rbf')
        self.last_sample_time = 0
        self.sample_delay = 1
        self.remaining_time = 0
        self.recognition_mode = False
        self.recognition_timer = None
        self.countdown_seconds = 5  # 5 seconds per repetition

        try:
            self.model = joblib.load('gesture_model.joblib')
        except:
            pass

        self.setup_gui()
        self.setup_camera()

    def extract_features(self, landmarks):
        features = []
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features, dtype=np.float32)

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
            
            if elapsed_time < 5:
                features = self.extract_features(landmarks)
                self.current_samples.append(features)

            elif elapsed_time >= 5:
                self.samples_count += 1
                self.repetition_timer = time.time()
                
                if self.samples_count < 5:
                    self.countdown_seconds = 5  # Reset countdown to 5 seconds
                    self.status_label.config(text=f"Repetition {self.samples_count+1}/5: {self.countdown_seconds} seconds remaining...")
                    self.update_countdown()
                else:
                    self.save_user_data()
                    self.recording = False
                    self.status_label.config(text="Registration Complete!")
                    messagebox.showinfo("Registration", "Registration complete!")

    def save_user_data(self):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO users (username) VALUES (?)", (self.username,))
        user_id = cursor.lastrowid

        for sample in self.current_samples:
            cursor.execute(
                "INSERT INTO gesture_samples (user_id, feature_data, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)", 
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

        if len(set(sample[0] for sample in data)) < 2:
            return

        X = [np.frombuffer(sample[1], dtype=np.float32) for sample in data]
        y = [sample[0] for sample in data]

        feature_length = max(len(features) for features in X)
        X = [np.pad(features, (0, feature_length - len(features)), mode='constant') for features in X]

        self.model.fit(X, y)
        joblib.dump(self.model, 'gesture_model.joblib')

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
                elif time.time() - self.recognition_timer >= 3:
                    if landmarks:
                        self.process_recognition(landmarks)
                    self.recognition_timer = None

            img = Image.fromarray(frame_with_landmarks)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.root.after(10, self.update_video_feed)

    def process_recognition(self, landmarks):
        if self.recognition_mode:
            features = self.extract_features(landmarks)
            try:
                feature_length = self.model.n_features_in_
                features = np.pad(features, (0, feature_length - len(features)), mode='constant')
                user_id = self.model.predict([features])[0]
                cursor = self.conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if result:
                    messagebox.showinfo("Recognition Result", f"Recognized user: {result[0]}")
                else:
                    messagebox.showerror("Error", "User not found.")
            except Exception as e:
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
            messagebox.showinfo("Recognition", "Perform the gesture to authenticate")
            self.recognition_mode = True
            self.recognition_timer = None
