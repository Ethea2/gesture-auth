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

        try:
            self.model = joblib.load('gesture_model.joblib')
        except:
            pass

        self.setup_gui()
        self.setup_camera()

    def extract_features(self, landmarks):
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)

    def register_user(self):
        if not self.recording:
            self.username = simpledialog.askstring("Input", "Enter username:")
            if self.username:
                self.recording = True
                self.samples_count = 0
                self.current_samples = []

    def process_registration(self, landmarks):
        current_time = time.time()
        
        if self.recording and self.samples_count < 5 and (current_time - self.last_sample_time > self.sample_delay):
            self.last_sample_time = current_time              
            features = self.extract_features(landmarks)
            self.current_samples.append(features)
            self.samples_count += 1
            
            if self.samples_count >= 5:
                self.save_user_data()
                self.recording = False
                messagebox.showinfo("Registration", "Registration complete!")

    def save_user_data(self):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO users (username) VALUES (?)", (self.username,))
        user_id = cursor.lastrowid
        
        for sample in self.current_samples:
            cursor.execute("INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", (user_id, sample.tobytes()))
        
        self.conn.commit()
        self.retrain_model()

    def retrain_model(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT users.user_id, gesture_samples.feature_data 
            FROM users 
            JOIN gesture_samples ON users.user_id = gesture_samples.user_id
        """)
        data = cursor.fetchall()

        if len(set(sample[0] for sample in data)) < 2:
            return

        X = [np.frombuffer(sample[1]) for sample in data]
        y = [sample[0] for sample in data]

        self.model.fit(X, y)
        joblib.dump(self.model, 'gesture_model.joblib')

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if self.recording:
                        self.process_registration(hand_landmarks)
                    elif hasattr(self, 'recognition_mode') and self.recognition_mode:
                        self.process_recognition(hand_landmarks)

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            if self.recording:
                self.canvas.create_text(
                    320, 50,
                    text=f"Remaining time {self.samples_count}/5",
                    fill="red",
                    font=("Arial", 20)
                )
                self.canvas.create_text(
                    320, 460,
                    text="Please perform the gesture a couple of times within 5 seconds. Starting recording...",
                    fill="red",
                    font=("Arial", 20)
                )

        self.root.after(10, self.update_video_feed)

    def setup_gui(self):
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
    
        self.status_label = tk.Label(self.root, text="Ready")
        self.status_label.pack()
    
        self.register_button = tk.Button(self.root, text="Register", command=self.register_user)
        self.register_button.pack(side=tk.LEFT, padx=10, pady=10)
    
        self.recognize_button = tk.Button(self.root, text="Recognize", command=self.recognize_gesture)
        self.recognize_button.pack(side=tk.LEFT, padx=10, pady=10)
                
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

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

    def process_recognition(self, landmarks):
        if hasattr(self, 'recognition_mode') and self.recognition_mode:
            features = self.extract_features(landmarks)
            try:
                user_id = self.model.predict([features])[0]
                cursor = self.conn.cursor()
                cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                username = cursor.fetchone()[0]
                messagebox.showinfo("Recognition Result", f"Recognized user: {username}")
            except Exception:
                messagebox.showerror("Error", "Could not recognize gesture. Please try again.")
            finally:
                self.recognition_mode = False
