import tkinter as tk
from tkinter import messagebox
import time

class FeedbackSystem:
    def __init__(self, conn, gesture_processor, model_manager):
        self.conn = conn
        self.gesture_processor = gesture_processor
        self.model_manager = model_manager
        
        # Store last recognition data for learning
        self.last_recognition_landmarks = None
        self.last_recognition_confidence = 0.0
        self.last_predicted_user_id = None

    def show_recognition_feedback(self, predicted_user_id, confidence, landmarks, parent_window):
        """Show feedback dialog after recognition and handle learning from mistakes"""
        # Store the landmarks for potential correction
        self.last_recognition_landmarks = landmarks
        self.last_recognition_confidence = confidence
        self.last_predicted_user_id = predicted_user_id
        
        # Create a custom dialog
        feedback_window = tk.Toplevel(parent_window)
        feedback_window.title("Recognition Feedback")
        feedback_window.geometry("400x350")
        feedback_window.grab_set()  # Make it modal
        
        # Center the window
        feedback_window.geometry("+%d+%d" % (
            (feedback_window.winfo_screenwidth() // 2) - 200,
            (feedback_window.winfo_screenheight() // 2) - 175
        ))
        
        # Get the predicted username
        cursor = self.conn.cursor()
        cursor.execute("SELECT username FROM users WHERE user_id = ?", (predicted_user_id,))
        result = cursor.fetchone()
        predicted_username = result[0] if result else f"Unknown (ID: {predicted_user_id})"
        
        # Show recognition result
        if confidence >= self.model_manager.confidence_threshold:
            result_text = f"✅ Recognized as: {predicted_username}"
            result_color = "#4CAF50"  # Green
        else:
            result_text = f"❌ Below threshold - Suggested: {predicted_username}"
            result_color = "#FF9800"  # Orange
        
        title_label = tk.Label(feedback_window, text=result_text, 
                            font=("Arial", 14), fg=result_color)
        title_label.pack(pady=(20, 10))
        
        tk.Label(feedback_window, text=f"Confidence: {confidence:.2f} (Threshold: {self.model_manager.confidence_threshold:.2f})", 
                font=("Arial", 12)).pack(pady=(0, 20))
        
        # Ask if recognition was correct
        tk.Label(feedback_window, text="Was this recognition correct?", 
                font=("Arial", 12, "bold")).pack(pady=(10, 20))
        
        button_frame = tk.Frame(feedback_window)
        button_frame.pack(pady=10)
        
        # Yes button - recognition was correct
        tk.Button(button_frame, text="✅ Yes, Correct", 
                command=lambda: self.handle_correct_recognition(feedback_window),
                font=("Arial", 12), bg="#4CAF50", fg="white", padx=20).pack(side=tk.LEFT, padx=10)
        
        # No button - recognition was incorrect
        tk.Button(button_frame, text="❌ No, Incorrect", 
                command=lambda: self.handle_incorrect_recognition(feedback_window),
                font=("Arial", 12), bg="#F44336", fg="white", padx=20).pack(side=tk.LEFT, padx=10)
        
        # Add a skip button
        tk.Button(feedback_window, text="⏭️ Skip Feedback", 
                command=lambda: feedback_window.destroy(),
                font=("Arial", 10), bg="#9E9E9E", fg="white").pack(pady=(20, 10))

    def handle_correct_recognition(self, feedback_window):
        """Handle case when recognition was correct"""
        feedback_window.destroy()
        messagebox.showinfo("Feedback", "Thank you! No changes needed.")

    def handle_incorrect_recognition(self, feedback_window):
        """Handle case when recognition was incorrect"""
        # Create a new dialog to handle error cases
        correction_window = tk.Toplevel()
        correction_window.title("Recognition Correction")
        correction_window.geometry("500x450")
        correction_window.grab_set()  # Make it modal
        
        # Center the window
        correction_window.geometry("+%d+%d" % (
            (correction_window.winfo_screenwidth() // 2) - 250,
            (correction_window.winfo_screenheight() // 2) - 225
        ))
        
        # Close the feedback window
        feedback_window.destroy()
        
        tk.Label(correction_window, text="Please select the correct scenario:", 
                font=("Arial", 14, "bold")).pack(pady=(20, 20))
        
        # Get all registered users
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id, username FROM users ORDER BY username")
        all_users = cursor.fetchall()
        
        # Create radio buttons for user selection
        user_frame = tk.LabelFrame(correction_window, text="If a registered user, select who:")
        user_frame.pack(fill=tk.X, padx=20, pady=10)
        
        user_var = tk.StringVar()
        for user_id, username in all_users:
            tk.Radiobutton(user_frame, text=username, variable=user_var, 
                        value=str(user_id), font=("Arial", 11)).pack(anchor=tk.W, padx=10, pady=2)
        
        # Scenario buttons
        scenario_frame = tk.Frame(correction_window)
        scenario_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Scenario A: Should have recognized me
        def scenario_a():
            selected_user_id = user_var.get()
            if selected_user_id:
                self.learn_from_missed_recognition(int(selected_user_id))
                correction_window.destroy()
            else:
                messagebox.showwarning("Selection Required", "Please select the correct user first.")
        
        # Scenario B: Recognized as wrong user
        def scenario_b():
            selected_user_id = user_var.get()
            if selected_user_id:
                self.learn_from_misrecognition(self.last_predicted_user_id, int(selected_user_id))
                correction_window.destroy()
            else:
                messagebox.showwarning("Selection Required", "Please select the correct user first.")
        
        # Scenario C: Should not recognize at all
        def scenario_c():
            self.learn_from_false_recognition(self.last_predicted_user_id)
            correction_window.destroy()
        
        tk.Button(scenario_frame, text="A: Should have recognized me as selected user", 
                command=scenario_a, font=("Arial", 11), bg="#2196F3", fg="white", 
                wraplength=400, justify=tk.CENTER).pack(fill=tk.X, pady=5)
        
        tk.Button(scenario_frame, text="B: Recognized wrong user (correct user is selected above)", 
                command=scenario_b, font=("Arial", 11), bg="#FF9800", fg="white",
                wraplength=400, justify=tk.CENTER).pack(fill=tk.X, pady=5)
        
        tk.Button(scenario_frame, text="C: Should not recognize anyone (unauthorized gesture)", 
                command=scenario_c, font=("Arial", 11), bg="#9C27B0", fg="white",
                wraplength=400, justify=tk.CENTER).pack(fill=tk.X, pady=5)
        
        # Cancel button
        tk.Button(correction_window, text="Cancel", 
                command=lambda: correction_window.destroy(),
                font=("Arial", 11), bg="#9E9E9E", fg="white").pack(pady=20)

    def learn_from_missed_recognition(self, correct_user_id):
        """Scenario A: Add samples to the correct user when recognition missed"""
        if not self.last_recognition_landmarks:
            messagebox.showerror("Error", "No gesture data available for correction")
            return
        
        try:
            # Extract features from the last recognition landmarks
            features = self.gesture_processor.extract_features(self.last_recognition_landmarks)
            
            # Create augmented samples to improve learning
            augmented_samples = self.gesture_processor.augment_samples(features)
            
            # Add all samples to the database for the correct user
            cursor = self.conn.cursor()
            for sample in augmented_samples:
                cursor.execute(
                    "INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", 
                    (correct_user_id, sample.tobytes())
                )
            
            self.conn.commit()
            messagebox.showinfo("Learning Complete", 
                            f"Added {len(augmented_samples)} samples to help improve recognition")
            
            # Retrain the model with the new data
            self.model_manager.retrain_model(self.conn)
            
        except Exception as e:
            print(f"Learning error: {str(e)}")
            messagebox.showerror("Learning Error", f"Error while updating model: {str(e)}")

    def learn_from_misrecognition(self, wrong_user_id, correct_user_id):
        """Scenario B: Add samples to correct user and negative samples for wrong user"""
        if not self.last_recognition_landmarks:
            messagebox.showerror("Error", "No gesture data available for correction")
            return
        
        try:
            # Extract features from the last recognition landmarks
            features = self.gesture_processor.extract_features(self.last_recognition_landmarks)
            
            # Create augmented samples
            augmented_samples = self.gesture_processor.augment_samples(features)
            
            cursor = self.conn.cursor()
            
            # Add samples to correct user
            for sample in augmented_samples:
                cursor.execute(
                    "INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", 
                    (correct_user_id, sample.tobytes())
                )
            
            # Add negative samples for the wrong user (fewer samples to avoid over-penalization)
            limited_samples = augmented_samples[:3]  # Only use first 3 augmented samples
            for sample in limited_samples:
                cursor.execute(
                    "INSERT INTO negative_samples (user_id, feature_data) VALUES (?, ?)", 
                    (wrong_user_id, sample.tobytes())
                )
            
            self.conn.commit()
            messagebox.showinfo("Learning Complete", 
                            f"Added samples for correct user and negative samples for misrecognized user")
            
            # Retrain the model with the new data
            self.model_manager.retrain_model(self.conn)
            
        except Exception as e:
            print(f"Learning error: {str(e)}")
            messagebox.showerror("Learning Error", f"Error while updating model: {str(e)}")

    def learn_from_false_recognition(self, wrong_user_id):
        """Scenario C: Add negative samples for falsely recognized user"""
        if not self.last_recognition_landmarks:
            messagebox.showerror("Error", "No gesture data available for correction")
            return
        
        try:
            # Extract features from the last recognition landmarks
            features = self.gesture_processor.extract_features(self.last_recognition_landmarks)
            
            # Create fewer augmented samples for negatives to avoid over-penalization
            limited_samples = [features]
            
            # Add just one slightly noisy sample (much less variation)
            import numpy as np
            noise_scale = 0.005
            noisy = features + np.random.normal(0, noise_scale, len(features))
            limited_samples.append(noisy)
            
            cursor = self.conn.cursor()
            
            # Add negative samples for the wrong user
            for sample in limited_samples:
                cursor.execute(
                    "INSERT INTO negative_samples (user_id, feature_data) VALUES (?, ?)", 
                    (wrong_user_id, sample.tobytes())
                )
            
            self.conn.commit()
            messagebox.showinfo("Learning Complete", 
                            f"Added {len(limited_samples)} negative samples to help prevent false recognition")
            
            # Retrain the model with the new data
            self.model_manager.retrain_model(self.conn)
            
        except Exception as e:
            print(f"Learning error: {str(e)}")
            messagebox.showerror("Learning Error", f"Error while updating model: {str(e)}")

    def should_show_feedback(self, confidence):
        """Determine if feedback should be shown based on confidence"""
        # Show feedback for low confidence predictions or randomly for high confidence
        import random
        if confidence < self.model_manager.confidence_threshold:
            return True  # Always show for low confidence
        else:
            return random.random() < 0.1  # 10% chance for high confidence