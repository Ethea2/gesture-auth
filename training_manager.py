import time
import tkinter as tk
from tkinter import messagebox

class TrainingManager:
    def __init__(self, conn, gesture_processor, model_manager):
        self.conn = conn
        self.gesture_processor = gesture_processor
        self.model_manager = model_manager
        
        # Training state
        self.recording = False
        self.samples_count = 0
        self.current_samples = []
        self.username = None
        
        # 4-phase training system
        self.total_repetitions = 20
        self.repetitions_per_phase = 5
        self.current_phase = 1
        
        # Timing
        self.repetition_timer = 0
        self.countdown_seconds = 5
        self.last_sample_time = 0
        
        # Pause system
        self.registration_paused = False
        self.no_hand_start_time = None
        self.pause_delay = 3.0
        self.pause_dialog = None

    def get_phase_instruction(self, phase):
        """Get instruction text for each phase"""
        instructions = {
            1: "Phase 1: Perform your gesture naturally",
            2: "Phase 2: Tilt your hand slightly to the opposite side\n(Right hand → tilt left, Left hand → tilt right\nTwo hands → continue naturally)",
            3: "Phase 3: Return to your original gesture position",
            4: "Phase 4: Return to the tilted position from Phase 2"
        }
        return instructions.get(phase, "Perform your gesture")

    def get_detailed_phase_instruction(self, phase):
        """Get detailed instruction for phase transition dialogs"""
        instructions = {
            1: "Phase 1: Natural Gesture\n\nPerform your gesture in its most natural, comfortable position. This establishes your baseline gesture pattern.",
            2: "Phase 2: Slight Tilt\n\nNow tilt your hand slightly to the opposite direction:\n• Right hand → tilt slightly left\n• Left hand → tilt slightly right\n• Two hands → continue your gesture naturally\n\nThis helps the model learn variations in hand positioning.",
            3: "Phase 3: Return to Original\n\nReturn to your original, natural gesture position from Phase 1. This reinforces the primary gesture pattern.",
            4: "Phase 4: Return to Tilt\n\nReturn to the tilted position from Phase 2. This completes the variation training for better recognition accuracy."
        }
        return instructions.get(phase, "Continue with your gesture")

    def start_registration(self, username):
        """Start the registration process"""
        self.username = username
        
        # Show comprehensive instructions
        messagebox.showinfo("Registration Instructions", 
                        "Enhanced Registration Process (20 repetitions in 4 phases):\n\n" +
                        "For optimal recognition results:\n" +
                        "1. Keep your hand INSIDE the blue box at all times\n" +
                        "2. Follow the phase-specific instructions\n" +
                        "3. Maintain consistent gesture core while following tilt instructions\n" +
                        "4. Each phase has 5 repetitions (5 seconds each)\n\n" +
                        "The system will guide you through each phase.\n" +
                        "NOTE: Registration will pause if no hand is detected for 3 seconds.")
        
        self.recording = True
        self.registration_paused = False
        self.no_hand_start_time = None
        self.samples_count = 0 
        self.current_phase = 1
        self.current_samples = []
        self.repetition_timer = time.time()
        self.countdown_seconds = 5
        
        # Show first phase instruction
        messagebox.showinfo("Phase 1 Starting", self.get_detailed_phase_instruction(1))
        
        return f"Phase 1 - Repetition 1/{self.repetitions_per_phase}: Starting in {self.countdown_seconds} seconds..."

    def check_hand_presence(self, landmarks):
        """Check if hand is present and handle pausing logic"""
        current_time = time.time()
        
        if landmarks:  # Hand detected
            self.no_hand_start_time = None
            if self.registration_paused:
                # Hand detected while paused, resume automatically
                self.registration_paused = False
                if self.pause_dialog:
                    self.pause_dialog.destroy()
                    self.pause_dialog = None
        else:  # No hand detected
            if self.no_hand_start_time is None:
                self.no_hand_start_time = current_time
            elif not self.registration_paused and (current_time - self.no_hand_start_time) >= self.pause_delay:
                # Pause registration
                self.registration_paused = True
                self.show_pause_dialog()

    def show_pause_dialog(self):
        """Show dialog when registration is paused due to no hand detection"""
        if self.pause_dialog is not None:
            return  # Dialog already open
            
        self.pause_dialog = tk.Toplevel()
        self.pause_dialog.title("Registration Paused")
        self.pause_dialog.geometry("400x200")
        self.pause_dialog.transient()
        self.pause_dialog.grab_set()
        
        # Message
        message_frame = tk.Frame(self.pause_dialog)
        message_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        icon_label = tk.Label(message_frame, text="⚠️", font=("Arial", 24))
        icon_label.pack()
        
        title_label = tk.Label(message_frame, text="Registration Paused", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(10, 5))
        
        message_label = tk.Label(message_frame, 
                                text="No hand detected in the blue box for 3 seconds.\n\n" +
                                     "Place your hand in the blue detection box to continue\n" +
                                     "or choose an option below:",
                                font=("Arial", 11),
                                justify=tk.CENTER)
        message_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(self.pause_dialog)
        button_frame.pack(pady=20, padx=20, fill=tk.X)
        
        def continue_registration():
            self.registration_paused = False
            self.no_hand_start_time = None
            self.pause_dialog.destroy()
            self.pause_dialog = None
            
        def cancel_registration():
            self.stop_registration()
            self.pause_dialog.destroy()
            self.pause_dialog = None
            
        continue_btn = tk.Button(button_frame, text="Continue Registration", 
                               command=continue_registration,
                               font=("Arial", 11), bg="#4CAF50", fg="white", 
                               padx=20, pady=5)
        continue_btn.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        cancel_btn = tk.Button(button_frame, text="Cancel Registration", 
                             command=cancel_registration,
                             font=("Arial", 11), bg="#f44336", fg="white", 
                             padx=20, pady=5)
        cancel_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Handle window close
        self.pause_dialog.protocol("WM_DELETE_WINDOW", cancel_registration)

    def process_registration(self, landmarks):
        """Process registration with landmarks"""
        current_time = time.time()
        
        # Check for hand presence and handle pausing
        self.check_hand_presence(landmarks)
        
        # Don't process if paused
        if self.registration_paused:
            return "Registration Paused - Place hand in blue box"
            
        if self.recording and self.samples_count < self.total_repetitions:
            elapsed_time = current_time - self.repetition_timer
            
            # Collect multiple samples during each repetition window
            if elapsed_time < 5 and landmarks and time.time() - self.last_sample_time > 0.2:
                features = self.gesture_processor.extract_features(landmarks)
                
                # Add original sample
                self.current_samples.append(features)
                
                # Add augmented samples
                augmented = self.gesture_processor.augment_samples(features)
                for aug_sample in augmented[1:]:  # Skip the first one as it's the original
                    self.current_samples.append(aug_sample)
                    
                self.last_sample_time = time.time()
                # Display sample count
                samples_collected = len([s for s in self.current_samples if len(s) > 0])
                rep_in_phase = (self.samples_count % self.repetitions_per_phase) + 1
                return f"Phase {self.current_phase} - Rep {rep_in_phase}/{self.repetitions_per_phase}: {self.countdown_seconds}s - Samples: {samples_collected}"

            elif elapsed_time >= 5:
                self.samples_count += 1
                self.repetition_timer = time.time()
                
                if self.samples_count < self.total_repetitions:
                    # Check if we need to move to next phase
                    if self.samples_count % self.repetitions_per_phase == 0:
                        self.current_phase += 1
                        # Show phase transition dialog
                        messagebox.showinfo(f"Phase {self.current_phase} Starting", 
                                          self.get_detailed_phase_instruction(self.current_phase))
                    
                    self.countdown_seconds = 5  # Reset countdown to 5 seconds
                    rep_in_phase = (self.samples_count % self.repetitions_per_phase) + 1
                    
                    # Give variation tips at specific points
                    if self.samples_count == self.repetitions_per_phase:  # After phase 1
                        messagebox.showinfo("Phase Transition", "Great! Now moving to Phase 2 with slight hand tilting.")
                    elif self.samples_count == self.repetitions_per_phase * 2:  # After phase 2
                        messagebox.showinfo("Phase Transition", "Excellent! Return to your original position for Phase 3.")
                    elif self.samples_count == self.repetitions_per_phase * 3:  # After phase 3
                        messagebox.showinfo("Phase Transition", "Almost done! Final phase with tilted position.")
                    
                    return f"Phase {self.current_phase} - Repetition {rep_in_phase}/{self.repetitions_per_phase}: {self.countdown_seconds} seconds remaining..."
                        
                else:
                    # Registration complete
                    self.save_user_data()
                    self.stop_registration()
                    messagebox.showinfo("Registration Complete", 
                                      f"Registration successful!\n\n" +
                                      f"Completed all 4 phases with {len(self.current_samples)} total samples.\n" +
                                      f"Your gesture recognition model has been updated.")
                    return "Registration Complete!"
        
        return ""

    def update_countdown(self):
        """Update countdown timer"""
        if self.recording and not self.registration_paused:
            current_time = time.time()
            elapsed_time = current_time - self.repetition_timer
            self.countdown_seconds = max(0, 5 - int(elapsed_time))
            return True
        return False

    def get_status_message(self):
        """Get current status message for UI"""
        if not self.recording:
            return "Ready for registration"
        elif self.registration_paused:
            return "Registration Paused - Place hand in blue box"
        else:
            rep_in_phase = (self.samples_count % self.repetitions_per_phase) + 1
            return f"Phase {self.current_phase}/4 - Rep {rep_in_phase}/{self.repetitions_per_phase}"

    def save_user_data(self):
        """Save collected samples to database"""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO users (username) VALUES (?)", (self.username,))
        user_id = cursor.lastrowid

        for sample in self.current_samples:
            cursor.execute(
                "INSERT INTO gesture_samples (user_id, feature_data) VALUES (?, ?)", 
                (user_id, sample.tobytes())
            )

        self.conn.commit()
        self.model_manager.retrain_model(self.conn)

    def stop_registration(self):
        """Stop the registration process"""
        self.recording = False
        self.registration_paused = False
        self.current_samples = []
        self.samples_count = 0
        self.current_phase = 1
        if self.pause_dialog:
            self.pause_dialog.destroy()
            self.pause_dialog = None

    def continue_from_pause(self):
        """Continue registration from pause"""
        self.registration_paused = False
        self.no_hand_start_time = None
        if self.pause_dialog:
            self.pause_dialog.destroy()
            self.pause_dialog = None

    def cancel_registration(self):
        """Cancel registration with confirmation"""
        result = messagebox.askyesno("Cancel Registration", 
                                   "Are you sure you want to cancel the registration?")
        if result:
            self.stop_registration()
            return True
        return False