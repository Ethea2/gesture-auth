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
        
        # NEW: Prompt pause system
        self.prompt_paused = False
        self.pause_reason = None
        self.pause_start_time = None
        self.accumulated_pause_time = 0

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
        
        # Pause registration for initial instructions
        self.pause_for_prompt("initial_instructions")
        
        # Show comprehensive instructions
        def on_instructions_ok():
            self.resume_from_prompt()
            # Show first phase instruction
            self.pause_for_prompt("phase_1_start")
            
            def on_phase_1_ok():
                self.resume_from_prompt()
                self._initialize_registration()
            
            self.show_phase_instruction_dialog(1, on_phase_1_ok)
        
        self.show_initial_instructions_dialog(on_instructions_ok)
        
        return "Registration starting - please read instructions"

    def show_initial_instructions_dialog(self, callback):
        """Show initial registration instructions"""
        dialog = tk.Toplevel()
        dialog.title("Registration Instructions")
        dialog.geometry("500x400")
        dialog.transient()
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (
            (dialog.winfo_screenwidth() // 2) - 250,
            (dialog.winfo_screenheight() // 2) - 200
        ))
        
        # Instructions text
        instructions = """Enhanced Registration Process (20 repetitions in 4 phases)

For optimal recognition results:

1. Keep your hand INSIDE the blue box at all times
2. Follow the phase-specific instructions
3. Maintain consistent gesture core while following tilt instructions
4. Each phase has 5 repetitions (5 seconds each)

The system will guide you through each phase.

NOTE: Registration will pause if no hand is detected for 3 seconds.

Click OK when ready to begin!"""
        
        text_widget = tk.Text(dialog, wrap=tk.WORD, font=("Arial", 12), padx=20, pady=20)
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # OK button
        ok_button = tk.Button(dialog, text="OK - Start Registration", command=lambda: [dialog.destroy(), callback()],
                             font=("Arial", 12), bg="#4CAF50", fg="white", padx=30, pady=10)
        ok_button.pack(pady=20)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog.destroy(), callback()])

    def show_phase_instruction_dialog(self, phase, callback):
        """Show phase-specific instruction dialog"""
        dialog = tk.Toplevel()
        dialog.title(f"Phase {phase} Instructions")
        dialog.geometry("450x300")
        dialog.transient()
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (
            (dialog.winfo_screenwidth() // 2) - 225,
            (dialog.winfo_screenheight() // 2) - 150
        ))
        
        # Phase instructions
        instruction_text = self.get_detailed_phase_instruction(phase)
        
        text_widget = tk.Text(dialog, wrap=tk.WORD, font=("Arial", 12), padx=15, pady=15)
        text_widget.insert(tk.END, instruction_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # OK button
        ok_button = tk.Button(dialog, text="OK - Continue", command=lambda: [dialog.destroy(), callback()],
                             font=("Arial", 12), bg="#2196F3", fg="white", padx=30, pady=10)
        ok_button.pack(pady=20)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog.destroy(), callback()])

    def _initialize_registration(self):
        """Initialize registration after all prompts are done"""
        self.recording = True
        self.registration_paused = False
        self.no_hand_start_time = None
        self.samples_count = 0 
        self.current_phase = 1
        self.current_samples = []
        self.repetition_timer = time.time()
        self.countdown_seconds = 5
        self.accumulated_pause_time = 0

    def pause_for_prompt(self, reason):
        """Pause registration for a prompt/dialog"""
        if self.recording:
            self.prompt_paused = True
            self.pause_reason = reason
            self.pause_start_time = time.time()
            print(f"Registration paused for: {reason}")

    def resume_from_prompt(self):
        """Resume registration after prompt is dismissed"""
        if self.prompt_paused and self.pause_start_time:
            # Calculate how long we were paused
            pause_duration = time.time() - self.pause_start_time
            self.accumulated_pause_time += pause_duration
            
            # Adjust the repetition timer to account for pause time
            self.repetition_timer += pause_duration
            
            self.prompt_paused = False
            self.pause_reason = None
            self.pause_start_time = None
            print("Registration resumed after prompt")

    def update_countdown(self):
        """Update countdown timer"""
        if self.recording and not self.registration_paused and not self.prompt_paused:
            current_time = time.time()
            elapsed_time = current_time - self.repetition_timer
            self.countdown_seconds = max(0, 5 - int(elapsed_time))
            return True
        return False

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
            elif not self.registration_paused and not self.prompt_paused and (current_time - self.no_hand_start_time) >= self.pause_delay:
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
        
        # Don't process if paused for any reason
        if self.registration_paused or self.prompt_paused:
            if self.registration_paused:
                return "Registration Paused - Place hand in blue box"
            elif self.prompt_paused:
                return f"Registration Paused - {self.pause_reason}"
        
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
                        
                        # Pause for phase transition
                        self.pause_for_prompt(f"phase_{self.current_phase}_transition")
                        
                        def on_phase_transition_ok():
                            self.resume_from_prompt()
                        
                        # Show phase transition dialog
                        self.show_phase_instruction_dialog(self.current_phase, on_phase_transition_ok)
                    
                    self.countdown_seconds = 5  # Reset countdown to 5 seconds
                    rep_in_phase = (self.samples_count % self.repetitions_per_phase) + 1
                    
                    return f"Phase {self.current_phase} - Repetition {rep_in_phase}/{self.repetitions_per_phase}: {self.countdown_seconds} seconds remaining..."
                        
                else:
                    # Registration complete
                    self.save_user_data()
                    self.stop_registration()
                    
                    # Show completion dialog
                    messagebox.showinfo("Registration Complete", 
                                      f"Registration successful!\n\n" +
                                      f"Completed all 4 phases with {len(self.current_samples)} total samples.\n" +
                                      f"Your gesture recognition model has been updated.")
                    return "Registration Complete!"
        
        return ""

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
        self.prompt_paused = False
        self.current_samples = []
        self.samples_count = 0
        self.current_phase = 1
        self.accumulated_pause_time = 0
        
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

    def get_status_message(self):
        """Get current status message for UI"""
        if not self.recording:
            return "Ready for registration"
        elif self.prompt_paused:
            return f"Registration Paused - {self.pause_reason}"
        elif self.registration_paused:
            return "Registration Paused - Place hand in blue box"
        else:
            rep_in_phase = (self.samples_count % self.repetitions_per_phase) + 1
            return f"Phase {self.current_phase}/4 - Rep {rep_in_phase}/{self.repetitions_per_phase}"

    def is_paused(self):
        """Check if registration is currently paused"""
        return self.registration_paused or self.prompt_paused