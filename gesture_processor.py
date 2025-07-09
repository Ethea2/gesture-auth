import numpy as np
import cv2
import mediapipe as mp

from practical_biometric_enhancement import PracticalBiometricEnhancement

class GestureProcessor:
    def __init__(self, frame_width=640, frame_height=480, box_width=600, box_height=300):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.box_width = box_width
        self.box_height = box_height
        self.box_x = (frame_width - box_width) // 2
        self.box_y = (frame_height - 200) // 2
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.biometric_enhancer = PracticalBiometricEnhancement()
        self.use_biometric_features = True 

    def update_box_size(self, width, height):
        """Update detection box size and recalculate position"""
        self.box_width = width
        self.box_height = height
        self.box_x = (self.frame_width - self.box_width) // 2
        self.box_y = (self.frame_height - self.box_height) // 2

    def is_hand_in_box(self, hand_landmarks):
        """Check if hand landmarks are within the detection box"""
        if not hand_landmarks:
            return False
        
        # Get bounding box of hand landmarks
        x_coords = [lm.x * self.frame_width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * self.frame_height for lm in hand_landmarks.landmark]
        
        hand_min_x = min(x_coords)
        hand_max_x = max(x_coords)
        hand_min_y = min(y_coords)
        hand_max_y = max(y_coords)
        
        # Check if majority of hand is within the detection box
        box_left = self.box_x
        box_right = self.box_x + self.box_width
        box_top = self.box_y
        box_bottom = self.box_y + self.box_height
        
        # Calculate overlap percentage
        overlap_left = max(hand_min_x, box_left)
        overlap_right = min(hand_max_x, box_right)
        overlap_top = max(hand_min_y, box_top)
        overlap_bottom = min(hand_max_y, box_bottom)
        
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
            hand_area = (hand_max_x - hand_min_x) * (hand_max_y - hand_min_y)
            
            if hand_area > 0:
                overlap_percentage = overlap_area / hand_area
                return overlap_percentage > 0.7  # At least 70% of hand should be in box
        
        return False

    def filter_hands_in_box(self, hand_landmarks_list):
        """Filter hands to only include those within the detection box"""
        if not hand_landmarks_list:
            return []
        
        filtered_hands = []
        for hand_landmarks in hand_landmarks_list:
            if self.is_hand_in_box(hand_landmarks):
                filtered_hands.append(hand_landmarks)
        
        return filtered_hands

    def combine_landmarks(self, hand_landmarks):
        """Combine landmarks from hands that are within the detection box"""
        if not hand_landmarks:
            return []
        
        # Filter hands to only include those in the detection box
        filtered_hands = self.filter_hands_in_box(hand_landmarks)
        
        combined_landmarks = []
        if filtered_hands:
            for hand in filtered_hands:
                combined_landmarks.extend(hand.landmark)
        return combined_landmarks

    def process_frame(self, frame):
        """Process frame and return hand landmarks within detection box"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)
        landmarks = self.combine_landmarks(hand_results.multi_hand_landmarks)
        
        return hand_results, landmarks

    def draw_detection_box(self, frame, mode="training"):
        """Draw the detection box on the frame"""
        # Draw the main detection box
        cv2.rectangle(frame, 
                     (self.box_x, self.box_y), 
                     (self.box_x + self.box_width, self.box_y + self.box_height), 
                     (0, 0, 255), 3)  # Blue color, thickness 3
        
        # Add corner markers for better visibility
        corner_size = 20
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(frame, (self.box_x, self.box_y), (self.box_x + corner_size, self.box_y), (0, 0, 255), corner_thickness)
        cv2.line(frame, (self.box_x, self.box_y), (self.box_x, self.box_y + corner_size), (0, 0, 255), corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (self.box_x + self.box_width, self.box_y), (self.box_x + self.box_width - corner_size, self.box_y), (0, 0, 255), corner_thickness)
        cv2.line(frame, (self.box_x + self.box_width, self.box_y), (self.box_x + self.box_width, self.box_y + corner_size), (0, 0, 255), corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (self.box_x, self.box_y + self.box_height), (self.box_x + corner_size, self.box_y + self.box_height), (0, 0, 255), corner_thickness)
        cv2.line(frame, (self.box_x, self.box_y + self.box_height), (self.box_x, self.box_y + self.box_height - corner_size), (0, 0, 255), corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (self.box_x + self.box_width, self.box_y + self.box_height), (self.box_x + self.box_width - corner_size, self.box_y + self.box_height), (0, 0, 255), corner_thickness)
        cv2.line(frame, (self.box_x + self.box_width, self.box_y + self.box_height), (self.box_x + self.box_width, self.box_y + self.box_height - corner_size), (0, 0, 255), corner_thickness)
        
        # Add instruction text
        font = cv2.FONT_HERSHEY_SIMPLEX
        if mode == "training":
            text = "Place hand in blue box"
        else:
            text = "Place hand here to authenticate"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = self.box_x + (self.box_width - text_size[0]) // 2
        text_y = self.box_y - 10
        
        # Add background for text
        cv2.rectangle(frame, (text_x - 5, text_y - 20), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        return frame

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks only for hands within the detection box"""
        if hand_landmarks:
            # Filter hands to only show those in the detection box
            filtered_hands = self.filter_hands_in_box(hand_landmarks)
            
            for hand_lms in filtered_hands:
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

    def extract_features(self, landmarks):
        """Extract more robust features from landmarks with NaN handling"""
        if not landmarks:
            return np.array([])
        
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
                normalized_dist = dist / hand_size if hand_size > 0 else 0.0
                relative_features.append(normalized_dist)
                
            # Distances between adjacent fingertips (normalized)
            for i in range(len(fingertips)-1):
                tip1 = fingertips[i]
                tip2 = fingertips[i+1]
                dist = np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2 + (tip1.z - tip2.z)**2)
                normalized_dist = dist / hand_size if hand_size > 0 else 0.0
                relative_features.append(normalized_dist)
            
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
                
                if v1_norm > 1e-6 and v2_norm > 1e-6:  # More conservative check
                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
                    angle = np.arccos(cos_angle)
                    # Check if angle is valid
                    if np.isnan(angle) or np.isinf(angle):
                        angle = 0.0
                    relative_features.append(angle)
                else:
                    relative_features.append(0.0)  # Default angle for invalid vectors
                    
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
                
                if v_current_norm > 1e-6 and v_next_norm > 1e-6:
                    cos_angle = np.dot(v_current, v_next) / (v_current_norm * v_next_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
                    angle = np.arccos(cos_angle)
                    # Check if angle is valid
                    if np.isnan(angle) or np.isinf(angle):
                        angle = 0.0
                    relative_features.append(angle)
                else:
                    relative_features.append(0.0)  # Default angle for invalid vectors
        
        # Combine all features
        all_features = np.concatenate([np.array(basic_features, dtype=np.float32), 
                                    np.array(relative_features, dtype=np.float32)])
        
        # Final check for NaN or inf values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply biometric enhancement if enabled
        if self.use_biometric_features:
            try:
                enhanced_features = self.biometric_enhancer.enhance_existing_features(
                    landmarks, all_features
                )
                return enhanced_features
            except Exception as e:
                print(f"Biometric enhancement failed: {e}")
                return all_features
        else:
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

    def reset_user_context(self):
        """Call this when switching between users"""
        if self.use_biometric_features:
            self.biometric_enhancer.reset_history()
    
    def enable_biometric_features(self, enable=True):
        """Enable or disable biometric feature enhancement"""
        self.use_biometric_features = enable
        if enable:
            print("✅ Biometric features enabled")
        else:
            print("❌ Biometric features disabled")
    
    def get_feature_info(self):
        """Get information about current feature configuration"""
        if self.use_biometric_features:
            bio_features = self.biometric_enhancer.get_feature_names()
            total_additional = len(bio_features)
            return f"Enhanced features: Basic + {total_additional} biometric features"
        else:
            return "Basic features only"