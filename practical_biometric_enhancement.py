import numpy as np
from collections import deque
from distance_invariant_features import DistanceInvariantFeatures

class PracticalBiometricEnhancement:
    """
    Enhanced biometric features using distance-invariant characteristics
    No hand size dependencies - works at any camera distance
    """
    
    def __init__(self):
        # Store recent frames for temporal analysis
        self.recent_landmarks = deque(maxlen=5)
        self.distance_invariant_extractor = DistanceInvariantFeatures()
        
    def enhance_existing_features(self, landmarks, basic_features):
        """
        Add distance-invariant biometric features to existing feature vector
        
        Args:
            landmarks: MediaPipe hand landmarks
            basic_features: Your existing feature vector
            
        Returns:
            Enhanced feature vector with distance-invariant biometric features
        """
        # Store landmarks for temporal analysis
        self.recent_landmarks.append(landmarks)
        
        # Extract distance-invariant features
        distance_invariant_features = self.distance_invariant_extractor.extract_distance_invariant_features(landmarks)
        
        # Extract temporal consistency features
        temporal_features = self.extract_temporal_consistency()
        
        # Combine all features
        enhanced_features = np.concatenate([
            basic_features,                    # Your existing features
            distance_invariant_features,       # Distance-invariant biometric features (35)
            temporal_features                  # Temporal consistency features (5)
        ])
        
        return enhanced_features
    
    def extract_temporal_consistency(self):
        """Extract temporal consistency without hand size dependency"""
        if len(self.recent_landmarks) < 3:
            return np.zeros(5)  # Not enough data yet
        
        features = []
        
        # 1. ANGULAR STABILITY (how consistent are finger angles)
        if len(self.recent_landmarks) >= 3:
            angular_stability = self.calculate_angular_stability()
            features.append(angular_stability)
        else:
            features.append(0.0)
        
        # 2. GESTURE SHAPE CONSISTENCY (how consistent is overall hand shape)
        if len(self.recent_landmarks) >= 3:
            shape_consistency = self.calculate_shape_consistency()
            features.append(shape_consistency)
        else:
            features.append(1.0)
        
        # 3. FINGER RATIO STABILITY (how consistent are finger ratios)
        if len(self.recent_landmarks) >= 3:
            ratio_stability = self.calculate_ratio_stability()
            features.append(ratio_stability)
        else:
            features.append(0.0)
        
        # 4. JOINT ANGLE CONSISTENCY (how consistent are joint angles)
        if len(self.recent_landmarks) >= 3:
            joint_consistency = self.calculate_joint_consistency()
            features.append(joint_consistency)
        else:
            features.append(0.0)
        
        # 5. OVERALL GESTURE STABILITY (combined stability metric)
        if len(self.recent_landmarks) >= 3:
            overall_stability = np.mean(features[:4])  # Average of other stabilities
            features.append(overall_stability)
        else:
            features.append(1.0)
        
        return np.array(features, dtype=np.float32)
    
    def calculate_angular_stability(self):
        """Calculate how stable finger angles are across frames"""
        recent_frames = list(self.recent_landmarks)[-3:]
        
        angle_variations = []
        for i in range(len(recent_frames) - 1):
            landmarks1 = recent_frames[i]
            landmarks2 = recent_frames[i + 1]
            
            if len(landmarks1) >= 21 and len(landmarks2) >= 21:
                # Compare key angles between frames
                angles1 = self.extract_key_angles(landmarks1)
                angles2 = self.extract_key_angles(landmarks2)
                
                # Calculate angular differences
                angle_diff = np.abs(np.array(angles1) - np.array(angles2))
                angle_variations.extend(angle_diff)
        
        if angle_variations:
            # Low variation = high stability
            stability = 1.0 / (1.0 + np.mean(angle_variations))
            return stability
        return 1.0
    
    def calculate_shape_consistency(self):
        """Calculate how consistent the overall hand shape is"""
        recent_frames = list(self.recent_landmarks)[-3:]
        
        shape_similarities = []
        for i in range(len(recent_frames) - 1):
            landmarks1 = recent_frames[i]
            landmarks2 = recent_frames[i + 1]
            
            if len(landmarks1) >= 21 and len(landmarks2) >= 21:
                # Compare shape features
                shape1 = self.extract_shape_features(landmarks1)
                shape2 = self.extract_shape_features(landmarks2)
                
                # Calculate shape similarity
                similarity = self.calculate_feature_similarity(shape1, shape2)
                shape_similarities.append(similarity)
        
        return np.mean(shape_similarities) if shape_similarities else 1.0
    
    def calculate_ratio_stability(self):
        """Calculate how stable finger ratios are"""
        recent_frames = list(self.recent_landmarks)[-3:]
        
        ratio_variations = []
        for i in range(len(recent_frames) - 1):
            landmarks1 = recent_frames[i]
            landmarks2 = recent_frames[i + 1]
            
            if len(landmarks1) >= 21 and len(landmarks2) >= 21:
                # Compare finger ratios
                ratios1 = self.extract_finger_ratios(landmarks1)
                ratios2 = self.extract_finger_ratios(landmarks2)
                
                # Calculate ratio differences
                ratio_diff = np.abs(np.array(ratios1) - np.array(ratios2))
                ratio_variations.extend(ratio_diff)
        
        if ratio_variations:
            # Low variation = high stability
            stability = 1.0 / (1.0 + np.mean(ratio_variations) * 10)
            return stability
        return 1.0
    
    def calculate_joint_consistency(self):
        """Calculate how consistent joint angles are"""
        recent_frames = list(self.recent_landmarks)[-3:]
        
        joint_variations = []
        for i in range(len(recent_frames) - 1):
            landmarks1 = recent_frames[i]
            landmarks2 = recent_frames[i + 1]
            
            if len(landmarks1) >= 21 and len(landmarks2) >= 21:
                # Compare joint angles
                joints1 = self.extract_joint_angles(landmarks1)
                joints2 = self.extract_joint_angles(landmarks2)
                
                # Calculate joint differences
                joint_diff = np.abs(np.array(joints1) - np.array(joints2))
                joint_variations.extend(joint_diff)
        
        if joint_variations:
            # Low variation = high stability
            stability = 1.0 / (1.0 + np.mean(joint_variations) * 5)
            return stability
        return 1.0
    
    def extract_key_angles(self, landmarks):
        """Extract key angles for comparison"""
        angles = []
        
        # Finger spread angles
        wrist = landmarks[0]
        finger_mcps = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        for i in range(len(finger_mcps) - 1):
            vec1 = np.array([finger_mcps[i].x - wrist.x, finger_mcps[i].y - wrist.y])
            vec2 = np.array([finger_mcps[i+1].x - wrist.x, finger_mcps[i+1].y - wrist.y])
            
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
        
        return angles
    
    def extract_shape_features(self, landmarks):
        """Extract shape features for comparison"""
        return self.distance_invariant_extractor.extract_distance_invariant_features(landmarks)
    
    def extract_finger_ratios(self, landmarks):
        """Extract finger ratios for comparison"""
        ratios = []
        
        # Finger bone ratios
        finger_segments = [
            [landmarks[5], landmarks[6], landmarks[7], landmarks[8]],   # Index
            [landmarks[9], landmarks[10], landmarks[11], landmarks[12]], # Middle
            [landmarks[13], landmarks[14], landmarks[15], landmarks[16]], # Ring
            [landmarks[17], landmarks[18], landmarks[19], landmarks[20]]  # Pinky
        ]
        
        for finger in finger_segments:
            seg1 = self.distance_3d(finger[0], finger[1])
            seg2 = self.distance_3d(finger[1], finger[2])
            
            if seg1 > 0:
                ratios.append(seg2 / seg1)
            else:
                ratios.append(1.0)
        
        return ratios
    
    def extract_joint_angles(self, landmarks):
        """Extract joint angles for comparison"""
        angles = []
        
        # Key joint angles
        finger_joints = [
            [landmarks[5], landmarks[6], landmarks[7]],   # Index
            [landmarks[9], landmarks[10], landmarks[11]], # Middle
            [landmarks[13], landmarks[14], landmarks[15]], # Ring
            [landmarks[17], landmarks[18], landmarks[19]]  # Pinky
        ]
        
        for joints in finger_joints:
            angle = self.calculate_angle(joints[0], joints[1], joints[2])
            angles.append(angle)
        
        return angles
    
    def distance_3d(self, p1, p2):
        """Calculate 3D distance between two points"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 1e-8 and v2_norm > 1e-8:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return angle if not (np.isnan(angle) or np.isinf(angle)) else 0.0
        return 0.0
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        if len(features1) != len(features2):
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            return max(0, similarity)  # Ensure non-negative
        return 0.0
    
    def reset_history(self):
        """Reset temporal history (call between different users)"""
        self.recent_landmarks.clear()
    
    def get_feature_names(self):
        """Return names of all distance-invariant biometric features"""
        invariant_names = self.distance_invariant_extractor.get_feature_names()
        temporal_names = [
            'angular_stability',
            'shape_consistency', 
            'ratio_stability',
            'joint_consistency',
            'overall_stability'
        ]
        
        return invariant_names + temporal_names