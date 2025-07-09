import numpy as np
from collections import deque

class DistanceInvariantFeatures:
    """
    Extract biometric features that are invariant to camera distance
    Focus on angles, ratios, and geometric relationships
    """
    
    def __init__(self):
        self.recent_landmarks = deque(maxlen=5)
        
    def extract_distance_invariant_features(self, landmarks):
        """Extract features that don't depend on absolute distances"""
        if len(landmarks) < 21:
            return np.zeros(35)  # Fixed feature length
        
        features = []
        
        # Key landmarks
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # MCP joints
        thumb_mcp = landmarks[1]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # PIP joints
        thumb_pip = landmarks[2]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        
        # === 1. FINGER BONE RATIOS (Distance Invariant) ===
        finger_ratios = self.calculate_finger_bone_ratios(landmarks)
        features.extend(finger_ratios)  # 15 features
        
        # === 2. JOINT ANGLES (Completely Distance Invariant) ===
        joint_angles = self.calculate_joint_angles(landmarks)
        features.extend(joint_angles)  # 10 features
        
        # === 3. FINGER SPREAD ANGLES (Distance Invariant) ===
        spread_angles = self.calculate_finger_spread_angles(landmarks)
        features.extend(spread_angles)  # 4 features
        
        # === 4. HAND CURVATURE PATTERNS (Distance Invariant) ===
        curvature_features = self.calculate_hand_curvature(landmarks)
        features.extend(curvature_features)  # 3 features
        
        # === 5. RELATIVE POSITIONING (Normalized by Palm) ===
        relative_features = self.calculate_relative_positioning(landmarks)
        features.extend(relative_features)  # 3 features
        
        return np.array(features[:35], dtype=np.float32)
    
    def calculate_finger_bone_ratios(self, landmarks):
        """Calculate ratios between finger bone segments (distance invariant)"""
        features = []
        
        # For each finger, calculate bone segment ratios
        finger_segments = [
            # [MCP, PIP, DIP, TIP]
            [landmarks[1], landmarks[2], landmarks[3], landmarks[4]],   # Thumb
            [landmarks[5], landmarks[6], landmarks[7], landmarks[8]],   # Index
            [landmarks[9], landmarks[10], landmarks[11], landmarks[12]], # Middle
            [landmarks[13], landmarks[14], landmarks[15], landmarks[16]], # Ring
            [landmarks[17], landmarks[18], landmarks[19], landmarks[20]]  # Pinky
        ]
        
        for finger in finger_segments:
            # Calculate segment lengths
            seg1 = self.distance_3d(finger[0], finger[1])  # Proximal phalanx
            seg2 = self.distance_3d(finger[1], finger[2])  # Middle phalanx
            seg3 = self.distance_3d(finger[2], finger[3])  # Distal phalanx
            
            # Calculate ratios (distance invariant)
            if seg1 > 0:
                features.append(seg2 / seg1)  # Middle/Proximal ratio
                features.append(seg3 / seg1)  # Distal/Proximal ratio
            else:
                features.extend([1.0, 1.0])
                
            if seg2 > 0:
                features.append(seg3 / seg2)  # Distal/Middle ratio
            else:
                features.append(1.0)
        
        return features  # 15 features (5 fingers × 3 ratios)
    
    def calculate_joint_angles(self, landmarks):
        """Calculate finger joint angles (completely distance invariant)"""
        features = []
        
        # Finger joint sequences
        finger_joints = [
            [landmarks[1], landmarks[2], landmarks[3]],   # Thumb joints
            [landmarks[5], landmarks[6], landmarks[7]],   # Index joints
            [landmarks[9], landmarks[10], landmarks[11]], # Middle joints
            [landmarks[13], landmarks[14], landmarks[15]], # Ring joints
            [landmarks[17], landmarks[18], landmarks[19]]  # Pinky joints
        ]
        
        for joints in finger_joints:
            for i in range(len(joints) - 2):
                # Calculate angle at middle joint
                angle = self.calculate_angle(joints[i], joints[i+1], joints[i+2])
                features.append(angle)
        
        return features  # 10 features (5 fingers × 2 angles)
    
    def calculate_finger_spread_angles(self, landmarks):
        """Calculate angles between adjacent fingers (distance invariant)"""
        features = []
        
        wrist = landmarks[0]
        finger_mcps = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]  # Index to Pinky MCPs
        
        # Calculate angles between adjacent fingers
        for i in range(len(finger_mcps) - 1):
            vec1 = np.array([finger_mcps[i].x - wrist.x, finger_mcps[i].y - wrist.y])
            vec2 = np.array([finger_mcps[i+1].x - wrist.x, finger_mcps[i+1].y - wrist.y])
            
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features.append(angle)
        
        return features  # 4 features
    
    def calculate_hand_curvature(self, landmarks):
        """Calculate hand curvature patterns (distance invariant)"""
        features = []
        
        # Palm curvature using knuckle line
        knuckles = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]  # MCP joints
        
        # Calculate curvature of knuckle line
        if len(knuckles) >= 4:
            # Use three points to calculate curvature
            for i in range(len(knuckles) - 2):
                curvature = self.calculate_curvature(knuckles[i], knuckles[i+1], knuckles[i+2])
                features.append(curvature)
        
        return features  # 3 features
    
    def calculate_relative_positioning(self, landmarks):
        """Calculate relative positions normalized by palm triangle"""
        features = []
        
        # Use palm triangle as reference (wrist, index MCP, pinky MCP)
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        # Calculate palm triangle area (for normalization)
        palm_area = self.triangle_area(wrist, index_mcp, pinky_mcp)
        
        if palm_area > 0:
            # Thumb position relative to palm triangle
            thumb_tip = landmarks[4]
            thumb_area = self.triangle_area(wrist, index_mcp, thumb_tip)
            features.append(thumb_area / palm_area)
            
            # Middle finger position relative to palm triangle
            middle_tip = landmarks[12]
            middle_area = self.triangle_area(wrist, index_mcp, middle_tip)
            features.append(middle_area / palm_area)
            
            # Ring finger position relative to palm triangle
            ring_tip = landmarks[16]
            ring_area = self.triangle_area(wrist, pinky_mcp, ring_tip)
            features.append(ring_area / palm_area)
        else:
            features.extend([1.0, 1.0, 1.0])
        
        return features  # 3 features
    
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
    
    def calculate_curvature(self, p1, p2, p3):
        """Calculate curvature at p2 using three points"""
        # Simplified curvature calculation
        a = self.distance_3d(p1, p2)
        b = self.distance_3d(p2, p3)
        c = self.distance_3d(p1, p3)
        
        if a > 0 and b > 0 and c > 0:
            # Use triangle area to approximate curvature
            s = (a + b + c) / 2  # Semi-perimeter
            area = max(0, s * (s - a) * (s - b) * (s - c))  # Heron's formula
            area = np.sqrt(area)
            
            # Curvature approximation
            curvature = 4 * area / (a * b * c) if (a * b * c) > 0 else 0
            return curvature
        return 0.0
    
    def triangle_area(self, p1, p2, p3):
        """Calculate area of triangle formed by three points"""
        # Using cross product for area calculation
        v1 = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        v2 = np.array([p3.x - p1.x, p3.y - p1.y, p3.z - p1.z])
        
        cross = np.cross(v1, v2)
        if cross.ndim == 0:  # 2D cross product
            return abs(cross) / 2
        else:  # 3D cross product
            return np.linalg.norm(cross) / 2
    
    def get_feature_names(self):
        """Return names of all distance-invariant features"""
        names = []
        
        # Finger bone ratios (15 features)
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for finger in fingers:
            names.extend([f'{finger}_mid_prox_ratio', f'{finger}_dis_prox_ratio', f'{finger}_dis_mid_ratio'])
        
        # Joint angles (10 features)
        for finger in fingers:
            names.extend([f'{finger}_angle1', f'{finger}_angle2'])
        
        # Finger spread angles (4 features)
        spreads = ['index_middle', 'middle_ring', 'ring_pinky']
        for spread in spreads:
            names.append(f'{spread}_spread_angle')
        
        # Hand curvature (3 features)
        names.extend(['palm_curvature1', 'palm_curvature2', 'palm_curvature3'])
        
        # Relative positioning (3 features)
        names.extend(['thumb_palm_ratio', 'middle_palm_ratio', 'ring_palm_ratio'])
        
        return names