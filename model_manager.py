import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

class ModelManager:
    def __init__(self, min_samples_per_user=15, confidence_threshold=0.95):
        self.model = None
        self.scaler = StandardScaler()
        self.min_samples_per_user = min_samples_per_user
        self.confidence_threshold = confidence_threshold
        self.negative_examples = {}
        
        # Try to load existing model
        try:
            self.model = joblib.load('gesture_model.joblib')
            self.scaler = joblib.load('feature_scaler.joblib')
            self.negative_examples = joblib.load('negative_examples.joblib')
            print("Loaded existing model with negative examples")
        except:
            print("No existing model found, will create a new one")

    def retrain_model(self, conn):
        """Retrain the model with all available data including negative samples"""
        cursor = conn.cursor()
        
        # Get positive samples
        cursor.execute(
            """
            SELECT users.user_id, gesture_samples.feature_data 
            FROM users 
            JOIN gesture_samples ON users.user_id = gesture_samples.user_id
            """
        )
        positive_data = cursor.fetchall()
        
        # Get negative samples
        cursor.execute(
            """
            SELECT user_id, feature_data 
            FROM negative_samples
            """
        )
        negative_data = cursor.fetchall()

        if not positive_data:
            print("No training data available")
            return
        
        users_samples = {}
        for user_id, _ in positive_data:
            if user_id not in users_samples:
                users_samples[user_id] = 0
            users_samples[user_id] += 1
        
        # Check if we have enough samples per user
        min_samples = min(users_samples.values()) if users_samples else 0
        if min_samples < self.min_samples_per_user:
            print(f"Warning: Some users have only {min_samples} samples, which may be insufficient")
        
        X = []
        y = []
        sample_weights = []
        
        # Process positive samples
        for sample in positive_data:
            user_id, feature_data = sample
            try:
                features = np.frombuffer(feature_data, dtype=np.float32)
                if len(features) > 0:  # Make sure we have valid features
                    # Clean features to remove any NaN or inf values
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    X.append(features)
                    y.append(user_id)
                    sample_weights.append(1.0)  # Normal weight for positive samples
            except Exception as e:
                print(f"Error processing positive sample: {e}")
        
        # Process negative samples - store in negative_examples dict
        self.negative_examples = {}
        for sample in negative_data:
            user_id, feature_data = sample
            try:
                features = np.frombuffer(feature_data, dtype=np.float32)
                if len(features) > 0:
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    if user_id not in self.negative_examples:
                        self.negative_examples[user_id] = []
                    self.negative_examples[user_id].append(features)
            except Exception as e:
                print(f"Error processing negative sample: {e}")
        
        if len(X) == 0:
            print("Not enough valid samples to train model")
            return
        
        # Find the most common feature length BEFORE creating numpy array
        feature_lengths = [len(features) for features in X]
        if not feature_lengths:
            print("No valid features found")
            return
            
        common_length = max(set(feature_lengths), key=feature_lengths.count)
        print(f"Feature lengths found: {set(feature_lengths)}")
        print(f"Using common length: {common_length}")
        
        # Process negative examples to match common length
        processed_negative_examples = {}
        for user_id, neg_features_list in self.negative_examples.items():
            processed_list = []
            for features in neg_features_list:
                features_array = np.array(features, dtype=np.float32)
                if len(features_array) < common_length:
                    features_array = np.pad(features_array, (0, common_length - len(features_array)), mode='constant')
                elif len(features_array) > common_length:
                    features_array = features_array[:common_length]
                features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
                processed_list.append(features_array)
            processed_negative_examples[user_id] = processed_list
        self.negative_examples = processed_negative_examples
        
        # Pad or truncate ALL features to common length BEFORE array conversion
        X_processed = []
        for i, features in enumerate(X):
            features_array = np.array(features, dtype=np.float32)
            if len(features_array) < common_length:
                features_array = np.pad(features_array, (0, common_length - len(features_array)), mode='constant')
            elif len(features_array) > common_length:
                features_array = features_array[:common_length]
            # Clean again after processing
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            X_processed.append(features_array)
        
        # Now all features should have the same length
        try:
            X_processed = np.array(X_processed, dtype=np.float32)
        except ValueError as e:
            print(f"Error creating numpy array: {e}")
            return
            
        # Check if we have at least two different classes
        if len(set(y)) < 2:
            print("Need at least two different users to train the model")
            # Save the scaled features but skip training until we have 2+ users
            if X_processed.size > 0:
                # Final check for NaN/inf values
                if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                    print("Still found NaN/inf after cleaning, replacing with zeros")
                    X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Scale features
                self.scaler = StandardScaler()
                self.scaler.fit(X_processed)
                joblib.dump(self.scaler, 'feature_scaler.joblib')
                joblib.dump(self.negative_examples, 'negative_examples.joblib')
            return

        # Continue with model training
        # Final validation before scaling
        if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
            print("Found NaN/inf in processed features, cleaning again...")
            X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale features
        self.scaler = StandardScaler()
        try:
            X_scaled = self.scaler.fit_transform(X_processed)
            
            # Check if scaling introduced NaN/inf
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                print("Scaling introduced NaN/inf, cleaning...")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                
        except Exception as e:
            print(f"Error during scaling: {e}")
            print("Using original features without scaling")
            X_scaled = X_processed
        
        # Scale negative examples too
        for user_id in self.negative_examples:
            scaled_negatives = []
            for neg_features in self.negative_examples[user_id]:
                try:
                    scaled_neg = self.scaler.transform([neg_features])[0]
                    scaled_negatives.append(scaled_neg)
                except:
                    scaled_negatives.append(neg_features)
            self.negative_examples[user_id] = scaled_negatives
        
        # Print diagnostics
        print(f"Training model with {len(X_scaled)} positive samples from {len(set(y))} users")
        print(f"Negative samples: {sum(len(v) for v in self.negative_examples.values())} total")
        print(f"Feature length: {common_length}")
        print(f"Feature range: min={np.min(X_scaled):.4f}, max={np.max(X_scaled):.4f}")
        
        # Train model with RandomForest (better for handling negative examples)
        self.model = RandomForestClassifier(n_estimators=150, 
                                          max_depth=None,
                                          class_weight='balanced',
                                          n_jobs=-1)
        
        # Train the model with error handling
        try:
            self.model.fit(X_scaled, y)
            print("Model training successful")
        except Exception as e:
            print(f"Error training model: {e}")
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
            self.model = None
            return
        
        # Save the model, scaler, and negative examples
        if self.model is not None:
            joblib.dump(self.model, 'gesture_model.joblib')
            joblib.dump(self.scaler, 'feature_scaler.joblib')
            joblib.dump(self.negative_examples, 'negative_examples.joblib')
            print("Model training complete with negative samples incorporated")

    def predict(self, features):
        """Make a prediction with confidence score, considering negative examples"""
        if self.model is None:
            return None, 0.0
            
        try:
            # Process features to match expected format
            if hasattr(self.model, 'n_features_in_'):
                expected_length = self.model.n_features_in_
            elif hasattr(self.model, 'support_vectors_') and len(self.model.support_vectors_) > 0:
                expected_length = len(self.model.support_vectors_[0])
            else:
                print("Cannot determine expected feature length from model")
                return None, 0.0
                
            # Pad or truncate features to match expected length
            if len(features) < expected_length:
                features = np.pad(features, (0, expected_length - len(features)), mode='constant')
            elif len(features) > expected_length:
                features = features[:expected_length]
            
            # Scale features
            features_scaled = self.scaler.transform([features])[0]
                
            # Make prediction with probability
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([features_scaled])[0]
                max_prob = np.max(probabilities)
                user_id = int(self.model.classes_[np.argmax(probabilities)])
                confidence = max_prob
            else:
                # Fallback if probabilities not available
                user_id = int(self.model.predict([features_scaled])[0])
                confidence = 1.0
            
            # Check against negative examples with nuanced penalty
            if hasattr(self, 'negative_examples') and user_id in self.negative_examples:
                similarity_scores = []
                for neg_example in self.negative_examples[user_id]:
                    similarity = np.linalg.norm(features_scaled - neg_example)
                    similarity_scores.append(similarity)
                
                # Apply graduated penalty based on similarity to negative examples
                if similarity_scores:
                    min_similarity = min(similarity_scores)
                    if min_similarity < 3.0:  # Threshold for similarity
                        # Graduated penalty - less aggressive than before
                        penalty = max(0.5, min_similarity / 3.0)
                        confidence *= penalty
                        print(f"Applied negative sample penalty: {penalty:.2f} (similarity: {min_similarity:.2f})")
            
            return user_id, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def calculate_similarity_weight(self, feature1, feature2):
        """Calculate weighted similarity between two feature vectors"""
        # Assume the first 63 features are position-based, the rest are angles
        if len(feature1) <= 63:
            return np.linalg.norm(feature1 - feature2)
            
        # Split features into position and angle components
        pos_feat1, angle_feat1 = feature1[:63], feature1[63:]
        pos_feat2, angle_feat2 = feature2[:63], feature2[63:]
        
        # Calculate weighted distance
        pos_dist = np.linalg.norm(pos_feat1 - pos_feat2)
        angle_dist = np.linalg.norm(angle_feat1 - angle_feat2)
        
        # Weight angles more heavily
        return pos_dist + 2.0 * angle_dist

    def add_negative_sample(self, conn, user_id, features):
        """Add a negative sample to the database"""
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO negative_samples (user_id, feature_data) VALUES (?, ?)", 
            (user_id, features.tobytes())
        )
        conn.commit()

    def is_confident_prediction(self, confidence):
        """Check if prediction meets confidence threshold"""
        return confidence >= self.confidence_threshold

    def update_settings(self, min_samples=None, confidence_threshold=None):
        """Update model settings"""
        if min_samples is not None:
            self.min_samples_per_user = min_samples
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold