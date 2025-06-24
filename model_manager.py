import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import sqlite3

class ModelManager:
    def __init__(self, model_path="gesture_model.pkl", scaler_path="gesture_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.confidence_threshold = 0.7
        self.min_samples_per_user = 20
        self.feature_length = None
        
        # Load existing model if available
        self.load_model()

    def load_model(self):
        """Load the trained model and scaler from file"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Model and scaler loaded successfully")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
        return False

    def save_model(self):
        """Save the trained model and scaler to file"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print("Model and scaler saved successfully")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def prepare_training_data(self, conn, include_negatives=True):
        """Prepare training data from database, including negative samples"""
        cursor = conn.cursor()
        
        # Get positive samples (authorized users)
        cursor.execute("""
            SELECT gs.feature_data, gs.user_id, u.username 
            FROM gesture_samples gs 
            JOIN users u ON gs.user_id = u.user_id 
            WHERE u.username != 'UNAUTHORIZED_SAMPLES' 
            AND (gs.is_negative IS NULL OR gs.is_negative = 0)
        """)
        positive_samples = cursor.fetchall()
        
        if not positive_samples:
            print("No positive training samples found")
            return None, None
        
        # Prepare positive data
        X_positive = []
        y_positive = []
        user_mapping = {}
        user_counter = 0
        
        for sample_data, user_id, username in positive_samples:
            try:
                features = np.frombuffer(sample_data, dtype=np.float32)
                if self.feature_length is None:
                    self.feature_length = len(features)
                elif len(features) != self.feature_length:
                    print(f"Feature length mismatch: expected {self.feature_length}, got {len(features)}")
                    continue
                
                if user_id not in user_mapping:
                    user_mapping[user_id] = user_counter
                    user_counter += 1
                
                X_positive.append(features)
                y_positive.append(user_mapping[user_id])
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if not X_positive:
            print("No valid positive samples found")
            return None, None
        
        X_positive = np.array(X_positive)
        y_positive = np.array(y_positive)
        
        print(f"Loaded {len(X_positive)} positive samples for {len(user_mapping)} users")
        
        # Get negative samples if requested
        if include_negatives:
            cursor.execute("""
                SELECT gs.feature_data 
                FROM gesture_samples gs 
                JOIN users u ON gs.user_id = u.user_id 
                WHERE u.username = 'UNAUTHORIZED_SAMPLES' 
                OR gs.is_negative = 1
            """)
            negative_samples = cursor.fetchall()
            
            if negative_samples:
                X_negative = []
                for sample_data, in negative_samples:
                    try:
                        features = np.frombuffer(sample_data, dtype=np.float32)
                        if len(features) == self.feature_length:
                            X_negative.append(features)
                    except Exception as e:
                        print(f"Error processing negative sample: {e}")
                        continue
                
                if X_negative:
                    X_negative = np.array(X_negative)
                    # Assign a special class label for negative samples
                    y_negative = np.full(len(X_negative), -1)  # -1 for unauthorized
                    
                    # Combine positive and negative data
                    X_combined = np.vstack([X_positive, X_negative])
                    y_combined = np.concatenate([y_positive, y_negative])
                    
                    print(f"Added {len(X_negative)} negative samples")
                    return X_combined, y_combined, user_mapping
        
        return X_positive, y_positive, user_mapping

    def train_model(self, conn):
        """Train the model with both positive and negative samples"""
        print("Starting model training with negative learning support...")
        
        # Prepare training data
        data = self.prepare_training_data(conn, include_negatives=True)
        if data is None:
            print("Failed to prepare training data")
            return False
        
        X, y, user_mapping = data
        
        if len(X) < 10:  # Minimum samples required
            print("Not enough training samples")
            return False
        
        # Create and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Create model with probability estimates for confidence scoring
        # Using RandomForest as it handles multi-class + outlier detection well
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced positive/negative samples
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model training completed. Accuracy: {accuracy:.3f}")
        
        # Print classification report if we have enough test samples
        if len(y_test) > 5:
            unique_classes = np.unique(y_test)
            class_names = []
            for cls in unique_classes:
                if cls == -1:
                    class_names.append("UNAUTHORIZED")
                else:
                    # Find username for this class
                    for user_id, mapped_id in user_mapping.items():
                        if mapped_id == cls:
                            cursor = conn.cursor()
                            cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                            result = cursor.fetchone()
                            if result:
                                class_names.append(result[0])
                                break
                    else:
                        class_names.append(f"User_{cls}")
            
            print("\nClassification Report:")
            try:
                print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
            except:
                print("Could not generate detailed classification report")
        
        # Save the model
        self.save_model()
        
        # Store user mapping for prediction
        self.user_mapping = user_mapping
        return True

    def retrain_model(self, conn):
        """Retrain the model (alias for train_model for compatibility)"""
        return self.train_model(conn)

    def retrain_with_negatives(self, conn):
        """Explicitly retrain with emphasis on negative samples"""
        return self.train_model(conn)

    def predict(self, features):
        """Predict user from features with negative sample detection"""
        if self.model is None or self.scaler is None:
            print("Model not trained yet")
            return None
        
        try:
            # Ensure features are the right shape and type
            if len(features) != self.feature_length:
                print(f"Feature length mismatch: expected {self.feature_length}, got {len(features)}")
                return None
            
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = self.model.predict(features_scaled)[0]
            
            # Handle negative class prediction
            if predicted_class == -1:
                print("Gesture classified as unauthorized")
                return None  # Unauthorized gesture detected
            
            # Get confidence for the predicted class
            class_indices = self.model.classes_
            if predicted_class in class_indices:
                class_idx = np.where(class_indices == predicted_class)[0][0]
                confidence = probabilities[class_idx]
            else:
                confidence = 0.0
            
            # Check if there's a strong negative class prediction
            if -1 in class_indices:
                negative_idx = np.where(class_indices == -1)[0][0]
                negative_confidence = probabilities[negative_idx]
                
                # If negative confidence is high, reject even if positive class has higher score
                if negative_confidence > 0.3:  # Threshold for negative detection
                    print(f"High negative confidence: {negative_confidence:.3f}, rejecting gesture")
                    return None
            
            # Find the corresponding user_id
            user_id = None
            for uid, mapped_id in self.user_mapping.items():
                if mapped_id == predicted_class:
                    user_id = uid
                    break
            
            if user_id is None:
                print("Could not map predicted class to user ID")
                return None
            
            print(f"Predicted user_id: {user_id}, confidence: {confidence:.3f}")
            return user_id, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_confident_prediction(self, confidence):
        """Check if prediction confidence is above threshold"""
        return confidence >= self.confidence_threshold

    def update_settings(self, min_samples=None, confidence_threshold=None):
        """Update model settings"""
        if min_samples is not None:
            self.min_samples_per_user = min_samples
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        print(f"Settings updated: min_samples={self.min_samples_per_user}, confidence_threshold={self.confidence_threshold}")

    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return "No model trained"
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_length': self.feature_length,
            'confidence_threshold': self.confidence_threshold,
            'min_samples_per_user': self.min_samples_per_user,
            'classes': len(self.model.classes_) if hasattr(self.model, 'classes_') else 'Unknown'
        }
        return info

    def load_or_create_model(self, conn):
        """Load existing model or create new one if needed"""
        if not self.load_model():
            print("No existing model found, training new model...")
            return self.train_model(conn)
        return True