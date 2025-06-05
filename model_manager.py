import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

class ModelManager:
    def __init__(self, min_samples_per_user=15, confidence_threshold=0.7):
        self.model = None
        self.scaler = StandardScaler()
        self.min_samples_per_user = min_samples_per_user
        self.confidence_threshold = confidence_threshold
        
        # Try to load existing model
        try:
            self.model = joblib.load('gesture_model.joblib')
            self.scaler = joblib.load('feature_scaler.joblib')
            print("Loaded existing model")
        except:
            print("No existing model found, will create a new one")

    def retrain_model(self, conn):
        """Retrain the model with all available data"""
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT users.user_id, gesture_samples.feature_data 
            FROM users 
            JOIN gesture_samples ON users.user_id = gesture_samples.user_id
            """
        )
        data = cursor.fetchall()

        if not data:
            print("No training data available")
            return
        
        users_samples = {}
        for user_id, feature_data in data:
            if user_id not in users_samples:
                users_samples[user_id] = 0
            users_samples[user_id] += 1
        
        # Check if we have enough samples per user
        min_samples = min(users_samples.values()) if users_samples else 0
        if min_samples < self.min_samples_per_user:
            print(f"Warning: Some users have only {min_samples} samples, which may be insufficient")
        
        X = []
        y = []
        
        # Process each sample
        for sample in data:
            user_id, feature_data = sample
            try:
                features = np.frombuffer(feature_data, dtype=np.float32)
                if len(features) > 0:  # Make sure we have valid features
                    # Clean features to remove any NaN or inf values
                    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                    X.append(features)
                    y.append(user_id)
            except Exception as e:
                print(f"Error processing sample: {e}")
        
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
            # Debug: print shapes of first few features
            for i in range(min(5, len(X_processed))):
                print(f"Feature {i} shape: {X_processed[i].shape}")
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
        
        # Print diagnostics
        print(f"Training model with {len(X_scaled)} samples from {len(set(y))} users")
        print(f"Feature length: {common_length}")
        print(f"Feature range: min={np.min(X_scaled):.4f}, max={np.max(X_scaled):.4f}")
        
        # Evaluate model with cross-validation 
        if len(set(y)) > 1 and len(X_scaled) >= 10:
            try:
                # Use both SVC and RandomForest for comparison
                svc_model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
                
                # Try cross-validation with better error handling
                try:
                    svc_scores = cross_val_score(svc_model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                    svc_mean = np.mean(svc_scores)
                except Exception as svc_error:
                    print(f"SVC cross-validation failed: {svc_error}")
                    svc_mean = 0.0
                    svc_scores = [0.0]
                
                try:
                    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=min(5, len(set(y))), scoring='accuracy')
                    rf_mean = np.mean(rf_scores)
                except Exception as rf_error:
                    print(f"RandomForest cross-validation failed: {rf_error}")
                    rf_mean = 0.0
                    rf_scores = [0.0]
                
                print(f"SVC cross-validation accuracy: {svc_mean:.2f} ± {np.std(svc_scores):.2f}")
                print(f"RandomForest cross-validation accuracy: {rf_mean:.2f} ± {np.std(rf_scores):.2f}")
                
                # Choose the better model
                if rf_mean > svc_mean:
                    self.model = rf_model
                    print("Using RandomForest classifier based on cross-validation")
                    model_accuracy = rf_mean
                else:
                    self.model = svc_model
                    print("Using SVC classifier based on cross-validation")
                    model_accuracy = svc_mean
                    
                # Set confidence threshold based on model accuracy
                self.confidence_threshold = min(0.9, max(0.6, 1.0 - 1.5 * (1.0 - model_accuracy)))
                print(f"Setting confidence threshold to {self.confidence_threshold:.2f}")
                
                if model_accuracy < 0.7:
                    print("Warning: Model accuracy is low. Consider collecting more varied samples.")
                    messagebox.showwarning("Warning", "Recognition accuracy may be low. Try adding more varied samples for each gesture.")
                    
            except Exception as e:
                print(f"Cross-validation error: {e}")
                # Fallback to RandomForest if cross-validation fails
                self.model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
        else:
            # If not enough data for cross-validation, use RandomForest as default
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
        
        # Train the model with error handling
        try:
            self.model.fit(X_scaled, y)
            print("Model training successful")
        except Exception as e:
            print(f"Error training model: {e}")
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
            self.model = None
            return
        
        # Save the model and scaler
        if self.model is not None:
            joblib.dump(self.model, 'gesture_model.joblib')
            joblib.dump(self.scaler, 'feature_scaler.joblib')
            print("Model training complete")

    def predict(self, features):
        """Make a prediction with confidence score"""
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
            features_scaled = self.scaler.transform([features])
                
            # Make prediction with probability
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                max_prob = np.max(probabilities)
                user_id = int(self.model.classes_[np.argmax(probabilities)])
                confidence = max_prob
            else:
                # Fallback if probabilities not available
                user_id = int(self.model.predict(features_scaled)[0])
                confidence = 1.0
                try:
                    decision_values = self.model.decision_function(features_scaled)
                    confidence = np.max(np.abs(decision_values)) / 10  # Scale to 0-1 range approximately
                except:
                    pass
            
            return user_id, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def is_confident_prediction(self, confidence):
        """Check if prediction meets confidence threshold"""
        return confidence >= self.confidence_threshold

    def update_settings(self, min_samples=None, confidence_threshold=None):
        """Update model settings"""
        if min_samples is not None:
            self.min_samples_per_user = min_samples
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold