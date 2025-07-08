import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sqlite3
import joblib
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelTesting:
    def __init__(self, db_path='gesture_auth.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.results = {}
        
        # Load existing model components if available
        try:
            self.base_model = joblib.load('gesture_model.joblib')
            self.base_scaler = joblib.load('feature_scaler.joblib')
            print("✅ Loaded existing model for baseline comparison")
        except:
            print("ℹ️  No existing model found - will train fresh models for testing")
            self.base_model = None
            self.base_scaler = None
    
    def load_user_data(self):
        """Load and prepare user data for analysis"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT users.user_id, users.username, gesture_samples.feature_data 
            FROM users 
            JOIN gesture_samples ON users.user_id = gesture_samples.user_id
        """)
        data = cursor.fetchall()
        
        # Organize data by user
        user_data = {}
        user_names = {}
        
        for user_id, username, feature_data in data:
            try:
                features = np.frombuffer(feature_data, dtype=np.float32)
                
                # Comprehensive data cleaning
                features = self.clean_features(features)
                
                if user_id not in user_data:
                    user_data[user_id] = []
                    user_names[user_id] = username
                
                user_data[user_id].append(features)
            except Exception as e:
                print(f"Error processing sample for user {username}: {e}")
                continue
        
        # Normalize feature lengths
        if user_data:
            all_features = [feat for user_feats in user_data.values() for feat in user_feats]
            if all_features:
                max_length = max(len(f) for f in all_features)
                
                for user_id in user_data:
                    normalized_features = []
                    for features in user_data[user_id]:
                        # Pad or truncate
                        if len(features) < max_length:
                            features = np.pad(features, (0, max_length - len(features)), mode='constant')
                        elif len(features) > max_length:
                            features = features[:max_length]
                        
                        # Clean again after padding
                        features = self.clean_features(features)
                        normalized_features.append(features)
                    user_data[user_id] = normalized_features
        
        return user_data, user_names
    
    def clean_features(self, features):
        """Comprehensive feature cleaning to handle problematic values"""
        # Convert to numpy array if not already
        features = np.array(features, dtype=np.float32)
        
        # Step 1: Replace NaN and infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Step 2: Clip extreme values that might cause overflow
        # Use a reasonable range for gesture features
        features = np.clip(features, -100.0, 100.0)
        
        # Step 3: Check for any remaining problematic values
        if not np.all(np.isfinite(features)):
            print("Warning: Found non-finite values after cleaning, replacing with zeros")
            features[~np.isfinite(features)] = 0.0
        
        # Step 4: Ensure no values are too large for float32
        max_float32 = np.finfo(np.float32).max
        features = np.clip(features, -max_float32/2, max_float32/2)
        
        return features
    
    def validate_array(self, array, name="array"):
        """Validate and clean array for scikit-learn compatibility"""
        array = np.array(array, dtype=np.float32)
        
        # Check for problematic values
        if not np.all(np.isfinite(array)):
            print(f"Warning: Found non-finite values in {name}, cleaning...")
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for extreme values
        max_val = np.max(np.abs(array))
        if max_val > 1e6:  # Very large values
            print(f"Warning: Found extreme values in {name} (max: {max_val}), clipping...")
            array = np.clip(array, -1e6, 1e6)
        
        # Final validation
        if not np.all(np.isfinite(array)):
            print(f"Error: Still found non-finite values in {name} after cleaning")
            array[~np.isfinite(array)] = 0.0
        
        return array
    
    def test_sample_size_effect(self, user_data, user_names, max_samples=100):
        """Test how the number of training samples affects recognition accuracy"""
        print("\n" + "="*60)
        print("📊 EXPERIMENT 1: EFFECT OF TRAINING SAMPLE SIZE")
        print("="*60)
        
        # Only test users with sufficient samples
        eligible_users = {uid: data for uid, data in user_data.items() if len(data) >= 20}
        
        if len(eligible_users) < 2:
            print("❌ Need at least 2 users with 20+ samples each for this test")
            return
        
        # Test with different sample sizes
        sample_sizes = list(range(5, min(max_samples, max(len(data) for data in eligible_users.values())), 5))
        
        results = {
            'sample_size': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'user_pair': []
        }
        
        print(f"Testing sample sizes: {sample_sizes}")
        print(f"Using users: {[user_names[uid] for uid in eligible_users.keys()]}")
        
        # Test all pairs of users
        for user1_id, user2_id in combinations(eligible_users.keys(), 2):
            user1_name = user_names[user1_id]
            user2_name = user_names[user2_id]
            pair_name = f"{user1_name} vs {user2_name}"
            
            print(f"\nTesting pair: {pair_name}")
            
            for sample_size in sample_sizes:
                # Prepare data for this sample size
                user1_samples = np.array(eligible_users[user1_id][:sample_size])
                user2_samples = np.array(eligible_users[user2_id][:sample_size])
                
                # Create training and test sets
                X = np.vstack([user1_samples, user2_samples])
                y = np.array([user1_id] * len(user1_samples) + [user2_id] * len(user2_samples))
                
                # Stratified split to ensure both users in train/test
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=42
                    )
                    
                    # Train model
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                    
                    results['sample_size'].append(sample_size)
                    results['accuracy'].append(accuracy)
                    results['precision'].append(precision)
                    results['recall'].append(recall)
                    results['f1'].append(f1)
                    results['user_pair'].append(pair_name)
                    
                    print(f"  Size {sample_size:2d}: Accuracy = {accuracy:.3f}")
                    
                except Exception as e:
                    print(f"  Size {sample_size:2d}: Failed - {e}")
        
        # Analysis and visualization
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            print(f"\n📈 SAMPLE SIZE ANALYSIS RESULTS:")
            
            # Average results across all user pairs
            avg_results = df_results.groupby('sample_size').agg({
                'accuracy': ['mean', 'std'],
                'precision': ['mean', 'std'],
                'recall': ['mean', 'std'],
                'f1': ['mean', 'std']
            }).round(3)
            
            print("\nAverage Performance Across All User Pairs:")
            print(avg_results)
            
            # Find minimum samples needed for good performance
            good_accuracy_threshold = 0.90
            sufficient_samples = df_results[df_results['accuracy'] >= good_accuracy_threshold]
            
            if not sufficient_samples.empty:
                min_samples_needed = sufficient_samples['sample_size'].min()
                print(f"\n🎯 KEY FINDING: Minimum samples needed for ≥{good_accuracy_threshold:.0%} accuracy: {min_samples_needed}")
            else:
                print(f"\n⚠️  No configuration achieved ≥{good_accuracy_threshold:.0%} accuracy")
            
            # Store results
            self.results['sample_size_effect'] = df_results
            
            return df_results
        else:
            print("❌ No valid results obtained")
            return None
    
    def test_confidence_threshold_effect(self, user_data, user_names):
        """Test how confidence threshold affects FAR and FRR"""
        print("\n" + "="*60)
        print("🎯 EXPERIMENT 2: CONFIDENCE THRESHOLD OPTIMIZATION")
        print("="*60)
        
        # Prepare data from all users
        all_users = list(user_data.keys())
        if len(all_users) < 2:
            print("❌ Need at least 2 users for confidence threshold testing")
            return
        
        # Create a comprehensive dataset
        X_all, y_all = [], []
        for user_id, features_list in user_data.items():
            for features in features_list:
                X_all.append(features)
                y_all.append(user_id)
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Split into train/test with validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.3, stratify=y_all, random_state=42
        )
        
        # Train model with enhanced validation
        scaler = StandardScaler()
        X_train = self.validate_array(X_train, "confidence_X_train")
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = self.validate_array(X_train_scaled, "confidence_X_train_scaled")
        
        X_test = self.validate_array(X_test, "confidence_X_test")
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = self.validate_array(X_test_scaled, "confidence_X_test_scaled")
        
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get prediction probabilities
        y_proba = model.predict_proba(X_test_scaled)
        y_pred_raw = model.predict(X_test_scaled)
        max_proba = np.max(y_proba, axis=1)
        
        # Test different confidence thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = {
            'threshold': [],
            'accuracy': [],
            'far': [],  # False Accept Rate
            'frr': [],  # False Reject Rate
            'accepted_ratio': [],
            'precision': [],
            'recall': []
        }
        
        print("Testing confidence thresholds...")
        
        for threshold in thresholds:
            # Apply threshold - reject predictions below threshold
            accepted_mask = max_proba >= threshold
            
            if np.sum(accepted_mask) == 0:
                continue  # No predictions accepted at this threshold
            
            # Calculate metrics only for accepted predictions
            y_test_accepted = y_test[accepted_mask]
            y_pred_accepted = y_pred_raw[accepted_mask]
            
            # Basic metrics
            accuracy = accuracy_score(y_test_accepted, y_pred_accepted)
            accepted_ratio = np.sum(accepted_mask) / len(y_test)
            
            # FAR and FRR calculation
            correct_predictions = (y_test_accepted == y_pred_accepted)
            incorrect_predictions = ~correct_predictions
            
            # False Accept Rate: incorrect predictions that were accepted
            far = np.sum(incorrect_predictions) / len(y_test) if len(y_test) > 0 else 0
            
            # False Reject Rate: correct predictions that were rejected due to low confidence
            rejected_mask = ~accepted_mask
            if np.sum(rejected_mask) > 0:
                # Estimate what would have been correct among rejected
                y_pred_rejected = y_pred_raw[rejected_mask]
                y_test_rejected = y_test[rejected_mask]
                would_be_correct = (y_test_rejected == y_pred_rejected)
                frr = np.sum(would_be_correct) / len(y_test) if len(y_test) > 0 else 0
            else:
                frr = 0
            
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test_accepted, y_pred_accepted, average='weighted', zero_division=0
            )
            
            results['threshold'].append(threshold)
            results['accuracy'].append(accuracy)
            results['far'].append(far)
            results['frr'].append(frr)
            results['accepted_ratio'].append(accepted_ratio)
            results['precision'].append(precision)
            results['recall'].append(recall)
            
            print(f"Threshold {threshold:.2f}: Acc={accuracy:.3f}, FAR={far:.3f}, FRR={frr:.3f}, Accept={accepted_ratio:.3f}")
        
        df_confidence = pd.DataFrame(results)
        
        if not df_confidence.empty:
            print(f"\n📊 CONFIDENCE THRESHOLD ANALYSIS:")
            
            # Find optimal threshold (minimize FAR + FRR)
            df_confidence['combined_error'] = df_confidence['far'] + df_confidence['frr']
            optimal_idx = df_confidence['combined_error'].idxmin()
            optimal_threshold = df_confidence.loc[optimal_idx, 'threshold']
            
            print(f"\n🎯 OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
            print(f"   Accuracy: {df_confidence.loc[optimal_idx, 'accuracy']:.3f}")
            print(f"   FAR: {df_confidence.loc[optimal_idx, 'far']:.3f}")
            print(f"   FRR: {df_confidence.loc[optimal_idx, 'frr']:.3f}")
            print(f"   Acceptance Rate: {df_confidence.loc[optimal_idx, 'accepted_ratio']:.3f}")
            
            # Equal Error Rate (where FAR ≈ FRR)
            df_confidence['error_diff'] = abs(df_confidence['far'] - df_confidence['frr'])
            eer_idx = df_confidence['error_diff'].idxmin()
            eer_threshold = df_confidence.loc[eer_idx, 'threshold']
            eer_rate = (df_confidence.loc[eer_idx, 'far'] + df_confidence.loc[eer_idx, 'frr']) / 2
            
            print(f"\n⚖️  EQUAL ERROR RATE (EER): {eer_rate:.3f} at threshold {eer_threshold:.2f}")
            
            self.results['confidence_threshold_effect'] = df_confidence
            return df_confidence
        else:
            print("❌ No valid results obtained")
            return None
    
    def test_imposter_recognition(self, user_data, user_names):
        """Test how well the system distinguishes legitimate users from imposters doing same gesture"""
        print("\n" + "="*60)
        print("👥 EXPERIMENT 3: LEGITIMATE USER vs IMPOSTER ANALYSIS")
        print("="*60)
        
        if len(user_data) < 3:
            print("❌ Need at least 3 users to test imposter scenarios")
            return
        
        results = {
            'legitimate_user': [],
            'imposter_user': [],
            'legitimate_accuracy': [],
            'imposter_rejection_rate': [],
            'samples_used': [],
            'confidence_threshold': []
        }
        
        # Test different sample sizes and confidence thresholds
        sample_sizes = [10, 20, 30, 50] if max(len(data) for data in user_data.values()) >= 50 else [5, 10, 15]
        confidence_thresholds = [0.6, 0.7, 0.8, 0.9]
        
        print(f"Testing with sample sizes: {sample_sizes}")
        print(f"Testing with confidence thresholds: {confidence_thresholds}")
        
        user_ids = list(user_data.keys())
        
        for target_user_id in user_ids:
            if len(user_data[target_user_id]) < max(sample_sizes):
                continue
                
            target_user_name = user_names[target_user_id]
            print(f"\nTesting legitimate user: {target_user_name}")
            
            # Test against each other user as potential imposter
            for imposter_user_id in user_ids:
                if imposter_user_id == target_user_id or len(user_data[imposter_user_id]) < max(sample_sizes):
                    continue
                
                imposter_user_name = user_names[imposter_user_id]
                
                for sample_size in sample_sizes:
                    try:
                        # Prepare training data (legitimate user only)
                        legitimate_samples = np.array(user_data[target_user_id][:sample_size])
                        legitimate_samples = self.validate_array(legitimate_samples, f"legitimate_samples_{target_user_name}")
                        
                        # Create a binary classification problem
                        # Positive class: legitimate user
                        # Negative class: everyone else (using other users as negative examples)
                        negative_samples = []
                        for other_user_id in user_ids:
                            if other_user_id != target_user_id:
                                # Use fewer samples from other users to balance
                                other_samples = user_data[other_user_id][:sample_size//len(user_ids)]
                                negative_samples.extend(other_samples)
                        
                        negative_samples = np.array(negative_samples[:sample_size])  # Limit negative samples
                        negative_samples = self.validate_array(negative_samples, f"negative_samples_{target_user_name}")
                        
                        # Create training set
                        X_train = np.vstack([legitimate_samples, negative_samples])
                        y_train = np.array([1] * len(legitimate_samples) + [0] * len(negative_samples))
                        
                        # Prepare test data
                        # Test on remaining samples from legitimate user
                        legitimate_test = np.array(user_data[target_user_id][sample_size:sample_size+10])
                        # Test on imposter samples
                        imposter_test = np.array(user_data[imposter_user_id][:10])
                        
                        if len(legitimate_test) == 0 or len(imposter_test) == 0:
                            continue
                        
                        # Clean test data
                        legitimate_test = self.validate_array(legitimate_test, f"legitimate_test_{target_user_name}")
                        imposter_test = self.validate_array(imposter_test, f"imposter_test_{imposter_user_name}")
                        
                        # Train model with enhanced data validation
                        scaler = StandardScaler()
                        
                        # Clean and validate training data
                        X_train = self.validate_array(X_train, "X_train")
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_train_scaled = self.validate_array(X_train_scaled, "X_train_scaled")
                        
                        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                        model.fit(X_train_scaled, y_train)
                        
                        for conf_threshold in confidence_thresholds:
                            try:
                                # Test legitimate user recognition with validation
                                legitimate_test_scaled = scaler.transform(legitimate_test)
                                legitimate_test_scaled = self.validate_array(legitimate_test_scaled, "legitimate_test_scaled")
                                
                                legitimate_proba = model.predict_proba(legitimate_test_scaled)[:, 1]  # Probability of being legitimate
                                legitimate_accepted = np.sum(legitimate_proba >= conf_threshold)
                                legitimate_accuracy = legitimate_accepted / len(legitimate_test)
                                
                                # Test imposter rejection with validation
                                imposter_test_scaled = scaler.transform(imposter_test)
                                imposter_test_scaled = self.validate_array(imposter_test_scaled, "imposter_test_scaled")
                                
                                imposter_proba = model.predict_proba(imposter_test_scaled)[:, 1]
                                imposter_rejected = np.sum(imposter_proba < conf_threshold)
                                imposter_rejection_rate = imposter_rejected / len(imposter_test)
                                
                                results['legitimate_user'].append(target_user_name)
                                results['imposter_user'].append(imposter_user_name)
                                results['legitimate_accuracy'].append(legitimate_accuracy)
                                results['imposter_rejection_rate'].append(imposter_rejection_rate)
                                results['samples_used'].append(sample_size)
                                results['confidence_threshold'].append(conf_threshold)
                                
                                print(f"  vs {imposter_user_name} (samples={sample_size}, threshold={conf_threshold}): "
                                      f"Legit_Acc={legitimate_accuracy:.3f}, Imp_Rej={imposter_rejection_rate:.3f}")
                                      
                            except Exception as e:
                                print(f"  vs {imposter_user_name} (samples={sample_size}, threshold={conf_threshold}): "
                                      f"Failed - {e}")
                                continue
                                
                    except Exception as e:
                        print(f"  Sample size {sample_size}: Failed - {e}")
                        continue
        
        df_imposter = pd.DataFrame(results)
        
        if not df_imposter.empty:
            print(f"\n🔍 IMPOSTER ANALYSIS RESULTS:")
            
            # Overall statistics
            print(f"\nOverall Performance:")
            print(f"  Average Legitimate User Accuracy: {df_imposter['legitimate_accuracy'].mean():.3f} ± {df_imposter['legitimate_accuracy'].std():.3f}")
            print(f"  Average Imposter Rejection Rate: {df_imposter['imposter_rejection_rate'].mean():.3f} ± {df_imposter['imposter_rejection_rate'].std():.3f}")
            
            # Best configuration
            df_imposter['combined_performance'] = (df_imposter['legitimate_accuracy'] + df_imposter['imposter_rejection_rate']) / 2
            best_idx = df_imposter['combined_performance'].idxmax()
            
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"  Samples: {df_imposter.loc[best_idx, 'samples_used']}")
            print(f"  Confidence Threshold: {df_imposter.loc[best_idx, 'confidence_threshold']}")
            print(f"  Legitimate Accuracy: {df_imposter.loc[best_idx, 'legitimate_accuracy']:.3f}")
            print(f"  Imposter Rejection: {df_imposter.loc[best_idx, 'imposter_rejection_rate']:.3f}")
            
            # Analysis by factors
            print(f"\nPerformance by Sample Size:")
            sample_analysis = df_imposter.groupby('samples_used').agg({
                'legitimate_accuracy': 'mean',
                'imposter_rejection_rate': 'mean'
            }).round(3)
            print(sample_analysis)
            
            print(f"\nPerformance by Confidence Threshold:")
            threshold_analysis = df_imposter.groupby('confidence_threshold').agg({
                'legitimate_accuracy': 'mean',
                'imposter_rejection_rate': 'mean'
            }).round(3)
            print(threshold_analysis)
            
            self.results['imposter_analysis'] = df_imposter
            return df_imposter
        else:
            print("❌ No valid results obtained")
            return None
    
    def test_gesture_complexity_effect(self, user_data, user_names):
        """Analyze how gesture complexity affects recognition performance"""
        print("\n" + "="*60)
        print("🤲 EXPERIMENT 4: GESTURE COMPLEXITY ANALYSIS")
        print("="*60)
        
        # Calculate complexity metrics for each user's gestures
        complexity_results = {
            'user': [],
            'avg_feature_variance': [],
            'feature_range': [],
            'movement_complexity': [],
            'recognition_accuracy': [],
            'confidence_score': []
        }
        
        print("Analyzing gesture complexity for each user...")
        
        for user_id, features_list in user_data.items():
            if len(features_list) < 10:
                continue
                
            username = user_names[user_id]
            features_array = np.array(features_list)
            
            # Complexity Metric 1: Feature Variance
            feature_variance = np.var(features_array, axis=0)
            avg_variance = np.mean(feature_variance)
            
            # Complexity Metric 2: Feature Range
            feature_range = np.max(features_array, axis=0) - np.min(features_array, axis=0)
            avg_range = np.mean(feature_range)
            
            # Complexity Metric 3: Movement Complexity (assuming first 63 features are positions)
            if features_array.shape[1] >= 63:
                position_features = features_array[:, :63]  # x,y,z coordinates
                # Calculate movement variation (how much positions vary between samples)
                movement_var = np.mean(np.var(position_features, axis=0))
            else:
                movement_var = avg_variance
            
            # Test recognition performance for this user against others
            other_users = [uid for uid in user_data.keys() if uid != user_id and len(user_data[uid]) >= 10]
            
            if len(other_users) > 0:
                # Create binary classification: this user vs others
                positive_samples = features_array[:15]  # Use first 15 samples for training
                negative_samples = []
                
                for other_id in other_users[:3]:  # Use up to 3 other users
                    negative_samples.extend(user_data[other_id][:5])  # 5 samples each
                
                negative_samples = np.array(negative_samples)
                
                # Prepare training data
                X_train = np.vstack([positive_samples, negative_samples])
                y_train = np.array([1] * len(positive_samples) + [0] * len(negative_samples))
                
                # Test data (remaining samples from target user)
                test_samples = features_array[15:25] if len(features_array) > 25 else features_array[-5:]
                
                if len(test_samples) > 0:
                    # Train and test
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    test_samples_scaled = scaler.transform(test_samples)
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    predictions = model.predict(test_samples_scaled)
                    probabilities = model.predict_proba(test_samples_scaled)[:, 1]
                    
                    accuracy = np.mean(predictions)  # Should be 1 for all samples of the legitimate user
                    avg_confidence = np.mean(probabilities)
                    
                    complexity_results['user'].append(username)
                    complexity_results['avg_feature_variance'].append(avg_variance)
                    complexity_results['feature_range'].append(avg_range)
                    complexity_results['movement_complexity'].append(movement_var)
                    complexity_results['recognition_accuracy'].append(accuracy)
                    complexity_results['confidence_score'].append(avg_confidence)
                    
                    print(f"  {username}: Variance={avg_variance:.4f}, Range={avg_range:.2f}, "
                          f"Movement={movement_var:.4f}, Accuracy={accuracy:.3f}, Confidence={avg_confidence:.3f}")
        
        df_complexity = pd.DataFrame(complexity_results)
        
        if not df_complexity.empty and len(df_complexity) > 1:
            print(f"\n📊 GESTURE COMPLEXITY ANALYSIS:")
            
            # Correlation analysis
            correlations = df_complexity[['avg_feature_variance', 'feature_range', 'movement_complexity', 
                                        'recognition_accuracy', 'confidence_score']].corr()
            
            print(f"\nCorrelations with Recognition Performance:")
            print(f"  Variance vs Accuracy: {correlations.loc['avg_feature_variance', 'recognition_accuracy']:.3f}")
            print(f"  Range vs Accuracy: {correlations.loc['feature_range', 'recognition_accuracy']:.3f}")
            print(f"  Movement vs Accuracy: {correlations.loc['movement_complexity', 'recognition_accuracy']:.3f}")
            
            print(f"\nCorrelations with Confidence:")
            print(f"  Variance vs Confidence: {correlations.loc['avg_feature_variance', 'confidence_score']:.3f}")
            print(f"  Range vs Confidence: {correlations.loc['feature_range', 'confidence_score']:.3f}")
            print(f"  Movement vs Confidence: {correlations.loc['movement_complexity', 'confidence_score']:.3f}")
            
            # Categorize users by complexity
            df_complexity['complexity_category'] = 'Medium'
            
            # High complexity: above median in at least 2 metrics
            high_complexity_mask = (
                (df_complexity['avg_feature_variance'] > df_complexity['avg_feature_variance'].median()) +
                (df_complexity['feature_range'] > df_complexity['feature_range'].median()) +
                (df_complexity['movement_complexity'] > df_complexity['movement_complexity'].median())
            ) >= 2
            
            df_complexity.loc[high_complexity_mask, 'complexity_category'] = 'High'
            
            # Low complexity: below median in at least 2 metrics
            low_complexity_mask = (
                (df_complexity['avg_feature_variance'] < df_complexity['avg_feature_variance'].median()) +
                (df_complexity['feature_range'] < df_complexity['feature_range'].median()) +
                (df_complexity['movement_complexity'] < df_complexity['movement_complexity'].median())
            ) >= 2
            
            df_complexity.loc[low_complexity_mask, 'complexity_category'] = 'Low'
            
            print(f"\nPerformance by Complexity Category:")
            complexity_summary = df_complexity.groupby('complexity_category').agg({
                'recognition_accuracy': ['mean', 'std'],
                'confidence_score': ['mean', 'std']
            }).round(3)
            print(complexity_summary)
            
            # Key insights
            high_complexity_users = df_complexity[df_complexity['complexity_category'] == 'High']
            low_complexity_users = df_complexity[df_complexity['complexity_category'] == 'Low']
            
            if len(high_complexity_users) > 0 and len(low_complexity_users) > 0:
                high_acc = high_complexity_users['recognition_accuracy'].mean()
                low_acc = low_complexity_users['recognition_accuracy'].mean()
                
                print(f"\n🔍 KEY INSIGHTS:")
                print(f"  High complexity gestures: {high_acc:.3f} average accuracy")
                print(f"  Low complexity gestures: {low_acc:.3f} average accuracy")
                
                if high_acc > low_acc:
                    print("  → More complex gestures tend to be MORE recognizable")
                else:
                    print("  → Simpler gestures tend to be MORE recognizable")
            
            self.results['complexity_analysis'] = df_complexity
            return df_complexity
        else:
            print("❌ Insufficient data for complexity analysis")
            return None
    
    def generate_comprehensive_report(self):
        """Generate the complete experimental analysis report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL TESTING REPORT")
        print("="*80)
        
        # Load data
        user_data, user_names = self.load_user_data()
        
        if len(user_data) < 2:
            print("❌ INSUFFICIENT DATA: Need at least 2 users with training samples")
            return
        
        print(f"📋 DATASET OVERVIEW:")
        print(f"  • Total users: {len(user_data)}")
        print(f"  • Total samples: {sum(len(samples) for samples in user_data.values())}")
        print(f"  • Users: {list(user_names.values())}")
        
        # Run all experiments
        print("\n🧪 RUNNING EXPERIMENTAL ANALYSIS...")
        
        # Experiment 1: Sample Size Effect
        sample_results = self.test_sample_size_effect(user_data, user_names)
        
        # Experiment 2: Confidence Threshold Effect
        confidence_results = self.test_confidence_threshold_effect(user_data, user_names)
        
        # Experiment 3: Legitimate vs Imposter Analysis
        imposter_results = self.test_imposter_recognition(user_data, user_names)
        
        # Experiment 4: Gesture Complexity Analysis
        complexity_results = self.test_gesture_complexity_effect(user_data, user_names)
        
        # Generate final summary
        self.generate_final_summary()
        
        return {
            'sample_size_effect': sample_results,
            'confidence_threshold_effect': confidence_results,
            'imposter_analysis': imposter_results,
            'complexity_analysis': complexity_results
        }
    
    def generate_final_summary(self):
        """Generate final summary and recommendations"""
        print("\n" + "="*80)
        print("📋 FINAL SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        # Sample Size Recommendations
        if 'sample_size_effect' in self.results:
            df_sample = self.results['sample_size_effect']
            if df_sample is not None and not df_sample.empty:
                best_sample_size = df_sample.loc[df_sample['accuracy'].idxmax(), 'sample_size']
                avg_accuracy = df_sample.groupby('sample_size')['accuracy'].mean()
                good_sizes = avg_accuracy[avg_accuracy >= 0.90]
                
                if not good_sizes.empty:
                    min_good_size = good_sizes.index.min()
                    recommendations.append(f"Use at least {min_good_size} samples per user for ≥90% accuracy")
                else:
                    recommendations.append(f"Current max tested: {best_sample_size} samples (best performance)")
        
        # Confidence Threshold Recommendations
        if 'confidence_threshold_effect' in self.results:
            df_conf = self.results['confidence_threshold_effect']
            if df_conf is not None and not df_conf.empty:
                optimal_threshold = df_conf.loc[df_conf['combined_error'].idxmin(), 'threshold']
                recommendations.append(f"Optimal confidence threshold: {optimal_threshold:.2f}")
        
        # Imposter Detection Recommendations
        if 'imposter_analysis' in self.results:
            df_imp = self.results['imposter_analysis']
            if df_imp is not None and not df_imp.empty:
                best_config = df_imp.loc[df_imp['combined_performance'].idxmax()]
                recommendations.append(f"Best imposter detection: {best_config['samples_used']} samples, "
                                     f"threshold {best_config['confidence_threshold']}")
        
        # Complexity Recommendations
        if 'complexity_analysis' in self.results:
            df_comp = self.results['complexity_analysis']
            if df_comp is not None and not df_comp.empty:
                high_complex = df_comp[df_comp['complexity_category'] == 'High']
                low_complex = df_comp[df_comp['complexity_category'] == 'Low']
                
                if len(high_complex) > 0 and len(low_complex) > 0:
                    if high_complex['recognition_accuracy'].mean() > low_complex['recognition_accuracy'].mean():
                        recommendations.append("Encourage users to use more complex gestures for better security")
                    else:
                        recommendations.append("Simple gestures may be more reliable for recognition")
        
        print("🎯 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Security Assessment
        print(f"\n🔐 SECURITY ASSESSMENT:")
        if 'imposter_analysis' in self.results:
            df_imp = self.results['imposter_analysis']
            if df_imp is not None and not df_imp.empty:
                avg_rejection = df_imp['imposter_rejection_rate'].mean()
                if avg_rejection >= 0.95:
                    print("  ✅ EXCELLENT: Very low false accept rate")
                elif avg_rejection >= 0.90:
                    print("  ✅ GOOD: Acceptable false accept rate")
                elif avg_rejection >= 0.80:
                    print("  ⚠️  MODERATE: Consider increasing security measures")
                else:
                    print("  ❌ POOR: High false accept rate - security risk")
        
        # Usability Assessment
        if 'sample_size_effect' in self.results:
            df_sample = self.results['sample_size_effect']
            if df_sample is not None and not df_sample.empty:
                min_samples = df_sample['sample_size'].min()
                if min_samples <= 20:
                    print("  ✅ GOOD: Reasonable training burden for users")
                elif min_samples <= 50:
                    print("  ⚠️  MODERATE: Significant training time required")
                else:
                    print("  ❌ POOR: Extensive training may discourage users")
        
        print(f"\n✅ COMPREHENSIVE TESTING COMPLETE!")
        print(f"📊 Use these results to optimize your gesture recognition system")

    def create_visualizations(self):
        """Create visualizations for all experimental results"""
        if not self.results:
            print("No results to visualize. Run generate_comprehensive_report() first.")
            return
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Sample Size Effect
        if 'sample_size_effect' in self.results and self.results['sample_size_effect'] is not None:
            plt.subplot(2, 3, 1)
            df_sample = self.results['sample_size_effect']
            avg_by_size = df_sample.groupby('sample_size')['accuracy'].agg(['mean', 'std'])
            
            plt.errorbar(avg_by_size.index, avg_by_size['mean'], yerr=avg_by_size['std'], 
                        marker='o', capsize=5, capthick=2)
            plt.xlabel('Number of Training Samples')
            plt.ylabel('Recognition Accuracy')
            plt.title('Effect of Training Sample Size')
            plt.grid(True, alpha=0.3)
        
        # Plot 2: Confidence Threshold ROC-style
        if 'confidence_threshold_effect' in self.results and self.results['confidence_threshold_effect'] is not None:
            plt.subplot(2, 3, 2)
            df_conf = self.results['confidence_threshold_effect']
            
            plt.plot(df_conf['far'], 1 - df_conf['frr'], 'b-o', label='ROC-style curve')
            plt.xlabel('False Accept Rate')
            plt.ylabel('True Accept Rate (1 - FRR)')
            plt.title('Security Performance Trade-off')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot 3: Threshold vs Accuracy
        if 'confidence_threshold_effect' in self.results and self.results['confidence_threshold_effect'] is not None:
            plt.subplot(2, 3, 3)
            df_conf = self.results['confidence_threshold_effect']
            
            plt.plot(df_conf['threshold'], df_conf['accuracy'], 'g-o', label='Accuracy')
            plt.plot(df_conf['threshold'], df_conf['accepted_ratio'], 'r-s', label='Acceptance Rate')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Rate')
            plt.title('Threshold vs Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Imposter Analysis Heatmap
        if 'imposter_analysis' in self.results and self.results['imposter_analysis'] is not None:
            plt.subplot(2, 3, 4)
            df_imp = self.results['imposter_analysis']
            
            # Create pivot table for heatmap
            pivot_data = df_imp.groupby(['samples_used', 'confidence_threshold'])['combined_performance'].mean().unstack()
            
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Training Samples')
            plt.title('Imposter Detection Performance')
        
        # Plot 5: Complexity Analysis
        if 'complexity_analysis' in self.results and self.results['complexity_analysis'] is not None:
            plt.subplot(2, 3, 5)
            df_comp = self.results['complexity_analysis']
            
            colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            for category in df_comp['complexity_category'].unique():
                mask = df_comp['complexity_category'] == category
                plt.scatter(df_comp[mask]['avg_feature_variance'], 
                           df_comp[mask]['recognition_accuracy'],
                           c=colors.get(category, 'blue'), label=category, s=100, alpha=0.7)
            
            plt.xlabel('Feature Variance (Complexity)')
            plt.ylabel('Recognition Accuracy')
            plt.title('Gesture Complexity vs Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Summary Performance Comparison
        plt.subplot(2, 3, 6)
        
        # Collect key metrics for comparison
        metrics = []
        values = []
        
        if 'sample_size_effect' in self.results and self.results['sample_size_effect'] is not None:
            max_acc = self.results['sample_size_effect']['accuracy'].max()
            metrics.append('Max Accuracy\n(Sample Size)')
            values.append(max_acc)
        
        if 'confidence_threshold_effect' in self.results and self.results['confidence_threshold_effect'] is not None:
            best_acc = self.results['confidence_threshold_effect']['accuracy'].max()
            metrics.append('Max Accuracy\n(Threshold)')
            values.append(best_acc)
        
        if 'imposter_analysis' in self.results and self.results['imposter_analysis'] is not None:
            avg_legit = self.results['imposter_analysis']['legitimate_accuracy'].mean()
            avg_reject = self.results['imposter_analysis']['imposter_rejection_rate'].mean()
            metrics.extend(['Legitimate\nRecognition', 'Imposter\nRejection'])
            values.extend([avg_legit, avg_reject])
        
        if metrics:
            bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'][:len(metrics)])
            plt.ylabel('Performance Rate')
            plt.title('Overall System Performance')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gesture_recognition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved as 'gesture_recognition_analysis.png'")

# Usage example
if __name__ == "__main__":
    tester = ComprehensiveModelTesting()
    results = tester.generate_comprehensive_report()
    
    # Optional: Create visualizations
    try:
        tester.create_visualizations()
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        print("Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")