"""
Random Search Explainer Experiment for Auto MPG Dataset
Demonstrates counterfactual generation using preference-based random search.

This script:
1. Loads Auto MPG dataset and trained NN_Residual model
2. Defines preferences (actionability and desirability) for features
3. Generates counterfactual explanations using RandomSearchExplainer
4. Compares results across different sample-target pairs
"""

import logging
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
from pathlib import Path
import tensorflow as tf

from explainit.priorities.nonlinear import exponential
from explainit.explainers.random_search import RandomSearchExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ARTICLE-BASED METRICS (Mothilal et al. 2020)
# ============================================================================

def calculate_mad_from_training(X_train: np.ndarray) -> np.ndarray:
    """
    Calculate Median Absolute Deviation (MAD) for each feature from training data.
    
    As defined in the paper (Section 3.2):
    MAD is used to normalize continuous features in distance calculations.
    
    Args:
        X_train: Training data (n_samples, n_features)
    
    Returns:
        mad_values: Array of MAD values for each feature (n_features,)
    """
    median = np.median(X_train, axis=0)
    mad = np.median(np.abs(X_train - median), axis=0)
    
    # Handle zero MAD (constant features) by replacing with 1.0
    mad = np.where(mad == 0, 1.0, mad)
    
    return mad


# ============================================================================
# DATASET AND MODEL LOADER
# ============================================================================

class AutoMPGExperiment:
    """
    Auto MPG Dataset Experiment
    
    Features: 4 continuous features
    - displacement: Engine displacement in cubic inches
    - horsepower: Engine horsepower
    - weight: Vehicle weight in pounds
    - acceleration: Time to accelerate from 0 to 60 mph (seconds)
    
    Target: Miles per gallon (MPG) - continuous regression target
    Model: NN_Residual (Skip connections, BatchNorm)
    """
    
    def __init__(self):
        self.dataset_name = "Auto_MPG"
        self.model_name = "NN_Residual"
        self.base_dir = Path(__file__).parent.parent.parent.parent / "ML_models_for_tests" / "ML_regression_results" / "Auto_MPG"
        self.feature_names = ['displacement', 'horsepower', 'weight', 'acceleration']
        self.target_name = 'mpg'
        
    def load_data(self):
        """Load Auto MPG dataset and trained model."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        # Load data
        X_data = pd.read_csv(self.base_dir / "X_data.csv")
        y_data = pd.read_csv(self.base_dir / "y_data.csv")
        
        # Load scaler
        import pickle
        with open(self.base_dir / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Load trained model
        try:
            import tensorflow as tf
            from tensorflow.keras import losses, metrics
            
            # Force CPU usage to avoid GPU issues
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Load model with compile=False to avoid deserialization issues
            model = tf.keras.models.load_model(
                self.base_dir / f"{self.model_name}_model.h5",
                compile=False
            )
            
            # Compile the model manually
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Build the model by calling it once
            logger.info("Building model...")
            _ = model(tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32))
            logger.info("Model built successfully")
            
            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Split data (same split as in training - 80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42
        )
        
        # Scale data
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to float32 for TensorFlow compatibility
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        
        logger.info(f"Data loaded: {len(X_data)} samples, {len(self.feature_names)} features")
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Target range: [{y_data.min().values[0]:.2f}, {y_data.max().values[0]:.2f}] MPG")
        
        return X_train_scaled, X_test_scaled, y_train.values.ravel(), y_test.values.ravel(), scaler, model, X_data, y_data
    
    def find_exemplar(self, model, X_test, target=25.0):
        """
        Find the exemplar - the sample in the dataset closest to the target MPG.
        
        Args:
            model: Trained regression model
            X_test: Test dataset
            target: Target MPG value
        
        Returns:
            exemplar: Sample closest to target
            exemplar_prediction: Prediction for the exemplar
            exemplar_index: Index of exemplar in X_test
        """
        logger.info(f"Finding exemplar closest to target={target} MPG...")
        
        # Get predictions for all test samples
        predictions = model.predict(X_test, verbose=0).ravel()
        
        # Find sample closest to target
        distances = np.abs(predictions - target)
        exemplar_index = np.argmin(distances)
        
        exemplar = X_test[exemplar_index].tolist()
        exemplar_prediction = predictions[exemplar_index]
        
        logger.info(f"Found exemplar at index {exemplar_index}")
        logger.info(f"Exemplar prediction: {exemplar_prediction:.2f} MPG (target: {target})")
        logger.info(f"Distance from target: {abs(exemplar_prediction - target):.2f} MPG")
        
        return exemplar, exemplar_prediction, exemplar_index
    
    def define_preferences(self, sample, exemplar, X_train, exemplar_weight=0.5):
        """
        Define preferences for Auto MPG features.
        
        Preferences:
        - displacement: Lower is better (smaller engines, more efficient)
        - horsepower: Lower is better (less power, more efficient) 
        - weight: Lower is better (lighter cars, more efficient)
        - acceleration: Higher is better (but less impact on MPG)
        
        All features are actionable (could be changed in car design).
        
        Args:
            sample: Sample to explain
            exemplar: Exemplar (sample closest to target)
            X_train: Training dataset to extract min/max values
            exemplar_weight: Weight assigned to exemplar value (default: 0.5)
        
        Returns:
            preferences: Dictionary defining preferences
        """
        logger.info("Defining preferences for Auto MPG features...")
        
        numerical_preferences = {}
        
        # All features are actionable with preference functions
        for idx in range(len(self.feature_names)):
            sample_val = sample[idx]
            exemplar_val = exemplar[idx]
            
            # Get actual min/max from dataset
            dataset_min = X_train[:, idx].min()
            dataset_max = X_train[:, idx].max()
            
            # For features where lower is better (displacement, horsepower, weight)
            # We want: f(sample_value) = 1, f(exemplar_value) = exemplar_weight
            # If sample_val < exemplar_val, we prefer lower values (decreasing direction)
            # If sample_val > exemplar_val, we prefer higher values (increasing direction)
            
            if idx in [0, 1, 2]:  # displacement, horsepower, weight - lower is better
                # Determine direction based on sample vs exemplar
                if sample_val <= exemplar_val:
                    # Sample is already lower or equal - prefer decreasing
                    preference_func, x0 = self.create_numerical_preference_function(
                        sample_value=sample_val,
                        exemplar_value=exemplar_val,
                        min_val=dataset_min,
                        max_val=dataset_max,
                        exemplar_weight=exemplar_weight,
                        increasing=False  # Prefer lower values
                    )
                else:
                    # Sample is higher - prefer decreasing to reach exemplar
                    preference_func, x0 = self.create_numerical_preference_function(
                        sample_value=sample_val,
                        exemplar_value=exemplar_val,
                        min_val=dataset_min,
                        max_val=dataset_max,
                        exemplar_weight=exemplar_weight,
                        increasing=False  # Prefer lower values
                    )
            else:  # acceleration - higher is better (faster acceleration)
                if sample_val >= exemplar_val:
                    # Sample is already higher - prefer increasing
                    preference_func, x0 = self.create_numerical_preference_function(
                        sample_value=sample_val,
                        exemplar_value=exemplar_val,
                        min_val=dataset_min,
                        max_val=dataset_max,
                        exemplar_weight=exemplar_weight,
                        increasing=True
                    )
                else:
                    # Sample is lower - prefer increasing to reach exemplar
                    preference_func, x0 = self.create_numerical_preference_function(
                        sample_value=sample_val,
                        exemplar_value=exemplar_val,
                        min_val=dataset_min,
                        max_val=dataset_max,
                        exemplar_weight=exemplar_weight,
                        increasing=True
                    )
            
            acceptable_min = max(dataset_min, x0) if idx in [0, 1, 2] else dataset_min
            acceptable_max = dataset_max if idx in [0, 1, 2] else min(dataset_max, x0)
            
            logger.info(f"Feature {idx} ({self.feature_names[idx]}):")
            logger.info(f"  Dataset min: {dataset_min:.4f}, Dataset max: {dataset_max:.4f}")
            logger.info(f"  Sample value: {sample_val:.4f} (weight = 1.0)")
            logger.info(f"  Exemplar value: {exemplar_val:.4f} (weight = {exemplar_weight})")
            logger.info(f"  Calculated x0: {x0:.4f} (weight = 0.0)")
            logger.info(f"  Acceptable min: {acceptable_min:.4f}, Acceptable max: {acceptable_max:.4f}")
            
            numerical_preferences[idx] = {
                'function': preference_func,
                'min': acceptable_min,
                'max': acceptable_max
            }
        
        preferences = {
            'numerical': numerical_preferences,
            'categorical': {}  # No categorical features in Auto MPG
        }
        
        return preferences
    
    def create_numerical_preference_function(self, sample_value, exemplar_value, 
                                            min_val=0.0, max_val=1.0, 
                                            exemplar_weight=0.5, increasing=True):
        """
        Create a preference function for a numerical feature.
        
        Args:
            sample_value: Value in the sample (weight = 1)
            exemplar_value: Value in the exemplar (weight = exemplar_weight)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            exemplar_weight: Weight assigned to exemplar value (default: 0.5)
            increasing: If True, prefer higher values; if False, prefer lower values
        
        Returns:
            Preference function and x0 value
        """
        a = 5
        t_target = np.log(1 + exemplar_weight * (np.exp(a) - 1)) / a
        
        if increasing:
            # f(x) = 1 at sample_value, f(x) = exemplar_weight at exemplar_value
            # f(x) -> 0 as x approaches x0
            x0 = (exemplar_value - t_target * sample_value) / (1 - t_target)
            x1 = sample_value
            
            def preference_func(x):
                return exponential(x, x0=x0, x1=x1, increasing=True, a=a)
        else:
            # f(x) = 1 at sample_value, f(x) = exemplar_weight at exemplar_value
            # f(x) -> 0 as x approaches x0 (but decreasing direction)
            x1 = (exemplar_value - t_target * sample_value) / (1 - t_target)
            x0 = sample_value
            
            def preference_func(x):
                return exponential(x, x0=x0, x1=x1, increasing=False, a=a)
        
        return preference_func, x0 if increasing else x1


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_counterfactual_experiment(model, X_train, X_test, y_train, config, experiment, mad_values):
    """
    Run a comprehensive counterfactual experiment with multiple samples and targets.
    
    Args:
        model: Trained model
        X_train: Training dataset
        X_test: Test dataset
        y_train: Training predictions
        config: Dictionary with experiment parameters
        experiment: Experiment instance (e.g., AutoMPGExperiment)
        mad_values: MAD values for article-based metrics
    
    Returns:
        results: List of experiment results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING COUNTERFACTUAL EXPERIMENT - RANDOM SEARCH METHOD")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    
    # Get predictions for all test samples
    logger.info("Getting predictions for test samples...")
    logger.info(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    sys.stderr.flush()
    
    # Test with a small batch first using direct call (bypasses predict machinery)
    logger.info("Testing prediction with first sample using model.__call__()...")
    sys.stderr.flush()
    test_input = tf.constant(X_test[:1].astype(np.float32))
    test_pred = model(test_input, training=False)
    logger.info(f"Test prediction successful: {test_pred.numpy()[0][0]:.2f}")
    sys.stderr.flush()
    
    logger.info("Getting all predictions using model.__call__()...")
    sys.stderr.flush()
    predictions = model(tf.constant(X_test.astype(np.float32)), training=False).numpy().ravel()
    logger.info(f"Predictions computed for {len(predictions)} samples")
    logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    sys.stderr.flush()
    
    # Select equally distributed prediction quantiles
    logger.info(f"Selecting {config['n_quantiles']} quantile points...")
    quantiles = np.linspace(0, 1, config['n_quantiles'])
    quantile_values = np.quantile(predictions, quantiles)
    
    # Find samples closest to each quantile
    sample_points = []
    for q_val in quantile_values:
        distances = np.abs(predictions - q_val)
        idx = np.argmin(distances)
        sample_points.append({
            'index': idx,
            'sample': X_test[idx].tolist(),
            'prediction': predictions[idx]
        })
    
    logger.info(f"Selected {len(sample_points)} sample points with predictions:")
    for i, sp in enumerate(sample_points):
        logger.info(f"  Sample {i+1}: prediction = {sp['prediction']:.2f} {experiment.target_name}")
    logger.info("\n")
    
    # Run experiment for each sample-target pair
    results = []
    total_combinations = len(sample_points) * (len(sample_points) - 1)
    current_combination = 0
    
    # For time estimation
    start_time = time.time()
    pair_times = []
    
    for sample_idx, sample_point in enumerate(sample_points):
        sample = sample_point['sample']
        sample_pred = sample_point['prediction']
        
        for target_idx, target_point in enumerate(sample_points):
            if sample_idx == target_idx:
                continue  # Skip same point
            
            current_combination += 1
            target_pred = target_point['prediction']
            exemplar = target_point['sample']
            
            # Time estimation
            if pair_times:
                avg_time = np.mean(pair_times)
                remaining_pairs = total_combinations - current_combination + 1
                est_remaining_sec = avg_time * remaining_pairs
                est_remaining_min = est_remaining_sec / 60
                logger.info(f"[{current_combination}/{total_combinations}] Sample {sample_idx+1} → Target {target_idx+1} (Est. {est_remaining_min:.1f} min remaining)")
            else:
                logger.info(f"[{current_combination}/{total_combinations}] Sample {sample_idx+1} → Target {target_idx+1}")
            
            logger.info(f"  Sample prediction: {sample_pred:.2f}, Target prediction: {target_pred:.2f}")
            
            pair_start_time = time.time()
            
            # Define preferences
            logger.info(f"  Defining preferences...")
            preferences = experiment.define_preferences(sample, exemplar, X_train, config['exemplar_weight'])
            
            # Generate counterfactuals
            logger.info(f"  Creating explainer...")
            model_pred = lambda x: model(tf.constant(x.astype(np.float32)), training=False).numpy().ravel()
            explainer = RandomSearchExplainer(
                model_pred=model_pred,
                priorities=preferences,
                sample=sample,
                target=target_pred
            )
            logger.info(f"  Generating counterfactuals...")
            # Time the generation
            method_start_time = time.time()
            # First get all counterfactuals to track statistics
            all_cf_samples, all_cf_predictions, all_cf_scores = explainer.generate_random_samples(
                n_samples=config['n_samples'],
                epsilon=config['epsilon'],
                use_monte_carlo=config['use_monte_carlo'],
                random_seed=42,
                max_tries=100,
                return_top_n=None  # Get all first
            )
            method_elapsed = time.time() - method_start_time
            
            # Track statistics before filtering
            total_cf_found = len(all_cf_samples)
            max_preference_score = max(all_cf_scores) if all_cf_scores else 0.0
            n_cf_with_max_score = sum(1 for score in all_cf_scores if score == max_preference_score) if all_cf_scores else 0
            
            logger.info(f"  Generation complete: {total_cf_found} counterfactuals found")
            logger.info(f"  Max preference score: {max_preference_score:.4f} ({n_cf_with_max_score} CFs with this score)")
            
            # ===== ARTICLE-BASED METRICS (Mothilal et al. 2020) =====
            # Compute metrics for THIS experiment's CFs (all_cf_samples compared to same original)
            k_requested = config['n_samples']
            n_generated = len(all_cf_samples)
            d = len(sample)  # Number of features
            
            # 1. % Valid CFs
            valid_mask_exp = np.abs(np.array(all_cf_predictions) - target_pred) <= config['epsilon']
            n_valid_exp = np.sum(valid_mask_exp)
            pct_valid_cfs_exp = (n_valid_exp / k_requested) if k_requested > 0 else 0.0
            
            # 2. Continuous Proximity: -(1/k) * Σ dist_cont(c_i, x)
            proximities_exp = []
            for cf in all_cf_samples:
                dist = np.mean(np.abs(np.array(cf) - np.array(sample)) / mad_values)
                proximities_exp.append(dist)
            continuous_proximity_exp = -np.mean(proximities_exp) if proximities_exp else 0.0
            
            # 3. Continuous Sparsity: 1 - (1/(k*d)) * Σ_i Σ_p 1[c_i,p ≠ x_p]
            total_changes_exp = 0
            for cf in all_cf_samples:
                n_changes = np.sum(np.abs(np.array(cf) - np.array(sample)) > 1e-6)
                total_changes_exp += n_changes
            continuous_sparsity_exp = 1.0 - (total_changes_exp / (n_generated * d)) if n_generated > 0 else 1.0
            
            # 4. Continuous Diversity: (1/C(k,2)) * Σ_(i<j) dist_cont(c_i, c_j)
            continuous_diversity_exp = 0.0
            if n_generated > 1:
                pairwise_distances_exp = []
                for i in range(n_generated):
                    for j in range(i + 1, n_generated):
                        dist = np.mean(np.abs(np.array(all_cf_samples[i]) - np.array(all_cf_samples[j])) / mad_values)
                        pairwise_distances_exp.append(dist)
                continuous_diversity_exp = np.mean(pairwise_distances_exp)
            
            # 5. Cont-Count Diversity: (1/C(k,2)) * Σ_(i<j) count(features differ)
            cont_count_diversity_exp = 0.0
            if n_generated > 1:
                count_diffs_exp = []
                for i in range(n_generated):
                    for j in range(i + 1, n_generated):
                        n_diff = np.sum(np.abs(np.array(all_cf_samples[i]) - np.array(all_cf_samples[j])) > 1e-6)
                        count_diffs_exp.append(n_diff)
                cont_count_diversity_exp = np.mean(count_diffs_exp) / d if count_diffs_exp else 0.0
            
            # Store article metrics for this experiment
            article_metrics_exp = {
                'k_requested': k_requested,
                'n_generated': n_generated,
                'n_valid': int(n_valid_exp),
                'pct_valid_cfs': pct_valid_cfs_exp,
                'continuous_proximity': continuous_proximity_exp,
                'categorical_proximity': 1.0,  # No categorical features
                'continuous_sparsity': continuous_sparsity_exp,
                'continuous_diversity': continuous_diversity_exp,
                'categorical_diversity': 0.0,  # No categorical features
                'cont_count_diversity': cont_count_diversity_exp
            }
            
            logger.info(f"\n  ARTICLE-BASED METRICS (Mothilal et al. 2020):")
            logger.info(f"    % Valid CFs: {pct_valid_cfs_exp:.2%} ({n_valid_exp}/{k_requested})")
            logger.info(f"    Continuous-Proximity: {continuous_proximity_exp:.4f}")
            logger.info(f"    Categorical-Proximity: 1.0000")
            logger.info(f"    Continuous-Sparsity: {continuous_sparsity_exp:.4f} (1.0 = sparse)")
            logger.info(f"    Continuous-Diversity: {continuous_diversity_exp:.4f}")
            logger.info(f"    Categorical-Diversity: 0.0000")
            logger.info(f"    Cont-Count-Diversity: {cont_count_diversity_exp:.4f}")
            
            # Keep ALL counterfactuals (no filtering to top N)
            if total_cf_found > 0:
                # Sort by preference score descending but keep all
                sorted_indices = np.argsort(all_cf_scores)[::-1]
                
                cf_samples = [all_cf_samples[i] for i in sorted_indices]
                cf_predictions = [all_cf_predictions[i] for i in sorted_indices]
                cf_scores = [all_cf_scores[i] for i in sorted_indices]
                
                # Calculate % valid after epsilon threshold
                n_valid_after_eps = sum(1 for cf_pred in cf_predictions if abs(cf_pred - target_pred) <= config['epsilon'])
                pct_valid_after_eps = (n_valid_after_eps / len(cf_predictions)) * 100 if len(cf_predictions) > 0 else 0.0
                logger.info(f"    % Valid CFs after eps: {pct_valid_after_eps:.2f}% ({n_valid_after_eps}/{len(cf_predictions)})")
                
                # Calculate priority metrics for valid CFs
                valid_cf_scores = [all_cf_scores[i] for i in range(len(all_cf_predictions)) 
                                  if abs(all_cf_predictions[i] - target_pred) <= config['epsilon']]
                if valid_cf_scores:
                    highest_priority = max(valid_cf_scores)
                    n_highest_priority = sum(1 for score in valid_cf_scores if score == highest_priority)
                    pct_highest = (n_highest_priority / len(valid_cf_scores)) * 100
                    logger.info(f"    Highest priority value: {highest_priority:.4f}")
                    logger.info(f"    Number of highest CFs: {pct_highest:.2f}% ({n_highest_priority}/{len(valid_cf_scores)})")
            else:
                cf_samples, cf_predictions, cf_scores = [], [], []
            
            # Store results with statistics and article metrics
            result = {
                'sample_idx': sample_idx + 1,
                'target_idx': target_idx + 1,
                'sample_prediction': sample_pred,
                'target_prediction': target_pred,
                'sample_values': sample,
                'exemplar_values': exemplar,
                'n_counterfactuals': len(cf_samples),  # Number saved (top N)
                'total_cf_found': total_cf_found,  # Total found before filtering
                'max_preference_score': max_preference_score,
                'n_cf_with_max_score': n_cf_with_max_score,
                'computation_time': method_elapsed,
                'article_metrics': article_metrics_exp,  # Per-experiment article metrics
                'counterfactuals': []
            }
            
            # Calculate metrics for each counterfactual (matching DiCE metrics)
            sample_array = np.array(sample)
            for i, (cf_sample, cf_pred, cf_score) in enumerate(zip(cf_samples, cf_predictions, cf_scores)):
                cf_array = np.array(cf_sample)
                l1_distance = np.sum(np.abs(cf_array - sample_array))
                l2_distance = np.sqrt(np.sum((cf_array - sample_array) ** 2))
                sparsity = np.sum(np.abs(cf_array - sample_array) > 1e-6)
                pred_error = abs(cf_pred - target_pred)
                is_valid = pred_error <= config['epsilon']
                
                result['counterfactuals'].append({
                    'rank': i + 1,
                    'prediction': cf_pred,
                    'preference_score': cf_score,
                    'sample': cf_sample,
                    'l1_distance': l1_distance,
                    'l2_distance': l2_distance,
                    'sparsity': int(sparsity),
                    'pred_error': pred_error,
                    'valid': is_valid
                })
            
            results.append(result)
            
            # Track time for this pair
            pair_elapsed = time.time() - pair_start_time
            pair_times.append(pair_elapsed)
            
            if len(cf_samples) > 0:
                logger.info(f"  Found {len(cf_samples)} counterfactuals")
                logger.info(f"  Prediction range: [{min(cf_predictions):.2f}, {max(cf_predictions):.2f}]")
                logger.info(f"  Preference score range: [{min(cf_scores):.2f}, {max(cf_scores):.2f}]")
            else:
                logger.info(f"  No counterfactuals found")
            logger.info("")
    
    # ============================================================================
    # AGGREGATE ARTICLE-BASED METRICS ACROSS ALL EXPERIMENTS (Mothilal et al. 2020)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ARTICLE-BASED METRICS (Mothilal et al. 2020):")
    
    # Collect article metrics from all experiments
    experiments_with_metrics = [r for r in results if 'article_metrics' in r]
    
    if experiments_with_metrics:
        # Average diversity metrics across all experiments (these are per-experiment measures)
        avg_continuous_proximity = np.mean([r['article_metrics']['continuous_proximity'] for r in experiments_with_metrics])
        avg_categorical_proximity = np.mean([r['article_metrics']['categorical_proximity'] for r in experiments_with_metrics])
        avg_continuous_sparsity = np.mean([r['article_metrics']['continuous_sparsity'] for r in experiments_with_metrics])
        avg_continuous_diversity = np.mean([r['article_metrics']['continuous_diversity'] for r in experiments_with_metrics])
        avg_categorical_diversity = np.mean([r['article_metrics']['categorical_diversity'] for r in experiments_with_metrics])
        avg_cont_count_diversity = np.mean([r['article_metrics']['cont_count_diversity'] for r in experiments_with_metrics])
        
        # Calculate OVERALL validity (not averaged) - total valid / total requested
        total_k_requested = sum(r['article_metrics']['k_requested'] for r in experiments_with_metrics)
        total_n_valid = sum(r['article_metrics']['n_valid'] for r in experiments_with_metrics)
        pct_valid_overall = (total_n_valid / total_k_requested) if total_k_requested > 0 else 0.0
        
        # Calculate validity AFTER filtering to top N
        n_filtered_cfs = sum(r['n_counterfactuals'] for r in results)
        n_filtered_valid = sum(1 for r in results for cf in r['counterfactuals'] if cf['valid'])
        pct_valid_after_filter = (n_filtered_valid / n_filtered_cfs) if n_filtered_cfs > 0 else 0.0
        
        # Calculate priority metrics for ALL valid CFs across experiments
        all_valid_cf_scores = [cf['preference_score'] for r in results for cf in r['counterfactuals'] if cf['valid']]
        if all_valid_cf_scores:
            overall_highest_priority = max(all_valid_cf_scores)
            n_overall_highest = sum(1 for score in all_valid_cf_scores if score == overall_highest_priority)
            pct_overall_highest = (n_overall_highest / len(all_valid_cf_scores)) * 100
        else:
            overall_highest_priority = 0.0
            n_overall_highest = 0
            pct_overall_highest = 0.0
        
        # Print metrics
        logger.info(f"  % Valid CFs: {pct_valid_overall:.2%} ({total_n_valid}/{total_k_requested})")
        logger.info(f"  % Valid CFs after eps: {pct_valid_after_filter:.2%} ({n_filtered_valid}/{n_filtered_cfs})")
        if all_valid_cf_scores:
            logger.info(f"  Highest priority value: {overall_highest_priority:.4f}")
            logger.info(f"  Number of highest CFs: {pct_overall_highest:.2f}% ({n_overall_highest}/{len(all_valid_cf_scores)})")
        logger.info(f"  Continuous-Proximity: {avg_continuous_proximity:.4f}")
        logger.info(f"  Categorical-Proximity: {avg_categorical_proximity:.4f}")
        logger.info(f"  Continuous-Sparsity: {avg_continuous_sparsity:.4f} (1.0 = sparse)")
        logger.info(f"  Continuous-Diversity: {avg_continuous_diversity:.4f}")
        logger.info(f"  Categorical-Diversity: {avg_categorical_diversity:.4f}")
        logger.info(f"  Cont-Count-Diversity: {avg_cont_count_diversity:.4f}")
        logger.info("=" * 80)
    else:
        logger.info("  No counterfactuals generated.")
        logger.info("=" * 80)
    
    return results


def save_results_csv(results, config, experiment, filename='experiment_results_random_search.csv'):
    """
    Save experiment results to CSV file.
    
    Args:
        results: List of experiment results
        config: Experiment configuration
        experiment: Experiment instance
        filename: Output filename
    """
    logger.info(f"Saving results to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header (matching DiCE metrics for comparison + per-experiment article metrics)
        writer.writerow([
            'sample_idx', 'target_idx', 
            'sample_prediction', 'target_prediction',
            'n_counterfactuals', 'total_cf_found',
            'max_preference_score', 'n_cf_with_max_score',
            'computation_time',
            'k_requested', 'n_generated', 'n_valid', 'pct_valid_cfs',
            'continuous_proximity', 'categorical_proximity',
            'continuous_sparsity', 'continuous_diversity',
            'categorical_diversity', 'cont_count_diversity',
            'cf_rank', 'cf_prediction', 'cf_preference_score',
            'cf_l1_distance', 'cf_l2_distance', 'cf_sparsity',
            'cf_pred_error', 'cf_valid'
        ])
        
        # Write results
        for result in results:
            article_m = result.get('article_metrics', {})
            base_row = [
                result['sample_idx'],
                result['target_idx'],
                result['sample_prediction'],
                result['target_prediction'],
                result['n_counterfactuals'],
                result['total_cf_found'],
                result['max_preference_score'],
                result['n_cf_with_max_score'],
                result['computation_time'],
                article_m.get('k_requested', 0),
                article_m.get('n_generated', 0),
                article_m.get('n_valid', 0),
                article_m.get('pct_valid_cfs', 0),
                article_m.get('continuous_proximity', 0),
                article_m.get('categorical_proximity', 1),
                article_m.get('continuous_sparsity', 1),
                article_m.get('continuous_diversity', 0),
                article_m.get('categorical_diversity', 0),
                article_m.get('cont_count_diversity', 0)
            ]
            
            if result['counterfactuals']:
                for cf in result['counterfactuals']:
                    writer.writerow(base_row + [
                        cf['rank'],
                        cf['prediction'],
                        cf['preference_score'],
                        cf['l1_distance'],
                        cf['l2_distance'],
                        cf['sparsity'],
                        cf['pred_error'],
                        cf['valid']
                    ])
            else:
                # No counterfactuals found
                writer.writerow(base_row + ['', '', '', '', '', '', ''])
    
    logger.info(f"Results saved to {filename}")


def print_experiment_results(results, config, experiment):
    """
    Print comprehensive summary of experiment results.
    
    Args:
        results: List of experiment results
        config: Experiment configuration
        experiment: Experiment instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY - RANDOM SEARCH METHOD")
    logger.info("=" * 80)
    
    # Overall statistics
    total_experiments = len(results)
    experiments_with_cfs = sum(1 for r in results if r['n_counterfactuals'] > 0)
    total_cfs = sum(r['n_counterfactuals'] for r in results)
    avg_cfs = total_cfs / experiments_with_cfs if experiments_with_cfs > 0 else 0
    total_valid_cfs = sum(1 for r in results for cf in r['counterfactuals'] if cf['valid'])
    
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Experiments with CFs: {experiments_with_cfs} ({100*experiments_with_cfs/total_experiments:.1f}%)")
    logger.info(f"  Total CFs generated: {total_cfs}")
    logger.info(f"  Valid CFs (within epsilon): {total_valid_cfs} ({100*total_valid_cfs/total_cfs:.1f}% of all CFs)" if total_cfs > 0 else "  Valid CFs: 0")
    logger.info(f"  Average CFs per experiment: {avg_cfs:.2f}")
    
    # Time statistics
    total_time = sum(r['computation_time'] for r in results)
    avg_time = total_time / total_experiments
    logger.info(f"\nComputation Time:")
    logger.info(f"  Total: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"  Average per experiment: {avg_time:.2f} seconds")
    
    # Distance statistics (for all CFs and for valid CFs separately)
    if total_cfs > 0:
        all_l1_distances = [cf['l1_distance'] for r in results for cf in r['counterfactuals']]
        all_l2_distances = [cf['l2_distance'] for r in results for cf in r['counterfactuals']]
        all_sparsities = [cf['sparsity'] for r in results for cf in r['counterfactuals']]
        all_pred_errors = [cf['pred_error'] for r in results for cf in r['counterfactuals']]
        
        logger.info(f"\nAll CFs Statistics:")
        logger.info(f"  L1 distance: min={min(all_l1_distances):.4f}, max={max(all_l1_distances):.4f}, avg={np.mean(all_l1_distances):.4f}")
        logger.info(f"  L2 distance: min={min(all_l2_distances):.4f}, max={max(all_l2_distances):.4f}, avg={np.mean(all_l2_distances):.4f}")
        logger.info(f"  Sparsity: min={min(all_sparsities)}, max={max(all_sparsities)}, avg={np.mean(all_sparsities):.2f}")
        logger.info(f"  Prediction Error: min={min(all_pred_errors):.4f}, max={max(all_pred_errors):.4f}, avg={np.mean(all_pred_errors):.4f}")
        
        # Statistics for valid CFs only
        valid_cfs = [cf for r in results for cf in r['counterfactuals'] if cf['valid']]
        if valid_cfs:
            valid_l1 = [cf['l1_distance'] for cf in valid_cfs]
            valid_l2 = [cf['l2_distance'] for cf in valid_cfs]
            valid_sparsity = [cf['sparsity'] for cf in valid_cfs]
            valid_pred_errors = [cf['pred_error'] for cf in valid_cfs]
            
            logger.info(f"\nValid CFs Only (within epsilon={config['epsilon']}) Statistics:")
            logger.info(f"  Count: {len(valid_cfs)}")
            logger.info(f"  L1 distance: min={min(valid_l1):.4f}, max={max(valid_l1):.4f}, avg={np.mean(valid_l1):.4f}")
            logger.info(f"  L2 distance: min={min(valid_l2):.4f}, max={max(valid_l2):.4f}, avg={np.mean(valid_l2):.4f}")
            logger.info(f"  Sparsity: min={min(valid_sparsity)}, max={max(valid_sparsity)}, avg={np.mean(valid_sparsity):.2f}")
            logger.info(f"  Prediction Error: min={min(valid_pred_errors):.4f}, max={max(valid_pred_errors):.4f}, avg={np.mean(valid_pred_errors):.4f}")
    
    logger.info("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Random Search Explainer Experiments - Auto MPG")
    logger.info("=" * 80)
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    config = {
        # Dataset and model
        'dataset': 'Auto_MPG',          # Dataset to use
        'model': 'NN_Residual',         # Model to use
        
        # Experiment settings
        'n_quantiles': 3,               # Number of equally distributed prediction points
        'epsilon': 2.0,                 # Target prediction tolerance (±2 MPG)
        
        # Preference-based method settings
        'return_top_n': None,           # Keep all counterfactuals (no filtering)
        'exemplar_weight': 0.01,        # Weight assigned to exemplar value
        'n_samples': 10000,             # Number of samples to generate per experiment
        'use_monte_carlo': True,        # Use Monte Carlo sampling
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    
    # ============================================================================
    # SELECT EXPERIMENT
    # ============================================================================
    if config['dataset'] == 'Auto_MPG':
        experiment = AutoMPGExperiment()
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    # ============================================================================
    # LOAD DATA AND MODEL
    # ============================================================================
    X_train, X_test, y_train, y_test, scaler, model, X_full, y_full = experiment.load_data()
    
    # Calculate MAD from training data for article-based metrics
    mad_values = calculate_mad_from_training(X_train)
    logger.info(f"Calculated MAD values for {len(mad_values)} features")
    
    # ============================================================================
    # RUN PREFERENCE-BASED RANDOM SEARCH EXPERIMENT
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING RANDOM SEARCH EXPERIMENT")
    logger.info("=" * 80)
    
    results_random_search = run_counterfactual_experiment(model, X_train, X_test, y_train, config, experiment, mad_values)
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS TO CSV")
    logger.info("=" * 80)
    
    save_results_csv(results_random_search, config, experiment, filename='experiment_results_random_search.csv')
    logger.info("CSV file saved successfully.")
    
    # ============================================================================
    # PRINT SUMMARY
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    
    try:
        print_experiment_results(results_random_search, config, experiment)
    except Exception as e:
        logger.error(f"Error printing results: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
