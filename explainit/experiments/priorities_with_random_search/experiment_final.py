"""
Final Counterfactual Explanation Experiments
Configurable framework for running experiments on different datasets and models.

Currently supported:
- Auto MPG dataset with NN_Residual model
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
from explainit.experiments.priorities_with_random_search.standard_methods import run_all_methods

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET AND MODEL LOADERS
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

def run_counterfactual_experiment(model, X_train, X_test, y_train, config, experiment):
    """
    Run a comprehensive counterfactual experiment with multiple samples and targets.
    
    Args:
        model: Trained model
        X_train: Training dataset
        X_test: Test dataset
        y_train: Training predictions
        config: Dictionary with experiment parameters
        experiment: Experiment instance (e.g., AutoMPGExperiment)
    
    Returns:
        results: List of experiment results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING COUNTERFACTUAL EXPERIMENT")
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
            
            # Sort by preference score descending and filter to top N
            if total_cf_found > 0:
                sorted_indices = np.argsort(all_cf_scores)[::-1]
                
                # Keep only top N if specified
                if config.get('return_top_n') and config['return_top_n'] > 0:
                    top_n = min(config['return_top_n'], total_cf_found)
                    sorted_indices = sorted_indices[:top_n]
                    logger.info(f"  Keeping top {top_n} counterfactuals (out of {total_cf_found})")
                
                cf_samples = [all_cf_samples[i] for i in sorted_indices]
                cf_predictions = [all_cf_predictions[i] for i in sorted_indices]
                cf_scores = [all_cf_scores[i] for i in sorted_indices]
            else:
                cf_samples, cf_predictions, cf_scores = [], [], []
            
            # Store results with statistics
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
                'counterfactuals': []
            }
            
            # Calculate L2 distances for each counterfactual
            sample_array = np.array(sample)
            for i, (cf_sample, cf_pred, cf_score) in enumerate(zip(cf_samples, cf_predictions, cf_scores)):
                cf_array = np.array(cf_sample)
                l1_distance = np.sum(np.abs(cf_array - sample_array))
                l2_distance = np.sqrt(np.sum((cf_array - sample_array) ** 2))
                sparsity = np.sum(np.abs(cf_array - sample_array) > 1e-6)
                
                result['counterfactuals'].append({
                    'rank': i + 1,
                    'prediction': cf_pred,
                    'preference_score': cf_score,
                    'sample': cf_sample,
                    'l1_distance': l1_distance,
                    'l2_distance': l2_distance,
                    'sparsity': int(sparsity)
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
    
    return results


def run_standard_methods_experiment(model, X_train, X_test, y_train, config, experiment):
    """
    Run experiment using standard counterfactual methods for comparison.
    
    Args:
        model: Trained model
        X_train: Training dataset
        X_test: Test dataset
        y_train: Training predictions
        config: Dictionary with experiment parameters
        experiment: Experiment instance (e.g., AutoMPGExperiment)
    
    Returns:
        results: List of experiment results from standard methods
    """
    # Get methods to run from config
    methods_to_run = config.get('standard_methods', ['wachter', 'growing_spheres', 'prototype', 'gradient_based'])
    
    logger.info("\n" + "=" * 80)
    logger.info("STARTING STANDARD METHODS EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    
    # Get predictions for all test samples
    logger.info("Getting predictions for test samples...")
    logger.info(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    sys.stderr.flush()
    
    import tensorflow as tf
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
            'sample': X_test[idx],
            'prediction': predictions[idx]
        })
    
    logger.info(f"Selected {len(sample_points)} sample points with predictions:")
    for i, sp in enumerate(sample_points):
        logger.info(f"  Sample {i+1}: prediction = {sp['prediction']:.2f} {experiment.target_name}")
    logger.info("\n")
    
    # Define feature ranges for optimization methods
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = X_train[:, idx].min()
        max_val = X_train[:, idx].max()
        feature_ranges.append((float(min_val), float(max_val)))
    
    # Model prediction wrapper
    def model_predict(X):
        import tensorflow as tf
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
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
            
            # Run selected standard methods
            logger.info(f"  Running standard methods: {', '.join(methods_to_run)}")
            method_start_time = time.time()
            method_results = run_all_methods(
                X_original=sample,
                model=model,
                model_predict=model_predict,
                target_value=target_pred,
                X_train=X_train,
                y_train=y_train,
                epsilon=config['epsilon'],
                feature_ranges=feature_ranges,
                methods_to_run=methods_to_run
            )
            total_standard_time = time.time() - method_start_time
            
            # Store results
            result = {
                'sample_idx': sample_idx + 1,
                'target_idx': target_idx + 1,
                'sample_prediction': sample_pred,
                'target_prediction': target_pred,
                'sample_values': sample.tolist(),
                'computation_time': total_standard_time,
                'methods': method_results
            }
            
            results.append(result)
            
            # Track time for this pair
            pair_elapsed = time.time() - pair_start_time
            pair_times.append(pair_elapsed)
            
            # Log summary
            for method_name, method_result in method_results.items():
                if method_result['counterfactual'] is not None:
                    metrics = method_result['metrics']
                    logger.info(f"    {method_name}: VALID - "
                               f"L2={metrics['l2_distance']:.4f}, "
                               f"Sparsity={metrics['sparsity']}, "
                               f"Pred={method_result['prediction']:.2f}")
                else:
                    logger.info(f"    {method_name}: FAILED")
            logger.info("")
    
    return results


def print_experiment_results(results, config, experiment):
    """
    Print comprehensive results from the counterfactual experiment.
    
    Args:
        results: List of experiment results
        config: Experiment configuration
        experiment: Experiment instance
    """
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 80)
    sys.stderr.flush()
    
    # Overall statistics
    total_experiments = len(results)
    experiments_with_cf = sum(1 for r in results if r['n_counterfactuals'] > 0)
    total_cf = sum(r['n_counterfactuals'] for r in results)
    
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Dataset: {experiment.dataset_name}")
    logger.info(f"  Model: {experiment.model_name}")
    logger.info(f"  Total sample-target pairs: {total_experiments}")
    logger.info(f"  Pairs with counterfactuals: {experiments_with_cf} ({100*experiments_with_cf/total_experiments:.1f}%)")
    logger.info(f"  Total counterfactuals generated: {total_cf}")
    logger.info(f"  Average counterfactuals per pair: {total_cf/total_experiments:.2f}")
    sys.stderr.flush()
    
    # Detailed results
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY BY SAMPLE-TARGET PAIR")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"\nSample {result['sample_idx']} → Target {result['target_idx']}: "
                    f"{result['n_counterfactuals']} counterfactuals found "
                    f"(distance: {abs(result['sample_prediction'] - result['target_prediction']):.2f} {experiment.target_name})")
        
        if result['n_counterfactuals'] > 0 and len(result['counterfactuals']) > 0:
            best_cf = result['counterfactuals'][0]
            best_score = best_cf['preference_score']
            
            # Count how many CFs have the same score as the best one
            count_with_best_score = sum(1 for cf in result['counterfactuals'] 
                                       if cf['preference_score'] == best_score)
            
            logger.info(f"  Best CF: prediction={best_cf['prediction']:.2f} {experiment.target_name}, "
                       f"preference={best_score:.2f}, "
                       f"distance_from_target={abs(best_cf['prediction'] - result['target_prediction']):.2f}")
            logger.info(f"  CFs with best score ({best_score:.2f}): {count_with_best_score} out of {result['n_counterfactuals']}")
    
    sys.stderr.flush()
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY COMPLETE")
    logger.info("=" * 80)
    sys.stderr.flush()


def save_results_csv(results, config, experiment, filename='experiment_results_final.csv'):
    """Save experiment results to CSV."""
    logger.info(f"Saving results to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        config_keys = list(config.keys())
        writer.writerow([
            'dataset', 'model',
            'sample_idx', 'sample_prediction', 'sample_values',
            'target_idx', 'target_prediction', 'exemplar_values',
            'total_cf_found', 'max_preference_score', 'n_cf_with_max_score',
            'computation_time',
            'cf_rank', 'cf_prediction', 'cf_distance_from_target',
            'cf_preference_score', 'cf_l1_distance', 'cf_l2_distance', 'cf_sparsity',
            'cf_values'
        ] + config_keys)
        
        # Config values
        config_values = [config[key] for key in config_keys]
        
        # Data rows
        for result in results:
            base_row = [
                experiment.dataset_name,
                experiment.model_name,
                result['sample_idx'],
                result['sample_prediction'],
                str(result['sample_values']),
                result['target_idx'],
                result['target_prediction'],
                str(result['exemplar_values']),
                result['total_cf_found'],
                result['max_preference_score'],
                result['n_cf_with_max_score'],
                result['computation_time']
            ]
            
            if result['n_counterfactuals'] == 0:
                writer.writerow(base_row + ['', '', '', '', '', '', '', ''] + config_values)
            else:
                for cf in result['counterfactuals']:
                    cf_row = base_row + [
                        cf['rank'],
                        cf['prediction'],
                        abs(cf['prediction'] - result['target_prediction']),
                        cf['preference_score'],
                        cf['l1_distance'],
                        cf['l2_distance'],
                        cf['sparsity'],
                        str(cf['sample'])
                    ] + config_values
                    writer.writerow(cf_row)
    
    logger.info(f"Results saved to {filename}")


def save_standard_methods_results_csv(results, config, experiment, filename='experiment_results_standard_methods.csv'):
    """Save standard methods experiment results to CSV."""
    logger.info(f"Saving standard methods results to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        config_keys = list(config.keys())
        writer.writerow([
            'dataset', 'model',
            'sample_idx', 'sample_prediction', 'sample_values',
            'target_idx', 'target_prediction',
            'computation_time',
            'method', 'valid', 'cf_prediction', 'cf_values',
            'l1_distance', 'l2_distance', 'sparsity',
            'prediction_error', 'additional_info'
        ] + config_keys)
        
        # Config values
        config_values = [config[key] for key in config_keys]
        
        # Data rows
        for result in results:
            base_row = [
                experiment.dataset_name,
                experiment.model_name,
                result['sample_idx'],
                result['sample_prediction'],
                str(result['sample_values']),
                result['target_idx'],
                result['target_prediction'],
                result['computation_time']
            ]
            
            # Write one row per method
            for method_name, method_result in result['methods'].items():
                if method_result['counterfactual'] is not None:
                    metrics = method_result['metrics']
                    method_row = base_row + [
                        method_name,
                        metrics['validity'],
                        method_result['prediction'],
                        str(method_result['counterfactual'].tolist()),
                        metrics['l1_distance'],
                        metrics['l2_distance'],
                        metrics['sparsity'],
                        metrics['prediction_error'],
                        str(method_result['info'])
                    ] + config_values
                else:
                    method_row = base_row + [
                        method_name,
                        False,
                        '', '',
                        '', '', '', '',
                        str(method_result['info'])
                    ] + config_values
                
                writer.writerow(method_row)
    
    logger.info(f"Standard methods results saved to {filename}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Final Counterfactual Explanation Experiments")
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
        
        # Method selection - Set to True/False to enable/disable methods
        'run_preference_method': True,  # Run preference-based random search method
        'run_standard_methods': True,   # Run standard methods
        
        # Standard methods to run (only used if run_standard_methods=True)
        # Options: 'wachter', 'growing_spheres', 'prototype', 'gradient_based'
        'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
        
        # Preference-based method settings (only used if run_preference_method=True)
        'return_top_n': 5,              # Number of top counterfactuals to return per experiment
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
    
    # ============================================================================
    # RUN PREFERENCE-BASED RANDOM SEARCH EXPERIMENT
    # ============================================================================
    if config.get('run_preference_method', True):
        logger.info("\n" + "=" * 80)
        logger.info("PART 1: PREFERENCE-BASED RANDOM SEARCH METHOD")
        logger.info("=" * 80)
        
        results_random_search = run_counterfactual_experiment(model, X_train, X_test, y_train, config, experiment)
        
        # ============================================================================
        # SAVE PREFERENCE-BASED RESULTS
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SAVING PREFERENCE-BASED RESULTS TO CSV")
        logger.info("=" * 80)
        
        save_results_csv(results_random_search, config, experiment, filename='experiment_results_preference_based.csv')
        logger.info("Preference-based CSV file saved successfully.")
        
        # ============================================================================
        # PRINT PREFERENCE-BASED SUMMARY
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PREFERENCE-BASED EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        
        try:
            print_experiment_results(results_random_search, config, experiment)
        except Exception as e:
            logger.error(f"Error printing preference-based results: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("\n" + "=" * 80)
        logger.info("SKIPPING PREFERENCE-BASED METHOD (disabled in config)")
        logger.info("=" * 80)
    
    # ============================================================================
    # RUN STANDARD METHODS EXPERIMENT
    # ============================================================================
    if config.get('run_standard_methods', True):
        logger.info("\n" + "=" * 80)
        logger.info("PART 2: STANDARD COUNTERFACTUAL METHODS")
        logger.info(f"Methods to run: {', '.join(config.get('standard_methods', []))}")
        logger.info("=" * 80)
        
        results_standard = run_standard_methods_experiment(model, X_train, X_test, y_train, config, experiment)
        
        # ============================================================================
        # SAVE STANDARD METHODS RESULTS
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("SAVING STANDARD METHODS RESULTS TO CSV")
        logger.info("=" * 80)
        
        save_standard_methods_results_csv(results_standard, config, experiment, filename='experiment_results_standard_methods.csv')
        logger.info("Standard methods CSV file saved successfully.")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("SKIPPING STANDARD METHODS (disabled in config)")
        logger.info("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
