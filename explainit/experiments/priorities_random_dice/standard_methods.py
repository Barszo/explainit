"""
Standard Counterfactual Explanation Methods
Implementation of counterfactual generation algorithms for comparison with priorities-based methods.

Primary focus: DiCE (Diverse Counterfactual Explanations)
Framework can be extended with additional standard methods.
"""

import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from typing import Callable, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# DiCE METHOD (Diverse Counterfactual Explanations)
# ============================================================================

def dice_counterfactual(
    X_original: np.ndarray,
    model,
    target_value: float,
    X_train_df: pd.DataFrame,
    feature_names: list,
    epsilon: float = 1.0,
    total_CFs: int = 5,
    desired_range: Optional[Tuple[float, float]] = None,
    feature_ranges: Optional[List[Tuple[float, float]]] = None,
    method: str = 'random',
    diversity_weight: float = 1.0,
    proximity_weight: float = 0.5
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    DiCE (Diverse Counterfactual Explanations) method.
    
    Generates diverse counterfactual explanations using the Microsoft DiCE library.
    Reference: https://interpret.ml/DiCE/
    
    Args:
        X_original: Original instance to explain (numpy array)
        model: TensorFlow/Keras model
        target_value: Target prediction value
        X_train_df: Training dataset as DataFrame (must include outcome column)
        feature_names: List of feature names (should match DataFrame columns)
        epsilon: Tolerance for target prediction
        total_CFs: Number of diverse counterfactuals to generate
        desired_range: Optional tuple (min, max) for desired prediction range.
                      If None, defaults to [target_value - epsilon, target_value + epsilon]
        feature_ranges: List of (min, max) tuples for each feature (optional)
        method: DiCE method to use ('random', 'genetic', or 'gradient')
                - 'random': Fastest, most reliable for general use
                - 'genetic': Good for exploring diverse solutions
                - 'gradient': Best for neural networks, may be slower
        diversity_weight: Weight for diversity in CF generation
        proximity_weight: Weight for proximity to original instance
    
    Returns:
        counterfactual: Best counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information including:
              - method: 'DiCE'
              - valid: Whether CF is within epsilon of target
              - distance: L2 distance from original
              - n_generated: Number of CFs generated
              - n_valid: Number of valid CFs (within epsilon)
              - diversity_score: Diversity score of generated CFs (if available)
              - all_cfs: All generated counterfactuals (if requested)
              - all_predictions: Predictions for all CFs (if requested)
    """
    try:
        import dice_ml
        from dice_ml import Dice
    except ImportError:
        logger.error("DiCE library not installed. Install with: pip install dice-ml")
        return None, None, {'method': 'DiCE', 'valid': False, 'error': 'DiCE not installed'}
    
    try:
        # Convert original sample to DataFrame
        X_original_df = pd.DataFrame([X_original], columns=feature_names)
        
        # Determine outcome name from X_train_df
        outcome_name = None
        for col in X_train_df.columns:
            if col not in feature_names:
                outcome_name = col
                break
        
        if outcome_name is None:
            logger.error("X_train_df must contain an outcome column not in feature_names")
            return None, None, {'method': 'DiCE', 'valid': False, 'error': 'No outcome column in training data'}
        
        # Prepare data for DiCE
        d = dice_ml.Data(
            dataframe=X_train_df,
            continuous_features=feature_names,
            outcome_name=outcome_name
        )
        
        # Create model wrapper for DiCE (TF2 backend for neural networks)
        m = dice_ml.Model(model=model, backend='TF2', model_type='regressor')
        
        # Initialize DiCE with specified method
        dice_exp = Dice(d, m, method=method)
        
        # Set desired range
        if desired_range is None:
            desired_range = [target_value - epsilon, target_value + epsilon]
        
        # Generate counterfactuals
        # Note: proximity_weight and diversity_weight are only supported by 'genetic' method
        cf_params = {
            'query_instances': X_original_df,
            'total_CFs': total_CFs,
            'desired_range': desired_range,
            'features_to_vary': feature_names,
            'verbose': False
        }
        
        # Add method-specific parameters
        if method == 'genetic':
            cf_params['proximity_weight'] = proximity_weight
            cf_params['diversity_weight'] = diversity_weight
        
        dice_result = dice_exp.generate_counterfactuals(**cf_params)
        
        # Extract counterfactuals
        cf_examples = dice_result.cf_examples_list[0]
        
        if cf_examples.final_cfs_df is None or len(cf_examples.final_cfs_df) == 0:
            logger.warning("DiCE failed to generate counterfactuals")
            return None, None, {'method': 'DiCE', 'valid': False, 'reason': 'no_cfs_generated'}
        
        # Get predictions for all generated CFs
        cfs_array = cf_examples.final_cfs_df[feature_names].values
        preds_result = model(tf.constant(cfs_array.astype(np.float32)), training=False)
        # Handle both tensor and numpy array returns
        if hasattr(preds_result, 'numpy'):
            predictions = preds_result.numpy().ravel()
        else:
            predictions = np.array(preds_result).ravel()
        
        # Find the best CF (closest to target and within epsilon)
        best_cf = None
        best_pred = None
        best_distance = float('inf')
        valid_cfs = []
        
        for i, (cf, pred) in enumerate(zip(cfs_array, predictions)):
            is_valid = abs(pred - target_value) <= epsilon
            distance = np.linalg.norm(cf - X_original)
            
            if is_valid:
                valid_cfs.append((cf, pred, distance))
                if distance < best_distance:
                    best_cf = cf
                    best_pred = pred
                    best_distance = distance
        
        # If no valid CF found, return the closest one anyway
        if best_cf is None:
            distances = np.linalg.norm(cfs_array - X_original, axis=1)
            pred_errors = np.abs(predictions - target_value)
            
            # Choose CF with smallest prediction error
            best_idx = np.argmin(pred_errors)
            best_cf = cfs_array[best_idx]
            best_pred = predictions[best_idx]
            best_distance = distances[best_idx]
        
        is_valid = abs(best_pred - target_value) <= epsilon
        
        info = {
            'method': 'DiCE',
            'valid': is_valid,
            'distance': float(best_distance),
            'n_generated': len(cfs_array),
            'n_valid': len(valid_cfs),
            'diversity_score': float(cf_examples.diversity_score) if hasattr(cf_examples, 'diversity_score') else None,
            'all_cfs': cfs_array,
            'all_predictions': predictions
        }
        
        return best_cf, best_pred, info
        
    except Exception as e:
        logger.error(f"DiCE method failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {'method': 'DiCE', 'valid': False, 'error': str(e)}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_metrics(
    X_original: np.ndarray,
    X_counterfactual: np.ndarray,
    prediction: float,
    target_value: float,
    epsilon: float
) -> dict:
    """
    Compute standard counterfactual quality metrics.
    
    Args:
        X_original: Original instance
        X_counterfactual: Generated counterfactual
        prediction: Prediction for counterfactual
        target_value: Target prediction value
        epsilon: Tolerance for target
    
    Returns:
        Dictionary with computed metrics:
        - l1_distance: L1 (Manhattan) distance from original
        - l2_distance: L2 (Euclidean) distance from original
        - sparsity: Number of features changed (with 0.001 threshold)
        - validity: Whether prediction is within epsilon of target
        - prediction_error: Absolute prediction error from target
    """
    # L1 and L2 distances
    l1_distance = float(np.sum(np.abs(X_counterfactual - X_original)))
    l2_distance = float(np.linalg.norm(X_counterfactual - X_original))
    
    # Sparsity (number of changed features)
    sparsity = int(np.sum(np.abs(X_counterfactual - X_original) > 0.001))
    
    # Validity (achieves target within epsilon)
    validity = abs(prediction - target_value) <= epsilon
    
    # Prediction error
    prediction_error = abs(prediction - target_value)
    
    return {
        'l1_distance': l1_distance,
        'l2_distance': l2_distance,
        'sparsity': sparsity,
        'validity': validity,
        'prediction_error': prediction_error
    }


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


def compute_article_metrics(
    X_original: np.ndarray,
    all_cfs: np.ndarray,
    all_predictions: np.ndarray,
    target_value: float,
    epsilon: float,
    mad_values: np.ndarray,
    k: int,
    categorical_features: Optional[List[int]] = None
) -> dict:
    """
    Compute counterfactual quality metrics as defined in Mothilal et al. 2020.
    
    This function computes metrics from the DiCE paper (Mothilal et al., FAT* 2020):
    1. Validity: Fraction of CFs that achieve the target prediction (within epsilon)
    2. Proximity: Average distance from original instance to each CF (MAD-normalized)
       - Continuous Proximity: For continuous features
       - Categorical Proximity: For categorical features (if present)
    3. Sparsity: Fraction of unchanged features (normalized to [0,1])
    4. Diversity: Average pairwise distance between CFs (MAD-normalized)
    5. Count Diversity: Fraction of differing features between CF pairs (normalized)
    
    Reference: "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"
               Mothilal et al., FAT* 2020
    
    Args:
        X_original: Original instance (n_features,)
        all_cfs: All generated counterfactuals (n_cfs, n_features)
        all_predictions: Predictions for all CFs (n_cfs,)
        target_value: Target prediction value
        epsilon: Tolerance for target prediction
        mad_values: MAD values for each feature (n_features,)
        k: Number of counterfactuals requested (for validity denominator)
        categorical_features: List of indices of categorical features (None = all continuous)
    
    Returns:
        Dictionary with article-based metrics (matching Mothilal et al. 2020 exactly):
        - pct_valid_cfs: Fraction of valid CFs (within epsilon of target)
        - n_valid: Number of valid CFs
        - n_generated: Number of CFs actually generated
        - continuous_proximity: Negative average MAD-normalized distance to original
        - categorical_proximity: 1 - average categorical distance to original
        - continuous_sparsity: Fraction of unchanged continuous features (1.0 = all unchanged)
        - continuous_diversity: Average MAD-normalized pairwise distance (continuous features)
        - categorical_diversity: Average categorical pairwise distance
        - cont_count_diversity: Fraction of continuous features that differ between CF pairs
    """
    n_generated = len(all_cfs)
    d = len(X_original)  # Total number of features
    
    # Separate continuous and categorical features
    if categorical_features is None:
        categorical_features = []
    
    continuous_features = [i for i in range(d) if i not in categorical_features]
    d_cont = len(continuous_features)
    d_cat = len(categorical_features)
    
    # 1. % VALID CFs (Article metric name)
    # For regression: fraction of CFs where |prediction - target| <= epsilon
    valid_mask = np.abs(all_predictions - target_value) <= epsilon
    n_valid = np.sum(valid_mask)
    pct_valid_cfs = float(n_valid) / k if k > 0 else 0.0
    
    # 2. PROXIMITY (Equations 3, 5, 7)
    
    # 2a. CONTINUOUS PROXIMITY
    # Continuous-Proximity: -(1/k) * Σ dist_cont(ci, x)
    # dist_cont(c, x) = (1/dcont) * Σ |cp - xp| / MADp
    continuous_proximity = 0.0
    if n_generated > 0 and d_cont > 0:
        distances_cont = []
        for cf in all_cfs:
            # MAD-normalized L1 distance for continuous features only
            cont_diff = np.abs(cf[continuous_features] - X_original[continuous_features]) / mad_values[continuous_features]
            dist = np.mean(cont_diff)  # Average over continuous features (divides by d_cont)
            distances_cont.append(dist)
        
        # Proximity is negative average (for minimization in paper)
        continuous_proximity = -np.mean(distances_cont)
    
    # 2b. CATEGORICAL PROXIMITY
    # Categorical-Proximity: 1 - (1/k) * Σ dist_cat(ci, x)
    # dist_cat(c, x) = (1/dcat) * Σ 1[cp ≠ xp]
    categorical_proximity = 1.0  # Default if no categorical features
    if n_generated > 0 and d_cat > 0:
        distances_cat = []
        for cf in all_cfs:
            # Count changed categorical features
            n_changed = np.sum(np.abs(cf[categorical_features] - X_original[categorical_features]) > 1e-6)
            dist = n_changed / d_cat  # Fraction changed
            distances_cat.append(dist)
        
        # Categorical proximity (1 = no changes, 0 = all changed)
        categorical_proximity = 1.0 - np.mean(distances_cat)
    
    # 3. CONTINUOUS-SPARSITY (Article version - for continuous features ONLY)
    # Continuous-Sparsity: 1 - (1/(k*d_cont)) * Σ_i Σ_p 1[c_i,p ≠ x_p]
    # Higher is better (1 = no changes, 0 = all continuous features changed)
    continuous_sparsity = 1.0
    if n_generated > 0 and d_cont > 0:
        total_changes = 0
        for cf in all_cfs:
            # Count changes in CONTINUOUS features only
            n_changes = np.sum(np.abs(cf[continuous_features] - X_original[continuous_features]) > 1e-6)
            total_changes += n_changes
        
        continuous_sparsity = 1.0 - (total_changes / (n_generated * d_cont))
    
    # 4. DIVERSITY METRICS (Distance-based, Section 4.1)
    # Separate metrics for continuous and categorical features
    
    continuous_diversity = 0.0
    categorical_diversity = 0.0
    cont_count_diversity = 0.0
    
    if n_generated > 1:
        # Calculate pairwise distances
        pairwise_distances_cont = []
        pairwise_distances_cat = []
        cont_count_differences = []  # For cont-count-diversity (continuous features only)
        
        for i in range(n_generated):
            for j in range(i + 1, n_generated):
                # CONTINUOUS-DIVERSITY: MAD-normalized distance for continuous features
                # (1/C(k,2)) * Σ_(i<j) dist_cont(c_i, c_j)
                if d_cont > 0:
                    cont_diff = np.abs(all_cfs[i][continuous_features] - all_cfs[j][continuous_features]) / mad_values[continuous_features]
                    dist_cont = np.mean(cont_diff)  # Average over continuous features (divides by d_cont)
                    pairwise_distances_cont.append(dist_cont)
                
                # CATEGORICAL-DIVERSITY: Categorical distance
                # (1/C(k,2)) * Σ_(i<j) dist_cat(c_i, c_j)
                if d_cat > 0:
                    n_cat_diff = np.sum(np.abs(all_cfs[i][categorical_features] - all_cfs[j][categorical_features]) > 1e-6)
                    dist_cat = n_cat_diff / d_cat  # Fraction different (divides by d_cat)
                    pairwise_distances_cat.append(dist_cat)
                
                # CONT-COUNT-DIVERSITY: Count of differing CONTINUOUS features (normalized)
                # (1/(C(k,2)*d_cont)) * Σ_(i<j) Σ_p 1[c_i,p ≠ c_j,p] for continuous features only
                if d_cont > 0:
                    n_cont_diff = np.sum(np.abs(all_cfs[i][continuous_features] - all_cfs[j][continuous_features]) > 1e-6)
                    cont_count_differences.append(n_cont_diff)
        
        # Compute final diversity metrics
        if d_cont > 0 and len(pairwise_distances_cont) > 0:
            continuous_diversity = np.mean(pairwise_distances_cont)
        
        if d_cat > 0 and len(pairwise_distances_cat) > 0:
            categorical_diversity = np.mean(pairwise_distances_cat)
        
        # Cont-Count-Diversity: normalized by d_cont
        if d_cont > 0 and len(cont_count_differences) > 0:
            cont_count_diversity = np.mean(cont_count_differences) / d_cont
    
    return {
        'pct_valid_cfs': pct_valid_cfs,
        'n_valid': int(n_valid),
        'n_generated': n_generated,
        'continuous_proximity': float(continuous_proximity),
        'categorical_proximity': float(categorical_proximity),
        'continuous_sparsity': float(continuous_sparsity),
        'continuous_diversity': float(continuous_diversity),
        'categorical_diversity': float(categorical_diversity),
        'cont_count_diversity': float(cont_count_diversity)
    }


def run_all_methods(
    X_original: np.ndarray,
    model,
    target_value: float,
    X_train_df: pd.DataFrame,
    feature_names: list,
    epsilon: float = 1.0,
    feature_ranges: Optional[List[Tuple[float, float]]] = None,
    methods_to_run: Optional[List[str]] = None,
    dice_params: Optional[dict] = None
) -> dict:
    """
    Run selected standard counterfactual methods and compare results.
    
    Currently supports: DiCE
    Can be extended with additional methods (Wachter, Growing Spheres, etc.)
    
    Args:
        X_original: Original instance to explain
        model: TensorFlow/Keras model
        target_value: Target prediction value
        X_train_df: Training dataset as DataFrame (with outcome column)
        feature_names: List of feature names
        epsilon: Tolerance for target prediction
        feature_ranges: List of (min, max) tuples for each feature
        methods_to_run: List of method names to run. Options: ['dice']
                       If None, runs all available methods.
        dice_params: Optional dictionary with DiCE-specific parameters:
                    - total_CFs: Number of CFs to generate (default: 5)
                    - method: 'random', 'genetic', or 'gradient' (default: 'random')
                    - diversity_weight: Weight for diversity (default: 1.0)
                    - proximity_weight: Weight for proximity (default: 0.5)
    
    Returns:
        Dictionary with results from selected methods.
        Each method entry contains:
        - counterfactual: Generated CF (or None if failed)
        - prediction: Prediction for CF
        - metrics: Quality metrics (if successful)
        - info: Method-specific information
    """
    # Default to all methods if not specified
    if methods_to_run is None:
        methods_to_run = ['dice']
    
    # Default DiCE parameters
    if dice_params is None:
        dice_params = {}
    dice_total_cfs = dice_params.get('total_CFs', 5)
    dice_method = dice_params.get('method', 'random')
    dice_diversity = dice_params.get('diversity_weight', 1.0)
    dice_proximity = dice_params.get('proximity_weight', 0.5)
    
    results = {}
    
    # DiCE method
    if 'dice' in methods_to_run:
        logger.info("  Running DiCE method...")
        try:
            cf_dice, pred_dice, info_dice = dice_counterfactual(
                X_original=X_original,
                model=model,
                target_value=target_value,
                X_train_df=X_train_df,
                feature_names=feature_names,
                epsilon=epsilon,
                total_CFs=dice_total_cfs,
                feature_ranges=feature_ranges,
                method=dice_method,
                diversity_weight=dice_diversity,
                proximity_weight=dice_proximity
            )
            
            if cf_dice is not None:
                metrics_dice = compute_metrics(X_original, cf_dice, pred_dice, target_value, epsilon)
                results['dice'] = {
                    'counterfactual': cf_dice,
                    'prediction': pred_dice,
                    'metrics': metrics_dice,
                    'info': info_dice
                }
            else:
                results['dice'] = {'counterfactual': None, 'info': info_dice}
        except Exception as e:
            logger.warning(f"  DiCE method failed: {e}")
            results['dice'] = {'counterfactual': None, 'info': {'error': str(e)}}
    
    # Additional methods can be added here
    # Example placeholders:
    
    # if 'wachter' in methods_to_run:
    #     logger.info("  Running Wachter's method...")
    #     # Implementation would go here
    #     pass
    
    # if 'growing_spheres' in methods_to_run:
    #     logger.info("  Running Growing Spheres...")
    #     # Implementation would go here
    #     pass
    
    return results


# ============================================================================
# DATA LOADING (for orchestrator)
# ============================================================================

def load_auto_mpg_data():
    """
    Load Auto MPG dataset and trained model for testing.
    
    Returns:
        X_train_df: Training data as DataFrame (with outcome column)
        X_test_df: Test data as DataFrame  
        y_train: Training predictions
        y_test: Test predictions
        model: Trained TensorFlow model
        scaler: Feature scaler
        feature_names: List of feature names
    """
    import pickle
    import os
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    # Navigate to Auto MPG data directory
    base_dir = Path(__file__).parent.parent.parent.parent / "ML_models_for_tests" / "ML_regression_results" / "Auto_MPG"
    
    logger.info(f"Loading Auto MPG dataset from {base_dir}...")
    
    # Load data
    X_data = pd.read_csv(base_dir / "X_data.csv")
    y_data = pd.read_csv(base_dir / "y_data.csv")
    
    # Load scaler
    with open(base_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Load trained model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    model = tf.keras.models.load_model(
        base_dir / "NN_Residual_model.h5",
        compile=False
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Build the model
    _ = model(tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32))
    
    # Split data (same as experiment)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Scale data
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Create dataframes for methods (DiCE requires outcome column)
    feature_names = list(X_data.columns)
    y_train_ravel = y_train.values.ravel()
    y_test_ravel = y_test.values.ravel()
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_train_df['mpg'] = y_train_ravel
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    logger.info(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train_df, X_test_df, y_train_ravel, y_test_ravel, model, scaler, feature_names


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_all_standard_methods_tests():
    """
    Main orchestrator function to run all standard method tests.
    
    This function:
    1. Loads Auto MPG dataset and model
    2. Selects test samples (low, medium, high predictions)
    3. Runs all available methods on all sample-target pairs
    4. Saves results to CSV files
    """
    from pathlib import Path
    
    logger.info("=" * 80)
    logger.info("STANDARD METHODS TESTING ORCHESTRATOR")
    logger.info("=" * 80)
    
    # Load data
    X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names = load_auto_mpg_data()
    X_train = X_train_df[feature_names].values
    X_test = X_test_df.values
    
    # Get predictions
    predictions = model(tf.constant(X_test), training=False).numpy().ravel()
    
    # Select 3 test samples (low, medium, high MPG predictions)
    quantiles = np.linspace(0, 1, 3)
    quantile_values = np.quantile(predictions, quantiles)
    
    samples = []
    for q_val in quantile_values:
        distances = np.abs(predictions - q_val)
        idx = np.argmin(distances)
        samples.append({
            'index': idx,
            'sample': X_test[idx],
            'prediction': predictions[idx],
            'actual': y_test[idx]
        })
    
    logger.info(f"\nSelected test samples:")
    for i, s in enumerate(samples):
        logger.info(f"  Sample {i+1}: Prediction={s['prediction']:.2f} MPG, Actual={s['actual']:.2f} MPG")
    
    # Calculate MAD from training data for article-based metrics
    logger.info("\nCalculating MAD (Median Absolute Deviation) from training data...")
    mad_values = calculate_mad_from_training(X_train)
    logger.info(f"MAD values calculated for {len(mad_values)} features")
    
    # Define feature ranges
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = X_train[:, idx].min()
        max_val = X_train[:, idx].max()
        feature_ranges.append((float(min_val), float(max_val)))
    
    # Test parameters
    epsilon = 2.0  # ±2 MPG tolerance
    methods_to_run = ['dice']  # Add more methods here as they are implemented
    
    # DiCE-specific parameters
    dice_params = {
        'total_CFs': 5,
        'method': 'random',
        'diversity_weight': 1.0,
        'proximity_weight': 0.5
    }
    
    # Run experiments on all sample-target pairs
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EXPERIMENTS ON ALL METHODS")
    logger.info("=" * 80)
    
    all_results = []
    article_results_aggregated = []  # Article-based metrics (per experiment)
    article_results_detailed = []  # Article-based metrics (per CF)
    
    for sample_idx, sample_info in enumerate(samples):
        for target_idx, target_info in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            sample = sample_info['sample']
            sample_pred = sample_info['prediction']
            target_pred = target_info['prediction']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Experiment: Sample {sample_idx+1} → Target {target_idx+1}")
            logger.info(f"  Sample prediction: {sample_pred:.2f} MPG")
            logger.info(f"  Target prediction: {target_pred:.2f} MPG")
            logger.info(f"  Distance: {abs(sample_pred - target_pred):.2f} MPG")
            logger.info('='*80)
            
            # Run all methods
            results = run_all_methods(
                X_original=sample,
                model=model,
                target_value=target_pred,
                X_train_df=X_train_df,
                feature_names=feature_names,
                epsilon=epsilon,
                feature_ranges=feature_ranges,
                methods_to_run=methods_to_run,
                dice_params=dice_params
            )
            
            # Process results for each method
            for method_name, method_result in results.items():
                if method_result['counterfactual'] is not None:
                    cf = method_result['counterfactual']
                    pred = method_result['prediction']
                    metrics = method_result['metrics']
                    info = method_result['info']
                    
                    logger.info(f"\n  {method_name.upper()}: {'VALID' if info['valid'] else 'INVALID'}")
                    logger.info(f"    CF Prediction: {pred:.2f} MPG")
                    logger.info(f"    L2 Distance: {metrics['l2_distance']:.4f}")
                    logger.info(f"    L1 Distance: {metrics['l1_distance']:.4f}")
                    logger.info(f"    Sparsity: {metrics['sparsity']} features changed")
                    logger.info(f"    Prediction Error: {metrics['prediction_error']:.4f}")
                    
                    # Add method-specific info
                    if method_name == 'dice':
                        logger.info(f"    Generated CFs: {info.get('n_generated', 0)}")
                        logger.info(f"    Valid CFs: {info.get('n_valid', 0)}")
                        if info.get('diversity_score') is not None:
                            logger.info(f"    Diversity Score: {info['diversity_score']:.4f}")
                    
                    # Store standard results (keep as-is)
                    all_results.append({
                        'method': method_name,
                        'sample_idx': sample_idx + 1,
                        'target_idx': target_idx + 1,
                        'sample_pred': sample_pred,
                        'target_pred': target_pred,
                        'valid': info['valid'],
                        'cf_pred': pred,
                        'l2_distance': metrics['l2_distance'],
                        'l1_distance': metrics['l1_distance'],
                        'sparsity': metrics['sparsity'],
                        'pred_error': metrics['prediction_error'],
                        **{k: v for k, v in info.items() if k not in ['method', 'valid', 'distance', 'all_cfs', 'all_predictions']}
                    })
                    
                    # ===== ARTICLE-BASED METRICS (NEW) =====
                    # Compute metrics from Mothilal et al. 2020
                    if 'all_cfs' in info and 'all_predictions' in info:
                        all_cfs = info['all_cfs']
                        all_predictions = info['all_predictions']
                        k_requested = dice_params.get('total_CFs', 5)
                        
                        article_metrics = compute_article_metrics(
                            X_original=sample,
                            all_cfs=all_cfs,
                            all_predictions=all_predictions,
                            target_value=target_pred,
                            epsilon=epsilon,
                            mad_values=mad_values,
                            k=k_requested,
                            categorical_features=None  # Auto MPG has no categorical features
                        )
                        
                        logger.info(f"\n  ARTICLE-BASED METRICS (Mothilal et al. 2020):")
                        logger.info(f"    % Valid CFs: {article_metrics['pct_valid_cfs']:.2%} ({article_metrics['n_valid']}/{k_requested})")
                        logger.info(f"    Continuous-Proximity: {article_metrics['continuous_proximity']:.4f}")
                        logger.info(f"    Categorical-Proximity: {article_metrics['categorical_proximity']:.4f}")
                        logger.info(f"    Continuous-Sparsity: {article_metrics['continuous_sparsity']:.4f} (1.0 = sparse)")
                        logger.info(f"    Continuous-Diversity: {article_metrics['continuous_diversity']:.4f}")
                        logger.info(f"    Categorical-Diversity: {article_metrics['categorical_diversity']:.4f}")
                        logger.info(f"    Cont-Count-Diversity: {article_metrics['cont_count_diversity']:.4f}")
                        
                        # Store aggregated article metrics (per experiment)
                        article_results_aggregated.append({
                            'method': method_name,
                            'sample_idx': sample_idx + 1,
                            'target_idx': target_idx + 1,
                            'sample_pred': sample_pred,
                            'target_pred': target_pred,
                            'k_requested': k_requested,
                            'n_generated': article_metrics['n_generated'],
                            'pct_valid_cfs': article_metrics['pct_valid_cfs'],
                            'n_valid': article_metrics['n_valid'],
                            'continuous_proximity': article_metrics['continuous_proximity'],
                            'categorical_proximity': article_metrics['categorical_proximity'],
                            'continuous_sparsity': article_metrics['continuous_sparsity'],
                            'continuous_diversity': article_metrics['continuous_diversity'],
                            'categorical_diversity': article_metrics['categorical_diversity'],
                            'cont_count_diversity': article_metrics['cont_count_diversity']
                        })
                        
                        # Store detailed article metrics (per CF)
                        for cf_idx, (cf, cf_pred) in enumerate(zip(all_cfs, all_predictions)):
                            is_valid = abs(cf_pred - target_pred) <= epsilon
                            
                            # Calculate MAD-normalized distance to original
                            mad_normalized_dist = np.mean(np.abs(cf - sample) / mad_values)
                            
                            # Count changed features
                            n_changed = np.sum(np.abs(cf - sample) > 1e-6)
                            
                            article_results_detailed.append({
                                'method': method_name,
                                'sample_idx': sample_idx + 1,
                                'target_idx': target_idx + 1,
                                'cf_idx': cf_idx + 1,
                                'sample_pred': sample_pred,
                                'target_pred': target_pred,
                                'cf_pred': cf_pred,
                                'is_valid': is_valid,
                                'pred_error': abs(cf_pred - target_pred),
                                'mad_normalized_distance': mad_normalized_dist,
                                'n_features_changed': n_changed
                            })
                    
                else:
                    logger.info(f"\n  {method_name.upper()}: FAILED")
                    logger.info(f"    Reason: {method_result['info'].get('reason', method_result['info'].get('error', 'unknown'))}")
                    
                    all_results.append({
                        'method': method_name,
                        'sample_idx': sample_idx + 1,
                        'target_idx': target_idx + 1,
                        'sample_pred': sample_pred,
                        'target_pred': target_pred,
                        'valid': False,
                        'cf_pred': None,
                        'l2_distance': None,
                        'l1_distance': None,
                        'sparsity': None,
                        'pred_error': None,
                        'error': str(method_result['info'].get('reason', method_result['info'].get('error', 'unknown')))
                    })
    
    # Save results (keep existing CSV unchanged)
    results_df = pd.DataFrame(all_results)
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'all_standard_methods_results.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nStandard results saved to: {output_file}")
    
    # Save article-based metrics to NEW CSV files
    if article_results_aggregated:
        article_agg_df = pd.DataFrame(article_results_aggregated)
        article_agg_file = output_dir / 'article_metrics_aggregated.csv'
        article_agg_df.to_csv(article_agg_file, index=False)
        logger.info(f"Article metrics (aggregated) saved to: {article_agg_file}")
    
    if article_results_detailed:
        article_det_df = pd.DataFrame(article_results_detailed)
        article_det_file = output_dir / 'article_metrics_detailed.csv'
        article_det_df.to_csv(article_det_file, index=False)
        logger.info(f"Article metrics (detailed per CF) saved to: {article_det_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - ALL METHODS")
    logger.info("=" * 80)
    
    for method_name in methods_to_run:
        method_results = results_df[results_df['method'] == method_name]
        logger.info(f"\n{method_name.upper()}:")
        logger.info(f"  Total experiments: {len(method_results)}")
        logger.info(f"  Valid counterfactuals: {method_results['valid'].sum()}")
        logger.info(f"  Success rate: {method_results['valid'].mean()*100:.1f}%")
        
        if method_results['valid'].sum() > 0:
            valid_results = method_results[method_results['valid']]
            logger.info(f"  Average L2 Distance: {valid_results['l2_distance'].mean():.4f}")
            logger.info(f"  Average L1 Distance: {valid_results['l1_distance'].mean():.4f}")
            logger.info(f"  Average Sparsity: {valid_results['sparsity'].mean():.2f}")
            logger.info(f"  Average Prediction Error: {valid_results['pred_error'].mean():.4f}")
    
    # Print article-based metrics summary
    if article_results_aggregated:
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY - ARTICLE-BASED METRICS (Mothilal et al. 2020)")
        logger.info("=" * 80)
        
        article_agg_df = pd.DataFrame(article_results_aggregated)
        for method_name in methods_to_run:
            method_article = article_agg_df[article_agg_df['method'] == method_name]
            if len(method_article) > 0:
                logger.info(f"\n{method_name.upper()}:")
                logger.info(f"  Total experiments: {len(method_article)}")
                logger.info(f"  Average % Valid CFs: {method_article['pct_valid_cfs'].mean()*100:.1f}%")
                logger.info(f"  Average Continuous-Proximity: {method_article['continuous_proximity'].mean():.4f}")
                logger.info(f"  Average Categorical-Proximity: {method_article['categorical_proximity'].mean():.4f}")
                logger.info(f"  Average Continuous-Sparsity: {method_article['continuous_sparsity'].mean():.4f}")
                logger.info(f"  Average Continuous-Diversity: {method_article['continuous_diversity'].mean():.4f}")
                logger.info(f"  Average Categorical-Diversity: {method_article['categorical_diversity'].mean():.4f}")
                logger.info(f"  Average Cont-Count-Diversity: {method_article['cont_count_diversity'].mean():.4f}")
                logger.info(f"  Average Valid CFs per experiment: {method_article['n_valid'].mean():.2f}")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    run_all_standard_methods_tests()
