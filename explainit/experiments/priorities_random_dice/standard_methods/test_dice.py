"""
DiCE (Diverse Counterfactual Explanations) Method Testing Script
Implementation based on Microsoft's DiCE library
https://interpret.ml/DiCE/

DiCE generates diverse counterfactual explanations that help users understand
what they can do to get a favorable prediction.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import sys
import os
from pathlib import Path
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(X_original, X_cf, prediction, target, epsilon):
    """Compute counterfactual quality metrics."""
    l2_distance = float(np.linalg.norm(X_cf - X_original))
    l1_distance = float(np.sum(np.abs(X_cf - X_original)))
    sparsity = int(np.sum(np.abs(X_cf - X_original) > 0.001))
    
    return {
        'l2_distance': l2_distance,
        'l1_distance': l1_distance,
        'sparsity': sparsity
    }


def load_auto_mpg_data():
    """Load Auto MPG dataset and trained model."""
    # Navigate to project root: standard_methods -> priorities_with_random_search -> experiments -> explainit -> project_root
    base_dir = Path(__file__).parent.parent.parent.parent.parent / "ML_models_for_tests" / "ML_regression_results" / "Auto_MPG"
    
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Scale data
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Create dataframes for DiCE (requires feature names AND outcome)
    feature_names = list(X_data.columns)
    y_train_ravel = y_train.values.ravel()
    y_test_ravel = y_test.values.ravel()
    
    # DiCE requires the outcome column in the DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_train_df['mpg'] = y_train_ravel
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    logger.info(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train_df, X_test_df, y_train_ravel, y_test_ravel, model, scaler, feature_names


def dice_counterfactual(
    X_original: np.ndarray,
    model,
    target_value: float,
    X_train_df: pd.DataFrame,
    feature_names: list,
    epsilon: float = 1.0,
    total_CFs: int = 5,
    desired_range=None,
    feature_ranges=None
):
    """
    DiCE counterfactual generation method.
    
    Args:
        X_original: Original instance to explain (numpy array)
        model: TensorFlow/Keras model
        target_value: Target prediction value
        X_train_df: Training dataset as DataFrame
        feature_names: List of feature names
        epsilon: Tolerance for target prediction
        total_CFs: Number of counterfactuals to generate
        desired_range: Tuple (min, max) for desired prediction range
        feature_ranges: List of (min, max) tuples for each feature
    
    Returns:
        counterfactual: Best counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
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
        
        # Prepare data for DiCE
        # DiCE expects a DataFrame with all features AND the outcome column
        d = dice_ml.Data(
            dataframe=X_train_df,
            continuous_features=feature_names,
            outcome_name='mpg'
        )
        
        # Create model wrapper for DiCE
        # Pass the Keras model directly for TF2 backend
        m = dice_ml.Model(model=model, backend='TF2', model_type='regressor')
        
        # Initialize DiCE with random method (faster and more reliable than gradient)
        dice_exp = Dice(d, m, method='random')
        
        # Set desired range if not provided
        if desired_range is None:
            desired_range = [target_value - epsilon, target_value + epsilon]
        
        # Generate counterfactuals
        dice_result = dice_exp.generate_counterfactuals(
            query_instances=X_original_df,
            total_CFs=total_CFs,
            desired_range=desired_range,
            features_to_vary=feature_names,
            verbose=False
        )
        
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
            'diversity_score': float(cf_examples.diversity_score) if hasattr(cf_examples, 'diversity_score') else None
        }
        
        return best_cf, best_pred, info
        
    except Exception as e:
        logger.error(f"DiCE method failed with error: {e}")
        return None, None, {'method': 'DiCE', 'valid': False, 'error': str(e)}


def test_dice_on_auto_mpg():
    """Test DiCE method on Auto MPG dataset."""
    logger.info("=" * 80)
    logger.info("TESTING DICE METHOD ON AUTO MPG")
    logger.info("=" * 80)
    
    # Load data
    X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names = load_auto_mpg_data()
    X_train = X_train_df.values
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
    
    # Define feature ranges
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = X_train[:, idx].min()
        max_val = X_train[:, idx].max()
        feature_ranges.append((float(min_val), float(max_val)))
    
    # Test parameters
    epsilon = 2.0  # ±2 MPG tolerance
    total_CFs = 5  # Generate 5 diverse counterfactuals
    
    # Test DiCE on sample-target pairs
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING DICE EXPERIMENTS")
    logger.info("=" * 80)
    
    results = []
    
    for sample_idx, sample_info in enumerate(samples):
        for target_idx, target_info in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            sample = sample_info['sample']
            sample_pred = sample_info['prediction']
            target_pred = target_info['prediction']
            
            logger.info(f"\nExperiment: Sample {sample_idx+1} → Target {target_idx+1}")
            logger.info(f"  Sample prediction: {sample_pred:.2f} MPG")
            logger.info(f"  Target prediction: {target_pred:.2f} MPG")
            logger.info(f"  Distance: {abs(sample_pred - target_pred):.2f} MPG")
            
            # Run DiCE
            cf, pred, info = dice_counterfactual(
                X_original=sample,
                model=model,
                target_value=target_pred,
                X_train_df=X_train_df,
                feature_names=feature_names,
                epsilon=epsilon,
                total_CFs=total_CFs,
                feature_ranges=feature_ranges
            )
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_pred, epsilon)
                
                logger.info(f"  Result: {'VALID' if info['valid'] else 'INVALID'}")
                logger.info(f"    CF Prediction: {pred:.2f} MPG")
                logger.info(f"    L2 Distance: {metrics['l2_distance']:.4f}")
                logger.info(f"    L1 Distance: {metrics['l1_distance']:.4f}")
                logger.info(f"    Sparsity: {metrics['sparsity']} features changed")
                logger.info(f"    Prediction Error: {abs(pred - target_pred):.4f}")
                logger.info(f"    Generated CFs: {info.get('n_generated', 0)}")
                logger.info(f"    Valid CFs: {info.get('n_valid', 0)}")
                if info.get('diversity_score') is not None:
                    logger.info(f"    Diversity Score: {info['diversity_score']:.4f}")
                
                results.append({
                    'sample_idx': sample_idx + 1,
                    'target_idx': target_idx + 1,
                    'sample_pred': sample_pred,
                    'target_pred': target_pred,
                    'valid': info['valid'],
                    'cf_pred': pred,
                    'l2_distance': metrics['l2_distance'],
                    'l1_distance': metrics['l1_distance'],
                    'sparsity': metrics['sparsity'],
                    'pred_error': abs(pred - target_pred),
                    'n_generated': info.get('n_generated', 0),
                    'n_valid': info.get('n_valid', 0)
                })
            else:
                logger.info(f"  Result: FAILED")
                logger.info(f"    Reason: {info.get('reason', info.get('error', 'unknown'))}")
                
                results.append({
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
                    'n_generated': 0,
                    'n_valid': 0
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path(__file__).parent / 'results' / 'dice_test_results.csv'
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Valid counterfactuals: {results_df['valid'].sum()}")
    logger.info(f"Success rate: {results_df['valid'].mean()*100:.1f}%")
    
    if results_df['valid'].sum() > 0:
        valid_results = results_df[results_df['valid']]
        logger.info(f"\nValid CF Statistics:")
        logger.info(f"  Average L2 Distance: {valid_results['l2_distance'].mean():.4f}")
        logger.info(f"  Average L1 Distance: {valid_results['l1_distance'].mean():.4f}")
        logger.info(f"  Average Sparsity: {valid_results['sparsity'].mean():.2f}")
        logger.info(f"  Average Prediction Error: {valid_results['pred_error'].mean():.4f}")
        logger.info(f"  Average Generated CFs: {valid_results['n_generated'].mean():.2f}")
        logger.info(f"  Average Valid CFs: {valid_results['n_valid'].mean():.2f}")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    test_dice_on_auto_mpg()
