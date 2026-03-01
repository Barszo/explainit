"""
Standard Methods Test Manager
============================

This script manages testing of all four standard counterfactual methods:
1. Wachter's Method (optimization-based)
2. Growing Spheres (geometric search with interpolation)
3. Prototype-Based (nearest real training instance)
4. Gradient-Based (neural network gradient descent)

HOW TO USE:
-----------
1. Configure dataset and model paths in DATASET_CONFIG
2. Set parameters for each method in METHOD_CONFIGS
3. Run: python stand_meth_test_manager.py
4. Results saved as CSV files: wachter_results.csv, growing_spheres_results.csv, etc.

PARAMETER SELECTION GUIDE:
--------------------------

WACHTER METHOD:
  - lambda (0.01-10.0): Weight for prediction vs distance
      * Low (0.01-0.1): Prioritizes staying close to original (smaller distance)
      * Medium (0.5-1.0): Balanced approach (recommended starting point)
      * High (5.0-10.0): Prioritizes reaching target (may travel far)
  - epsilon (0.5-5.0): Tolerance for target prediction (MPG units)
      * Small (0.5-1.0): Strict target matching (fewer solutions)
      * Large (3.0-5.0): Relaxed matching (more solutions, less precise)

GROWING SPHERES:
  - epsilon (0.5-10.0): Tolerance for target prediction
      * Smaller → Fewer training prototypes available → May fail
      * Larger → More prototypes → Higher success rate
  - n_search_samples (5-100): Interpolation granularity
      * Low (5-10): Fast but coarse, may miss closer points
      * Medium (20-50): Good balance (recommended)
      * High (50-100): Precise but slower
  - n_top_candidates (5-20): Number of prototypes to try
      * Low (5): Fast, tries nearest prototypes only
      * Medium (10): Good balance (recommended)
      * High (20): Explores more options, slower

PROTOTYPE-BASED:
  - epsilon (0.5-10.0): Tolerance for target prediction
      * Same as Growing Spheres - controls prototype availability
  - top_k (1-10): Which k-th nearest prototype to select
      * 1: Returns closest prototype (smallest distance)
      * 2-5: Alternative prototypes (more diverse)
      * Higher: More distant prototypes (less useful typically)

GRADIENT-BASED (Neural Networks Only):
  - learning_rate (0.001-0.5): Gradient descent step size
      * Too small (0.001): Slow convergence, may timeout
      * Optimal (0.01-0.05): Good balance (recommended)
      * Too large (0.1+): Unstable, may overshoot
  - lambda (0.01-5.0): Weight for prediction vs distance
      * Same interpretation as Wachter method
  - epsilon (0.5-5.0): Tolerance for target prediction
      * Same interpretation as other methods
  - max_iter (100-1000): Maximum optimization iterations
      * Typically converges in 50-200 iterations

DATASET CONFIGURATION:
----------------------
Update DATASET_CONFIG to point to your dataset:
  - base_dir: Path to dataset folder
  - X_data_file: Feature data CSV
  - y_data_file: Target data CSV
  - scaler_file: Fitted scaler pickle file
  - model_file: Trained model file (.h5 for Keras)
  - feature_names: List of feature names for reporting

"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# Dataset and Model Configuration (shared across all methods)
DATASET_CONFIG = {
    'name': 'Auto_MPG',
    'base_dir': Path(__file__).parent.parent.parent.parent.parent / "ML_models_for_tests" / "ML_regression_results" / "Auto_MPG",
    'X_data_file': 'X_data.csv',
    'y_data_file': 'y_data.csv',
    'scaler_file': 'scaler.pkl',
    'model_file': 'NN_Residual_model.h5',
    'feature_names': ['displacement', 'horsepower', 'weight', 'acceleration'],
    'test_size': 0.2,
    'random_state': 42
}

# Wachter Method Parameters
WACHTER_CONFIG = {
    'enabled': True,
    'lambda_values': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],  # Prediction vs distance weight
    'epsilon_values': [0.5, 1.0, 2.0, 3.0, 5.0],         # Target tolerance (MPG)
    'max_iter': 1000,                                     # Maximum optimization iterations
    'output_file': 'wachter_results.csv'
}

# Growing Spheres Parameters
GROWING_SPHERES_CONFIG = {
    'enabled': True,
    'epsilon_values': [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],  # Target tolerance (MPG)
    'n_search_samples_values': [5, 10, 20, 50, 100],    # Interpolation granularity
    'n_top_candidates_values': [5, 10, 20],              # Number of prototypes to try
    'output_file': 'growing_spheres_results.csv'
}

# Prototype-Based Parameters
PROTOTYPE_CONFIG = {
    'enabled': True,
    'epsilon_values': [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],  # Target tolerance (MPG)
    'top_k_values': [1, 2, 3, 5, 10],                    # Which k-th nearest prototype
    'output_file': 'prototype_results.csv'
}

# Gradient-Based Parameters (Neural Networks Only)
GRADIENT_CONFIG = {
    'enabled': True,
    'learning_rate_values': [0.001, 0.01, 0.05, 0.1, 0.5],  # Gradient step size
    'lambda_values': [0.01, 0.1, 0.5, 1.0, 5.0],            # Prediction vs distance weight
    'epsilon_values': [0.5, 1.0, 2.0, 3.0, 5.0],            # Target tolerance (MPG)
    'max_iter': 500,                                         # Maximum iterations
    'output_file': 'gradient_based_results.csv'
}

# Output directory for results
OUTPUT_DIR = Path(__file__).parent / 'results'

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


def load_dataset_and_model(config):
    """
    Load dataset and model based on configuration.
    
    Args:
        config: Dictionary with dataset configuration
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, y_train_pred, model, scaler, feature_names)
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    
    base_dir = config['base_dir']
    logger.info(f"Loading dataset: {config['name']}")
    logger.info(f"  From: {base_dir}")
    
    # Load data
    X_data = pd.read_csv(base_dir / config['X_data_file'])
    y_data = pd.read_csv(base_dir / config['y_data_file'])
    
    # Load scaler
    with open(base_dir / config['scaler_file'], 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    model = tf.keras.models.load_model(
        base_dir / config['model_file'],
        compile=False
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Build model
    _ = model(tf.constant([[0.0] * X_data.shape[1]], dtype=tf.float32))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # Scale data
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Get training predictions
    y_train_pred = model(tf.constant(X_train_scaled), training=False).numpy().ravel()
    
    logger.info(f"  Train samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {config['feature_names']}")
    
    return (X_train_scaled, X_test_scaled, 
            y_train.values.ravel(), y_test.values.ravel(), 
            y_train_pred, model, scaler, config['feature_names'])


def get_test_samples(model, X_test, y_test):
    """
    Select test samples for experimentation (3 quantiles).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        List of sample dictionaries
    """
    import tensorflow as tf
    
    # Get predictions
    predictions = model(tf.constant(X_test), training=False).numpy().ravel()
    
    # Select 3 quantile points (low, medium, high)
    quantiles = np.linspace(0, 1, 3)
    quantile_values = np.quantile(predictions, quantiles)
    
    samples = []
    for i, q_val in enumerate(quantile_values):
        distances = np.abs(predictions - q_val)
        idx = np.argmin(distances)
        samples.append({
            'index': idx,
            'sample': X_test[idx],
            'prediction': predictions[idx],
            'actual': y_test[idx],
            'quantile': i
        })
    
    logger.info(f"\nSelected test samples:")
    for i, s in enumerate(samples):
        logger.info(f"  Sample {i+1}: Prediction={s['prediction']:.2f}, Actual={s['actual']:.2f}")
    
    return samples


def run_wachter_tests(model, X_train, samples, config, feature_names):
    """Run Wachter method tests with configured parameters."""
    from test_wachter_method import wachter_always_return, compute_metrics
    import tensorflow as tf
    
    if not config['enabled']:
        logger.info("Wachter tests disabled")
        return None
    
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING WACHTER METHOD TESTS")
    logger.info("=" * 80)
    
    # Define feature ranges
    feature_ranges = [(float(X_train[:, i].min()), float(X_train[:, i].max())) 
                     for i in range(X_train.shape[1])]
    
    # Model prediction wrapper
    def model_predict(X):
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    results = []
    
    # Test all sample-target combinations
    for sample_idx, sample_point in enumerate(samples):
        for target_idx, target_point in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            logger.info(f"\nTesting: Sample {sample_idx+1} → Target {target_idx+1}")
            
            # Test all parameter combinations
            for epsilon in config['epsilon_values']:
                for lambda_param in config['lambda_values']:
                    cf, pred, info = wachter_always_return(
                        X_original=sample_point['sample'],
                        model_predict=model_predict,
                        target_value=target_point['prediction'],
                        epsilon=epsilon,
                        lambda_param=lambda_param,
                        feature_ranges=feature_ranges,
                        max_iter=config['max_iter']
                    )
                    
                    if cf is not None:
                        metrics = compute_metrics(sample_point['sample'], cf, pred, 
                                                 target_point['prediction'], epsilon)
                        
                        results.append({
                            'method': 'Wachter',
                            'sample_idx': sample_idx + 1,
                            'target_idx': target_idx + 1,
                            'sample_prediction': sample_point['prediction'],
                            'target_prediction': target_point['prediction'],
                            'prediction_change': target_point['prediction'] - sample_point['prediction'],
                            'epsilon': epsilon,
                            'lambda': lambda_param,
                            'valid': info['valid'],
                            'cf_prediction': pred,
                            'prediction_error': abs(pred - target_point['prediction']),
                            'l2_distance': metrics['l2_distance'],
                            'l1_distance': metrics['l1_distance'],
                            'sparsity': metrics['sparsity'],
                            'iterations': info['iterations'],
                            'optimizer': info.get('optimizer_used', 'unknown'),
                            'final_loss': info.get('final_loss', None)
                        })
    
    logger.info(f"\nWachter tests complete: {len(results)} results")
    return pd.DataFrame(results)


def run_growing_spheres_tests(model, X_train, y_train_pred, samples, config, feature_names):
    """Run Growing Spheres method tests with configured parameters."""
    from test_growing_spheres_method import growing_spheres_modified, compute_metrics
    import tensorflow as tf
    
    if not config['enabled']:
        logger.info("Growing Spheres tests disabled")
        return None
    
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING GROWING SPHERES METHOD TESTS")
    logger.info("=" * 80)
    
    # Model prediction wrapper
    def model_predict(X):
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    results = []
    
    # Test all sample-target combinations
    for sample_idx, sample_point in enumerate(samples):
        for target_idx, target_point in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            logger.info(f"\nTesting: Sample {sample_idx+1} → Target {target_idx+1}")
            
            # Test all parameter combinations
            for epsilon in config['epsilon_values']:
                for n_search in config['n_search_samples_values']:
                    for n_cand in config['n_top_candidates_values']:
                        cf, pred, info = growing_spheres_modified(
                            X_original=sample_point['sample'],
                            model_predict=model_predict,
                            target_value=target_point['prediction'],
                            X_train=X_train,
                            y_train=y_train_pred,
                            epsilon=epsilon,
                            n_search_samples=n_search,
                            n_top_candidates=n_cand
                        )
                        
                        if cf is not None and info['valid']:
                            metrics = compute_metrics(sample_point['sample'], cf, pred,
                                                     target_point['prediction'], epsilon)
                            
                            results.append({
                                'method': 'Growing Spheres',
                                'sample_idx': sample_idx + 1,
                                'target_idx': target_idx + 1,
                                'sample_prediction': sample_point['prediction'],
                                'target_prediction': target_point['prediction'],
                                'prediction_change': target_point['prediction'] - sample_point['prediction'],
                                'epsilon': epsilon,
                                'n_search_samples': n_search,
                                'n_top_candidates': n_cand,
                                'valid': info['valid'],
                                'cf_prediction': pred,
                                'prediction_error': abs(pred - target_point['prediction']),
                                'l2_distance': metrics['l2_distance'],
                                'l1_distance': metrics['l1_distance'],
                                'sparsity': metrics['sparsity'],
                                'n_candidates_found': info['n_candidates_found'],
                                'n_candidates_tried': info['n_candidates_tried']
                            })
                        else:
                            results.append({
                                'method': 'Growing Spheres',
                                'sample_idx': sample_idx + 1,
                                'target_idx': target_idx + 1,
                                'sample_prediction': sample_point['prediction'],
                                'target_prediction': target_point['prediction'],
                                'prediction_change': target_point['prediction'] - sample_point['prediction'],
                                'epsilon': epsilon,
                                'n_search_samples': n_search,
                                'n_top_candidates': n_cand,
                                'valid': False,
                                'cf_prediction': None,
                                'prediction_error': None,
                                'l2_distance': None,
                                'l1_distance': None,
                                'sparsity': None,
                                'n_candidates_found': info.get('n_candidates_found', 0),
                                'n_candidates_tried': info.get('n_candidates_tried', 0)
                            })
    
    logger.info(f"\nGrowing Spheres tests complete: {len(results)} results")
    return pd.DataFrame(results)


def run_prototype_tests(model, X_train, y_train_pred, samples, config, feature_names):
    """Run Prototype-Based method tests with configured parameters."""
    from test_prototype_based_method import prototype_based_modified, compute_metrics
    import tensorflow as tf
    
    if not config['enabled']:
        logger.info("Prototype tests disabled")
        return None
    
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING PROTOTYPE-BASED METHOD TESTS")
    logger.info("=" * 80)
    
    # Model prediction wrapper
    def model_predict(X):
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    results = []
    
    # Test all sample-target combinations
    for sample_idx, sample_point in enumerate(samples):
        for target_idx, target_point in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            logger.info(f"\nTesting: Sample {sample_idx+1} → Target {target_idx+1}")
            
            # Test all parameter combinations
            for epsilon in config['epsilon_values']:
                for top_k in config['top_k_values']:
                    cf, pred, info = prototype_based_modified(
                        X_original=sample_point['sample'],
                        model_predict=model_predict,
                        target_value=target_point['prediction'],
                        X_train=X_train,
                        y_train=y_train_pred,
                        epsilon=epsilon,
                        top_k=top_k
                    )
                    
                    if cf is not None and info['valid']:
                        metrics = compute_metrics(sample_point['sample'], cf, pred,
                                                 target_point['prediction'], epsilon)
                        
                        results.append({
                            'method': 'Prototype',
                            'sample_idx': sample_idx + 1,
                            'target_idx': target_idx + 1,
                            'sample_prediction': sample_point['prediction'],
                            'target_prediction': target_point['prediction'],
                            'prediction_change': target_point['prediction'] - sample_point['prediction'],
                            'epsilon': epsilon,
                            'top_k': top_k,
                            'valid': info['valid'],
                            'cf_prediction': pred,
                            'prediction_error': abs(pred - target_point['prediction']),
                            'l2_distance': metrics['l2_distance'],
                            'l1_distance': metrics['l1_distance'],
                            'sparsity': metrics['sparsity'],
                            'n_prototypes_available': info['n_candidates_found'],
                            'is_real_instance': info['is_real_instance']
                        })
                    else:
                        results.append({
                            'method': 'Prototype',
                            'sample_idx': sample_idx + 1,
                            'target_idx': target_idx + 1,
                            'sample_prediction': sample_point['prediction'],
                            'target_prediction': target_point['prediction'],
                            'prediction_change': target_point['prediction'] - sample_point['prediction'],
                            'epsilon': epsilon,
                            'top_k': top_k,
                            'valid': False,
                            'cf_prediction': None,
                            'prediction_error': None,
                            'l2_distance': None,
                            'l1_distance': None,
                            'sparsity': None,
                            'n_prototypes_available': info.get('n_candidates_found', 0),
                            'is_real_instance': False
                        })
    
    logger.info(f"\nPrototype tests complete: {len(results)} results")
    return pd.DataFrame(results)


def run_gradient_tests(model, X_train, samples, config, feature_names):
    """Run Gradient-Based method tests with configured parameters."""
    from test_gradient_based_method import is_neural_network, gradient_based_modified, compute_metrics
    import tensorflow as tf
    
    if not config['enabled']:
        logger.info("Gradient-Based tests disabled")
        return None
    
    # Check if model is neural network
    if not is_neural_network(model):
        logger.warning("Model is not a neural network - skipping Gradient-Based tests")
        return None
    
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING GRADIENT-BASED METHOD TESTS")
    logger.info("=" * 80)
    
    # Define feature ranges
    feature_ranges = [(float(X_train[:, i].min()), float(X_train[:, i].max())) 
                     for i in range(X_train.shape[1])]
    
    results = []
    
    # Test all sample-target combinations
    for sample_idx, sample_point in enumerate(samples):
        for target_idx, target_point in enumerate(samples):
            if sample_idx == target_idx:
                continue
            
            logger.info(f"\nTesting: Sample {sample_idx+1} → Target {target_idx+1}")
            
            # Test all parameter combinations
            for epsilon in config['epsilon_values']:
                for lr in config['learning_rate_values']:
                    for lambda_param in config['lambda_values']:
                        cf, pred, info = gradient_based_modified(
                            X_original=sample_point['sample'],
                            model=model,
                            target_value=target_point['prediction'],
                            feature_ranges=feature_ranges,
                            epsilon=epsilon,
                            lambda_param=lambda_param,
                            learning_rate=lr,
                            max_iter=config['max_iter']
                        )
                        
                        if cf is not None:
                            metrics = compute_metrics(sample_point['sample'], cf, pred,
                                                     target_point['prediction'], epsilon)
                            
                            results.append({
                                'method': 'Gradient-Based',
                                'sample_idx': sample_idx + 1,
                                'target_idx': target_idx + 1,
                                'sample_prediction': sample_point['prediction'],
                                'target_prediction': target_point['prediction'],
                                'prediction_change': target_point['prediction'] - sample_point['prediction'],
                                'epsilon': epsilon,
                                'learning_rate': lr,
                                'lambda': lambda_param,
                                'valid': info['valid'],
                                'cf_prediction': pred,
                                'prediction_error': abs(pred - target_point['prediction']),
                                'l2_distance': metrics['l2_distance'],
                                'l1_distance': metrics['l1_distance'],
                                'sparsity': metrics['sparsity'],
                                'iterations': info['iterations'],
                                'final_loss': info.get('final_loss', None)
                            })
    
    logger.info(f"\nGradient-Based tests complete: {len(results)} results")
    return pd.DataFrame(results)


def save_results(df, output_file, output_dir):
    """Save results to CSV file."""
    if df is None or len(df) == 0:
        logger.warning(f"No results to save for {output_file}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / output_file
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved to: {filepath}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Valid results: {df['valid'].sum() if 'valid' in df.columns else 'N/A'}")


def print_summary(results_dict):
    """Print summary statistics for all methods."""
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 80)
    
    for method, df in results_dict.items():
        if df is None or len(df) == 0:
            logger.info(f"\n{method}: No results")
            continue
        
        valid = df[df['valid'] == True]
        
        logger.info(f"\n{method}:")
        logger.info(f"  Total tests: {len(df)}")
        logger.info(f"  Valid counterfactuals: {len(valid)} ({100*len(valid)/len(df):.1f}%)")
        
        if len(valid) > 0:
            logger.info(f"  Average L2 distance: {valid['l2_distance'].mean():.4f}")
            logger.info(f"  Average sparsity: {valid['sparsity'].mean():.2f}")
            logger.info(f"  Average prediction error: {valid['prediction_error'].mean():.4f}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("STANDARD METHODS TEST MANAGER")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset and model
    data = load_dataset_and_model(DATASET_CONFIG)
    X_train, X_test, y_train, y_test, y_train_pred, model, scaler, feature_names = data
    
    # Get test samples
    samples = get_test_samples(model, X_test, y_test)
    
    # Run tests for each method
    results = {}
    
    if WACHTER_CONFIG['enabled']:
        results['Wachter'] = run_wachter_tests(
            model, X_train, samples, WACHTER_CONFIG, feature_names
        )
        save_results(results['Wachter'], WACHTER_CONFIG['output_file'], OUTPUT_DIR)
    
    if GROWING_SPHERES_CONFIG['enabled']:
        results['Growing Spheres'] = run_growing_spheres_tests(
            model, X_train, y_train_pred, samples, GROWING_SPHERES_CONFIG, feature_names
        )
        save_results(results['Growing Spheres'], GROWING_SPHERES_CONFIG['output_file'], OUTPUT_DIR)
    
    if PROTOTYPE_CONFIG['enabled']:
        results['Prototype'] = run_prototype_tests(
            model, X_train, y_train_pred, samples, PROTOTYPE_CONFIG, feature_names
        )
        save_results(results['Prototype'], PROTOTYPE_CONFIG['output_file'], OUTPUT_DIR)
    
    if GRADIENT_CONFIG['enabled']:
        results['Gradient-Based'] = run_gradient_tests(
            model, X_train, samples, GRADIENT_CONFIG, feature_names
        )
        save_results(results['Gradient-Based'], GRADIENT_CONFIG['output_file'], OUTPUT_DIR)
    
    # Print summary
    print_summary(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
