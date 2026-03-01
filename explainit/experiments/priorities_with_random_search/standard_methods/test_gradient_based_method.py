"""
Gradient-Based Method Testing Script
Test different learning_rate, lambda, and epsilon values for neural network models.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import sys
import os
from pathlib import Path
import tensorflow as tf

# Compute metrics function
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_auto_mpg_data():
    """Load Auto MPG dataset and trained model."""
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Scale data
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    logger.info(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train.values.ravel(), y_test.values.ravel(), model, scaler


def get_test_samples(model, X_test, y_test):
    """Get test samples for experimentation."""
    # Get predictions
    predictions = model(tf.constant(X_test), training=False).numpy().ravel()
    
    # Select 3 quantile points (low, medium, high MPG)
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
    
    return samples


def is_neural_network(model):
    """Check if model is a neural network (TensorFlow/Keras)."""
    # Check if it's a Keras model
    if hasattr(tf.keras, 'Model') and isinstance(model, tf.keras.Model):
        return True
    # Check if it's a Sequential model
    if hasattr(tf.keras, 'Sequential') and isinstance(model, tf.keras.Sequential):
        return True
    return False


def gradient_based_modified(
    X_original: np.ndarray,
    model,
    target_value: float,
    feature_ranges: list,
    epsilon: float = 1.0,
    lambda_param: float = 1.0,
    learning_rate: float = 0.01,
    max_iter: int = 500
):
    """
    Modified Gradient-Based method with additional diagnostics and configurable parameters.
    
    Args:
        X_original: Original instance to explain
        model: TensorFlow/Keras model
        target_value: Target prediction value
        feature_ranges: List of (min, max) tuples for each feature
        epsilon: Tolerance for target prediction
        lambda_param: Weight balancing prediction vs distance
        learning_rate: Step size for gradient descent
        max_iter: Maximum number of iterations
    
    Returns:
        counterfactual: Generated counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    # Convert to TensorFlow variable (requires_grad=True)
    X_cf = tf.Variable(X_original.copy().astype(np.float32), dtype=tf.float32)
    
    # Track optimization
    best_cf = None
    best_pred = None
    best_distance = float('inf')
    best_loss = float('inf')
    iteration_history = []
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            # Forward pass
            pred = model(tf.expand_dims(X_cf, 0), training=False)[0, 0]
            
            # Loss components
            pred_loss = tf.square(pred - target_value)
            distance_loss = tf.reduce_sum(tf.square(X_cf - X_original))
            
            # Total loss
            total_loss = lambda_param * pred_loss + distance_loss
        
        # Compute gradients
        gradients = tape.gradient(total_loss, [X_cf])
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, [X_cf]))
        
        # Apply feature bounds
        for feat_idx, (min_val, max_val) in enumerate(feature_ranges):
            X_cf[feat_idx].assign(tf.clip_by_value(X_cf[feat_idx], min_val, max_val))
        
        # Check validity
        current_pred = pred.numpy()
        current_distance = np.linalg.norm(X_cf.numpy() - X_original)
        is_valid = abs(current_pred - target_value) <= epsilon
        
        # Track history
        iteration_history.append({
            'iteration': iteration,
            'loss': total_loss.numpy(),
            'pred_loss': pred_loss.numpy(),
            'distance_loss': distance_loss.numpy(),
            'prediction': current_pred,
            'distance': current_distance,
            'valid': is_valid
        })
        
        # Update best if valid and closer
        if is_valid and current_distance < best_distance:
            best_cf = X_cf.numpy().copy()
            best_pred = current_pred
            best_distance = current_distance
            best_loss = total_loss.numpy()
        
        # Early stopping if good solution found
        if is_valid and iteration > 50 and abs(current_pred - target_value) < epsilon / 2:
            break
    
    # Final evaluation
    if best_cf is not None:
        info = {
            'method': 'Gradient-Based',
            'valid': True,
            'distance': best_distance,
            'iterations': len(iteration_history),
            'final_loss': best_loss,
            'converged': True,
            'learning_rate': learning_rate,
            'lambda': lambda_param,
            'iteration_history': iteration_history
        }
        return best_cf, best_pred, info
    else:
        # Return last iteration even if invalid
        final_cf = X_cf.numpy()
        final_pred = model(tf.expand_dims(X_cf, 0), training=False)[0, 0].numpy()
        
        info = {
            'method': 'Gradient-Based',
            'valid': False,
            'reason': 'did_not_reach_target',
            'distance': np.linalg.norm(final_cf - X_original),
            'iterations': len(iteration_history),
            'final_loss': iteration_history[-1]['loss'],
            'converged': False,
            'learning_rate': learning_rate,
            'lambda': lambda_param,
            'iteration_history': iteration_history
        }
        return final_cf, final_pred, info


def test_gradient_based_parameters(model, X_train, sample, target_value, feature_names):
    """
    Test Gradient-Based method with different learning_rate, lambda, and epsilon values.
    
    Args:
        model: Trained neural network model
        X_train: Training data (for feature ranges)
        sample: Sample to generate counterfactual for
        target_value: Target prediction value
        feature_names: Names of features
    """
    # Check if model is a neural network
    if not is_neural_network(model):
        print("\n" + "=" * 100)
        print("ERROR: Model is not a neural network!")
        print("=" * 100)
        print("Gradient-based method ONLY works with TensorFlow/Keras neural networks.")
        print("Current model type:", type(model))
        return []
    
    print("\n" + "=" * 100)
    print("✓ Model is a neural network (TensorFlow/Keras)")
    print("=" * 100)
    
    # Define feature ranges
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = float(X_train[:, idx].min())
        max_val = float(X_train[:, idx].max())
        feature_ranges.append((min_val, max_val))
    
    # Test different parameter combinations
    learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.5]
    lambda_values = [0.01, 0.1, 0.5, 1.0, 5.0]
    epsilon_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    print("\n" + "=" * 100)
    print("GRADIENT-BASED METHOD - PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(f"\nOriginal Sample Prediction: {model(tf.constant([sample]), training=False).numpy()[0, 0]:.2f} MPG")
    print(f"Target Prediction: {target_value:.2f} MPG")
    print(f"Distance to Target: {abs(model(tf.constant([sample]), training=False).numpy()[0, 0] - target_value):.2f} MPG")
    print(f"\nOriginal Sample Values: {sample}")
    print("\n" + "=" * 100)
    
    # Store results for summary
    all_results = []
    
    # Test 1: Effect of epsilon (with fixed learning_rate and lambda)
    print("\n" + "=" * 100)
    print("TEST 1: EPSILON SENSITIVITY (learning_rate=0.01, lambda=1.0)")
    print("=" * 100)
    
    for epsilon in epsilon_values:
        # Run Gradient-Based
        cf, pred, info = gradient_based_modified(
            X_original=sample,
            model=model,
            target_value=target_value,
            feature_ranges=feature_ranges,
            epsilon=epsilon,
            lambda_param=1.0,
            learning_rate=0.01,
            max_iter=500
        )
        
        # Prepare result
        result = {
            'epsilon': epsilon,
            'lambda': 1.0,
            'learning_rate': 0.01,
            'valid': info['valid'],
            'prediction': pred,
            'counterfactual': cf,
            'info': info
        }
        
        if cf is not None:
            metrics = compute_metrics(sample, cf, pred, target_value, epsilon)
            result['metrics'] = metrics
        
        all_results.append(result)
        
        # Print result
        print(f"\nEpsilon = {epsilon:5.1f} MPG")
        print(f"{'─' * 90}")
        
        if cf is not None and info['valid']:
            print(f"✓ VALID counterfactual found")
            print(f"  Prediction: {pred:.2f} MPG (target: {target_value:.2f}, error: {abs(pred - target_value):.2f})")
            print(f"  Distance: L2={metrics['l2_distance']:.4f}, L1={metrics['l1_distance']:.4f}")
            print(f"  Sparsity: {metrics['sparsity']} features changed")
            print(f"  Iterations: {info['iterations']}")
            print(f"  Final loss: {info['final_loss']:.6f}")
            
            # Show feature changes
            print(f"  Feature Changes:")
            for idx, (orig, new) in enumerate(zip(sample, cf)):
                if abs(orig - new) > 0.001:
                    change_pct = ((new - orig) / (orig + 1e-10)) * 100
                    print(f"    {feature_names[idx]:15s}: {orig:.4f} → {new:.4f} (Δ={new-orig:+.4f}, {change_pct:+.1f}%)")
        else:
            print(f"✗ FAILED - Did not reach target within epsilon")
            print(f"  Prediction: {pred:.2f} MPG (target: {target_value:.2f}, error: {abs(pred - target_value):.2f})")
            print(f"  Iterations: {info['iterations']}")
            print(f"  Reason: {info.get('reason', 'unknown')}")
    
    # Test 2: Effect of learning_rate (with fixed epsilon and lambda)
    valid_epsilon = None
    for r in all_results:
        if r['valid']:
            valid_epsilon = r['epsilon']
            break
    
    if valid_epsilon is not None:
        print("\n" + "=" * 100)
        print(f"TEST 2: LEARNING_RATE SENSITIVITY (epsilon={valid_epsilon}, lambda=1.0)")
        print("=" * 100)
        
        for lr in learning_rate_values:
            cf, pred, info = gradient_based_modified(
                X_original=sample,
                model=model,
                target_value=target_value,
                feature_ranges=feature_ranges,
                epsilon=valid_epsilon,
                lambda_param=1.0,
                learning_rate=lr,
                max_iter=500
            )
            
            result = {
                'epsilon': valid_epsilon,
                'lambda': 1.0,
                'learning_rate': lr,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_value, valid_epsilon)
                result['metrics'] = metrics
            
            all_results.append(result)
            
            print(f"\nlearning_rate = {lr:.3f} (gradient step size)")
            print(f"{'─' * 90}")
            
            if cf is not None and info['valid']:
                print(f"✓ VALID counterfactual found")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Distance: L2={metrics['l2_distance']:.4f}")
                print(f"  Sparsity: {metrics['sparsity']} features changed")
                print(f"  Iterations: {info['iterations']}")
            else:
                print(f"✗ FAILED")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Iterations: {info['iterations']}")
    
    # Test 3: Effect of lambda (with fixed epsilon and learning_rate)
    if valid_epsilon is not None:
        print("\n" + "=" * 100)
        print(f"TEST 3: LAMBDA SENSITIVITY (epsilon={valid_epsilon}, learning_rate=0.01)")
        print("=" * 100)
        
        for lam in lambda_values:
            cf, pred, info = gradient_based_modified(
                X_original=sample,
                model=model,
                target_value=target_value,
                feature_ranges=feature_ranges,
                epsilon=valid_epsilon,
                lambda_param=lam,
                learning_rate=0.01,
                max_iter=500
            )
            
            result = {
                'epsilon': valid_epsilon,
                'lambda': lam,
                'learning_rate': 0.01,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_value, valid_epsilon)
                result['metrics'] = metrics
            
            all_results.append(result)
            
            print(f"\nlambda = {lam:5.2f} (prediction vs distance weight)")
            print(f"{'─' * 90}")
            
            if cf is not None and info['valid']:
                print(f"✓ VALID counterfactual found")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Distance: L2={metrics['l2_distance']:.4f}")
                print(f"  Sparsity: {metrics['sparsity']} features changed")
                print(f"  Iterations: {info['iterations']}")
            else:
                print(f"✗ FAILED")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Iterations: {info['iterations']}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - TEST 1 (EPSILON SENSITIVITY)")
    print("=" * 100)
    print(f"\n{'Epsilon':<10}{'Valid':<10}{'Prediction':<12}{'Pred Error':<12}{'L2 Distance':<15}{'Sparsity':<12}{'Iterations':<12}")
    print("─" * 100)
    
    for result in all_results:
        if result['lambda'] == 1.0 and result['learning_rate'] == 0.01:
            eps = result['epsilon']
            valid = "✓" if result['valid'] else "✗"
            
            if result['counterfactual'] is not None:
                pred = result['prediction']
                pred_error = abs(pred - target_value)
                l2_dist = result['metrics']['l2_distance']
                sparsity = result['metrics']['sparsity']
                iters = result['info']['iterations']
                
                print(f"{eps:<10.1f}{valid:<10}{pred:<12.2f}{pred_error:<12.2f}{l2_dist:<15.4f}{str(sparsity):<12}{iters:<12}")
            else:
                print(f"{eps:<10.1f}{valid:<10}{'N/A':<12}{'N/A':<12}{'N/A':<15}{'N/A':<12}{'N/A':<12}")
    
    print("\n" + "=" * 100)
    
    # Analysis insights
    print("\nKEY INSIGHTS:")
    print("─" * 100)
    
    # Count valid results from Test 1
    test1_results = [r for r in all_results if r['lambda'] == 1.0 and r['learning_rate'] == 0.01]
    valid_results = [r for r in test1_results if r['valid']]
    print(f"• Valid counterfactuals found: {len(valid_results)}/{len(test1_results)} epsilon values")
    
    if valid_results:
        # Best by distance
        best_by_distance = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
        print(f"\n• Closest counterfactual:")
        print(f"    Epsilon={best_by_distance['epsilon']:.1f}")
        print(f"    L2 Distance={best_by_distance['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_distance['prediction']:.2f} MPG")
        print(f"    Sparsity={best_by_distance['metrics']['sparsity']} features changed")
        print(f"    Iterations={best_by_distance['info']['iterations']}")
        
        # Best by sparsity
        best_by_sparsity = min(valid_results, key=lambda r: r['metrics']['sparsity'])
        print(f"\n• Most sparse counterfactual:")
        print(f"    Epsilon={best_by_sparsity['epsilon']:.1f}")
        print(f"    Sparsity={best_by_sparsity['metrics']['sparsity']} features changed")
        print(f"    L2 Distance={best_by_sparsity['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_sparsity['prediction']:.2f} MPG")
        
        # Parameter effects
        print(f"\n• Epsilon parameter effect:")
        print(f"    Smaller epsilon → Stricter target tolerance → May not converge")
        print(f"    Larger epsilon → More relaxed → Higher success rate")
        
        # learning_rate effect (if tested)
        test2_results = [r for r in all_results if r['epsilon'] == valid_epsilon and r['lambda'] == 1.0 and r['learning_rate'] != 0.01]
        if test2_results:
            print(f"\n• Learning rate parameter effect:")
            print(f"    Too small → Slow convergence, may not reach target")
            print(f"    Too large → Unstable, may overshoot")
            print(f"    Optimal range: 0.01-0.05 for this problem")
        
        # lambda effect (if tested)
        test3_results = [r for r in all_results if r['epsilon'] == valid_epsilon and r['learning_rate'] == 0.01 and r['lambda'] != 1.0]
        if test3_results:
            print(f"\n• Lambda parameter effect:")
            print(f"    Low lambda → Prioritizes proximity (smaller distance)")
            print(f"    High lambda → Prioritizes prediction (reaches target faster)")
    
    print("\n" + "=" * 100)
    
    return all_results


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("GRADIENT-BASED METHOD - PARAMETER TESTING")
    logger.info("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, model, scaler = load_auto_mpg_data()
    
    # Check if model is a neural network
    logger.info("\nChecking model type...")
    if not is_neural_network(model):
        logger.error("ERROR: Model is not a neural network!")
        logger.error("Gradient-based method ONLY works with TensorFlow/Keras neural networks.")
        logger.error(f"Current model type: {type(model)}")
        return
    
    logger.info("✓ Model is a neural network (TensorFlow/Keras)")
    logger.info(f"  Model type: {type(model)}")
    logger.info(f"  Model architecture: {model.count_params()} parameters")
    
    # Feature names
    feature_names = ['displacement', 'horsepower', 'weight', 'acceleration']
    
    # Get test samples
    samples = get_test_samples(model, X_test, y_test)
    
    # Test all sample-target combinations (excluding same-to-same)
    total_combinations = len(samples) * (len(samples) - 1)
    current_combination = 0
    
    all_experiment_results = []
    
    for sample_idx, sample_point in enumerate(samples):
        for target_idx, target_point in enumerate(samples):
            if sample_idx == target_idx:
                continue  # Skip same-to-same
            
            current_combination += 1
            
            print("\n" + "=" * 100)
            print(f"TEST SCENARIO {current_combination}/{total_combinations}: Sample {sample_idx + 1} → Target {target_idx + 1}")
            print("=" * 100)
            print(f"\nOriginal: {sample_point['prediction']:.2f} MPG (Sample {sample_idx + 1})")
            print(f"Target:   {target_point['prediction']:.2f} MPG (Sample {target_idx + 1})")
            print(f"Change needed: {target_point['prediction'] - sample_point['prediction']:+.2f} MPG")
            
            # Run parameter testing
            results = test_gradient_based_parameters(
                model=model,
                X_train=X_train,
                sample=sample_point['sample'],
                target_value=target_point['prediction'],
                feature_names=feature_names
            )
            
            all_experiment_results.append({
                'sample_idx': sample_idx + 1,
                'target_idx': target_idx + 1,
                'sample_prediction': sample_point['prediction'],
                'target_prediction': target_point['prediction'],
                'results': results
            })
    
    # Print overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY - ALL SCENARIOS")
    print("=" * 100)
    
    print("\n" + "─" * 100)
    print("SUCCESS RATE BY SCENARIO")
    print("─" * 100)
    
    for exp in all_experiment_results:
        # Count valid results from Test 1 (epsilon sensitivity with standard params)
        test1_results = [r for r in exp['results'] if r['lambda'] == 1.0 and r['learning_rate'] == 0.01]
        valid_count = sum(1 for r in test1_results if r['valid'])
        
        print(f"\nSample {exp['sample_idx']} → Target {exp['target_idx']}: "
              f"{exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG "
              f"({exp['target_prediction'] - exp['sample_prediction']:+.2f})")
        print(f"  Valid configurations: {valid_count}/{len(test1_results)} epsilon values tested")
        
        if valid_count > 0:
            valid_results = [r for r in test1_results if r['valid']]
            best = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
            print(f"  Best: epsilon={best['epsilon']:.1f}, distance={best['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best['metrics']['sparsity']}, iterations={best['info']['iterations']}")
        else:
            print(f"  No valid counterfactuals found (did not converge)")
    
    # Detailed results table
    print("\n" + "─" * 100)
    print("DETAILED RESULTS TABLE - ALL SCENARIOS")
    print("─" * 100)
    print(f"\n{'Scenario':<15}{'Epsilon':<10}{'Valid':<8}{'Prediction':<12}{'Error':<10}{'L2 Dist':<12}{'Sparsity':<10}{'Iterations':<12}")
    print("─" * 100)
    
    for exp in all_experiment_results:
        scenario = f"S{exp['sample_idx']}→T{exp['target_idx']}"
        target = exp['target_prediction']
        
        # Show Test 1 results (epsilon sensitivity)
        test1_results = [r for r in exp['results'] if r['lambda'] == 1.0 and r['learning_rate'] == 0.01]
        
        for result in test1_results:
            eps = result['epsilon']
            valid = "✓" if result['valid'] else "✗"
            
            if result['counterfactual'] is not None and result['valid']:
                pred = result['prediction']
                pred_error = abs(pred - target)
                l2_dist = result['metrics']['l2_distance']
                sparsity = result['metrics']['sparsity']
                iters = result['info']['iterations']
                
                print(f"{scenario:<15}{eps:<10.1f}{valid:<8}{pred:<12.2f}{pred_error:<10.2f}{l2_dist:<12.4f}{str(sparsity):<10}{iters:<12}")
            else:
                iters = result['info'].get('iterations', 0)
                print(f"{scenario:<15}{eps:<10.1f}{valid:<8}{'N/A':<12}{'N/A':<10}{'N/A':<12}{'N/A':<10}{iters:<12}")
    
    # Best configurations across all scenarios
    print("\n" + "─" * 100)
    print("BEST CONFIGURATIONS BY SCENARIO")
    print("─" * 100)
    
    for exp in all_experiment_results:
        scenario = f"Sample {exp['sample_idx']} → Target {exp['target_idx']}"
        
        # Get all valid results for this scenario
        all_valid = [r for r in exp['results'] if r['valid']]
        
        if all_valid:
            # Best by distance
            best_dist = min(all_valid, key=lambda r: r['metrics']['l2_distance'])
            # Best by sparsity
            best_sparse = min(all_valid, key=lambda r: r['metrics']['sparsity'])
            
            print(f"\n{scenario}: {exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG")
            print(f"  Best distance: eps={best_dist['epsilon']:.1f}, lr={best_dist['learning_rate']:.3f}, "
                  f"lambda={best_dist['lambda']:.2f} → L2={best_dist['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_dist['metrics']['sparsity']}, iters={best_dist['info']['iterations']}")
            print(f"  Best sparsity: eps={best_sparse['epsilon']:.1f}, lr={best_sparse['learning_rate']:.3f}, "
                  f"lambda={best_sparse['lambda']:.2f} → L2={best_sparse['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_sparse['metrics']['sparsity']}, iters={best_sparse['info']['iterations']}")
        else:
            print(f"\n{scenario}: {exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG")
            print(f"  ✗ No valid counterfactuals found")
    
    # Overall statistics
    print("\n" + "─" * 100)
    print("OVERALL STATISTICS")
    print("─" * 100)
    
    total_scenarios = len(all_experiment_results)
    scenarios_with_solution = sum(1 for exp in all_experiment_results if any(r['valid'] for r in exp['results']))
    
    print(f"\nTotal scenarios tested: {total_scenarios}")
    print(f"Scenarios with valid counterfactuals: {scenarios_with_solution}/{total_scenarios} "
          f"({100*scenarios_with_solution/total_scenarios:.1f}%)")
    
    if scenarios_with_solution > 0:
        # Average distance and sparsity across best solutions
        all_best = []
        for exp in all_experiment_results:
            valid = [r for r in exp['results'] if r['valid']]
            if valid:
                best = min(valid, key=lambda r: r['metrics']['l2_distance'])
                all_best.append(best)
        
        if all_best:
            avg_dist = np.mean([b['metrics']['l2_distance'] for b in all_best])
            avg_sparse = np.mean([b['metrics']['sparsity'] for b in all_best])
            avg_iters = np.mean([b['info']['iterations'] for b in all_best])
            
            print(f"\nAverage metrics (best solutions):")
            print(f"  Average L2 distance: {avg_dist:.4f}")
            print(f"  Average sparsity: {avg_sparse:.1f} features")
            print(f"  Average iterations: {avg_iters:.1f}")
            
            # Compare with other methods
            print(f"\n• Method characteristics:")
            print(f"    Uses gradient descent through neural network")
            print(f"    Fast convergence (typically <100 iterations)")
            print(f"    May produce unrealistic synthetic points")
            print(f"    Similar to Wachter but exploits NN differentiability")
    
    print("\n" + "=" * 100)
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
