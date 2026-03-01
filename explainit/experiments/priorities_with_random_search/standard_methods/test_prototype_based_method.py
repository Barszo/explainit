"""
Prototype-Based Method Testing Script
Test different epsilon and top_k values to understand their impact on counterfactual generation.
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
    
    # Get training predictions
    y_train_pred = model(tf.constant(X_train_scaled), training=False).numpy().ravel()
    
    logger.info(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train.values.ravel(), y_test.values.ravel(), y_train_pred, model, scaler


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


def prototype_based_modified(
    X_original: np.ndarray,
    model_predict,
    target_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.0,
    top_k: int = 1
):
    """
    Modified Prototype-Based method with additional diagnostics and configurable parameters.
    
    Args:
        X_original: Original instance to explain
        model_predict: Model prediction function
        target_value: Target prediction value
        X_train: Training dataset
        y_train: Training predictions
        epsilon: Tolerance for target prediction
        top_k: Which k-th nearest prototype to return (1=closest, 2=2nd closest, etc.)
    
    Returns:
        counterfactual: Generated counterfactual (real training instance or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    # Find training instances within epsilon of target
    target_mask = np.abs(y_train - target_value) <= epsilon
    target_instances = X_train[target_mask]
    target_predictions = y_train[target_mask]
    
    if len(target_instances) == 0:
        logger.warning("No training instances found within epsilon of target")
        return None, None, {
            'method': 'Prototype-Based',
            'valid': False,
            'reason': 'no_target_instances',
            'n_candidates_found': 0
        }
    
    # Calculate distances to all target instances
    distances = np.linalg.norm(target_instances - X_original, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Check if we have enough prototypes
    if top_k > len(sorted_indices):
        logger.warning(f"Requested top_k={top_k} but only {len(sorted_indices)} prototypes available")
        return None, None, {
            'method': 'Prototype-Based',
            'valid': False,
            'reason': 'insufficient_prototypes',
            'n_candidates_found': len(target_instances),
            'top_k_requested': top_k
        }
    
    # Select the k-th closest prototype (k-1 because 0-indexed)
    selected_idx = sorted_indices[top_k - 1]
    prototype = target_instances[selected_idx]
    prototype_pred = target_predictions[selected_idx]
    prototype_distance = distances[selected_idx]
    
    # Verify prediction is still within epsilon
    actual_pred = model_predict([prototype])[0]
    is_valid = abs(actual_pred - target_value) <= epsilon
    
    info = {
        'method': 'Prototype-Based',
        'valid': is_valid,
        'distance': prototype_distance,
        'n_candidates_found': len(target_instances),
        'top_k_used': top_k,
        'prototype_training_pred': prototype_pred,
        'prototype_actual_pred': actual_pred,
        'is_real_instance': True
    }
    
    return prototype, actual_pred, info


def test_prototype_parameters(model, X_train, y_train, sample, target_value, feature_names):
    """
    Test Prototype-Based method with different epsilon and top_k values.
    
    Args:
        model: Trained model
        X_train: Training data
        y_train: Training predictions
        sample: Sample to generate counterfactual for
        target_value: Target prediction value
        feature_names: Names of features
    """
    # Model prediction wrapper
    def model_predict(X):
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    # Test different parameter combinations
    epsilon_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    top_k_values = [1, 2, 3, 5, 10]
    
    print("\n" + "=" * 100)
    print("PROTOTYPE-BASED METHOD - PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(f"\nOriginal Sample Prediction: {model_predict([sample])[0]:.2f} MPG")
    print(f"Target Prediction: {target_value:.2f} MPG")
    print(f"Distance to Target: {abs(model_predict([sample])[0] - target_value):.2f} MPG")
    print(f"\nOriginal Sample Values: {sample}")
    
    # First, analyze the training data distribution
    print("\n" + "=" * 100)
    print("TRAINING DATA ANALYSIS")
    print("=" * 100)
    
    for eps in epsilon_values:
        target_mask = np.abs(y_train - target_value) <= eps
        n_instances = np.sum(target_mask)
        if n_instances > 0:
            target_instances = X_train[target_mask]
            distances = np.linalg.norm(target_instances - sample, axis=1)
            min_dist = distances.min()
            max_dist = distances.max()
            mean_dist = distances.mean()
            print(f"Epsilon={eps:5.1f}: {n_instances:4d} prototypes available | "
                  f"Distance range: [{min_dist:.4f}, {max_dist:.4f}], mean={mean_dist:.4f}")
        else:
            print(f"Epsilon={eps:5.1f}: {n_instances:4d} prototypes available")
    
    print("\n" + "=" * 100)
    
    # Store results for summary
    all_results = []
    
    # Test 1: Effect of epsilon (with fixed top_k=1)
    print("\n" + "=" * 100)
    print("TEST 1: EPSILON SENSITIVITY (top_k=1, returns closest prototype)")
    print("=" * 100)
    
    for epsilon in epsilon_values:
        # Run Prototype-Based
        cf, pred, info = prototype_based_modified(
            X_original=sample,
            model_predict=model_predict,
            target_value=target_value,
            X_train=X_train,
            y_train=y_train,
            epsilon=epsilon,
            top_k=1
        )
        
        # Prepare result
        result = {
            'epsilon': epsilon,
            'top_k': 1,
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
            print(f"✓ VALID prototype found")
            print(f"  Prediction: {pred:.2f} MPG (target: {target_value:.2f}, error: {abs(pred - target_value):.2f})")
            print(f"  Distance: L2={metrics['l2_distance']:.4f}, L1={metrics['l1_distance']:.4f}")
            print(f"  Sparsity: {metrics['sparsity']} features changed")
            print(f"  Available prototypes: {info['n_candidates_found']}")
            print(f"  Prototype rank: {info['top_k_used']} (closest)")
            print(f"  Real training instance: Yes")
            
            # Show feature changes
            print(f"  Feature Changes (original → prototype):")
            for idx, (orig, new) in enumerate(zip(sample, cf)):
                if abs(orig - new) > 0.001:
                    change_pct = ((new - orig) / (orig + 1e-10)) * 100
                    print(f"    {feature_names[idx]:15s}: {orig:.4f} → {new:.4f} (Δ={new-orig:+.4f}, {change_pct:+.1f}%)")
        else:
            print(f"✗ FAILED - No valid prototype found")
            print(f"  Reason: {info.get('reason', 'unknown')}")
            print(f"  Available prototypes: {info.get('n_candidates_found', 0)}")
    
    # Test 2: Effect of top_k (with fixed epsilon that found results)
    valid_epsilon = None
    for r in all_results:
        if r['valid']:
            valid_epsilon = r['epsilon']
            break
    
    if valid_epsilon is not None:
        print("\n" + "=" * 100)
        print(f"TEST 2: TOP_K SENSITIVITY (epsilon={valid_epsilon}, returns k-th closest)")
        print("=" * 100)
        
        for k in top_k_values:
            cf, pred, info = prototype_based_modified(
                X_original=sample,
                model_predict=model_predict,
                target_value=target_value,
                X_train=X_train,
                y_train=y_train,
                epsilon=valid_epsilon,
                top_k=k
            )
            
            result = {
                'epsilon': valid_epsilon,
                'top_k': k,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_value, valid_epsilon)
                result['metrics'] = metrics
            
            all_results.append(result)
            
            print(f"\ntop_k = {k:2d} (select {k}{'st' if k==1 else 'nd' if k==2 else 'rd' if k==3 else 'th'} closest prototype)")
            print(f"{'─' * 90}")
            
            if cf is not None and info['valid']:
                print(f"✓ VALID prototype found")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Distance: L2={metrics['l2_distance']:.4f}")
                print(f"  Sparsity: {metrics['sparsity']} features changed")
                print(f"  Prototype rank: {k} out of {info['n_candidates_found']} available")
            else:
                print(f"✗ FAILED - {info.get('reason', 'unknown')}")
                if 'n_candidates_found' in info:
                    print(f"  Available prototypes: {info['n_candidates_found']} (requested top_k={k})")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - TEST 1 (EPSILON SENSITIVITY)")
    print("=" * 100)
    print(f"\n{'Epsilon':<10}{'Valid':<10}{'Prediction':<12}{'Pred Error':<12}{'L2 Distance':<15}{'Sparsity':<12}{'Prototypes':<12}")
    print("─" * 100)
    
    for result in all_results:
        if result['top_k'] == 1:
            eps = result['epsilon']
            valid = "✓" if result['valid'] else "✗"
            
            if result['counterfactual'] is not None:
                pred = result['prediction']
                pred_error = abs(pred - target_value)
                l2_dist = result['metrics']['l2_distance']
                sparsity = result['metrics']['sparsity']
                n_cand = result['info']['n_candidates_found']
                
                print(f"{eps:<10.1f}{valid:<10}{pred:<12.2f}{pred_error:<12.2f}{l2_dist:<15.4f}{str(sparsity):<12}{n_cand:<12}")
            else:
                n_cand = result['info'].get('n_candidates_found', 0)
                print(f"{eps:<10.1f}{valid:<10}{'N/A':<12}{'N/A':<12}{'N/A':<15}{'N/A':<12}{n_cand:<12}")
    
    print("\n" + "=" * 100)
    
    # Analysis insights
    print("\nKEY INSIGHTS:")
    print("─" * 100)
    
    # Count valid results from Test 1
    test1_results = [r for r in all_results if r['top_k'] == 1]
    valid_results = [r for r in test1_results if r['valid']]
    print(f"• Valid prototypes found: {len(valid_results)}/{len(test1_results)} epsilon values")
    
    if valid_results:
        # Best by distance
        best_by_distance = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
        print(f"\n• Closest prototype:")
        print(f"    Epsilon={best_by_distance['epsilon']:.1f}")
        print(f"    L2 Distance={best_by_distance['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_distance['prediction']:.2f} MPG")
        print(f"    Sparsity={best_by_distance['metrics']['sparsity']} features changed")
        
        # Best by sparsity
        best_by_sparsity = min(valid_results, key=lambda r: r['metrics']['sparsity'])
        print(f"\n• Most sparse prototype:")
        print(f"    Epsilon={best_by_sparsity['epsilon']:.1f}")
        print(f"    Sparsity={best_by_sparsity['metrics']['sparsity']} features changed")
        print(f"    L2 Distance={best_by_sparsity['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_sparsity['prediction']:.2f} MPG")
        
        # Epsilon effect
        print(f"\n• Epsilon parameter effect:")
        print(f"    Smaller epsilon → Fewer prototypes available → May fail")
        print(f"    Larger epsilon → More prototypes → Higher success rate, but less precise")
        print(f"    Prototypes are REAL training instances (not synthetic)")
        
        # top_k effect (if tested)
        test2_results = [r for r in all_results if r['epsilon'] == valid_epsilon and r['top_k'] != 1]
        if test2_results:
            valid_test2 = [r for r in test2_results if r['valid']]
            print(f"\n• top_k parameter effect:")
            print(f"    top_k=1 → Returns closest prototype (smallest distance)")
            print(f"    top_k>1 → Returns more distant prototypes (may offer diversity)")
            if valid_test2:
                distances_by_k = [(r['top_k'], r['metrics']['l2_distance']) for r in valid_test2]
                distances_by_k.sort()
                print(f"    Distance increases with k: {', '.join([f'k={k}: {d:.4f}' for k, d in distances_by_k[:3]])}")
    
    print("\n" + "=" * 100)
    
    return all_results


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("PROTOTYPE-BASED METHOD - PARAMETER TESTING")
    logger.info("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, y_train_pred, model, scaler = load_auto_mpg_data()
    
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
            results = test_prototype_parameters(
                model=model,
                X_train=X_train,
                y_train=y_train_pred,
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
        # Count valid results from Test 1 (epsilon sensitivity with top_k=1)
        test1_results = [r for r in exp['results'] if r['top_k'] == 1]
        valid_count = sum(1 for r in test1_results if r['valid'])
        
        print(f"\nSample {exp['sample_idx']} → Target {exp['target_idx']}: "
              f"{exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG "
              f"({exp['target_prediction'] - exp['sample_prediction']:+.2f})")
        print(f"  Valid configurations: {valid_count}/{len(test1_results)} epsilon values tested")
        
        if valid_count > 0:
            valid_results = [r for r in test1_results if r['valid']]
            best = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
            print(f"  Best: epsilon={best['epsilon']:.1f}, distance={best['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best['metrics']['sparsity']}")
        else:
            print(f"  No valid prototypes found (try larger epsilon)")
    
    # Detailed results table
    print("\n" + "─" * 100)
    print("DETAILED RESULTS TABLE - ALL SCENARIOS")
    print("─" * 100)
    print(f"\n{'Scenario':<15}{'Epsilon':<10}{'Valid':<8}{'Prediction':<12}{'Error':<10}{'L2 Dist':<12}{'Sparsity':<10}{'Prototypes':<12}")
    print("─" * 100)
    
    for exp in all_experiment_results:
        scenario = f"S{exp['sample_idx']}→T{exp['target_idx']}"
        target = exp['target_prediction']
        
        # Show Test 1 results (epsilon sensitivity with top_k=1)
        test1_results = [r for r in exp['results'] if r['top_k'] == 1]
        
        for result in test1_results:
            eps = result['epsilon']
            valid = "✓" if result['valid'] else "✗"
            
            if result['counterfactual'] is not None and result['valid']:
                pred = result['prediction']
                pred_error = abs(pred - target)
                l2_dist = result['metrics']['l2_distance']
                sparsity = result['metrics']['sparsity']
                n_cand = result['info']['n_candidates_found']
                
                print(f"{scenario:<15}{eps:<10.1f}{valid:<8}{pred:<12.2f}{pred_error:<10.2f}{l2_dist:<12.4f}{str(sparsity):<10}{n_cand:<12}")
            else:
                n_cand = result['info'].get('n_candidates_found', 0)
                print(f"{scenario:<15}{eps:<10.1f}{valid:<8}{'N/A':<12}{'N/A':<10}{'N/A':<12}{'N/A':<10}{n_cand:<12}")
    
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
            print(f"  Best distance: eps={best_dist['epsilon']:.1f}, top_k={best_dist['top_k']} "
                  f"→ L2={best_dist['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_dist['metrics']['sparsity']}")
            print(f"  Best sparsity: eps={best_sparse['epsilon']:.1f}, top_k={best_sparse['top_k']} "
                  f"→ L2={best_sparse['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_sparse['metrics']['sparsity']}")
        else:
            print(f"\n{scenario}: {exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG")
            print(f"  ✗ No valid prototypes found")
    
    # Overall statistics
    print("\n" + "─" * 100)
    print("OVERALL STATISTICS")
    print("─" * 100)
    
    total_scenarios = len(all_experiment_results)
    scenarios_with_solution = sum(1 for exp in all_experiment_results if any(r['valid'] for r in exp['results']))
    
    print(f"\nTotal scenarios tested: {total_scenarios}")
    print(f"Scenarios with valid prototypes: {scenarios_with_solution}/{total_scenarios} "
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
            
            print(f"\nAverage metrics (best solutions):")
            print(f"  Average L2 distance: {avg_dist:.4f}")
            print(f"  Average sparsity: {avg_sparse:.1f} features")
            
            # Compare with Growing Spheres if available
            print(f"\n• Method characteristics:")
            print(f"    Returns REAL training instances (not synthetic)")
            print(f"    Guarantees realistic/observed combinations")
            print(f"    Distance typically LARGER than Growing Spheres")
            print(f"    (Growing Spheres interpolates, finding closer points)")
    
    print("\n" + "=" * 100)
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
