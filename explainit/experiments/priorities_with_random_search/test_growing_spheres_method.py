"""
Growing Spheres Method Testing Script
Test different epsilon and n_search_samples values to understand their impact on counterfactual generation.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import sys
import os
from pathlib import Path
import tensorflow as tf

from explainit.experiments.priorities_with_random_search.standard_methods import compute_metrics

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


def growing_spheres_modified(
    X_original: np.ndarray,
    model_predict,
    target_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.0,
    n_search_samples: int = 20,
    n_top_candidates: int = 10
):
    """
    Modified Growing Spheres with additional diagnostics and configurable parameters.
    
    Args:
        X_original: Original instance to explain
        model_predict: Model prediction function
        target_value: Target prediction value
        X_train: Training dataset
        y_train: Training predictions
        epsilon: Tolerance for target prediction
        n_search_samples: Number of interpolation samples for binary search
        n_top_candidates: Number of closest prototypes to try
    
    Returns:
        counterfactual: Generated counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    # Find training instances within epsilon of target
    target_mask = np.abs(y_train - target_value) <= epsilon
    target_instances = X_train[target_mask]
    
    if len(target_instances) == 0:
        logger.warning("No training instances found within epsilon of target")
        return None, None, {
            'method': 'Growing Spheres',
            'valid': False,
            'reason': 'no_target_instances',
            'n_candidates_found': 0
        }
    
    # Calculate distances to all target instances
    distances = np.linalg.norm(target_instances - X_original, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Try closest instances with binary search
    best_cf = None
    best_pred = None
    best_distance = float('inf')
    candidates_tried = 0
    valid_found = 0
    
    for idx in sorted_indices[:n_top_candidates]:
        candidates_tried += 1
        candidate = target_instances[idx]
        
        # Binary search along the line from original to candidate
        alphas = np.linspace(0, 1, n_search_samples)
        for alpha in alphas:
            interpolated = (1 - alpha) * X_original + alpha * candidate
            pred = model_predict([interpolated])[0]
            
            # Check if valid
            if abs(pred - target_value) <= epsilon:
                dist = np.linalg.norm(interpolated - X_original)
                if dist < best_distance:
                    best_cf = interpolated
                    best_pred = pred
                    best_distance = dist
                    valid_found += 1
                break
    
    if best_cf is not None:
        info = {
            'method': 'Growing Spheres',
            'valid': True,
            'distance': best_distance,
            'n_candidates_found': len(target_instances),
            'n_candidates_tried': candidates_tried,
            'n_valid_found': valid_found
        }
        return best_cf, best_pred, info
    else:
        return None, None, {
            'method': 'Growing Spheres',
            'valid': False,
            'reason': 'no_valid_cf_found',
            'n_candidates_found': len(target_instances),
            'n_candidates_tried': candidates_tried,
            'n_valid_found': 0
        }


def test_growing_spheres_parameters(model, X_train, y_train, sample, target_value, feature_names):
    """
    Test Growing Spheres method with different epsilon and n_search_samples values.
    
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
    n_search_samples_values = [5, 10, 20, 50, 100]
    n_top_candidates_values = [5, 10, 20]
    
    print("\n" + "=" * 100)
    print("GROWING SPHERES METHOD - PARAMETER SENSITIVITY ANALYSIS")
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
            print(f"Epsilon={eps:5.1f}: {n_instances:4d} training instances found | "
                  f"Distance range: [{min_dist:.4f}, {max_dist:.4f}], mean={mean_dist:.4f}")
        else:
            print(f"Epsilon={eps:5.1f}: {n_instances:4d} training instances found")
    
    print("\n" + "=" * 100)
    
    # Store results for summary
    all_results = []
    
    # Test 1: Effect of epsilon (with fixed n_search_samples and n_top_candidates)
    print("\n" + "=" * 100)
    print("TEST 1: EPSILON SENSITIVITY (n_search_samples=20, n_top_candidates=10)")
    print("=" * 100)
    
    for epsilon in epsilon_values:
        # Run Growing Spheres
        cf, pred, info = growing_spheres_modified(
            X_original=sample,
            model_predict=model_predict,
            target_value=target_value,
            X_train=X_train,
            y_train=y_train,
            epsilon=epsilon,
            n_search_samples=20,
            n_top_candidates=10
        )
        
        # Prepare result
        result = {
            'epsilon': epsilon,
            'n_search_samples': 20,
            'n_top_candidates': 10,
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
            print(f"  Training instances available: {info['n_candidates_found']}")
            print(f"  Candidates tried: {info['n_candidates_tried']}, Valid found: {info['n_valid_found']}")
            
            # Show feature changes
            print(f"  Feature Changes:")
            for idx, (orig, new) in enumerate(zip(sample, cf)):
                if abs(orig - new) > 0.001:
                    change_pct = ((new - orig) / (orig + 1e-10)) * 100
                    print(f"    {feature_names[idx]:15s}: {orig:.4f} → {new:.4f} (Δ={new-orig:+.4f}, {change_pct:+.1f}%)")
        else:
            print(f"✗ FAILED - No valid counterfactual found")
            print(f"  Reason: {info.get('reason', 'unknown')}")
            print(f"  Training instances available: {info.get('n_candidates_found', 0)}")
            if info.get('n_candidates_tried', 0) > 0:
                print(f"  Candidates tried: {info['n_candidates_tried']}")
    
    # Test 2: Effect of n_search_samples (with fixed epsilon and n_top_candidates)
    # Use an epsilon that found results in Test 1
    valid_epsilon = None
    for r in all_results:
        if r['valid']:
            valid_epsilon = r['epsilon']
            break
    
    if valid_epsilon is not None:
        print("\n" + "=" * 100)
        print(f"TEST 2: N_SEARCH_SAMPLES SENSITIVITY (epsilon={valid_epsilon}, n_top_candidates=10)")
        print("=" * 100)
        
        for n_samples in n_search_samples_values:
            cf, pred, info = growing_spheres_modified(
                X_original=sample,
                model_predict=model_predict,
                target_value=target_value,
                X_train=X_train,
                y_train=y_train,
                epsilon=valid_epsilon,
                n_search_samples=n_samples,
                n_top_candidates=10
            )
            
            result = {
                'epsilon': valid_epsilon,
                'n_search_samples': n_samples,
                'n_top_candidates': 10,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_value, valid_epsilon)
                result['metrics'] = metrics
            
            all_results.append(result)
            
            print(f"\nn_search_samples = {n_samples:3d} (interpolation points)")
            print(f"{'─' * 90}")
            
            if cf is not None and info['valid']:
                print(f"✓ VALID counterfactual found")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Distance: L2={metrics['l2_distance']:.4f}")
                print(f"  Sparsity: {metrics['sparsity']} features changed")
            else:
                print(f"✗ FAILED")
    
    # Test 3: Effect of n_top_candidates (with fixed epsilon and n_search_samples)
    if valid_epsilon is not None:
        print("\n" + "=" * 100)
        print(f"TEST 3: N_TOP_CANDIDATES SENSITIVITY (epsilon={valid_epsilon}, n_search_samples=20)")
        print("=" * 100)
        
        for n_candidates in n_top_candidates_values:
            cf, pred, info = growing_spheres_modified(
                X_original=sample,
                model_predict=model_predict,
                target_value=target_value,
                X_train=X_train,
                y_train=y_train,
                epsilon=valid_epsilon,
                n_search_samples=20,
                n_top_candidates=n_candidates
            )
            
            result = {
                'epsilon': valid_epsilon,
                'n_search_samples': 20,
                'n_top_candidates': n_candidates,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            if cf is not None:
                metrics = compute_metrics(sample, cf, pred, target_value, valid_epsilon)
                result['metrics'] = metrics
            
            all_results.append(result)
            
            print(f"\nn_top_candidates = {n_candidates:2d} (prototypes to try)")
            print(f"{'─' * 90}")
            
            if cf is not None and info['valid']:
                print(f"✓ VALID counterfactual found")
                print(f"  Prediction: {pred:.2f} MPG (error: {abs(pred - target_value):.2f})")
                print(f"  Distance: L2={metrics['l2_distance']:.4f}")
                print(f"  Sparsity: {metrics['sparsity']} features changed")
            else:
                print(f"✗ FAILED")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - TEST 1 (EPSILON SENSITIVITY)")
    print("=" * 100)
    print(f"\n{'Epsilon':<10}{'Valid':<10}{'Prediction':<12}{'Pred Error':<12}{'L2 Distance':<15}{'Sparsity':<12}{'Candidates':<12}")
    print("─" * 100)
    
    for result in all_results:
        if result['n_search_samples'] == 20 and result['n_top_candidates'] == 10:
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
    test1_results = [r for r in all_results if r['n_search_samples'] == 20 and r['n_top_candidates'] == 10]
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
        
        # Best by sparsity
        best_by_sparsity = min(valid_results, key=lambda r: r['metrics']['sparsity'])
        print(f"\n• Most sparse counterfactual:")
        print(f"    Epsilon={best_by_sparsity['epsilon']:.1f}")
        print(f"    Sparsity={best_by_sparsity['metrics']['sparsity']} features changed")
        print(f"    L2 Distance={best_by_sparsity['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_sparsity['prediction']:.2f} MPG")
        
        # Epsilon effect
        print(f"\n• Epsilon parameter effect:")
        print(f"    Smaller epsilon → Fewer training instances available → May fail")
        print(f"    Larger epsilon → More training instances → Higher success rate, but less precise")
        
        # n_search_samples effect (if tested)
        test2_results = [r for r in all_results if r['epsilon'] == valid_epsilon and r['n_top_candidates'] == 10 and r['n_search_samples'] != 20]
        if test2_results:
            print(f"\n• n_search_samples parameter effect:")
            print(f"    More interpolation points → Finer search granularity")
            print(f"    But results are similar (distance varies by < 1% typically)")
        
        # n_top_candidates effect (if tested)
        test3_results = [r for r in all_results if r['epsilon'] == valid_epsilon and r['n_search_samples'] == 20 and r['n_top_candidates'] != 10]
        if test3_results:
            print(f"\n• n_top_candidates parameter effect:")
            print(f"    More candidates → More options to find closer counterfactual")
            print(f"    But usually first few candidates are sufficient")
    
    print("\n" + "=" * 100)
    
    return all_results


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("GROWING SPHERES METHOD - PARAMETER TESTING")
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
            results = test_growing_spheres_parameters(
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
        # Count valid results from Test 1 (epsilon sensitivity with standard params)
        test1_results = [r for r in exp['results'] if r['n_search_samples'] == 20 and r['n_top_candidates'] == 10]
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
            print(f"  No valid counterfactuals found (try larger epsilon or different method)")
    
    # Detailed results table
    print("\n" + "─" * 100)
    print("DETAILED RESULTS TABLE - ALL SCENARIOS")
    print("─" * 100)
    print(f"\n{'Scenario':<15}{'Epsilon':<10}{'Valid':<8}{'Prediction':<12}{'Error':<10}{'L2 Dist':<12}{'Sparsity':<10}{'Candidates':<12}")
    print("─" * 100)
    
    for exp in all_experiment_results:
        scenario = f"S{exp['sample_idx']}→T{exp['target_idx']}"
        target = exp['target_prediction']
        
        # Show Test 1 results (epsilon sensitivity)
        test1_results = [r for r in exp['results'] if r['n_search_samples'] == 20 and r['n_top_candidates'] == 10]
        
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
            print(f"  Best distance: eps={best_dist['epsilon']:.1f}, n_search={best_dist['n_search_samples']}, "
                  f"n_cand={best_dist['n_top_candidates']} → L2={best_dist['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_dist['metrics']['sparsity']}")
            print(f"  Best sparsity: eps={best_sparse['epsilon']:.1f}, n_search={best_sparse['n_search_samples']}, "
                  f"n_cand={best_sparse['n_top_candidates']} → L2={best_sparse['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_sparse['metrics']['sparsity']}")
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
            
            print(f"\nAverage metrics (best solutions):")
            print(f"  Average L2 distance: {avg_dist:.4f}")
            print(f"  Average sparsity: {avg_sparse:.1f} features")
    
    print("\n" + "=" * 100)
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
