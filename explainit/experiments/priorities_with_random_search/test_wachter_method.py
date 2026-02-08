"""
Wachter Method Testing Script
Test different lambda and epsilon values to understand their impact on counterfactual generation.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import sys
import os
from pathlib import Path
import tensorflow as tf

from explainit.experiments.priorities_with_random_search.standard_methods import wachter_counterfactual, compute_metrics

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


def test_wachter_parameters(model, X_train, sample, target_value, feature_names):
    """
    Test Wachter's method with different lambda and epsilon values.
    
    Args:
        model: Trained model
        X_train: Training data (for feature ranges)
        sample: Sample to generate counterfactual for
        target_value: Target prediction value
        feature_names: Names of features
    """
    # Define feature ranges
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = X_train[:, idx].min()
        max_val = X_train[:, idx].max()
        feature_ranges.append((float(min_val), float(max_val)))
    
    # Model prediction wrapper
    def model_predict(X):
        return model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    # Test different parameter combinations
    lambda_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    epsilon_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # First, test if the loss function has meaningful gradients
    def test_gradient_sensitivity(X_original, model_predict, target_value, lambda_param):
        """Test if small changes in features produce detectable changes in loss."""
        def loss_function(X_cf):
            pred = model_predict([X_cf])[0]
            pred_loss = (pred - target_value) ** 2
            distance_loss = np.sum((X_cf - X_original) ** 2)
            return lambda_param * pred_loss + distance_loss
        
        base_loss = loss_function(X_original)
        
        # Test small perturbations in each feature
        sensitivities = []
        step_sizes = [0.001, 0.01, 0.1]
        
        for feat_idx in range(len(X_original)):
            for step in step_sizes:
                X_perturbed = X_original.copy()
                X_perturbed[feat_idx] += step
                new_loss = loss_function(X_perturbed)
                delta_loss = abs(new_loss - base_loss)
                
                if delta_loss > 1e-10:
                    sensitivities.append({
                        'feature': feat_idx,
                        'step': step,
                        'delta_loss': delta_loss,
                        'gradient_approx': delta_loss / step
                    })
                    break
        
        return sensitivities
    
    # Test gradient sensitivity once with first lambda value
    print(f"\n{'=' * 100}")
    print(f"GRADIENT SENSITIVITY TEST (lambda={lambda_values[0]})")
    print(f"{'=' * 100}")
    sensitivities = test_gradient_sensitivity(sample, model_predict, target_value, lambda_values[0])
    
    if sensitivities:
        print(f"✓ Loss function IS sensitive to feature changes:")
        for s in sensitivities:
            print(f"  Feature {s['feature']} ({feature_names[s['feature']]}): "
                  f"step={s['step']:.3f} → Δloss={s['delta_loss']:.6f}, "
                  f"grad≈{s['gradient_approx']:.2f}")
    else:
        print(f"✗ Loss function NOT sensitive - gradients effectively zero")
        print(f"  This explains why optimizer cannot move!")
    print(f"{'=' * 100}\n")
    
    # Modified Wachter wrapper that ALWAYS returns the counterfactual
    def wachter_always_return(X_original, model_predict, target_value, epsilon, lambda_param, feature_ranges, max_iter):
        """Wrapper that returns counterfactual even if not valid."""
        from scipy.optimize import minimize
        
        # Track function evaluations for debugging
        eval_count = [0]
        
        def loss_function(X_cf):
            pred = model_predict([X_cf])[0]
            pred_loss = (pred - target_value) ** 2
            distance_loss = np.sum((X_cf - X_original) ** 2)
            total_loss = lambda_param * pred_loss + distance_loss
            eval_count[0] += 1
            return total_loss
        
        # Test initial loss
        initial_loss = loss_function(X_original)
        eval_count[0] = 0  # Reset counter
        
        # Try multiple optimization methods
        methods_to_try = [
            ('L-BFGS-B', {
                'maxiter': max_iter,
                'ftol': 1e-15,  # Much tighter convergence tolerance
                'gtol': 1e-10,  # Much tighter gradient tolerance  
                'eps': 1e-4,    # Larger step for gradient approximation
                'maxfun': 15000,
                'maxls': 50,
                'disp': False
            }),
            ('SLSQP', {
                'maxiter': max_iter,
                'ftol': 1e-15,
                'eps': 1e-4,
                'disp': False
            }),
            ('Powell', {  # Derivative-free method
                'maxiter': max_iter,
                'ftol': 1e-10,
                'disp': False
            })
        ]
        
        best_result = None
        best_method = None
        best_pred_error = float('inf')
        
        for method_name, options in methods_to_try:
            try:
                if method_name == 'Powell':
                    # Powell doesn't support bounds directly, so we skip it if bounds are tight
                    result = minimize(
                        loss_function,
                        X_original,
                        method=method_name,
                        options=options
                    )
                else:
                    result = minimize(
                        loss_function,
                        X_original,
                        method=method_name,
                        bounds=feature_ranges,
                        options=options
                    )
                
                # Check if this result is better
                X_cf_test = result.x
                pred_test = model_predict([X_cf_test])[0]
                pred_error = abs(pred_test - target_value)
                
                if best_result is None or pred_error < best_pred_error:
                    best_result = result
                    best_method = method_name
                    best_pred_error = pred_error
                
                # If we found a valid solution, stop trying other methods
                if pred_error <= epsilon:
                    break
                    
            except Exception as e:
                continue
        
        # Use best result
        if best_result is None:
            # Fallback to L-BFGS-B if all failed
            result = minimize(
                loss_function,
                X_original,
                method='L-BFGS-B',
                bounds=feature_ranges,
                options=methods_to_try[0][1]
            )
            best_method = 'L-BFGS-B (fallback)'
        else:
            result = best_result
        
        X_cf = result.x
        prediction = model_predict([X_cf])[0]
        is_valid = abs(prediction - target_value) <= epsilon
        
        info = {
            'method': f'Wachter ({best_method})',
            'valid': is_valid,
            'distance': np.linalg.norm(X_cf - X_original),
            'iterations': result.nit,
            'success': result.success,
            'final_loss': result.fun,
            'function_evals': eval_count[0],
            'initial_loss': initial_loss,
            'message': result.message,
            'optimizer_used': best_method
        }
        
        # ALWAYS return counterfactual, even if not valid
        return X_cf, prediction, info
    
    
    print("\n" + "=" * 100)
    print("WACHTER'S METHOD - PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(f"\nOriginal Sample Prediction: {model_predict([sample])[0]:.2f} MPG")
    print(f"Target Prediction: {target_value:.2f} MPG")
    print(f"Distance to Target: {abs(model_predict([sample])[0] - target_value):.2f} MPG")
    print(f"\nOriginal Sample Values: {sample}")
    print("\n" + "=" * 100)
    
    # Store results for summary
    all_results = []
    
    for epsilon in epsilon_values:
        print(f"\n{'=' * 100}")
        print(f"EPSILON = {epsilon:.1f} MPG (tolerance for reaching target)")
        print(f"{'=' * 100}")
        
        for lambda_param in lambda_values:
            # Run Wachter's method (using wrapper that always returns result)
            cf, pred, info = wachter_always_return(
                X_original=sample,
                model_predict=model_predict,
                target_value=target_value,
                epsilon=epsilon,
                lambda_param=lambda_param,
                feature_ranges=feature_ranges,
                max_iter=1000
            )
            
            # Prepare result
            result = {
                'epsilon': epsilon,
                'lambda': lambda_param,
                'valid': info['valid'],
                'prediction': pred,
                'counterfactual': cf,
                'info': info
            }
            
            # Always compute metrics since we always have a counterfactual
            metrics = compute_metrics(sample, cf, pred, target_value, epsilon)
            result['metrics'] = metrics
            
            all_results.append(result)
            
            # Print result
            print(f"\n  Lambda = {lambda_param:5.2f} (weight: prediction vs distance)")
            print(f"  {'─' * 90}")
            
            if info['valid']:
                print(f"  ✓ VALID counterfactual found")
                print(f"    Optimizer: {info['optimizer_used']}")
                print(f"    Prediction: {pred:.2f} MPG (target: {target_value:.2f}, error: {abs(pred - target_value):.2f})")
                print(f"    Distance: L2={metrics['l2_distance']:.4f}, L1={metrics['l1_distance']:.4f}")
                print(f"    Sparsity: {metrics['sparsity']} features changed")
                print(f"    Optimization: {info['iterations']} iters, {info['function_evals']} func evals")
                print(f"    Loss: Initial={info['initial_loss']:.6f} → Final={info['final_loss']:.6f} "
                      f"(reduction: {((info['initial_loss']-info['final_loss'])/info['initial_loss']*100):.1f}%)")
                
                # Show feature changes
                print(f"    Feature Changes:")
                for idx, (orig, new) in enumerate(zip(sample, cf)):
                    if abs(orig - new) > 0.001:
                        change_pct = ((new - orig) / (orig + 1e-10)) * 100
                        print(f"      {feature_names[idx]:15s}: {orig:.4f} → {new:.4f} (Δ={new-orig:+.4f}, {change_pct:+.1f}%)")
            
            else:
                print(f"  ✗ INVALID - Did not reach target within epsilon")
                print(f"    Optimizer: {info['optimizer_used']}")
                print(f"    Prediction: {pred:.2f} MPG (target: {target_value:.2f}, error: {abs(pred - target_value):.2f})")
                print(f"    Distance: L2={metrics['l2_distance']:.4f}, L1={metrics['l1_distance']:.4f}")
                print(f"    Sparsity: {metrics['sparsity']} features changed")
                print(f"    Optimization: {info['iterations']} iters, {info['function_evals']} func evals")
                
                if info['initial_loss'] != info['final_loss']:
                    loss_reduction = ((info['initial_loss']-info['final_loss'])/info['initial_loss']*100)
                    print(f"    Loss: Initial={info['initial_loss']:.6f} → Final={info['final_loss']:.6f} "
                          f"(reduction: {loss_reduction:.1f}%)")
                else:
                    print(f"    Loss: {info['initial_loss']:.6f} (NO CHANGE)")
                
                if info['iterations'] == 0:
                    print(f"    ⚠️  WARNING: Optimizer stopped immediately (0 iterations)!")
                    print(f"    Message: {info['message']}")
                    print(f"    Possible causes:")
                    print(f"      - Numerical gradient ≈ 0 (step size too small or NN insensitive)")
                    print(f"      - Initial point seen as local minimum")
                    print(f"      - Trying alternative optimizers...")
                
                # Show feature changes for invalid too (to see what it tried)
                if metrics['sparsity'] > 0:
                    print(f"    Feature Changes (attempted):")
                    for idx, (orig, new) in enumerate(zip(sample, cf)):
                        if abs(orig - new) > 0.001:
                            change_pct = ((new - orig) / (orig + 1e-10)) * 100
                            print(f"      {feature_names[idx]:15s}: {orig:.4f} → {new:.4f} (Δ={new-orig:+.4f}, {change_pct:+.1f}%)")
                else:
                    print(f"    No features were changed (optimization failed to move)")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"\n{'Epsilon':<10}{'Lambda':<10}{'Valid':<10}{'Prediction':<12}{'Pred Error':<12}{'L2 Distance':<15}{'Sparsity':<12}{'Iterations':<12}")
    print("─" * 100)
    
    for result in all_results:
        eps = result['epsilon']
        lam = result['lambda']
        valid = "✓" if result['valid'] else "✗"
        pred = result['prediction']
        pred_error = abs(pred - target_value)
        l2_dist = result['metrics']['l2_distance']
        sparsity = result['metrics']['sparsity']
        iterations = result['info']['iterations']
        
        print(f"{eps:<10.1f}{lam:<10.2f}{valid:<10}{pred:<12.2f}{pred_error:<12.2f}{l2_dist:<15.4f}{str(sparsity):<12}{iterations:<12}")
    
    print("\n" + "=" * 100)
    
    # Analysis insights
    print("\nKEY INSIGHTS:")
    print("─" * 100)
    
    # Count valid results
    valid_results = [r for r in all_results if r['valid']]
    print(f"• Valid counterfactuals found: {len(valid_results)}/{len(all_results)} ({100*len(valid_results)/len(all_results):.1f}%)")
    
    if valid_results:
        # Best by distance
        best_by_distance = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
        print(f"\n• Closest counterfactual (best distance):")
        print(f"    Lambda={best_by_distance['lambda']:.2f}, Epsilon={best_by_distance['epsilon']:.1f}")
        print(f"    L2 Distance={best_by_distance['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_distance['prediction']:.2f} MPG")
        print(f"    Sparsity={best_by_distance['metrics']['sparsity']} features changed")
        
        # Best by sparsity
        best_by_sparsity = min(valid_results, key=lambda r: r['metrics']['sparsity'])
        print(f"\n• Most sparse counterfactual (fewest changes):")
        print(f"    Lambda={best_by_sparsity['lambda']:.2f}, Epsilon={best_by_sparsity['epsilon']:.1f}")
        print(f"    Sparsity={best_by_sparsity['metrics']['sparsity']} features changed")
        print(f"    L2 Distance={best_by_sparsity['metrics']['l2_distance']:.4f}")
        print(f"    Prediction={best_by_sparsity['prediction']:.2f} MPG")
        
        # Lambda effect
        print(f"\n• Lambda parameter effect:")
        print(f"    Low lambda (0.01-0.1): Prioritizes proximity → smaller distance, may not reach target")
        print(f"    Medium lambda (0.5-1.0): Balanced approach → good tradeoff")
        print(f"    High lambda (5.0-10.0): Prioritizes prediction → reaches target, larger distance")
        
        # Epsilon effect
        print(f"\n• Epsilon parameter effect:")
        print(f"    Small epsilon (0.5-1.0): Strict target tolerance → fewer valid solutions")
        print(f"    Large epsilon (3.0-5.0): Relaxed target tolerance → more valid solutions, less precise")
    
    print("\n" + "=" * 100)
    
    return all_results


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("WACHTER METHOD - PARAMETER TESTING")
    logger.info("=" * 80)
    
    # Load data
    X_train, X_test, y_train, y_test, model, scaler = load_auto_mpg_data()
    
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
            results = test_wachter_parameters(
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
        valid_count = sum(1 for r in exp['results'] if r['valid'])
        
        print(f"\nSample {exp['sample_idx']} → Target {exp['target_idx']}: "
              f"{exp['sample_prediction']:.2f} → {exp['target_prediction']:.2f} MPG "
              f"({exp['target_prediction'] - exp['sample_prediction']:+.2f})")
        print(f"  Valid configurations: {valid_count}/{len(exp['results'])} (lambda×epsilon combinations tested)")
        
        if valid_count > 0:
            valid_results = [r for r in exp['results'] if r['valid']]
            best = min(valid_results, key=lambda r: r['metrics']['l2_distance'])
            print(f"  Best: lambda={best['lambda']:.2f}, epsilon={best['epsilon']:.1f}, "
                  f"distance={best['metrics']['l2_distance']:.4f}, sparsity={best['metrics']['sparsity']}, "
                  f"optimizer={best['info']['optimizer']}")
        else:
            print(f"  No valid counterfactuals found (optimization failed or target unreachable)")
    
    # Detailed results table
    print("\n" + "─" * 100)
    print("DETAILED RESULTS TABLE - ALL SCENARIOS")
    print("─" * 100)
    print(f"\n{'Scenario':<15}{'Lambda':<10}{'Epsilon':<10}{'Valid':<8}{'Prediction':<12}{'Error':<10}{'L2 Dist':<12}{'Sparsity':<10}{'Optimizer':<12}")
    print("─" * 100)
    
    for exp in all_experiment_results:
        scenario = f"S{exp['sample_idx']}→T{exp['target_idx']}"
        target = exp['target_prediction']
        
        for result in exp['results']:
            lam = result['lambda']
            eps = result['epsilon']
            valid = "✓" if result['valid'] else "✗"
            
            if result['counterfactual'] is not None and result['valid']:
                pred = result['prediction']
                pred_error = abs(pred - target)
                l2_dist = result['metrics']['l2_distance']
                sparsity = result['metrics']['sparsity']
                optimizer = result['info']['optimizer']
                
                print(f"{scenario:<15}{lam:<10.2f}{eps:<10.1f}{valid:<8}{pred:<12.2f}{pred_error:<10.2f}{l2_dist:<12.4f}{str(sparsity):<10}{optimizer:<12}")
            else:
                optimizer = result['info'].get('optimizer', 'N/A')
                print(f"{scenario:<15}{lam:<10.2f}{eps:<10.1f}{valid:<8}{'N/A':<12}{'N/A':<10}{'N/A':<12}{'N/A':<10}{optimizer:<12}")
    
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
            print(f"  Best distance: lambda={best_dist['lambda']:.2f}, eps={best_dist['epsilon']:.1f}, "
                  f"optimizer={best_dist['info']['optimizer']} → L2={best_dist['metrics']['l2_distance']:.4f}, "
                  f"sparsity={best_dist['metrics']['sparsity']}")
            print(f"  Best sparsity: lambda={best_sparse['lambda']:.2f}, eps={best_sparse['epsilon']:.1f}, "
                  f"optimizer={best_sparse['info']['optimizer']} → L2={best_sparse['metrics']['l2_distance']:.4f}, "
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
            
            # Optimizer success breakdown
            optimizer_counts = {}
            for b in all_best:
                opt = b['info']['optimizer']
                optimizer_counts[opt] = optimizer_counts.get(opt, 0) + 1
            
            print(f"\nBest solutions by optimizer:")
            for opt, count in sorted(optimizer_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {opt}: {count}/{len(all_best)} scenarios ({100*count/len(all_best):.1f}%)")
    
    print("\n" + "=" * 100)
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
