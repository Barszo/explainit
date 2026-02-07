"""
Test script for counterfactual explanation experiments.
This script helps you test the experiment setup with different configurations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_standard_methods_only():
    """
    Test only standard methods with a quick configuration.
    Useful for verifying the implementation works.
    """
    logger.info("=" * 80)
    logger.info("TESTING STANDARD METHODS (QUICK TEST)")
    logger.info("=" * 80)
    
    from experiment_final import AutoMPGExperiment, run_standard_methods_experiment, save_standard_methods_results_csv
    
    # Quick test configuration
    config = {
        'dataset': 'Auto_MPG',
        'model': 'NN_Residual',
        'n_quantiles': 2,  # Only 2 points = 2 experiments (2x1)
        'epsilon': 3.0,  # Larger tolerance for quicker success
        
        # Method selection
        'run_preference_method': False,  # Don't run preference method
        'run_standard_methods': True,
        'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],  # All methods
        
        'exemplar_weight': 0.01,
        'n_samples': 100,  # Not used for standard methods
        'use_monte_carlo': True
    }
    
    # Load data
    experiment = AutoMPGExperiment()
    X_train, X_test, y_train, y_test, scaler, model, X_full, y_full = experiment.load_data()
    
    # Run standard methods
    results = run_standard_methods_experiment(model, X_train, X_test, y_train, config, experiment)
    
    # Save results
    save_standard_methods_results_csv(results, config, experiment, filename='test_standard_methods.csv')
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"\nSample {result['sample_idx']} → Target {result['target_idx']}:")
        for method_name, method_result in result['methods'].items():
            if method_result['counterfactual'] is not None:
                metrics = method_result['metrics']
                logger.info(f"  {method_name}: ✓ VALID")
                logger.info(f"    L2 distance: {metrics['l2_distance']:.4f}")
                logger.info(f"    Sparsity: {metrics['sparsity']}")
                logger.info(f"    Prediction: {method_result['prediction']:.2f}")
            else:
                logger.info(f"  {method_name}: ✗ FAILED")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE - Check test_standard_methods.csv")
    logger.info("=" * 80)


def test_preference_method_only():
    """
    Test only preference-based random search method.
    """
    logger.info("=" * 80)
    logger.info("TESTING PREFERENCE-BASED METHOD (QUICK TEST)")
    logger.info("=" * 80)
    
    from experiment_final import AutoMPGExperiment, run_counterfactual_experiment, save_results_csv
    
    # Quick test configuration
    config = {
        'dataset': 'Auto_MPG',
        'model': 'NN_Residual',
        'n_quantiles': 2,  # Only 2 points = 2 experiments
        'epsilon': 3.0,
        
        # Method selection
        'run_preference_method': True,
        'run_standard_methods': False,  # Don't run standard methods
        'standard_methods': [],
        
        'return_top_n': 3,  # Keep top 3 CFs
        'exemplar_weight': 0.01,
        'n_samples': 1000,  # Quick generation
        'use_monte_carlo': True
    }
    
    # Load data
    experiment = AutoMPGExperiment()
    X_train, X_test, y_train, y_test, scaler, model, X_full, y_full = experiment.load_data()
    
    # Run preference-based method
    results = run_counterfactual_experiment(model, X_train, X_test, y_train, config, experiment)
    
    # Save results
    save_results_csv(results, config, experiment, filename='test_preference_based.csv')
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"\nSample {result['sample_idx']} → Target {result['target_idx']}: "
                   f"{result['n_counterfactuals']} counterfactuals found")
        if result['n_counterfactuals'] > 0:
            best_cf = result['counterfactuals'][0]
            logger.info(f"  Best CF: prediction={best_cf['prediction']:.2f}, "
                       f"preference={best_cf['preference_score']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE - Check test_preference_based.csv")
    logger.info("=" * 80)


def test_single_comparison():
    """
    Test both methods on a single sample-target pair for detailed comparison.
    """
    logger.info("=" * 80)
    logger.info("TESTING SINGLE SAMPLE-TARGET PAIR COMPARISON")
    logger.info("=" * 80)
    
    from experiment_final import AutoMPGExperiment
    from standard_methods import run_all_methods
    from explainit.explainers.random_search import RandomSearchExplainer
    from explainit.priorities.nonlinear import exponential
    import tensorflow as tf
    
    # Load data
    experiment = AutoMPGExperiment()
    X_train, X_test, y_train, y_test, scaler, model, X_full, y_full = experiment.load_data()
    
    # Select a sample and target
    sample = X_test[0]
    target_sample = X_test[10]
    
    sample_pred = model(tf.constant([sample.astype(np.float32)]), training=False).numpy()[0][0]
    target_pred = model(tf.constant([target_sample.astype(np.float32)]), training=False).numpy()[0][0]
    
    logger.info(f"\nSample prediction: {sample_pred:.2f} MPG")
    logger.info(f"Target prediction: {target_pred:.2f} MPG")
    logger.info(f"Distance to target: {abs(sample_pred - target_pred):.2f} MPG\n")
    
    epsilon = 2.0
    
    # ========================================================================
    # Test Standard Methods
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STANDARD METHODS")
    logger.info("=" * 80)
    
    feature_ranges = []
    for idx in range(X_train.shape[1]):
        min_val = X_train[:, idx].min()
        max_val = X_train[:, idx].max()
        feature_ranges.append((float(min_val), float(max_val)))
    
    model_predict = lambda X: model(tf.constant(np.array(X).astype(np.float32)), training=False).numpy().ravel()
    
    standard_results = run_all_methods(
        X_original=sample,
        model=model,
        model_predict=model_predict,
        target_value=target_pred,
        X_train=X_train,
        y_train=y_train,
        epsilon=epsilon,
        feature_ranges=feature_ranges
    )
    
    for method_name, result in standard_results.items():
        logger.info(f"\n{method_name.upper()}:")
        if result['counterfactual'] is not None:
            metrics = result['metrics']
            logger.info(f"  ✓ VALID")
            logger.info(f"  Prediction: {result['prediction']:.2f} MPG")
            logger.info(f"  L2 Distance: {metrics['l2_distance']:.4f}")
            logger.info(f"  Sparsity: {metrics['sparsity']} features changed")
            logger.info(f"  Prediction error: {metrics['prediction_error']:.4f}")
        else:
            logger.info(f"  ✗ FAILED")
    
    # ========================================================================
    # Test Preference-Based Method
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREFERENCE-BASED RANDOM SEARCH")
    logger.info("=" * 80)
    
    preferences = experiment.define_preferences(sample.tolist(), target_sample.tolist(), X_train, exemplar_weight=0.01)
    
    explainer = RandomSearchExplainer(
        model_pred=model_predict,
        priorities=preferences,
        sample=sample.tolist(),
        target=target_pred
    )
    
    cf_samples, cf_predictions, cf_scores = explainer.generate_random_samples(
        n_samples=5000,
        epsilon=epsilon,
        use_monte_carlo=True,
        random_seed=42,
        max_tries=100,
        return_top_n=5
    )
    
    logger.info(f"\nFound {len(cf_samples)} counterfactuals")
    
    if len(cf_samples) > 0:
        best_cf = cf_samples[0]
        best_pred = cf_predictions[0]
        best_score = cf_scores[0]
        
        # Calculate metrics
        l2_dist = np.linalg.norm(np.array(best_cf) - sample)
        sparsity = np.sum(np.abs(np.array(best_cf) - sample) > 0.01)
        
        logger.info(f"\nBest counterfactual:")
        logger.info(f"  Prediction: {best_pred:.2f} MPG")
        logger.info(f"  Preference score: {best_score:.4f}")
        logger.info(f"  L2 Distance: {l2_dist:.4f}")
        logger.info(f"  Sparsity: {int(sparsity)} features changed")
        logger.info(f"  Prediction error: {abs(best_pred - target_pred):.4f}")
    else:
        logger.info("  ✗ No counterfactuals found")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


def show_test_options():
    """Display available test options."""
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL EXPERIMENT TEST OPTIONS")
    print("=" * 80)
    print("\nAvailable tests:")
    print("  1. test_standard_methods_only() - Quick test of standard methods (2 experiments)")
    print("  2. test_preference_method_only() - Quick test of preference-based method (2 experiments)")
    print("  3. test_single_comparison() - Detailed comparison on a single sample-target pair")
    print("\nTo run a test:")
    print("  python test_experiment.py 1  # Run test 1")
    print("  python test_experiment.py 2  # Run test 2")
    print("  python test_experiment.py 3  # Run test 3")
    print("\nOr import in Python:")
    print("  from test_experiment import test_standard_methods_only")
    print("  test_standard_methods_only()")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_num = sys.argv[1]
        
        if test_num == "1":
            test_standard_methods_only()
        elif test_num == "2":
            test_preference_method_only()
        elif test_num == "3":
            test_single_comparison()
        else:
            print(f"Unknown test number: {test_num}")
            show_test_options()
    else:
        show_test_options()
