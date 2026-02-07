"""
Standard Counterfactual Explanation Methods
Implementation of popular counterfactual generation algorithms for comparison.
"""

import numpy as np
import logging
from scipy.optimize import minimize
from typing import Callable, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# WACHTER'S METHOD (2017)
# ============================================================================

def wachter_counterfactual(
    X_original: np.ndarray,
    model_predict: Callable,
    target_value: float,
    epsilon: float = 1.0,
    lambda_param: float = 0.5,
    feature_ranges: Optional[List[Tuple[float, float]]] = None,
    max_iter: int = 1000
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Wachter et al. (2017) counterfactual generation method.
    
    Minimizes: lambda * (f(x') - target)^2 + ||x' - x||^2
    
    Args:
        X_original: Original instance to explain
        model_predict: Model prediction function
        target_value: Target prediction value
        epsilon: Tolerance for target (valid if |pred - target| <= epsilon)
        lambda_param: Weight for prediction loss vs distance
        feature_ranges: List of (min, max) tuples for each feature
        max_iter: Maximum optimization iterations
    
    Returns:
        counterfactual: Generated counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    def loss_function(X_cf):
        # Prediction loss
        pred = model_predict([X_cf])[0]
        pred_loss = (pred - target_value) ** 2
        
        # Distance loss (L2 norm)
        distance_loss = np.sum((X_cf - X_original) ** 2)
        
        return lambda_param * pred_loss + distance_loss
    
    # Set bounds if provided
    bounds = None
    if feature_ranges is not None:
        bounds = feature_ranges
    
    # Optimize
    result = minimize(
        loss_function,
        X_original,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter}
    )
    
    X_cf = result.x
    prediction = model_predict([X_cf])[0]
    
    # Check if valid (within epsilon of target)
    is_valid = abs(prediction - target_value) <= epsilon
    
    info = {
        'method': 'Wachter',
        'valid': is_valid,
        'distance': np.linalg.norm(X_cf - X_original),
        'iterations': result.nit,
        'success': result.success
    }
    
    if is_valid:
        return X_cf, prediction, info
    else:
        return None, None, info


# ============================================================================
# GROWING SPHERES
# ============================================================================

def growing_spheres(
    X_original: np.ndarray,
    model_predict: Callable,
    target_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.0,
    n_search_samples: int = 20
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Growing Spheres algorithm for counterfactual generation.
    
    Searches in expanding spheres around the original instance to find
    instances with target prediction, then uses binary search to find
    the closest counterfactual on the decision boundary.
    
    Args:
        X_original: Original instance to explain
        model_predict: Model prediction function
        target_value: Target prediction value
        X_train: Training dataset
        y_train: Training labels/predictions
        epsilon: Tolerance for target prediction
        n_search_samples: Number of interpolation samples for binary search
    
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
        return None, None, {'method': 'Growing Spheres', 'valid': False, 'reason': 'no_target_instances'}
    
    # Calculate distances to all target instances
    distances = np.linalg.norm(target_instances - X_original, axis=1)
    sorted_indices = np.argsort(distances)
    
    # Try closest instances with binary search
    best_cf = None
    best_pred = None
    best_distance = float('inf')
    
    for idx in sorted_indices[:10]:  # Try top 10 closest
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
                break
    
    if best_cf is not None:
        info = {
            'method': 'Growing Spheres',
            'valid': True,
            'distance': best_distance
        }
        return best_cf, best_pred, info
    else:
        return None, None, {'method': 'Growing Spheres', 'valid': False, 'reason': 'no_valid_cf_found'}


# ============================================================================
# PROTOTYPE-BASED
# ============================================================================

def prototype_counterfactual(
    X_original: np.ndarray,
    model_predict: Callable,
    target_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.0,
    top_k: int = 1
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Prototype-based counterfactual generation.
    
    Finds real instances from the training data that achieve the target
    prediction and are closest to the original instance.
    
    Args:
        X_original: Original instance to explain
        model_predict: Model prediction function
        target_value: Target prediction value
        X_train: Training dataset
        y_train: Training labels/predictions
        epsilon: Tolerance for target prediction
        top_k: Return k-th closest prototype (1 = closest)
    
    Returns:
        counterfactual: Selected prototype (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    # Find training instances within epsilon of target
    target_mask = np.abs(y_train - target_value) <= epsilon
    target_instances = X_train[target_mask]
    
    if len(target_instances) == 0:
        logger.warning("No training instances found within epsilon of target")
        return None, None, {'method': 'Prototype', 'valid': False, 'reason': 'no_target_instances'}
    
    # Calculate distances to all target instances
    distances = np.linalg.norm(target_instances - X_original, axis=1)
    
    # Get top-k closest
    if len(distances) < top_k:
        logger.warning(f"Only {len(distances)} prototypes available, requested top-{top_k}")
        top_k = len(distances)
    
    closest_idx = np.argsort(distances)[top_k - 1]
    prototype = target_instances[closest_idx]
    
    # Verify prediction
    prediction = model_predict([prototype])[0]
    is_valid = abs(prediction - target_value) <= epsilon
    
    info = {
        'method': 'Prototype',
        'valid': is_valid,
        'distance': distances[closest_idx],
        'n_prototypes': len(target_instances)
    }
    
    return prototype, prediction, info


# ============================================================================
# GRADIENT-BASED (For TensorFlow/Keras models)
# ============================================================================

def gradient_based_counterfactual(
    X_original: np.ndarray,
    model,  # TensorFlow/Keras model
    target_value: float,
    epsilon: float = 1.0,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    lambda_param: float = 0.01,
    feature_ranges: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Gradient-based counterfactual generation using TensorFlow.
    
    Uses gradient descent to find counterfactual by directly optimizing
    the instance with respect to the model's output.
    
    Args:
        X_original: Original instance to explain
        model: TensorFlow/Keras model
        target_value: Target prediction value
        epsilon: Tolerance for target prediction
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum iterations
        lambda_param: Weight for distance penalty
        feature_ranges: List of (min, max) tuples for each feature
    
    Returns:
        counterfactual: Generated counterfactual (or None if failed)
        prediction: Prediction for counterfactual
        info: Dictionary with additional information
    """
    import tensorflow as tf
    
    # Convert to TensorFlow variable
    X_cf = tf.Variable(X_original.astype(np.float32), dtype=tf.float32)
    X_orig_tensor = tf.constant(X_original.astype(np.float32), dtype=tf.float32)
    target_tensor = tf.constant(target_value, dtype=tf.float32)
    
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    
    best_cf = None
    best_pred = None
    best_distance = float('inf')
    
    for iteration in range(max_iter):
        with tf.GradientTape() as tape:
            # Forward pass
            output = model(tf.expand_dims(X_cf, 0), training=False)
            prediction = output[0, 0]
            
            # Loss: prediction error + distance penalty
            pred_loss = tf.square(prediction - target_tensor)
            distance_loss = tf.reduce_sum(tf.square(X_cf - X_orig_tensor))
            
            loss = pred_loss + lambda_param * distance_loss
        
        # Backward pass
        gradients = tape.gradient(loss, [X_cf])
        optimizer.apply_gradients(zip(gradients, [X_cf]))
        
        # Apply bounds if provided
        if feature_ranges is not None:
            clipped_values = []
            for i, (min_val, max_val) in enumerate(feature_ranges):
                clipped_values.append(tf.clip_by_value(X_cf[i], min_val, max_val))
            X_cf.assign(tf.stack(clipped_values))
        
        # Check if valid
        current_pred = prediction.numpy()
        if abs(current_pred - target_value) <= epsilon:
            current_distance = np.linalg.norm(X_cf.numpy() - X_original)
            if current_distance < best_distance:
                best_cf = X_cf.numpy().copy()
                best_pred = current_pred
                best_distance = current_distance
                
                # Early stopping if we found a good solution
                if iteration > 100:
                    break
    
    if best_cf is not None:
        info = {
            'method': 'Gradient-Based',
            'valid': True,
            'distance': best_distance,
            'iterations': iteration + 1
        }
        return best_cf, best_pred, info
    else:
        # Return final result even if not valid
        final_pred = model(tf.expand_dims(X_cf, 0), training=False)[0, 0].numpy()
        info = {
            'method': 'Gradient-Based',
            'valid': False,
            'distance': np.linalg.norm(X_cf.numpy() - X_original),
            'iterations': max_iter,
            'final_error': abs(final_pred - target_value)
        }
        return X_cf.numpy(), final_pred, info


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
    Compute standard counterfactual metrics.
    
    Args:
        X_original: Original instance
        X_counterfactual: Generated counterfactual
        prediction: Prediction for counterfactual
        target_value: Target prediction value
        epsilon: Tolerance for target
    
    Returns:
        Dictionary with computed metrics
    """
    # L1 and L2 distances
    l1_distance = np.sum(np.abs(X_counterfactual - X_original))
    l2_distance = np.linalg.norm(X_counterfactual - X_original)
    
    # Sparsity (number of changed features)
    sparsity = np.sum(np.abs(X_counterfactual - X_original) > 0.01)
    
    # Validity (achieves target within epsilon)
    validity = abs(prediction - target_value) <= epsilon
    
    # Prediction error
    prediction_error = abs(prediction - target_value)
    
    return {
        'l1_distance': l1_distance,
        'l2_distance': l2_distance,
        'sparsity': int(sparsity),
        'validity': validity,
        'prediction_error': prediction_error
    }


def run_all_methods(
    X_original: np.ndarray,
    model,
    model_predict: Callable,
    target_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 1.0,
    feature_ranges: Optional[List[Tuple[float, float]]] = None,
    methods_to_run: Optional[List[str]] = None
) -> dict:
    """
    Run selected standard counterfactual methods and compare results.
    
    Args:
        X_original: Original instance to explain
        model: TensorFlow/Keras model (for gradient-based)
        model_predict: Model prediction function
        target_value: Target prediction value
        X_train: Training dataset
        y_train: Training predictions
        epsilon: Tolerance for target prediction
        feature_ranges: List of (min, max) tuples for each feature
        methods_to_run: List of method names to run. Options: ['wachter', 'growing_spheres', 'prototype', 'gradient_based']
                       If None, runs all methods.
    
    Returns:
        Dictionary with results from selected methods
    """
    # Default to all methods if not specified
    if methods_to_run is None:
        methods_to_run = ['wachter', 'growing_spheres', 'prototype', 'gradient_based']
    
    results = {}
    
    # Wachter's Method
    if 'wachter' in methods_to_run:
        logger.info("  Running Wachter's method...")
        cf_wachter, pred_wachter, info_wachter = wachter_counterfactual(
            X_original, model_predict, target_value, epsilon,
            lambda_param=0.5, feature_ranges=feature_ranges
        )
        if cf_wachter is not None:
            metrics_wachter = compute_metrics(X_original, cf_wachter, pred_wachter, target_value, epsilon)
            results['wachter'] = {
                'counterfactual': cf_wachter,
                'prediction': pred_wachter,
                'metrics': metrics_wachter,
                'info': info_wachter
            }
        else:
            results['wachter'] = {'counterfactual': None, 'info': info_wachter}
    
    # Growing Spheres
    if 'growing_spheres' in methods_to_run:
        logger.info("  Running Growing Spheres...")
        cf_spheres, pred_spheres, info_spheres = growing_spheres(
            X_original, model_predict, target_value, X_train, y_train, epsilon
        )
        if cf_spheres is not None:
            metrics_spheres = compute_metrics(X_original, cf_spheres, pred_spheres, target_value, epsilon)
            results['growing_spheres'] = {
                'counterfactual': cf_spheres,
                'prediction': pred_spheres,
                'metrics': metrics_spheres,
                'info': info_spheres
            }
        else:
            results['growing_spheres'] = {'counterfactual': None, 'info': info_spheres}
    
    # Prototype-based
    if 'prototype' in methods_to_run:
        logger.info("  Running Prototype-based method...")
        cf_prototype, pred_prototype, info_prototype = prototype_counterfactual(
            X_original, model_predict, target_value, X_train, y_train, epsilon
        )
        if cf_prototype is not None:
            metrics_prototype = compute_metrics(X_original, cf_prototype, pred_prototype, target_value, epsilon)
            results['prototype'] = {
                'counterfactual': cf_prototype,
                'prediction': pred_prototype,
                'metrics': metrics_prototype,
                'info': info_prototype
            }
        else:
            results['prototype'] = {'counterfactual': None, 'info': info_prototype}
    
    # Gradient-based (requires TensorFlow model)
    if 'gradient_based' in methods_to_run:
        try:
            logger.info("  Running Gradient-based method...")
            cf_gradient, pred_gradient, info_gradient = gradient_based_counterfactual(
                X_original, model, target_value, epsilon,
                learning_rate=0.01, max_iter=1000, feature_ranges=feature_ranges
            )
            if cf_gradient is not None:
                metrics_gradient = compute_metrics(X_original, cf_gradient, pred_gradient, target_value, epsilon)
                results['gradient_based'] = {
                    'counterfactual': cf_gradient,
                    'prediction': pred_gradient,
                    'metrics': metrics_gradient,
                    'info': info_gradient
                }
            else:
                results['gradient_based'] = {'counterfactual': None, 'info': info_gradient}
        except Exception as e:
            logger.warning(f"  Gradient-based method failed: {e}")
            results['gradient_based'] = {'counterfactual': None, 'info': {'error': str(e)}}
    
    return results
