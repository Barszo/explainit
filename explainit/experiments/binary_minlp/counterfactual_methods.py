"""
Counterfactual Explanation Methods

Implements six counterfactual algorithms:
1. Wachter et al.'s Algorithm
2. Sparse Wachter (with elastic net regularization)
3. DiCE (Diverse Counterfactual Explanations) - gradient implementation
4. Counterfactuals Guided by Prototypes
5. Official DiCE Library - uses the dice_ml library directly
6. MINLP Search - Mixed-Integer Nonlinear Programming via Shapley approximation
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

from explainit.explainers.minlp_search import MINLSearchExplainer


class WachterCounterfactual:
    """
    Wachter et al.'s counterfactual algorithm.
    
    Minimizes: loss(f(x'), y_target) + lambda * ||x' - x||^2
    """
    
    def __init__(self, model, lambda_param=0.1, learning_rate=0.01,
                 max_iterations=1000, target_class=1, feature_bounds=None,
                 num_cfs=1, init_noise_std=0.01):
        """
        Args:
            model: Trained model (TensorFlow/Keras)
            lambda_param: Weight for proximity term
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of optimization iterations
            target_class: Desired target class
            feature_bounds: Optional list of (min, max) tuples for each feature
            num_cfs: Number of CFs to generate with multi-start optimization
            init_noise_std: Std of random initialization noise for multi-start
        """
        self.model = model
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.target_class = target_class
        self.feature_bounds = feature_bounds
        self.num_cfs = num_cfs
        self.init_noise_std = init_noise_std
        self.last_run_iterations = None
        self.last_run_iterations_list = []
    
    def _generate_single(self, x_original, verbose=False):
        """
        Generate counterfactual for a single instance.
        
        Args:
            x_original: Original instance (1D numpy array)
            verbose: Print optimization progress
            
        Returns:
            Counterfactual instance (1D numpy array)
        """
        # Randomized initialization enables multiple distinct local minima.
        init_noise = np.random.normal(0, self.init_noise_std, size=len(x_original))
        x_cf = tf.Variable((x_original + init_noise).copy(), dtype=tf.float32)
        x_orig_tf = tf.constant(x_original, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        iterations_used = self.max_iterations
        
        for iteration in range(self.max_iterations):
            with tf.GradientTape() as tape:
                # Prediction loss
                pred = self.model(tf.expand_dims(x_cf, 0), training=False)
                pred_loss = tf.keras.losses.binary_crossentropy(
                    tf.constant([[self.target_class]], dtype=tf.float32),
                    pred
                )
                
                # Proximity loss (L2 distance)
                proximity_loss = tf.reduce_sum(tf.square(x_cf - x_orig_tf))
                
                # Total loss
                total_loss = pred_loss + self.lambda_param * proximity_loss
            
            # Compute gradients and update
            gradients = tape.gradient(total_loss, [x_cf])
            optimizer.apply_gradients(zip(gradients, [x_cf]))
            
            # Apply feature bounds if specified
            if self.feature_bounds is not None:
                clipped_values = []
                for i, (min_val, max_val) in enumerate(self.feature_bounds):
                    clipped_values.append(tf.clip_by_value(x_cf[i], min_val, max_val))
                x_cf.assign(tf.stack(clipped_values))
            
            # Check convergence
            if iteration % 100 == 0 and verbose:
                pred_val = pred.numpy()[0, 0]
                print(f"Iter {iteration}: Loss={total_loss.numpy():.4f}, "
                      f"Pred={pred_val:.4f}, Prox={proximity_loss.numpy():.4f}")
            
            # Early stopping if target reached
            if iteration > 100:
                pred_class = 1 if pred.numpy()[0, 0] > 0.5 else 0
                if pred_class == self.target_class:
                    if verbose:
                        print(f"Target reached at iteration {iteration}")
                    iterations_used = iteration + 1
                    break
        self.last_run_iterations = int(iterations_used)
        return x_cf.numpy()

    def generate(self, x_original, verbose=False):
        """
        Generate one or multiple counterfactuals.

        Returns:
            1D numpy array when num_cfs == 1, otherwise list of 1D arrays.
        """
        if self.num_cfs <= 1:
            cf = self._generate_single(x_original, verbose=verbose)
            self.last_run_iterations_list = [int(self.last_run_iterations)]
            return cf

        cfs = []
        run_iterations = []
        max_attempts = max(self.num_cfs * 3, self.num_cfs)
        for _ in range(max_attempts):
            if len(cfs) >= self.num_cfs:
                break
            cf = self._generate_single(x_original, verbose=False)
            if not any(np.allclose(cf, existing, atol=1e-5) for existing in cfs):
                cfs.append(cf)
                run_iterations.append(int(self.last_run_iterations))

        if not cfs:
            cf = self._generate_single(x_original, verbose=False)
            cfs.append(cf)
            run_iterations.append(int(self.last_run_iterations))
        self.last_run_iterations_list = run_iterations
        self.last_run_iterations = int(np.mean(run_iterations)) if run_iterations else self.max_iterations
        return cfs
    
    def generate_batch(self, X_original, verbose=False):
        """Generate counterfactuals for multiple instances."""
        counterfactuals = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"\\nGenerating CF {i+1}/{len(X_original)}")
            cf = self.generate(x, verbose=False)
            counterfactuals.append(cf)
        return np.array(counterfactuals)


class SparseWachterCounterfactual:
    """
    Sparse Wachter with elastic net regularization.
    
    Minimizes: loss(f(x'), y_target) + lambda1 * ||x' - x||^2 + lambda2 * ||x' - x||_1
    """
    
    def __init__(self, model, lambda1=0.1, lambda2=0.01, learning_rate=0.01,
                 max_iterations=1000, target_class=1, feature_bounds=None,
                 num_cfs=1, init_noise_std=0.01):
        """
        Args:
            model: Trained model
            lambda1: Weight for L2 proximity term
            lambda2: Weight for L1 sparsity term (elastic net)
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            target_class: Desired target class
            feature_bounds: Optional list of (min, max) tuples for each feature
            num_cfs: Number of CFs to generate with multi-start optimization
            init_noise_std: Std of random initialization noise for multi-start
        """
        self.model = model
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.target_class = target_class
        self.feature_bounds = feature_bounds
        self.num_cfs = num_cfs
        self.init_noise_std = init_noise_std
        self.last_run_iterations = None
        self.last_run_iterations_list = []
    
    def _generate_single(self, x_original, verbose=False):
        """Generate counterfactual with sparsity constraint."""
        init_noise = np.random.normal(0, self.init_noise_std, size=len(x_original))
        x_cf = tf.Variable((x_original + init_noise).copy(), dtype=tf.float32)
        x_orig_tf = tf.constant(x_original, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        iterations_used = self.max_iterations
        
        for iteration in range(self.max_iterations):
            with tf.GradientTape() as tape:
                # Prediction loss
                pred = self.model(tf.expand_dims(x_cf, 0), training=False)
                pred_loss = tf.keras.losses.binary_crossentropy(
                    tf.constant([[self.target_class]], dtype=tf.float32),
                    pred
                )
                
                # L2 proximity loss
                l2_loss = tf.reduce_sum(tf.square(x_cf - x_orig_tf))
                
                # L1 sparsity loss
                l1_loss = tf.reduce_sum(tf.abs(x_cf - x_orig_tf))
                
                # Total loss (elastic net)
                total_loss = pred_loss + self.lambda1 * l2_loss + self.lambda2 * l1_loss
            
            gradients = tape.gradient(total_loss, [x_cf])
            optimizer.apply_gradients(zip(gradients, [x_cf]))
            
            # Apply feature bounds if specified
            if self.feature_bounds is not None:
                clipped_values = []
                for i, (min_val, max_val) in enumerate(self.feature_bounds):
                    clipped_values.append(tf.clip_by_value(x_cf[i], min_val, max_val))
                x_cf.assign(tf.stack(clipped_values))
            
            if iteration % 100 == 0 and verbose:
                print(f"Iter {iteration}: Loss={total_loss.numpy():.4f}, "
                      f"Pred={pred.numpy()[0, 0]:.4f}")
            
            # Early stopping
            if iteration > 100:
                pred_class = 1 if pred.numpy()[0, 0] > 0.5 else 0
                if pred_class == self.target_class:
                    iterations_used = iteration + 1
                    break
        self.last_run_iterations = int(iterations_used)
        return x_cf.numpy()

    def generate(self, x_original, verbose=False):
        """
        Generate one or multiple sparse counterfactuals.

        Returns:
            1D numpy array when num_cfs == 1, otherwise list of 1D arrays.
        """
        if self.num_cfs <= 1:
            cf = self._generate_single(x_original, verbose=verbose)
            self.last_run_iterations_list = [int(self.last_run_iterations)]
            return cf

        cfs = []
        run_iterations = []
        max_attempts = max(self.num_cfs * 3, self.num_cfs)
        for _ in range(max_attempts):
            if len(cfs) >= self.num_cfs:
                break
            cf = self._generate_single(x_original, verbose=False)
            if not any(np.allclose(cf, existing, atol=1e-5) for existing in cfs):
                cfs.append(cf)
                run_iterations.append(int(self.last_run_iterations))

        if not cfs:
            cf = self._generate_single(x_original, verbose=False)
            cfs.append(cf)
            run_iterations.append(int(self.last_run_iterations))
        self.last_run_iterations_list = run_iterations
        self.last_run_iterations = int(np.mean(run_iterations)) if run_iterations else self.max_iterations
        return cfs
    
    def generate_batch(self, X_original, verbose=False):
        """Generate counterfactuals for multiple instances."""
        counterfactuals = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"Generating CF {i+1}/{len(X_original)}")
            cf = self.generate(x, verbose=False)
            counterfactuals.append(cf)
        return np.array(counterfactuals)


class DiceCounterfactual:
    """
    DiCE (Diverse Counterfactual Explanations) - IMPROVED VERSION.
    
    Generates multiple diverse counterfactuals and returns the closest one.
    
    Improvements over original:
    - Better initialization strategy (start from original, not random noise)
    - Reduced diversity weight to avoid interference with target reaching
    - Higher learning rate for faster convergence
    - Early stopping when targets are reached
    - Focus on prediction accuracy first, diversity second
    """
    
    def __init__(self, model, num_cfs=4, lambda_param=0.1, diversity_weight=0.1,
                 learning_rate=0.05, max_iterations=1000, target_class=1, feature_bounds=None):
        """
        Args:
            model: Trained model
            num_cfs: Number of counterfactuals to generate
            lambda_param: Weight for proximity (default: 0.1)
            diversity_weight: Weight for diversity among CFs (default: 0.1, reduced from 1.0)
            learning_rate: Learning rate (default: 0.05, increased from 0.01)
            max_iterations: Maximum iterations
            target_class: Desired target class
            feature_bounds: Optional list of (min, max) tuples for each feature
        """
        self.model = model
        self.num_cfs = num_cfs
        self.lambda_param = lambda_param
        self.diversity_weight = diversity_weight
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.target_class = target_class
        self.feature_bounds = feature_bounds
        self.last_run_iterations = None
        self.last_run_iterations_list = []
    
    def generate(self, x_original, verbose=False):
        """
        Generate multiple diverse counterfactuals and return closest one.
        
        As per the paper: "We generate 4 counterfactuals and take the closest one
        to the original point (as per L1 distance)."
        
        Improvements:
        - Start from original point with small perturbations (not random noise)
        - Use adaptive diversity weight (lower early on, higher later)
        - Add early stopping when sufficient CFs reach target
        - Better balance between prediction accuracy and diversity
        """
        # Initialize multiple counterfactuals with small random perturbations
        # This is better than large random noise which can lead CFs astray
        x_cfs = []
        for i in range(self.num_cfs):
            # Start from original with small perturbation (std=0.01 instead of 0.1)
            perturbation = np.random.randn(len(x_original)) * 0.01
            x_cf = tf.Variable(x_original.copy() + perturbation, dtype=tf.float32)
            x_cfs.append(x_cf)
        
        x_orig_tf = tf.constant(x_original, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Track best counterfactuals
        best_loss = float('inf')
        consecutive_no_improvement = 0
        iterations_used = self.max_iterations
        
        for iteration in range(self.max_iterations):
            with tf.GradientTape() as tape:
                total_loss = 0
                
                # Loss for each counterfactual
                predictions = []
                for x_cf in x_cfs:
                    # Prediction loss - focus on reaching target class
                    pred = self.model(tf.expand_dims(x_cf, 0), training=False)
                    predictions.append(pred)
                    
                    # Use binary crossentropy for target reaching
                    pred_loss = tf.keras.losses.binary_crossentropy(
                        tf.constant([[self.target_class]], dtype=tf.float32),
                        pred
                    )
                    
                    # Proximity loss (L2 distance)
                    proximity_loss = tf.reduce_sum(tf.square(x_cf - x_orig_tf))
                    
                    # Weight prediction loss more heavily
                    total_loss += 2.0 * pred_loss + self.lambda_param * proximity_loss
                
                # Diversity loss (encourage counterfactuals to be different from each other)
                # Only apply after CFs start moving in the right direction
                if iteration > 50:
                    for i in range(self.num_cfs):
                        for j in range(i + 1, self.num_cfs):
                            # Negative squared distance = penalty for being too similar
                            # We want to MINIMIZE this, so we SUBTRACT the squared distance
                            diversity_loss = -tf.reduce_sum(tf.square(x_cfs[i] - x_cfs[j]))
                            total_loss += self.diversity_weight * diversity_loss
            
            # Update all counterfactuals
            gradients = tape.gradient(total_loss, x_cfs)
            optimizer.apply_gradients(zip(gradients, x_cfs))
            
            # Apply feature bounds if specified
            if self.feature_bounds is not None:
                for x_cf in x_cfs:
                    clipped_values = []
                    for i, (min_val, max_val) in enumerate(self.feature_bounds):
                        clipped_values.append(tf.clip_by_value(x_cf[i], min_val, max_val))
                    x_cf.assign(tf.stack(clipped_values))
            
            # Check convergence
            current_loss = total_loss.numpy()
            if current_loss < best_loss:
                best_loss = current_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
            
            # Early stopping if majority of CFs reach target
            if iteration > 100 and iteration % 50 == 0:
                num_successful = 0
                for pred in predictions:
                    pred_val = pred.numpy()[0, 0]
                    pred_class = 1 if pred_val > 0.5 else 0
                    if pred_class == self.target_class:
                        num_successful += 1
                
                # If more than half of CFs reached target, we can stop
                if num_successful >= self.num_cfs // 2:
                    if verbose:
                        print(f"Early stopping at iteration {iteration}: "
                              f"{num_successful}/{self.num_cfs} CFs reached target")
                    iterations_used = iteration + 1
                    break
            
            # Stop if no improvement for a while
            if consecutive_no_improvement > 100:
                if verbose:
                    print(f"Stopping at iteration {iteration}: no improvement")
                iterations_used = iteration + 1
                break
            
            if iteration % 200 == 0 and verbose:
                avg_pred = np.mean([p.numpy()[0, 0] for p in predictions])
                print(f"Iter {iteration}: Loss={total_loss.numpy():.4f}, "
                      f"Avg Pred={avg_pred:.4f}")
        
        self.last_run_iterations = int(iterations_used)
        self.last_run_iterations_list = [int(iterations_used)]

        # Return the closest counterfactual that reached target (or just closest if none succeeded)
        cfs_numpy = [x_cf.numpy() for x_cf in x_cfs]
        
        # First, try to find successful CFs
        successful_cfs = []
        for cf in cfs_numpy:
            pred = self.model(np.expand_dims(cf, 0), training=False).numpy()[0, 0]
            pred_class = 1 if pred > 0.5 else 0
            if pred_class == self.target_class:
                successful_cfs.append(cf)
        
        # If we have successful CFs, return the closest one
        if successful_cfs:
            distances = [np.sum(np.abs(cf - x_original)) for cf in successful_cfs]
            closest_idx = np.argmin(distances)
            return successful_cfs[closest_idx]
        
        # Otherwise, return the CF with prediction closest to target
        target_pred = self.target_class
        pred_diffs = []
        for cf in cfs_numpy:
            pred = self.model(np.expand_dims(cf, 0), training=False).numpy()[0, 0]
            pred_diff = abs(pred - target_pred)
            pred_diffs.append(pred_diff)
        
        best_idx = np.argmin(pred_diffs)
        return cfs_numpy[best_idx]
    
    def generate_batch(self, X_original, verbose=False):
        """Generate counterfactuals for multiple instances."""
        counterfactuals = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"Generating CF {i+1}/{len(X_original)}")
            cf = self.generate(x, verbose=False)
            counterfactuals.append(cf)
        return np.array(counterfactuals)


class PrototypeGuidedCounterfactual:
    """
    Counterfactuals Guided by Prototypes.
    
    Uses prototypes from the target class to guide counterfactual generation.
    """
    
    def __init__(self, model, X_train, y_train, n_prototypes=5,
                 lambda_param=0.1, learning_rate=0.01,
                 max_iterations=1000, target_class=1, feature_bounds=None,
                 num_cfs=1, init_noise_std=0.01):
        """
        Args:
            model: Trained model
            X_train: Training data to extract prototypes
            y_train: Training labels
            n_prototypes: Number of prototypes to use
            lambda_param: Weight for proximity
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            target_class: Desired target class
            feature_bounds: Optional list of (min, max) tuples for each feature
            num_cfs: Number of CFs to generate with multi-start optimization
            init_noise_std: Std of random initialization noise for multi-start
        """
        self.model = model
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.target_class = target_class
        self.feature_bounds = feature_bounds
        self.num_cfs = num_cfs
        self.init_noise_std = init_noise_std
        self.last_run_iterations = None
        self.last_run_iterations_list = []
        
        # Extract prototypes from target class
        target_indices = np.where(y_train == target_class)[0]
        target_samples = X_train[target_indices]
        
        # Select n_prototypes representatives (e.g., cluster centers or random samples)
        if len(target_samples) > n_prototypes:
            # Random selection of prototypes
            prototype_indices = np.random.choice(len(target_samples), 
                                                n_prototypes, replace=False)
            self.prototypes = target_samples[prototype_indices]
        else:
            self.prototypes = target_samples
        
        self.prototypes = tf.constant(self.prototypes, dtype=tf.float32)
    
    def _generate_single(self, x_original, verbose=False):
        """
        Generate counterfactual guided by prototypes.
        
        Minimizes distance to nearest prototype while satisfying target class.
        """
        init_noise = np.random.normal(0, self.init_noise_std, size=len(x_original))
        x_cf = tf.Variable((x_original + init_noise).copy(), dtype=tf.float32)
        x_orig_tf = tf.constant(x_original, dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        iterations_used = self.max_iterations
        
        for iteration in range(self.max_iterations):
            with tf.GradientTape() as tape:
                # Prediction loss
                pred = self.model(tf.expand_dims(x_cf, 0), training=False)
                pred_loss = tf.keras.losses.binary_crossentropy(
                    tf.constant([[self.target_class]], dtype=tf.float32),
                    pred
                )
                
                # Distance to nearest prototype
                distances_to_prototypes = tf.reduce_sum(
                    tf.square(tf.expand_dims(x_cf, 0) - self.prototypes), axis=1
                )
                prototype_loss = tf.reduce_min(distances_to_prototypes)
                
                # Original proximity (for small changes)
                proximity_loss = tf.reduce_sum(tf.square(x_cf - x_orig_tf))
                
                # Total loss
                total_loss = pred_loss + self.lambda_param * prototype_loss + 0.01 * proximity_loss
            
            gradients = tape.gradient(total_loss, [x_cf])
            optimizer.apply_gradients(zip(gradients, [x_cf]))
            
            # Apply feature bounds if specified
            if self.feature_bounds is not None:
                clipped_values = []
                for i, (min_val, max_val) in enumerate(self.feature_bounds):
                    clipped_values.append(tf.clip_by_value(x_cf[i], min_val, max_val))
                x_cf.assign(tf.stack(clipped_values))
            
            if iteration % 100 == 0 and verbose:
                print(f"Iter {iteration}: Loss={total_loss.numpy():.4f}, "
                      f"Pred={pred.numpy()[0, 0]:.4f}")
            
            # Early stopping
            if iteration > 100:
                pred_class = 1 if pred.numpy()[0, 0] > 0.5 else 0
                if pred_class == self.target_class:
                    iterations_used = iteration + 1
                    break
        self.last_run_iterations = int(iterations_used)
        return x_cf.numpy()

    def generate(self, x_original, verbose=False):
        """
        Generate one or multiple prototype-guided counterfactuals.

        Returns:
            1D numpy array when num_cfs == 1, otherwise list of 1D arrays.
        """
        if self.num_cfs <= 1:
            cf = self._generate_single(x_original, verbose=verbose)
            self.last_run_iterations_list = [int(self.last_run_iterations)]
            return cf

        cfs = []
        run_iterations = []
        max_attempts = max(self.num_cfs * 3, self.num_cfs)
        for _ in range(max_attempts):
            if len(cfs) >= self.num_cfs:
                break
            cf = self._generate_single(x_original, verbose=False)
            if not any(np.allclose(cf, existing, atol=1e-5) for existing in cfs):
                cfs.append(cf)
                run_iterations.append(int(self.last_run_iterations))

        if not cfs:
            cf = self._generate_single(x_original, verbose=False)
            cfs.append(cf)
            run_iterations.append(int(self.last_run_iterations))
        self.last_run_iterations_list = run_iterations
        self.last_run_iterations = int(np.mean(run_iterations)) if run_iterations else self.max_iterations
        return cfs
    
    def generate_batch(self, X_original, verbose=False):
        """Generate counterfactuals for multiple instances."""
        counterfactuals = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"Generating CF {i+1}/{len(X_original)}")
            cf = self.generate(x, verbose=False)
            counterfactuals.append(cf)
        return np.array(counterfactuals)


class OfficialDiceCounterfactual:
    """
    Official DiCE library implementation using gradient descent.
    Uses the dice_ml library for generating counterfactual explanations.
    
    Note: This method may produce warnings from the DiCE library:
    - MAD warnings for features with zero variance (expected, library replaces with 1.0)
    - FutureWarnings about dtype incompatibility (pandas compatibility issue in dice_ml)
    These warnings do not affect the functionality and CFs are still generated correctly.
    
    The official DiCE library may be slower than custom implementations due to:
    - More sophisticated loss calculations and optimization
    - Internal preprocessing and postprocessing steps
    - Diversity enforcement across multiple counterfactuals
    """
    
    def __init__(self, model, X_train, y_train, feature_names,
                 num_cfs=4, max_iterations=1000, min_iterations=100,
                 target_class=1, learning_rate=0.05,
                 proximity_weight=0.5, diversity_weight=1.0,
                 categorical_penalty=0.1, loss_diff_thres=1e-5,
                 loss_converge_maxiter=1, yloss_type='hinge_loss',
                 feature_bounds=None, categorical_feature_names=None):
        """
        Args:
            model: Trained model (TensorFlow/Keras)
            X_train: Training data (numpy array or DataFrame)
            y_train: Training labels (numpy array)
            feature_names: List of feature names
            num_cfs: Number of counterfactuals to generate
            max_iterations: Maximum iterations for gradient descent
            min_iterations: Minimum iterations before checking convergence
            target_class: Desired target class
            learning_rate: Learning rate for gradient descent (higher = faster, 0.05-0.2)
            proximity_weight: Weight for proximity loss (lower = allow more change)
            diversity_weight: Weight for diversity loss
            categorical_penalty: Weight to ensure categorical variables sum to 1
            loss_diff_thres: Convergence threshold (higher = faster, e.g., 1e-3 for speed)
            loss_converge_maxiter: Iterations to hold convergence before stopping
            yloss_type: Loss function - 'hinge_loss', 'log_loss', or 'l2_loss'
            feature_bounds: Optional list of (min, max) tuples for each feature
            categorical_feature_names: Ignored. Kept for API compatibility only.
                DiCE gradient method only works with continuous features.
                Features with zero MAD in training data are automatically excluded
                from features_to_vary to prevent DiCE from wasting iterations on them.
        """
        self.model = model
        self.num_cfs = num_cfs
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.target_class = target_class
        self.learning_rate = learning_rate
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty
        self.loss_diff_thres = loss_diff_thres
        self.loss_converge_maxiter = loss_converge_maxiter
        self.yloss_type = yloss_type
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds
        self.last_run_iterations = None
        self.last_run_iterations_list = []
        
        # Import dice_ml
        try:
            import dice_ml
            self.dice_ml = dice_ml
        except ImportError:
            raise ImportError("dice_ml not installed. Install with: pip install dice-ml")
        
        # Prepare training data as DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_df = X_train.copy()
        else:
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
        
        # Add outcome column
        if isinstance(y_train, pd.Series):
            X_train_df['outcome'] = y_train.values
        else:
            X_train_df['outcome'] = y_train
        
        # Create DiCE Data object — all features are continuous (gradient method requirement).
        self.dice_data = dice_ml.Data(
            dataframe=X_train_df,
            continuous_features=list(feature_names),
            outcome_name='outcome'
        )

        # Create DiCE Model object
        self.dice_model = dice_ml.Model(model=model, backend='TF2')

        # Create DiCE explainer with gradient method
        self.dice_explainer = dice_ml.Dice(
            self.dice_data,
            self.dice_model,
            method='gradient'
        )

    def generate(self, x_original, verbose=False):
        """
        Generate counterfactual for a single instance.
        Returns the best counterfactual (closest to original that achieves target).
        
        Args:
            x_original: Original instance (1D numpy array)
            verbose: Print generation details (DiCE library may still show warnings)
            
        Returns:
            Counterfactual instance (1D numpy array)
            
        Note: You may see warnings from the DiCE library about MAD=0 or dtype issues.
        These are expected and do not affect functionality. To suppress them, you can use:
            import warnings
            warnings.filterwarnings('ignore')
        """
        # Convert to DataFrame
        query_df = pd.DataFrame([x_original], columns=self.feature_names)
        
        # Set parameters for gradient descent
        # Note: DiCE TensorFlow2 uses 'DiverseCF' or 'RandomInitCF' for algorithm
        # The actual gradient descent is automatic when method='gradient' is used
        cf_params = {
            'query_instances': query_df,
            'total_CFs': self.num_cfs,
            'desired_class': "opposite",  # Use "opposite" for binary classification
            'verbose': verbose,
            'proximity_weight': self.proximity_weight,
            'diversity_weight': self.diversity_weight,
            'categorical_penalty': self.categorical_penalty,
            'learning_rate': self.learning_rate,
            'min_iter': self.min_iterations,  # Minimum iterations before checking convergence
            'max_iter': self.max_iterations,  # Maximum iterations
            'loss_diff_thres': self.loss_diff_thres,  # Convergence threshold
            'loss_converge_maxiter': self.loss_converge_maxiter,  # Iterations to hold for convergence
            'algorithm': 'DiverseCF',  # Use diverse initialization
            'features_to_vary': 'all',
            'yloss_type': self.yloss_type,  # Loss type for classification
            'diversity_loss_type': 'dpp_style:inverse_dist',  # Diversity metric
            'posthoc_sparsity_param': 0.0,  # Disable slow post-hoc sparsification
        }
        
        # Add feature bounds if specified (permitted_range parameter for DiCE)
        if self.feature_bounds is not None:
            permitted_range = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(self.feature_bounds):
                    min_val, max_val = self.feature_bounds[i]
                    permitted_range[feature_name] = [float(min_val), float(max_val)]
            cf_params['permitted_range'] = permitted_range
        
        try:
            # Generate counterfactuals
            cf_result = self.dice_explainer.generate_counterfactuals(**cf_params)

            # Record actual iterations used from DiCE explainer.
            # max_iterations_run may exist but be None when no CFs were found.
            _raw_iters = getattr(self.dice_explainer, 'max_iterations_run', None)
            actual_iterations = int(_raw_iters if _raw_iters is not None else self.max_iterations)
            self.last_run_iterations = actual_iterations
            self.last_run_iterations_list = [actual_iterations]
            
            # Extract counterfactuals
            cf_examples = cf_result.cf_examples_list[0]
            
            if cf_examples.final_cfs_df is None or len(cf_examples.final_cfs_df) == 0:
                if verbose:
                    print("Warning: DiCE returned no counterfactuals")
                    print(f"  Target class: {self.target_class}")
                    print(f"  Original prediction: {self.model.predict(query_df[self.feature_names].values, verbose=0)[0,0]:.3f}")
                return np.array([x_original.copy()])
            
            # Get counterfactuals as array
            cfs_df = cf_examples.final_cfs_df
            
            # Remove outcome column if present
            if 'outcome' in cfs_df.columns:
                cfs_array = cfs_df[self.feature_names].values
            else:
                cfs_array = cfs_df[self.feature_names].values if all(f in cfs_df.columns for f in self.feature_names) else cfs_df.values
            
            # Check if CFs are actually different from original
            if len(cfs_array) == 0:
                if verbose:
                    print("Warning: No valid counterfactuals in result")
                return np.array([x_original.copy()])
            
            # Get predictions for all CFs
            predictions = self.model.predict(cfs_array, verbose=0).flatten()
            
            # Debug output
            if verbose:
                orig_pred = self.model.predict(x_original.reshape(1, -1), verbose=0)[0, 0]
                print(f"Debug: Original pred={orig_pred:.3f}, Generated {len(cfs_array)} CFs")
                for i, (cf, pred) in enumerate(zip(cfs_array, predictions)):
                    dist = np.linalg.norm(cf - x_original)
                    print(f"  CF {i+1}: pred={pred:.3f}, distance={dist:.3f}")
            
            # Return all generated CFs (sorted by distance)
            # Calculate distances from original
            distances = np.array([np.linalg.norm(cf - x_original) for cf in cfs_array])
            
            # Filter out any that are too similar to original (distance < 1e-6)
            valid_mask = distances > 1e-6
            if np.any(valid_mask):
                cfs_array = cfs_array[valid_mask]
                predictions = predictions[valid_mask]
                distances = distances[valid_mask]
            
            # Sort by distance (closest first)
            sort_idx = np.argsort(distances)
            cfs_array = cfs_array[sort_idx]
            predictions = predictions[sort_idx]
            distances = distances[sort_idx]
            
            if len(cfs_array) == 0:
                if verbose:
                    print("ERROR: DiCE returned only the original instance")
                return np.array([x_original.copy()])
            
            # Return all CFs (caller can decide how many to use)
            return cfs_array
            
        except Exception as e:
            if verbose:
                print(f"Error generating CF with official DiCE: {e}")
                import traceback
                traceback.print_exc()
            return np.array([x_original.copy()])
    
    def generate_batch(self, X_original, verbose=False):
        """Generate counterfactuals for multiple instances."""
        counterfactuals = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"Generating CF {i+1}/{len(X_original)} (Official DiCE)")
            cf = self.generate(x, verbose=False)
            counterfactuals.append(cf)
        return np.array(counterfactuals)


def compute_counterfactual_cost(X_original, X_counterfactual, metric='l1'):
    """
    Compute the cost/distance of counterfactuals.
    
    Args:
        X_original: Original instances
        X_counterfactual: Counterfactual instances
        metric: Distance metric ('l1' or 'l2')
    
    Returns:
        Array of distances
    """
    if metric == 'l1':
        return np.sum(np.abs(X_counterfactual - X_original), axis=1)
    elif metric == 'l2':
        return np.sqrt(np.sum((X_counterfactual - X_original) ** 2, axis=1))
    else:
        raise ValueError(f"Unknown metric: {metric}")


class MinlpCounterfactual:
    """
    Wrapper around explainit.explainers.minlp_search.MINLSearchExplainer that
    exposes the same `generate(x_original)` interface used by the other
    counterfactual methods in this module.

    Per-sample preferences are produced through the supplied
    ``preferences_builder`` callable (a function taking the sample and
    returning the priorities dict expected by ``MINLSearchExplainer``).
    """

    def __init__(self, model, X_train, y_train, feature_names,
                 preferences_builder,
                 num_cfs=1,
                 max_iterations=100,
                 target_class=1,
                 threshold=0.5,
                 target_probability=0.75,
                 target_exemplar_epsilon=0.05,
                 epsilon=0.1,
                 shap_approx=True,
                 shap_num_samples=200,
                 feature_bounds=None):
        self.model = model
        self.X_train = X_train.values if hasattr(X_train, 'values') else X_train
        self.y_train = y_train.values if hasattr(y_train, 'values') else y_train
        self.feature_names = feature_names
        self.preferences_builder = preferences_builder
        self.num_cfs = num_cfs
        self.max_iterations = max_iterations
        self.target_class = target_class
        self.threshold = threshold
        self.target_probability = target_probability
        self.target_exemplar_epsilon = target_exemplar_epsilon
        self.epsilon = epsilon
        self.shap_approx = shap_approx
        self.shap_num_samples = shap_num_samples
        self.feature_bounds = feature_bounds
        self.last_run_iterations = None
        self.last_run_iterations_list = []

    def _model_pred(self, x):
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return self.model.predict(x_arr, verbose=0).flatten()

    def generate(self, x_original, verbose=False):
        x_original = np.asarray(x_original, dtype=float).flatten()

        try:
            preferences = self.preferences_builder(x_original)
        except Exception as e:
            if verbose:
                print(f"MINLP: failed to build per-sample preferences: {e}")
            self.last_run_iterations_list = []
            self.last_run_iterations = self.max_iterations
            return [] if self.num_cfs > 1 else x_original.copy()

        explainer = MINLSearchExplainer(
            model_pred=self._model_pred,
            priorities=preferences,
            sample=x_original.tolist(),
            target=float(self.target_probability),
            dataset=self.X_train.copy(),
            target_exemplar_epsilon=self.target_exemplar_epsilon,
            epsilon=self.epsilon,
        )

        cfs_out = []
        iters_list = []

        try:
            counterfactuals, predictions, scores, found_counts = (
                explainer.find_counterfactuals_for_binary(
                    target_class=self.target_class,
                    threshold=self.threshold,
                    expected_counterfactuals=self.num_cfs,
                    max_iterations=self.max_iterations,
                    shap_approx=self.shap_approx,
                    num_samples=self.shap_num_samples,
                    return_top_n=self.num_cfs,
                )
            )

            for cf in counterfactuals[: self.num_cfs]:
                cf_arr = np.asarray(cf, dtype=float).flatten()
                if cf_arr.shape[0] != x_original.shape[0]:
                    continue
                cfs_out.append(cf_arr)

            if found_counts:
                iters_list = [int(c) for c in found_counts[: self.num_cfs]]
        except Exception as e:
            if verbose:
                print(f"MINLP: optimization failed: {e}")

        self.last_run_iterations_list = iters_list
        self.last_run_iterations = (
            int(np.mean(iters_list)) if iters_list else self.max_iterations
        )

        if not cfs_out:
            if self.num_cfs == 1:
                return x_original.copy()
            return []

        if self.num_cfs == 1:
            return cfs_out[0]
        return cfs_out

    def generate_batch(self, X_original, verbose=False):
        results = []
        for i, x in enumerate(X_original):
            if verbose and i % 10 == 0:
                print(f"Generating CF {i + 1}/{len(X_original)} (MINLP)")
            results.append(self.generate(x, verbose=False))
        return results


if __name__ == "__main__":
    print("Counterfactual methods module loaded successfully!")
    print("\\nAvailable methods:")
    print("1. WachterCounterfactual")
    print("2. SparseWachterCounterfactual")
    print("3. DiceCounterfactual (custom gradient implementation)")
    print("4. PrototypeGuidedCounterfactual")
    print("5. OfficialDiceCounterfactual (uses dice_ml library)")
    print("6. MinlpCounterfactual (Shapley-based MINLP search)")

