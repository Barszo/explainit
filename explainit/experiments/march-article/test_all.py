"""
Combined Counterfactual Methods Test

This script tests both standard counterfactual methods and preference-based methods
on the same datasets and samples for fair comparison.

Workflow:
1. Load configuration from config.yaml
2. Load/prepare datasets and train models
3. Select test samples from source class
4. Run standard CF methods (Wachter, DiCE, etc.)
5. Run preference-based methods on the SAME samples
6. Save results with unified measures for comparison

All methods save results with consistent metrics:
- prediction: Model prediction probability
- predicted_class: Binary class (0 or 1)
- target_achieved: Whether target class was reached
- l2_distance: L2 distance from original
- l1_distance: L1 distance (for preference methods)
- sparsity: Number of features changed (for preference methods)
"""

import os
import sys
import csv
import yaml
import pickle
import logging
import time
import concurrent.futures
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Add current directory to path for local modules
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import local modules
from data_downloader import load_communities_and_crime, load_german_credit, load_lending_club_selected_features, load_credit_card_default
from model_builder import create_baseline_model
from counterfactual_methods import (
    WachterCounterfactual,
    SparseWachterCounterfactual,
    DiceCounterfactual,
    PrototypeGuidedCounterfactual,
    OfficialDiceCounterfactual
)
from explainit.priorities.nonlinear import exponential
from explainit.explainers.random_search import RandomSearchExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Configuration loaded from: {config_path}")
    return config


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "reports"
RESULTS_DIR = SCRIPT_DIR / "results"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset name mapping
DATASET_NAMES = {
    'lending_club': 'Lending Club',
    'communities_crime': 'Communities and Crime',
    'german_credit': 'German Credit',
    'credit_card_default': 'Credit Card Default',
}

# CF Method name mapping
CF_METHOD_NAMES = {
    'wachter': 'Wachter',
    'sparse_wachter': 'Sparse Wachter',
    'dice': 'DiCE (gradient mode)',
    'prototype': 'Prototype-Guided',
    'dice_official': 'DiCE (official library)'
}


def setup_dataset_logging(dataset_key):
    """
    Add a FileHandler to the root logger so all log messages produced in the
    current process for this dataset are also written to log_{dataset_key}.log.
    Returns the path of the log file.
    """
    log_path = SCRIPT_DIR / f"log_{dataset_key}.log"
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    return log_path


class RunStatusTracker:
    """Track and persist run progress to status.txt."""

    def __init__(self, status_file_path, total_steps):
        self.status_file_path = status_file_path
        self.total_steps = max(int(total_steps), 0)
        self.started_steps = 0

    def reset_file(self):
        """Clear status file at the beginning of each execution."""
        self.status_file_path.write_text("", encoding='utf-8')

    def record_method_start(self, dataset_name, method_name):
        """Append method start status with timestamp and remaining steps."""
        self.started_steps += 1
        steps_left = max(self.total_steps - self.started_steps, 0)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = (
            f"[{timestamp}] dataset={dataset_name} | "
            f"method_started={method_name} | steps_left={steps_left}\n"
        )
        with open(self.status_file_path, 'a', encoding='utf-8') as status_file:
            status_file.write(line)


def get_selected_standard_method_display_names(config):
    """Get selected standard method display names from config."""
    selected = config['standard_methods']['selection']
    available_method_keys = list(CF_METHOD_NAMES.keys())

    if selected == 'all':
        selected_method_keys = available_method_keys
    elif isinstance(selected, str):
        selected_method_keys = [selected]
    elif isinstance(selected, list):
        selected_method_keys = selected
    else:
        raise ValueError(
            "config['standard_methods']['selection'] must be 'all', a method name string, or a list of method names"
        )

    invalid_methods = [m for m in selected_method_keys if m not in CF_METHOD_NAMES]
    if invalid_methods:
        raise ValueError(
            f"Unknown standard method(s): {invalid_methods}. Available methods: {available_method_keys}"
        )

    return [CF_METHOD_NAMES[m] for m in selected_method_keys]


def get_selected_preference_method_display_names(config):
    """Get selected preference method display names from config."""
    mode = config['preference_method']['mode']
    # Support both string and list
    if isinstance(mode, str):
        modes = [mode]
    elif isinstance(mode, list):
        modes = mode
    else:
        logger.warning(f"Invalid preference method mode: {mode}, defaulting to ['binary']")
        modes = ['binary']
    display_names = [f'Preference ({m})' for m in modes]
    return display_names


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def save_dataset(X_train, X_test, y_train, y_test, feature_names, scaler, dataset_name):
    """Save dataset to disk for reproducibility."""
    data_path = DATA_DIR / f"{dataset_name}_data.pkl"
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'timestamp': datetime.now().isoformat(),
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'n_features': X_train.shape[1]
    }
    
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"✓ Dataset saved to: {data_path}")
    return data_path


def load_dataset(dataset_name):
    """Load dataset from disk if it exists."""
    data_path = DATA_DIR / f"{dataset_name}_data.pkl"
    
    if not data_path.exists():
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"✓ Dataset loaded from: {data_path}")
    logger.info(f"  Saved on: {data['timestamp']}")
    logger.info(f"  Train shape: {data['train_shape']}, Test shape: {data['test_shape']}")
    
    return data


def train_simple_model(X_train, y_train, X_test, y_test, dataset_name, config):
    """
    Train a simple baseline model.
    Load from disk if available, otherwise train and save.
    
    Returns:
        Tuple of (trained model, training history, model info dict)
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    model_path = MODELS_DIR / f"{clean_name}_model.keras"
    history_path = MODELS_DIR / f"{clean_name}_history.pkl"
    
    # Try to load existing model
    if model_path.exists() and not config['model']['force_retrain']:
        try:
            # Ensure tf is always available
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"\n{'='*60}")
            logger.info(f"Model Loaded from Disk: {dataset_name}")
            logger.info(f"{'='*60}")
            logger.info(f"Training Accuracy:   {train_acc:.4f}")
            logger.info(f"Test Accuracy:       {test_acc:.4f}")
            logger.info(f"✓ Model loaded from: {model_path}")
            model_info = {
                'loaded_from_disk': True,
                'model_path': str(model_path),
                'epochs_trained': len(history['loss']) if history else 'N/A'
            }
            return model, history, model_info
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Training new model...")
    
    # Train new model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Model on {dataset_name}")
    logger.info(f"{'='*60}")
    
    input_dim = X_train.shape[1]
    model = create_baseline_model(input_dim)
    
    # Add Keras callback for progress logging
    import tensorflow as tf
    class ProgressLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            logger.info(f"Epoch {epoch+1}/{config['model']['epochs']} started.")
        def on_epoch_end(self, epoch, logs=None):
            logger.info(f"Epoch {epoch+1} ended. Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}")
        def on_batch_end(self, batch, logs=None):
            if batch % 100 == 0:
                logger.info(f"  Batch {batch} ended. Loss: {logs.get('loss'):.4f}")

    # Speed up training for Lending Club
    epochs = config['model']['epochs']
    # Debug: print dataset_name and batch size selection
    logger.info(f"Dataset name received: {dataset_name}")
    if dataset_name.lower().replace(' ', '_') == 'lending_club':
        batch_size = config['model'].get('lending_club_batch_size', config['model']['batch_size'])
        logger.info(f"Lending Club detected, using batch_size={batch_size}")
    else:
        batch_size = config['model']['batch_size']
        logger.info(f"Non-Lending Club, using batch_size={batch_size}")
    logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}")
    # Add EarlyStopping callback for Lending Club
    import tensorflow as tf
    callbacks = [ProgressLogger()]
    if dataset_name.lower() == 'lending_club':
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            min_delta=0.001
        )
        callbacks.append(early_stop)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"Training Accuracy:   {train_acc:.4f}")
    logger.info(f"Test Accuracy:       {test_acc:.4f}")
    
    # Save model and history
    model.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    logger.info(f"✓ Model saved to: {model_path}")
    logger.info(f"✓ History saved to: {history_path}")
    
    model_info = {
        'loaded_from_disk': False,
        'model_path': str(model_path),
        'epochs_trained': config['model']['epochs']
    }
    
    return model, history.history, model_info


def select_test_samples(X_test, y_test, model, config):
    """
    Select test samples from source class for counterfactual generation.
    
    Returns:
        Tuple of (test_indices, X_test_samples)
    """
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Select from true source class labels, then enforce uniqueness.
    source_class = config['datasets']['source_class']
    requested_n_samples = int(config['samples']['n_samples'])

    # Optional synthetic fallback settings when unique real samples are insufficient.
    synthetic_cfg = config.get('samples', {}).get('synthetic_fallback', {})
    enable_synthetic_fallback = bool(synthetic_cfg.get('enabled', True))
    noise_std_fraction = float(synthetic_cfg.get('noise_std_fraction', 0.05))
    max_noise_as_range_fraction = float(synthetic_cfg.get('max_noise_as_range_fraction', 0.10))
    max_attempts_per_sample = int(synthetic_cfg.get('max_attempts_per_sample', 100))

    source_indices = np.where(y_test_np == source_class)[0]

    if len(source_indices) == 0:
        raise ValueError(f"No test samples found with true class {source_class}")

    selected_indices = []
    selected_samples_set = set()

    for idx in source_indices:
        sample_tuple = tuple(X_test_np[idx])
        if sample_tuple not in selected_samples_set:
            selected_samples_set.add(sample_tuple)
            selected_indices.append(idx)

    unique_available = len(selected_indices)
    logger.info(
        f"Available unique test samples in class {source_class}: "
        f"{unique_available} (requested: {requested_n_samples})"
    )

    selected_indices_np = np.array(selected_indices, dtype=int)

    if unique_available >= requested_n_samples:
        # Use exactly requested_n_samples unique samples, chosen at random without replacement.
        chosen_positions = np.random.choice(
            len(selected_indices_np),
            size=requested_n_samples,
            replace=False
        )
        test_indices = selected_indices_np[chosen_positions]
        X_test_samples = X_test_np[test_indices]

        logger.info(
            f"\n✓ Selected {len(test_indices)} unique test samples from class {source_class}"
        )
        return test_indices, X_test_samples

    if not enable_synthetic_fallback:
        raise ValueError(
            f"Insufficient unique test samples for class {source_class}: "
            f"available {unique_available}, requested {requested_n_samples}. "
            f"Enable samples.synthetic_fallback.enabled to synthesize missing samples."
        )

    # Use all available unique real samples, then synthesize the missing amount.
    n_missing = requested_n_samples - unique_available
    logger.warning(
        f"Insufficient unique test samples for class {source_class}: "
        f"available {unique_available}, requested {requested_n_samples}. "
        f"Generating {n_missing} bounded synthetic sample(s)."
    )

    feature_min = np.min(X_test_np, axis=0)
    feature_max = np.max(X_test_np, axis=0)
    feature_range = np.maximum(feature_max - feature_min, 1e-9)

    source_samples_unique = X_test_np[selected_indices_np]
    source_feature_std = np.std(source_samples_unique, axis=0)
    noise_scale = np.minimum(
        np.maximum(source_feature_std * noise_std_fraction, 1e-6),
        feature_range * max_noise_as_range_fraction
    )

    synthetic_samples = []
    synthetic_ids = []
    selected_samples_set = set(tuple(X_test_np[idx]) for idx in selected_indices_np)
    next_synth_id = int(np.max(selected_indices_np)) + 1

    for _ in range(n_missing):
        accepted = False
        for _attempt in range(max_attempts_per_sample):
            base_pos = np.random.randint(0, len(source_samples_unique))
            base_sample = source_samples_unique[base_pos]
            noise = np.random.normal(loc=0.0, scale=noise_scale)
            candidate = np.clip(base_sample + noise, feature_min, feature_max)

            candidate_tuple = tuple(candidate)
            if candidate_tuple in selected_samples_set:
                continue

            selected_samples_set.add(candidate_tuple)
            synthetic_samples.append(candidate)
            synthetic_ids.append(next_synth_id)
            next_synth_id += 1
            accepted = True
            break

        if not accepted:
            raise ValueError(
                f"Could not synthesize enough bounded unique samples for class {source_class}. "
                f"Needed {n_missing}, generated {len(synthetic_samples)}."
            )

    real_indices = selected_indices_np
    real_samples = X_test_np[real_indices]
    test_indices = np.concatenate([real_indices, np.array(synthetic_ids, dtype=int)])
    X_test_samples = np.vstack([real_samples, np.array(synthetic_samples)])

    logger.info(
        f"\n✓ Selected {len(real_indices)} unique real + {len(synthetic_samples)} synthetic samples "
        f"for class {source_class} (total {len(test_indices)})"
    )

    return test_indices, X_test_samples


def generate_global_priorities_and_bounds(X_train, y_train, model, feature_names, config):
    """
    Generate global priorities and extract feature bounds once for all samples.
    
    This function:
    1. Finds a representative target sample from the training data
    2. Uses a representative source sample to generate priorities
    3. Extracts min/max bounds from the priorities
    4. Returns bounds that can be applied to all CF methods
    
    Args:
        X_train: Training data
        y_train: Training labels
        model: Trained model
        feature_names: List of feature names
        config: Configuration dictionary
        
    Returns:
        Tuple of (feature_bounds, target_sample, priority_params)
        - feature_bounds: List of (min, max) tuples for each feature
        - target_sample: The target sample used for priority generation
        - priority_params: List of parameter dicts for documentation
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATING GLOBAL PRIORITIES AND BOUNDS")
    logger.info(f"{'='*60}")
    
    # Find a representative target sample
    target_sample = find_target_sample(
        X_train, y_train, model,
        target_class=config['datasets']['target_class'],
        target_probability=config['preference_method']['target_probability']
    )
    
    if target_sample is None:
        logger.error("Could not find target sample for priority generation")
        return None, None, None
    
    # Use the mean of the training data from source class as representative sample
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_train_np = y_train_np.flatten()
    
    source_class = config['datasets']['source_class']
    source_mask = y_train_np == source_class
    source_samples = X_train_np[source_mask]
    
    if len(source_samples) == 0:
        logger.error(f"No training samples found for source class {source_class}")
        return None, None, None
    
    # Use mean as representative sample
    representative_sample = np.mean(source_samples, axis=0)
    
    logger.info(f"  Using mean of {len(source_samples)} source class samples as representative")
    logger.info(f"  Target sample prediction: {model.predict(target_sample.reshape(1, -1), verbose=0)[0, 0]:.4f}")
    logger.info(f"  Representative sample prediction: {model.predict(representative_sample.reshape(1, -1), verbose=0)[0, 0]:.4f}")
    
    # Generate priorities based on representative sample and target
    preferences, priority_params = define_preferences(
        sample=representative_sample,
        target_sample=target_sample,
        X_train=X_train,
        feature_names=feature_names,
        exemplar_weight=config['preference_method']['exemplar_weight'],
        steepness=config['preference_method'].get('preference_steepness', 5)
    )
    
    # Extract feature bounds
    feature_bounds = []
    for idx in range(len(representative_sample)):
        # Get dataset limits for this feature
        dataset_min = float(X_train_np[:, idx].min())
        dataset_max = float(X_train_np[:, idx].max())
        
        if idx in preferences['numerical']:
            min_val = preferences['numerical'][idx]['min']
            max_val = preferences['numerical'][idx]['max']
            
            # Ensure bounds don't exceed dataset limits
            min_val = max(dataset_min, min(min_val, dataset_max))
            max_val = max(dataset_min, min(max_val, dataset_max))
            
            # Ensure valid range
            if min_val > max_val:
                min_val = dataset_min
                max_val = dataset_max
            
            feature_bounds.append((min_val, max_val))
        else:
            # If no preference defined, use training data range
            feature_bounds.append((dataset_min, dataset_max))
    
    logger.info(f"  ✓ Generated {len(feature_bounds)} feature bounds")
    
    # Verify all bounds are valid
    invalid_bounds = []
    for idx, (min_val, max_val) in enumerate(feature_bounds):
        if min_val > max_val:
            fname = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
            invalid_bounds.append(f"{fname}: [{min_val:.4f}, {max_val:.4f}]")
    
    if invalid_bounds:
        logger.error(f"  ⚠️  Found {len(invalid_bounds)} invalid bounds (min > max):")
        for bound_str in invalid_bounds[:5]:
            logger.error(f"    {bound_str}")
    
    # Log some example bounds for verification
    if config['output']['verbose']:
        logger.info(f"\n  Sample feature bounds (clamped to dataset limits):")
        for i in range(min(5, len(feature_bounds))):
            fname = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            min_val, max_val = feature_bounds[i]
            dataset_min = float(X_train_np[:, i].min())
            dataset_max = float(X_train_np[:, i].max())
            logger.info(f"    {fname}: [{min_val:.4f}, {max_val:.4f}] (dataset: [{dataset_min:.4f}, {dataset_max:.4f}])")
        if len(feature_bounds) > 5:
            logger.info(f"    ... and {len(feature_bounds) - 5} more")
    
    return feature_bounds, target_sample, priority_params, preferences


def save_feature_bounds_statistics(dataset_name, X_train, feature_names, feature_bounds):
    """
    Save detailed statistics about feature bounds to a CSV file.
    
    Computes and saves:
    - Dataset min/max for each feature
    - Applied bounds min/max for each feature
    - Restriction statistics (how much bounds restrict the search space)
    - Overall statistics per dataset
    
    Args:
        dataset_name: Name of the dataset
        X_train: Training data
        feature_names: List of feature names
        feature_bounds: List of (min, max) tuples for each feature
    """
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    
    # Compute per-feature statistics
    feature_stats = []
    for idx in range(len(feature_bounds)):
        fname = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        
        # Dataset statistics
        dataset_min = float(X_train_np[:, idx].min())
        dataset_max = float(X_train_np[:, idx].max())
        dataset_range = dataset_max - dataset_min
        dataset_mean = float(X_train_np[:, idx].mean())
        dataset_std = float(X_train_np[:, idx].std())
        
        # Bound statistics
        bound_min, bound_max = feature_bounds[idx]
        bound_range = bound_max - bound_min
        
        # Restriction statistics
        if dataset_range > 0:
            range_restriction_pct = (bound_range / dataset_range) * 100
            min_restriction = (bound_min - dataset_min) / dataset_range * 100
            max_restriction = (dataset_max - bound_max) / dataset_range * 100
        else:
            range_restriction_pct = 100.0
            min_restriction = 0.0
            max_restriction = 0.0
        
        # Check if bounds are restricted
        is_restricted = (bound_min > dataset_min) or (bound_max < dataset_max)

        # % of actual training data points that fall within the applied bounds
        col_values = X_train_np[:, idx]
        coverage_mask = (col_values >= bound_min) & (col_values <= bound_max)
        feature_coverage_pct = float(100.0 * np.mean(coverage_mask))

        feature_stats.append({
            'dataset': dataset_name,
            'feature_name': fname,
            'feature_index': idx,
            'dataset_min': dataset_min,
            'dataset_max': dataset_max,
            'dataset_range': dataset_range,
            'dataset_mean': dataset_mean,
            'dataset_std': dataset_std,
            'bound_min': bound_min,
            'bound_max': bound_max,
            'bound_range': bound_range,
            'bound_center': (bound_min + bound_max) / 2,
            'range_restriction_pct': range_restriction_pct,
            'min_restriction_pct': min_restriction,
            'max_restriction_pct': max_restriction,
            'is_restricted': is_restricted,
            'feature_coverage_pct': feature_coverage_pct
        })
    
    # Save per-feature statistics
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    per_feature_path = RESULTS_DIR / f'{clean_name}_feature_bounds_details.csv'
    pd.DataFrame(feature_stats).to_csv(per_feature_path, index=False)
    logger.info(f"  ✓ Saved per-feature bounds statistics to: {per_feature_path}")
    
    # Compute overall dataset statistics
    bound_mins = [fb[0] for fb in feature_bounds]
    bound_maxs = [fb[1] for fb in feature_bounds]
    dataset_mins = [float(X_train_np[:, i].min()) for i in range(X_train_np.shape[1])]
    dataset_maxs = [float(X_train_np[:, i].max()) for i in range(X_train_np.shape[1])]
    
    # Count features with restrictions
    n_features = len(feature_bounds)
    n_restricted_features = sum(1 for stat in feature_stats if stat['is_restricted'])
    
    # Calculate average restriction
    avg_range_restriction = np.mean([stat['range_restriction_pct'] for stat in feature_stats])
    
    # Calculate bound range statistics
    bound_ranges = [fb[1] - fb[0] for fb in feature_bounds]
    dataset_ranges = [dataset_maxs[i] - dataset_mins[i] for i in range(n_features)]
    
    overall_stats = {
        'dataset': dataset_name,
        'n_features': n_features,
        'n_restricted_features': n_restricted_features,
        'restriction_ratio': n_restricted_features / n_features if n_features > 0 else 0,
        
        # Dataset statistics (across all features)
        'dataset_min_mean': np.mean(dataset_mins),
        'dataset_min_std': np.std(dataset_mins),
        'dataset_max_mean': np.mean(dataset_maxs),
        'dataset_max_std': np.std(dataset_maxs),
        'dataset_range_mean': np.mean(dataset_ranges),
        'dataset_range_std': np.std(dataset_ranges),
        
        # Bound statistics (across all features)
        'bound_min_mean': np.mean(bound_mins),
        'bound_min_std': np.std(bound_mins),
        'bound_max_mean': np.mean(bound_maxs),
        'bound_max_std': np.std(bound_maxs),
        'bound_range_mean': np.mean(bound_ranges),
        'bound_range_std': np.std(bound_ranges),
        
        # Restriction statistics
        'avg_range_restriction_pct': avg_range_restriction,
        'min_range_restriction_pct': min([stat['range_restriction_pct'] for stat in feature_stats]),
        'max_range_restriction_pct': max([stat['range_restriction_pct'] for stat in feature_stats]),
        
        # Search space metrics
        'effective_search_volume_ratio': np.prod([stat['range_restriction_pct'] / 100 for stat in feature_stats]),

        # Data coverage: % of training data points within bounds (averaged across features)
        'avg_feature_coverage_pct': float(np.mean([stat['feature_coverage_pct'] for stat in feature_stats])),
        'min_feature_coverage_pct': float(np.min([stat['feature_coverage_pct'] for stat in feature_stats])),
        'max_feature_coverage_pct': float(np.max([stat['feature_coverage_pct'] for stat in feature_stats]))
    }
    
    # Save overall statistics to a per-dataset file (safe for parallel runs)
    overall_path = RESULTS_DIR / f'feature_bounds_summary_{clean_name}.csv'
    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(overall_path, index=False)
    
    logger.info(f"  ✓ Saved overall bounds summary to: {overall_path}")
    logger.info(f"    - {n_restricted_features}/{n_features} features restricted ({n_restricted_features/n_features*100:.1f}%)")
    logger.info(f"    - Average range restriction: {avg_range_restriction:.1f}%")
    logger.info(f"    - Effective search volume: {overall_stats['effective_search_volume_ratio']*100:.2f}% of full space")
    logger.info(f"    - Avg feature coverage (data points within bounds): {overall_stats['avg_feature_coverage_pct']:.1f}%")


def calculate_mad_from_training(X_train: np.ndarray) -> tuple:
    """
    Calculate Median Absolute Deviation (MAD) for each feature from training data.
    Used for normalized distance calculations in metrics.
    
    Args:
        X_train: Training data (n_samples, n_features)
    
    Returns:
        Tuple of:
        - mad_values: Array of raw MAD values for each feature (n_features,)
        - nonzero_mad_mask: Boolean mask, True where MAD > 0
        - zero_mad_indices: Indices of features with MAD == 0
    """
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    median = np.median(X_train_np, axis=0)
    mad = np.median(np.abs(X_train_np - median), axis=0)

    nonzero_mad_mask = mad > 0
    zero_mad_indices = np.where(~nonzero_mad_mask)[0]

    return mad, nonzero_mad_mask, zero_mad_indices


def compute_binary_cf_metrics(
    X_original: np.ndarray,
    all_cfs: np.ndarray,
    all_predictions: np.ndarray,
    all_predicted_classes: np.ndarray,
    target_class: int,
    mad_values: np.ndarray,
    nonzero_mad_mask: np.ndarray,
    k: int,
    categorical_features: list = None
) -> dict:
    """
    Compute counterfactual quality metrics for binary classification.
    Adapted from Mothilal et al. 2020 for binary targets.
    
    Reference: "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"
               Mothilal et al., FAT* 2020
    
    Args:
        X_original: Original instance (n_features,)
        all_cfs: All generated counterfactuals (n_cfs, n_features)
        all_predictions: Prediction probabilities for all CFs (n_cfs,)
        all_predicted_classes: Predicted classes for all CFs (n_cfs,)
        target_class: Target class (0 or 1)
        mad_values: MAD values for each feature (n_features,)
        nonzero_mad_mask: Boolean mask, True where MAD > 0
        k: Number of counterfactuals requested (for validity denominator)
        categorical_features: List of indices of categorical features (None = all continuous)
    
    Returns:
        Dictionary with metrics:
        - pct_valid_cfs: % of CFs that achieve target class
        - continuous_proximity: Average MAD-normalized distance (negative, higher=better)
        - categorical_proximity: Average categorical similarity (1=best)
        - continuous_sparsity: Fraction of unchanged continuous features (1=best)
        - continuous_diversity: Average pairwise distance between CFs
        - categorical_diversity: Average categorical difference between CFs
        - cont_count_diversity: Average number of differing features
    """
    # Consider only unique CF examples for metric computation.
    # This follows the paper's evaluation protocol to avoid counting duplicates.
    n_generated = len(all_cfs)
    if n_generated > 0:
        duplicate_eps = 1e-6
        quantized_cfs = np.round(all_cfs / duplicate_eps).astype(np.int64)
        _, unique_indices = np.unique(quantized_cfs, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        all_cfs = all_cfs[unique_indices]
        all_predictions = all_predictions[unique_indices]
        all_predicted_classes = all_predicted_classes[unique_indices]
        n_generated = len(all_cfs)

    d = len(X_original)  # Total number of features
    
    # Separate continuous and categorical features
    if categorical_features is None:
        categorical_features = []
    
    continuous_features = [
        i for i in range(d)
        if i not in categorical_features and bool(nonzero_mad_mask[i])
    ]
    d_cont = len(continuous_features)
    d_cat = len(categorical_features)
    
    # 1. % VALID CFs (those that achieve target class)
    valid_mask = all_predicted_classes == target_class
    n_valid = np.sum(valid_mask)
    pct_valid_cfs = float(n_valid) / k if k > 0 else 0.0
    
    # 2. CONTINUOUS PROXIMITY
    # Negative requested-k-normalized MAD distance (higher = closer)
    continuous_proximity = 0.0
    if n_generated > 0 and d_cont > 0 and k > 0:
        distances_cont = []
        for cf in all_cfs:
            cont_diff = np.abs(cf[continuous_features] - X_original[continuous_features]) / mad_values[continuous_features]
            dist = np.mean(cont_diff)
            distances_cont.append(dist)
        continuous_proximity = -np.sum(distances_cont) / k
    
    # 3. CATEGORICAL PROXIMITY
    categorical_proximity = 1.0
    if n_generated > 0 and d_cat > 0 and k > 0:
        distances_cat = []
        for cf in all_cfs:
            n_changed = np.sum(np.abs(cf[categorical_features] - X_original[categorical_features]) > 1e-6)
            dist = n_changed / d_cat
            distances_cat.append(dist)
        categorical_proximity = 1.0 - (np.sum(distances_cat) / k)
    
    # 4. CONTINUOUS-SPARSITY
    # Fraction of features that remain unchanged (requested-k normalized)
    continuous_sparsity = 1.0
    if n_generated > 0 and d_cont > 0 and k > 0:
        total_changes = 0
        for cf in all_cfs:
            n_changes = np.sum(np.abs(cf[continuous_features] - X_original[continuous_features]) > 1e-6)
            total_changes += n_changes
        continuous_sparsity = 1.0 - (total_changes / (k * d_cont))
    
    # 5. CONTINUOUS-DIVERSITY
    # Requested-k pair-normalized average MAD distance
    continuous_diversity = 0.0
    if n_generated > 1 and d_cont > 0 and k > 1:
        pairwise_distances_cont = []
        for i in range(n_generated):
            for j in range(i + 1, n_generated):
                cont_diff = np.abs(all_cfs[i][continuous_features] - all_cfs[j][continuous_features]) / mad_values[continuous_features]
                dist_cont = np.mean(cont_diff)
                pairwise_distances_cont.append(dist_cont)
        if pairwise_distances_cont:
            n_requested_pairs = k * (k - 1) / 2
            continuous_diversity = np.sum(pairwise_distances_cont) / n_requested_pairs
    
    # 6. CATEGORICAL-DIVERSITY
    categorical_diversity = 0.0
    if n_generated > 1 and d_cat > 0 and k > 1:
        pairwise_distances_cat = []
        for i in range(n_generated):
            for j in range(i + 1, n_generated):
                n_cat_diff = np.sum(np.abs(all_cfs[i][categorical_features] - all_cfs[j][categorical_features]) > 1e-6)
                dist_cat = n_cat_diff / d_cat
                pairwise_distances_cat.append(dist_cat)
        if pairwise_distances_cat:
            n_requested_pairs = k * (k - 1) / 2
            categorical_diversity = np.sum(pairwise_distances_cat) / n_requested_pairs
    
    # 7. CONT-COUNT-DIVERSITY
    # Requested-k pair-normalized count diversity for continuous features
    cont_count_diversity = 0.0
    if n_generated > 1 and d_cont > 0 and k > 1:
        cont_count_differences = []
        for i in range(n_generated):
            for j in range(i + 1, n_generated):
                n_cont_diff = np.sum(np.abs(all_cfs[i][continuous_features] - all_cfs[j][continuous_features]) > 1e-6)
                cont_count_differences.append(n_cont_diff)
        if cont_count_differences:
            n_requested_pairs = k * (k - 1) / 2
            cont_count_diversity = np.sum(cont_count_differences) / (n_requested_pairs * d_cont)
    
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


def compute_preference_score_from_preferences(cf_array, preferences):
    """
    Compute preference score for a CF using given preference functions.

    Mirrors RandomSearchExplainer.calculate_preference_score but operates
    on a standalone preferences dict returned by define_preferences().

    Args:
        cf_array: Numpy array of CF feature values
        preferences: Dict with 'numerical' and 'categorical' keys

    Returns:
        Float preference score (sum of per-feature preference weights)
    """
    scores = []
    for idx, constraint in preferences['numerical'].items():
        if isinstance(constraint, dict) and 'function' in constraint:
            weight = float(np.asarray(constraint['function'](cf_array[idx])).squeeze())
            scores.append(weight)
    for group_indices, possible_values in preferences['categorical'].items():
        current_combo = tuple(cf_array[idx] for idx in group_indices)
        weight = possible_values.get(current_combo, 0)
        scores.append(float(weight))
    return float(np.sum(scores)) if scores else 0.0


def get_configured_num_cfs(config):
    """Return the single configured CF count used by all methods."""
    num_cfs = int(config.get('standard_methods', {}).get('num_cfs', 1))
    return max(1, num_cfs)


def enforce_expected_cf_count(items, expected_count, method_name, sample_id):
    """
    Enforce that the returned CF list does not exceed expected_count.

    If fewer than expected are returned, keep them and log a warning.
    If more than expected are returned, truncate to expected_count and log a warning.
    """
    expected_count = max(int(expected_count), 0)
    if items is None:
        items = []

    current_count = len(items)
    if current_count > expected_count:
        logger.warning(
            "  %s sample %s: generated %d CFs, expected %d. Truncating extras.",
            method_name,
            sample_id,
            current_count,
            expected_count
        )
        return items[:expected_count]

    if current_count < expected_count:
        logger.warning(
            "  %s sample %s: generated %d CFs, expected %d.",
            method_name,
            sample_id,
            current_count,
            expected_count
        )

    return items


# ============================================================================
# STANDARD COUNTERFACTUAL METHODS
# ============================================================================

def get_standard_cf_methods(config, model, X_train, y_train, feature_names, feature_bounds=None,
                            categorical_feature_names=None):
    """
    Initialize all standard counterfactual methods based on configuration.

    Args:
        config: Configuration dictionary
        model: Trained model
        X_train: Training data
        y_train: Training labels
        feature_names: List of feature names
        feature_bounds: Optional list of (min, max) tuples for each feature
        categorical_feature_names: Feature names to treat as categorical by DiCE (finite-valued
            target-encoded columns). If None all features are treated as continuous.

    Returns:
        Dictionary mapping method display names to method instances
    """
    target_class = config['datasets']['target_class']
    max_iter = config['standard_methods']['max_iterations']
    num_cfs = get_configured_num_cfs(config)

    # Select method keys first so only enabled methods are initialized.
    selected = config['standard_methods']['selection']
    available_method_keys = list(CF_METHOD_NAMES.keys())

    if selected == 'all':
        selected_method_keys = available_method_keys
    elif isinstance(selected, str):
        selected_method_keys = [selected]
    elif isinstance(selected, list):
        selected_method_keys = selected
    else:
        raise ValueError(
            "config['standard_methods']['selection'] must be 'all', a method name string, or a list of method names"
        )

    invalid_methods = [m for m in selected_method_keys if m not in CF_METHOD_NAMES]
    if invalid_methods:
        raise ValueError(
            f"Unknown standard method(s): {invalid_methods}. Available methods: {available_method_keys}"
        )

    if not selected_method_keys:
        logger.warning("No standard methods selected!")
        return {}

    methods = {}
    for method_key in selected_method_keys:
        if method_key == 'wachter':
            methods['Wachter'] = WachterCounterfactual(
                model,
                num_cfs=num_cfs,
                max_iterations=max_iter,
                target_class=target_class,
                feature_bounds=feature_bounds
            )
        elif method_key == 'sparse_wachter':
            methods['Sparse Wachter'] = SparseWachterCounterfactual(
                model,
                num_cfs=num_cfs,
                max_iterations=max_iter,
                target_class=target_class,
                feature_bounds=feature_bounds
            )
        elif method_key == 'dice':
            methods['DiCE (gradient mode)'] = DiceCounterfactual(
                model,
                num_cfs=num_cfs,
                max_iterations=max_iter,
                target_class=target_class,
                learning_rate=config['standard_methods']['dice']['learning_rate'],
                diversity_weight=config['standard_methods']['dice']['diversity_weight'],
                lambda_param=config['standard_methods']['dice']['lambda'],
                feature_bounds=feature_bounds
            )
        elif method_key == 'prototype':
            methods['Prototype-Guided'] = PrototypeGuidedCounterfactual(
                model,
                X_train.values if hasattr(X_train, 'values') else X_train,
                y_train.values if hasattr(y_train, 'values') else y_train,
                n_prototypes=config['standard_methods']['prototype']['n_prototypes'],
                num_cfs=num_cfs,
                max_iterations=max_iter,
                target_class=target_class,
                feature_bounds=feature_bounds
            )
        elif method_key == 'dice_official':
            methods['DiCE (official library)'] = OfficialDiceCounterfactual(
                model,
                X_train.values if hasattr(X_train, 'values') else X_train,
                y_train.values if hasattr(y_train, 'values') else y_train,
                feature_names,
                num_cfs=num_cfs,
                max_iterations=config['standard_methods']['dice_official']['max_iter'],
                min_iterations=config['standard_methods']['dice_official']['min_iter'],
                target_class=target_class,
                learning_rate=config['standard_methods']['dice_official']['learning_rate'],
                proximity_weight=config['standard_methods']['dice_official']['proximity_weight'],
                diversity_weight=config['standard_methods']['dice_official']['diversity_weight'],
                categorical_penalty=config['standard_methods']['dice_official']['categorical_penalty'],
                loss_diff_thres=config['standard_methods']['dice_official']['loss_diff_thres'],
                loss_converge_maxiter=config['standard_methods']['dice_official']['loss_converge_maxiter'],
                yloss_type=config['standard_methods']['dice_official']['yloss_type'],
                feature_bounds=feature_bounds,
                categorical_feature_names=categorical_feature_names
            )

    logger.info("Selected standard methods: %s", ", ".join(methods.keys()))

    return methods


def save_standard_cf_results(method_name, dataset_name, sample_id, x_original,
                             x_cf, model, feature_names, threshold=0.5,
                             preference_score=None):
    """
    Save both original sample and counterfactual to CSV files.
    Uses append mode to add one result at a time.
    
    Saves with unified measure names for comparison:
    - prediction, predicted_class, target_achieved, l2_distance
    
    Returns:
        Tuple of (pred_orig, class_orig, pred_cf, class_cf, distance, target_achieved)
        or None if validation fails
    """
    method_clean = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    dataset_clean = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    
    # Validate dimensions
    if x_cf.shape != x_original.shape:
        logger.error(f"  Dimension mismatch: CF has shape {x_cf.shape}, original has {x_original.shape}. Skipping.")
        return None
    
    if len(x_cf) != len(feature_names):
        logger.error(f"  Feature count mismatch: CF has {len(x_cf)} features, expected {len(feature_names)}. Skipping.")
        return None
    
    orig_path = RESULTS_DIR / f'{method_clean}_{dataset_clean}_originals.csv'
    cf_path = RESULTS_DIR / f'{method_clean}_{dataset_clean}_counterfactuals.csv'
    
    # Initialize files if they don't exist
    if not orig_path.exists():
        orig_columns = ['sample_id'] + list(feature_names) + ['prediction', 'predicted_class']
        pd.DataFrame(columns=orig_columns).to_csv(orig_path, index=False)
    
    if not cf_path.exists():
        cf_columns = ['sample_id'] + list(feature_names) + [
            'prediction', 'predicted_class', 'l2_distance', 'target_achieved',
            'preference_score'
        ]
        pd.DataFrame(columns=cf_columns).to_csv(cf_path, index=False)
    
    try:
        # Save original
        pred_orig = model.predict(x_original.reshape(1, -1), verbose=0)[0, 0]
        class_orig = 1 if pred_orig >= threshold else 0
        
        orig_row = {'sample_id': sample_id}
        for i, fname in enumerate(feature_names):
            orig_row[fname] = x_original[i]
        orig_row['prediction'] = pred_orig
        orig_row['predicted_class'] = class_orig
        
        pd.DataFrame([orig_row]).to_csv(orig_path, mode='a', header=False, index=False)
        
        # Save counterfactual
        pred_cf = model.predict(x_cf.reshape(1, -1), verbose=0)[0, 0]
        class_cf = 1 if pred_cf >= threshold else 0
        distance = np.linalg.norm(x_cf - x_original)
        target_achieved = (class_cf == config['datasets']['target_class'])
        
        cf_row = {'sample_id': sample_id}
        for i, fname in enumerate(feature_names):
            cf_row[fname] = x_cf[i]
        cf_row['prediction'] = pred_cf
        cf_row['predicted_class'] = class_cf
        cf_row['l2_distance'] = distance
        cf_row['target_achieved'] = target_achieved
        cf_row['preference_score'] = preference_score
        
        pd.DataFrame([cf_row]).to_csv(cf_path, mode='a', header=False, index=False)
        
        return pred_orig, class_orig, pred_cf, class_cf, distance, target_achieved
        
    except Exception as e:
        logger.error(f"  Error saving results: {e}. Skipping this counterfactual.")
        return None


def test_standard_methods(model, X_train, y_train, test_indices, X_test_samples,
                         dataset_name, feature_names, config, feature_bounds=None,
                         status_tracker=None, target_sample=None,
                         categorical_feature_names=None):
    """
    Test all standard counterfactual methods on selected samples.

    Args:
        model: Trained model
        X_train: Training data
        y_train: Training labels
        test_indices: Indices of test samples
        X_test_samples: Test samples
        dataset_name: Name of dataset
        feature_names: List of feature names
        config: Configuration dictionary
        feature_bounds: Optional list of (min, max) tuples for constraining features
        categorical_feature_names: Feature names that DiCE should treat as categorical.

    Returns:
        Dictionary with results summary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING STANDARD CF METHODS ON {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    if feature_bounds is not None:
        logger.info(f"✓ Using {len(feature_bounds)} feature bounds to constrain CF generation")
    
    methods = get_standard_cf_methods(config, model, X_train, y_train, feature_names, feature_bounds,
                                      categorical_feature_names=categorical_feature_names)
    
    if not methods:
        logger.warning("No standard methods to test")
        return {}
    
    results_summary = []
    threshold = config['samples']['threshold']
    target_class = config['datasets']['target_class']
    
    # Collect all CFs and run stats for metrics computation
    all_method_cfs = {}  # method_name -> list of CF data
    all_method_run_stats = {}  # method_name -> list of per-sample run stats
    
    for method_name, cf_method in methods.items():
        if status_tracker is not None:
            status_tracker.record_method_start(dataset_name, method_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Method: {method_name}")
        logger.info(f"{'='*60}")
        
        # Clear/initialize result files
        method_clean = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        dataset_clean = dataset_name.lower().replace(' ', '_').replace('&', 'and')
        orig_path = RESULTS_DIR / f'{method_clean}_{dataset_clean}_originals.csv'
        cf_path = RESULTS_DIR / f'{method_clean}_{dataset_clean}_counterfactuals.csv'
        
        if orig_path.exists():
            orig_path.unlink()
        if cf_path.exists():
            cf_path.unlink()
        
        # Test on each sample
        successes = 0
        total_cfs = 0
        distances = []
        
        # Collect data for this method
        method_cf_data = []
        method_run_stats = []
        expected_cfs = get_configured_num_cfs(config)
        
        for i, (idx, x_original) in enumerate(zip(test_indices, X_test_samples)):
            sample_id = int(idx)
            
            if config['output']['verbose']:
                logger.info(f"\n  Sample {i+1}/{len(X_test_samples)} (ID: {sample_id})")

            # Compute per-sample preferences for preference score evaluation
            per_sample_preferences = None
            if target_sample is not None:
                per_sample_preferences, _ = define_preferences(
                    sample=x_original,
                    target_sample=target_sample,
                    X_train=X_train,
                    feature_names=feature_names,
                    exemplar_weight=config['preference_method']['exemplar_weight'],
                    steepness=config['preference_method'].get('preference_steepness', 5)
                )

            try:
                calc_start = time.perf_counter()

                # Generate counterfactual(s)
                result = cf_method.generate(x_original)
                calc_duration = time.perf_counter() - calc_start

                # Iteration statistics from method (if provided)
                run_iter_list = getattr(cf_method, 'last_run_iterations_list', None)
                if run_iter_list:
                    run_iterations = float(np.mean(run_iter_list))
                else:
                    # last_run_iterations may be None if generate() threw before setting it
                    _iters = getattr(cf_method, 'last_run_iterations', None)
                    if _iters is None:
                        _iters = getattr(cf_method, 'max_iterations', 0)
                    run_iterations = float(_iters)
                
                # Handle multiple CFs (e.g., from DiCE)
                if isinstance(result, list):
                    cfs = result
                elif isinstance(result, np.ndarray):
                    # Handle 2D array (multiple CFs) or 1D array (single CF)
                    if result.ndim == 2:
                        # Multiple CFs returned as rows
                        cfs = [result[i] for i in range(len(result))]
                    else:
                        # Single CF
                        cfs = [result]
                else:
                    cfs = [result]

                cfs = enforce_expected_cf_count(cfs, expected_cfs, method_name, sample_id)
                
                # Save each CF and compute per-calculation validity
                sample_total_cfs = 0
                sample_valid_cfs = 0
                for x_cf in cfs:
                    if x_cf is not None:
                        # Ensure CF is 1D array
                        x_cf = np.array(x_cf).flatten()
                        
                        pref_score = None
                        if per_sample_preferences is not None:
                            pref_score = compute_preference_score_from_preferences(
                                x_cf, per_sample_preferences
                            )

                        result = save_standard_cf_results(
                            method_name, dataset_name, sample_id, x_original,
                            x_cf, model, feature_names, threshold,
                            preference_score=pref_score
                        )

                        # Check if save was successful
                        if result is not None:
                            pred_orig, class_orig, pred_cf, class_cf, distance, target_achieved = result

                            total_cfs += 1
                            distances.append(distance)

                            if target_achieved:
                                successes += 1
                                sample_valid_cfs += 1
                            sample_total_cfs += 1

                            # Store for metrics computation
                            method_cf_data.append({
                                'x_original': x_original.copy(),
                                'x_cf': x_cf.copy(),
                                'pred_cf': pred_cf,
                                'class_cf': class_cf,
                                'preference_score': pref_score
                            })
                            
                            if config['output']['verbose']:
                                logger.info(f"    CF: pred={pred_cf:.4f}, class={class_cf}, "
                                          f"distance={distance:.4f}, target_achieved={target_achieved}")
                        else:
                            # Validation or save failed, skip this CF
                            if config['output']['verbose']:
                                logger.warning(f"    Skipped invalid CF")

                is_valid_expected = (
                    sample_total_cfs >= expected_cfs and sample_valid_cfs >= expected_cfs
                )
                method_run_stats.append({
                    'duration_sec': float(calc_duration),
                    'iterations': float(run_iterations),
                    'is_valid_expected': bool(is_valid_expected)
                })
                
            except Exception as e:
                logger.error(f"    Error: {e}")
                method_run_stats.append({
                    'duration_sec': float(time.perf_counter() - calc_start),
                    'iterations': float(getattr(cf_method, 'max_iterations', 0)),
                    'is_valid_expected': False
                })
                continue
        
        # Store CF data for this method
        all_method_cfs[method_name] = method_cf_data
        all_method_run_stats[method_name] = method_run_stats
        
        # Summary for this method
        success_rate = (successes / total_cfs * 100) if total_cfs > 0 else 0
        avg_distance = np.mean(distances) if distances else 0
        
        logger.info(f"\n  Summary: {successes}/{total_cfs} successful ({success_rate:.1f}%), "
                   f"avg distance: {avg_distance:.4f}")
        
        results_summary.append({
            'method': method_name,
            'n_samples': len(X_test_samples),
            'n_cfs': total_cfs,
            'n_successful': successes,
            'success_rate': success_rate,
            'avg_distance': avg_distance
        })
    
    return {
        'summary': results_summary,
        'cf_data': all_method_cfs,
        'run_stats': all_method_run_stats
    }


def compute_and_save_method_metrics(dataset_name, all_method_cfs, X_train, target_class, config,
                                    all_method_run_stats=None, feature_names=None):
    """
    Compute Mothilal et al. 2020 metrics for all methods and save to CSV.
    
    Args:
        dataset_name: Name of the dataset
        all_method_cfs: Dict mapping method_name -> list of CF data dicts
        X_train: Training data for MAD calculation
        target_class: Target class for validity check
        config: Configuration dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPUTING MOTHILAL ET AL. 2020 METRICS")
    logger.info(f"{'='*60}")
    
    # Calculate MAD values and exclude zero-MAD features from continuous metrics
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    mad_values, nonzero_mad_mask, zero_mad_indices = calculate_mad_from_training(X_train_np)

    if len(zero_mad_indices) > 0:
        if feature_names is not None and len(feature_names) == len(mad_values):
            zero_names = [str(feature_names[i]) for i in zero_mad_indices]
        else:
            zero_names = [f'feature_{i}' for i in zero_mad_indices]

        logger.warning(
            "  Excluding %d zero-MAD feature(s) from MAD-normalized continuous metrics: %s",
            len(zero_mad_indices),
            ', '.join(zero_names)
        )
    
    # Compute metrics for each method
    metrics_results = []
    all_method_run_stats = all_method_run_stats or {}

    # Mothilal et al. define k as the requested number of CFs to generate.
    # We use method-specific requested counts from config.
    requested_k_standard = get_configured_num_cfs(config)
    requested_k_preference = requested_k_standard
    
    for method_name, cf_data_list in all_method_cfs.items():
        if not cf_data_list:
            logger.warning(f"  No CFs generated for {method_name}, skipping metrics")
            continue

        if method_name.startswith('Preference ('):
            requested_k = requested_k_preference
        else:
            requested_k = requested_k_standard
        
        logger.info(f"\n  Computing metrics for: {method_name}")
        
        # Group by original sample
        samples_map = {}
        for cf_data in cf_data_list:
            x_orig_tuple = tuple(cf_data['x_original'])
            if x_orig_tuple not in samples_map:
                samples_map[x_orig_tuple] = {
                    'x_original': cf_data['x_original'],
                    'cfs': [],
                    'predictions': [],
                    'classes': []
                }
            samples_map[x_orig_tuple]['cfs'].append(cf_data['x_cf'])
            samples_map[x_orig_tuple]['predictions'].append(cf_data['pred_cf'])
            samples_map[x_orig_tuple]['classes'].append(cf_data['class_cf'])
        
        # Compute metrics per sample, then average
        sample_metrics = []
        for x_orig_tuple, data in samples_map.items():
            x_original = data['x_original']
            all_cfs = np.array(data['cfs'])
            all_preds = np.array(data['predictions'])
            all_classes = np.array(data['classes'])
            
            # Number of CFs requested per sample (Mothilal et al. denominator)
            k = requested_k
            
            metrics = compute_binary_cf_metrics(
                X_original=x_original,
                all_cfs=all_cfs,
                all_predictions=all_preds,
                all_predicted_classes=all_classes,
                target_class=target_class,
                mad_values=mad_values,
                nonzero_mad_mask=nonzero_mad_mask,
                k=k,
                categorical_features=None  # Assuming all continuous for these datasets
            )
            
            sample_metrics.append(metrics)
        
        # Average across samples
        if sample_metrics:
            run_stats = all_method_run_stats.get(method_name, [])
            time_all = [r['duration_sec'] for r in run_stats]
            iter_all = [r['iterations'] for r in run_stats]
            valid_run_stats = [r for r in run_stats if r.get('is_valid_expected', False)]
            time_valid = [r['duration_sec'] for r in valid_run_stats]
            iter_valid = [r['iterations'] for r in valid_run_stats]

            # Average preference over valid CFs only
            valid_pref_scores = [
                cf_data['preference_score']
                for cf_data in cf_data_list
                if cf_data.get('class_cf') == target_class
                and cf_data.get('preference_score') is not None
            ]
            avg_preference_valid = np.mean(valid_pref_scores) if valid_pref_scores else np.nan

            avg_metrics = {
                'dataset': dataset_name,
                'method': method_name,
                'n_samples': len(samples_map),
                'total_cfs': len(cf_data_list),
                'n_zero_mad_features_excluded': int(len(zero_mad_indices)),
                'n_cont_features_used_for_mad_metrics': int(np.sum(nonzero_mad_mask)),
                'pct_valid_cfs': np.mean([m['pct_valid_cfs'] for m in sample_metrics]),
                'continuous_proximity': np.mean([m['continuous_proximity'] for m in sample_metrics]),
                'categorical_proximity': np.mean([m['categorical_proximity'] for m in sample_metrics]),
                'continuous_sparsity': np.mean([m['continuous_sparsity'] for m in sample_metrics]),
                'continuous_diversity': np.mean([m['continuous_diversity'] for m in sample_metrics]),
                'categorical_diversity': np.mean([m['categorical_diversity'] for m in sample_metrics]),
                'cont_count_diversity': np.mean([m['cont_count_diversity'] for m in sample_metrics]),
                'avg_preference_valid_cfs': avg_preference_valid,
                'avg_time_valid_expected': np.mean(time_valid) if time_valid else np.nan,
                'avg_time_all': np.mean(time_all) if time_all else np.nan,
                'avg_iterations_valid_expected': np.mean(iter_valid) if iter_valid else np.nan,
                'avg_iterations_all': np.mean(iter_all) if iter_all else np.nan
            }
            
            logger.info(f"    % Valid CFs: {avg_metrics['pct_valid_cfs']*100:.2f}%")
            logger.info(f"    Continuous-Proximity: {avg_metrics['continuous_proximity']:.4f}")
            logger.info(f"    Continuous-Sparsity: {avg_metrics['continuous_sparsity']:.4f}")
            logger.info(f"    Distance-Diversity: {avg_metrics['continuous_diversity']:.4f}")
            logger.info(f"    Count-Diversity: {avg_metrics['cont_count_diversity']:.4f}")
            pref_display = f"{avg_preference_valid:.4f}" if not np.isnan(avg_preference_valid) else "N/A"
            logger.info(f"    Avg Preference (valid CFs): {pref_display}")
            logger.info(f"    Avg Time (valid+expected): {avg_metrics['avg_time_valid_expected']:.4f}s")
            logger.info(f"    Avg Time (all): {avg_metrics['avg_time_all']:.4f}s")
            logger.info(f"    Avg Iter (valid+expected): {avg_metrics['avg_iterations_valid_expected']:.2f}")
            logger.info(f"    Avg Iter (all): {avg_metrics['avg_iterations_all']:.2f}")
            
            metrics_results.append(avg_metrics)
    
    # Save to a per-dataset CSV (safe for parallel runs)
    if metrics_results:
        dataset_clean = dataset_name.lower().replace(' ', '_').replace('&', 'and')
        metrics_path = RESULTS_DIR / f'metrics_{dataset_clean}.csv'
        metrics_df = pd.DataFrame(metrics_results)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"\n  ✓ Metrics saved to: {metrics_path}")
    else:
        logger.warning("  No metrics to save")


# ============================================================================
# PREFERENCE-BASED COUNTERFACTUAL METHOD
# ============================================================================

def create_numerical_preference_function(sample_value, target_value, 
                                        dataset_min, dataset_max,
                                        exemplar_weight=0.5, steepness=5):
    """
    Create an exponential preference function for a numerical feature.
    
    Args:
        sample_value: Value of the feature in the original sample
        target_value: Value of the feature in the target exemplar
        dataset_min: Minimum value in the dataset for this feature
        dataset_max: Maximum value in the dataset for this feature
        exemplar_weight: Weight at the boundary (0.01=strict, 0.9=permissive)
        steepness: Steepness of exponential transition (1=gentle, 10=steep)
    
    Returns:
        Tuple of (preference_function, acceptable_min, acceptable_max, info_dict)
    """
    a = steepness  # Use configurable steepness parameter

    # Piecewise shape:
    #   - Linear 0.5 → 1.0 from sample to midpoint  (halfway to target)
    #   - Exponential drop 1.0 → 0 from midpoint onward
    # f(target) = exemplar_weight (controls x1 position).
    midpoint = (sample_value + target_value) / 2.0

    # t values: position in [0,1] within [x0,x1] where the exponential equals exemplar_weight
    # For increasing: t_inc = log(1 + ew*(e^a-1)) / a
    # For decreasing: t_dec = log(1 + (1-ew)*(e^a-1)) / a
    t_inc = np.log(1 + exemplar_weight * (np.exp(a) - 1)) / a
    t_dec = np.log(1 + (1 - exemplar_weight) * (np.exp(a) - 1)) / a

    if sample_value < target_value:
        # Search rightward: linear 0.5→1.0 from sample to midpoint,
        # then exponential drop from midpoint; f(target) = exemplar_weight.
        x0 = midpoint
        x1 = x0 + (target_value - x0) / t_dec
        increasing = False
        direction = "decreasing"
        acceptable_min = sample_value
        acceptable_max = dataset_max

        # Capture as locals to avoid closure issues
        _sv, _mp = float(sample_value), float(midpoint)
        _x0, _x1, _a = float(x0), float(x1), a

        def preference_func(x):
            x_arr = np.asarray(x, dtype=float)
            scalar_input = x_arr.ndim == 0
            x_arr = np.atleast_1d(x_arr)
            result = np.zeros(x_arr.shape)
            # Linear ramp: [sample, midpoint] → weight [0.5, 1.0]
            mask_lin = (x_arr >= _sv) & (x_arr <= _mp)
            if np.any(mask_lin):
                denom = _mp - _sv
                if abs(denom) >= 1e-12:
                    t = (x_arr[mask_lin] - _sv) / denom
                    result[mask_lin] = 0.5 + 0.5 * t
                else:
                    result[mask_lin] = 1.0
            # Exponential drop: (midpoint, ∞)
            mask_exp = x_arr > _mp
            if np.any(mask_exp):
                result[mask_exp] = exponential(x_arr[mask_exp], x0=_x0, x1=_x1, increasing=False, a=_a)
            return float(result.item()) if scalar_input else result

    else:
        # Search leftward: exponential rise to 1.0 at midpoint (for x < midpoint),
        # then linear 1.0→0.5 from midpoint to sample; f(target) = exemplar_weight.
        x1 = midpoint
        x0 = (target_value - t_inc * x1) / (1 - t_inc)
        increasing = True
        direction = "increasing"
        acceptable_min = dataset_min
        acceptable_max = sample_value

        _sv, _mp = float(sample_value), float(midpoint)
        _x0, _x1, _a = float(x0), float(x1), a

        def preference_func(x):
            x_arr = np.asarray(x, dtype=float)
            scalar_input = x_arr.ndim == 0
            x_arr = np.atleast_1d(x_arr)
            result = np.zeros(x_arr.shape)
            # Exponential rise: (-∞, midpoint)
            mask_exp = x_arr < _mp
            if np.any(mask_exp):
                result[mask_exp] = exponential(x_arr[mask_exp], x0=_x0, x1=_x1, increasing=True, a=_a)
            # Linear ramp: [midpoint, sample] → weight [1.0, 0.5]
            mask_lin = (x_arr >= _mp) & (x_arr <= _sv)
            if np.any(mask_lin):
                denom = _sv - _mp
                if abs(denom) >= 1e-12:
                    t = (x_arr[mask_lin] - _mp) / denom
                    result[mask_lin] = 1.0 - 0.5 * t
                else:
                    result[mask_lin] = 1.0
            return float(result.item()) if scalar_input else result

    # Ensure bounds are within dataset limits and valid (min <= max)
    acceptable_min = max(dataset_min, min(acceptable_min, dataset_max))
    acceptable_max = max(dataset_min, min(acceptable_max, dataset_max))

    # If range is invalid (min > max), constrain to dataset range
    if acceptable_min > acceptable_max:
        acceptable_min = dataset_min
        acceptable_max = dataset_max

    info = {
        'sample_value': float(sample_value),
        'target_value': float(target_value),
        'midpoint': float(midpoint),
        'x0': float(x0),
        'x1': float(x1),
        'direction': direction,
        'increasing': increasing,
        'a': a,
        'exemplar_weight': exemplar_weight,
        'acceptable_min': float(acceptable_min),
        'acceptable_max': float(acceptable_max),
        'dataset_min': float(dataset_min),
        'dataset_max': float(dataset_max)
    }

    return preference_func, acceptable_min, acceptable_max, info


def define_preferences(sample, target_sample, X_train, feature_names, exemplar_weight=0.5, steepness=5):
    """
    Automatically define preferences based on sample and target.
    
    Args:
        sample: Original sample
        target_sample: Target exemplar sample
        X_train: Training data
        feature_names: List of feature names
        exemplar_weight: Weight at boundary (0.01=strict, 0.9=permissive)
        steepness: Steepness of exponential transition (1=gentle, 10=steep)
    
    Returns:
        Tuple of (preferences dict, params list for saving)
    """
    sample = np.array(sample).flatten()
    target_sample = np.array(target_sample).flatten()
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    
    numerical_preferences = {}
    params_list = []
    
    for idx in range(len(sample)):
        sample_val = sample[idx]
        target_val = target_sample[idx]
        
        # Get dataset range
        dataset_min = float(X_train_np[:, idx].min())
        dataset_max = float(X_train_np[:, idx].max())
        
        # Create preference function
        pref_func, acceptable_min, acceptable_max, info = create_numerical_preference_function(
            sample_value=sample_val,
            target_value=target_val,
            dataset_min=dataset_min,
            dataset_max=dataset_max,
            exemplar_weight=exemplar_weight,
            steepness=steepness
        )

        # % of training data values that fall within the acceptable bounds
        coverage_mask = (
            (X_train_np[:, idx] >= acceptable_min) &
            (X_train_np[:, idx] <= acceptable_max)
        )
        info['feature_coverage_pct'] = float(100.0 * np.mean(coverage_mask))

        numerical_preferences[idx] = {
            'function': pref_func,
            'min': acceptable_min,
            'max': acceptable_max
        }
        
        params_list.append({
            'feature_index': idx,
            'feature_name': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
            **info
        })
    
    preferences = {
        'numerical': numerical_preferences,
        'categorical': {}
    }
    
    return preferences, params_list


def find_target_sample(X_train, y_train, model, target_class, target_probability):
    """
    Find a sample from training data close to the target probability.
    
    Returns:
        Target sample (numpy array)
    """
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_train_np = y_train_np.flatten()
    
    # Get predictions for training set
    predictions = model.predict(X_train_np, verbose=0).flatten()
    
    # Find samples from target class
    target_mask = y_train_np == target_class
    target_samples = X_train_np[target_mask]
    target_predictions = predictions[target_mask]
    
    if len(target_samples) == 0:
        logger.warning(f"No training samples found for class {target_class}")
        return None
    
    # Find sample closest to target probability
    distances = np.abs(target_predictions - target_probability)
    closest_idx = np.argmin(distances)
    
    target_sample = target_samples[closest_idx]
    actual_prob = target_predictions[closest_idx]
    
    if config['output']['verbose']:
        logger.debug(f"  Target sample: prediction={actual_prob:.4f} (target={target_probability})")
    
    return target_sample


def generate_preference_cfs_for_sample(sample, sample_id, model, X_train, y_train,
                                       feature_names, config, method='binary', target_sample=None):
    """
    Generate preference-based counterfactuals for a single sample.
    
    Args:
        sample: Sample to generate CFs for
        sample_id: Sample identifier
        model: Trained model
        X_train: Training data
        y_train: Training labels
        feature_names: List of feature names
        config: Configuration dictionary
        method: 'binary' or 'continuous'
        target_sample: Optional pre-computed target sample
    
    Returns:
        Tuple of (results_list, params_list)
    """
    # Find target sample if not provided
    if target_sample is None:
        target_sample = find_target_sample(
            X_train, y_train, model,
            target_class=config['datasets']['target_class'],
            target_probability=config['preference_method']['target_probability']
        )
        
        if target_sample is None:
            logger.warning(f"Could not find target sample for sample {sample_id}")
            return [], []
    
    # Define preferences
    preferences, params_list = define_preferences(
        sample=sample,
        target_sample=target_sample,
        X_train=X_train,
        feature_names=feature_names,
        exemplar_weight=config['preference_method']['exemplar_weight'],
        steepness=config['preference_method'].get('preference_steepness', 5)
    )
    
    # Define model prediction function
    model_pred = lambda x: model.predict(np.array(x), verbose=0).flatten()
    
    # Calculate target value
    target_value = config['preference_method']['target_probability']
    
    # Initialize explainer
    explainer = RandomSearchExplainer(
        model_pred=model_pred,
        priorities=preferences,
        sample=sample.tolist(),
        target=target_value
    )

    # Get n_candidates_per_cf from config if present, else default to 1
    n_candidates_per_cf = config['preference_method'].get('n_candidates_per_cf', 1)
    num_cfs = get_configured_num_cfs(config)
    return_top_n = min(config['preference_method'].get('return_top_n', num_cfs), num_cfs)
    
    # Generate counterfactuals
    try:
        if method == 'binary':
            cf_samples, cf_predictions, cf_scores, cf_iterations = explainer.generate_for_binary(
                expected_counterfactuals=num_cfs,
                max_iterations=config['preference_method']['max_iterations'],
                target_class=config['datasets']['target_class'],
                threshold=config['samples']['threshold'],
                random_seed=config['random_seed'],
                use_monte_carlo=config['preference_method']['use_monte_carlo'],
                max_tries=config['preference_method']['max_tries'],
                return_top_n=return_top_n,
                n_candidates_per_cf=n_candidates_per_cf
            )
        else:  # continuous
            cf_samples, cf_predictions, cf_scores, cf_iterations = explainer.generate_random_samples(
                expected_counterfactuals=num_cfs,
                max_iterations=config['preference_method']['max_iterations'],
                epsilon=config['preference_method']['epsilon'],
                random_seed=config['random_seed'],
                use_monte_carlo=config['preference_method']['use_monte_carlo'],
                max_tries=config['preference_method']['max_tries'],
                return_top_n=return_top_n,
                n_candidates_per_cf=n_candidates_per_cf
            )
        
        # Diagnostic logging if no CFs found
        if len(cf_samples) == 0:
            logger.warning(f"  No counterfactuals found for sample {sample_id}")
            if config['output']['verbose']:
                orig_pred = model_pred([sample])[0]
                logger.warning(f"    Original prediction: {orig_pred:.4f}")
                logger.warning(f"    Target: {target_value:.4f} +/- {config['preference_method'].get('epsilon', 0.25):.4f}")
        
    except Exception as e:
        logger.error(f"  Error generating CFs for sample {sample_id}: {e}")
        return [], params_list
    
    # Package results
    results = []
    for i, (cf_sample, cf_pred, cf_score, cf_iter) in enumerate(zip(
            cf_samples, cf_predictions, cf_scores, cf_iterations)):
        cf_array = np.array(cf_sample)
        sample_array = np.array(sample)
        
        # Calculate unified metrics
        l1_distance = float(np.sum(np.abs(cf_array - sample_array)))
        l2_distance = float(np.linalg.norm(cf_array - sample_array))
        sparsity = int(np.sum(np.abs(cf_array - sample_array) > 0.01))
        cf_class = int(cf_pred >= 0.5)
        target_achieved = (cf_class == config['datasets']['target_class'])
        
        result = {
            'sample_id': sample_id,
            'cf_rank': i + 1,
            'cf_values': cf_sample,
            'prediction': float(cf_pred),
            'predicted_class': cf_class,
            'l2_distance': l2_distance,
            'target_achieved': target_achieved,
            'preference_score': float(cf_score),
            'iteration_found': int(cf_iter),
            'l1_distance': l1_distance,
            'sparsity': sparsity
        }
        
        results.append(result)
    
    if config['output']['verbose'] and len(results) > 0:
        max_pref = max([r['preference_score'] for r in results])
        logger.info(f"    Generated {len(results)} CFs (max preference={max_pref:.4f})")
    
    return results, params_list


def initialize_preference_result_files(dataset_name, method, feature_names):
    """
    Initialize/clear preference result CSV files at the start of testing.
    Creates empty files with headers.
    
    Returns:
        Tuple of (cf_path, params_path)
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    cf_path = RESULTS_DIR / f"preference_based_{method}_{clean_name}_counterfactuals.csv"
    params_path = RESULTS_DIR / f"preference_based_{method}_{clean_name}_priority_parameters.csv"
    
    # Clear/create counterfactuals file
    with open(cf_path, 'w', newline='') as f:
        header = ['sample_id', 'cf_rank'] + feature_names + [
            'prediction', 'predicted_class', 'l2_distance', 'target_achieved',
            'preference_score', 'iteration_found', 'l1_distance', 'sparsity'
        ]
        writer = csv.writer(f)
        writer.writerow(header)
    
    # Clear/create priority parameters file
    params_header = [
        'sample_id', 'feature_index', 'feature_name', 'sample_value', 'target_value',
        'x0', 'x1', 'direction', 'increasing', 'a', 'exemplar_weight',
        'acceptable_min', 'acceptable_max', 'dataset_min', 'dataset_max',
        'feature_coverage_pct'
    ]
    pd.DataFrame(columns=params_header).to_csv(params_path, index=False)
    
    return cf_path, params_path


def save_preference_cf(result, dataset_name, method, feature_names):
    """
    Append a single preference-based counterfactual to the CSV file.
    
    Args:
        result: Dictionary with CF result
        dataset_name: Name of dataset
        method: 'binary' or 'continuous'
        feature_names: List of feature names
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    cf_path = RESULTS_DIR / f"preference_based_{method}_{clean_name}_counterfactuals.csv"
    
    cf_values_list = result['cf_values'].tolist() if hasattr(
        result['cf_values'], 'tolist') else list(result['cf_values'])
    
    row = [
        result['sample_id'],
        result['cf_rank']
    ] + cf_values_list + [
        result['prediction'],
        result['predicted_class'],
        result['l2_distance'],
        result['target_achieved'],
        result['preference_score'],
        result['iteration_found'],
        result['l1_distance'],
        result['sparsity']
    ]
    
    # Append to CSV
    row_df = pd.DataFrame([dict(zip(
        ['sample_id', 'cf_rank'] + feature_names + [
            'prediction', 'predicted_class', 'l2_distance', 'target_achieved',
            'preference_score', 'iteration_found', 'l1_distance', 'sparsity'
        ],
        row
    ))])
    row_df.to_csv(cf_path, mode='a', header=False, index=False)


def save_preference_params(params_list, sample_id, dataset_name, method):
    """
    Append priority parameters for a single sample to the CSV file.
    
    Args:
        params_list: List of parameter dictionaries for this sample
        sample_id: Sample identifier
        dataset_name: Name of dataset
        method: 'binary' or 'continuous'
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    params_path = RESULTS_DIR / f"preference_based_{method}_{clean_name}_priority_parameters.csv"
    
    # Add sample_id to each parameter dict
    for param in params_list:
        param['sample_id'] = sample_id

    # Write columns in the same order as the header row created by
    # initialize_preference_result_files so data aligns correctly.
    params_columns = [
        'sample_id', 'feature_index', 'feature_name', 'sample_value', 'target_value',
        'x0', 'x1', 'direction', 'increasing', 'a', 'exemplar_weight',
        'acceptable_min', 'acceptable_max', 'dataset_min', 'dataset_max',
        'feature_coverage_pct'
    ]
    df = pd.DataFrame(params_list)[params_columns]
    df.to_csv(params_path, mode='a', header=False, index=False)


def test_preference_methods(model, X_train, y_train, test_indices, X_test_samples,
                           dataset_name, feature_names, config, target_sample=None,
                           status_tracker=None):
    """
    Test preference-based counterfactual methods on selected samples.
    Saves results dynamically as they are generated.
    
    Args:
        model: Trained model
        X_train: Training data
        y_train: Training labels
        test_indices: Indices of test samples
        X_test_samples: Test samples
        dataset_name: Name of dataset
        feature_names: List of feature names
        config: Configuration dictionary
        target_sample: Optional pre-computed target sample for priority generation
    
    Returns:
        Dictionary with results for each method mode
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING PREFERENCE-BASED METHODS ON {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    # If no target sample provided, find one
    if target_sample is None:
        target_sample = find_target_sample(
            X_train, y_train, model,
            target_class=config['datasets']['target_class'],
            target_probability=config['preference_method']['target_probability']
        )
        if target_sample is None:
            logger.error("Could not find target sample")
            return {}
    else:
        logger.info(f"✓ Using pre-computed target sample")
    
    # Determine which methods to test
    mode = config['preference_method']['mode']
    if isinstance(mode, str):
        methods_to_test = [mode]
    elif isinstance(mode, list):
        methods_to_test = mode
    else:
        logger.warning(f"Invalid preference method mode: {mode}, defaulting to ['binary']")
        methods_to_test = ['binary']
    
    method_results = {}
    all_preference_cf_data = {}  # method_name -> list of CF data
    all_preference_run_stats = {}  # method_name -> list of per-sample run stats
    
    for method in methods_to_test:
        method_display_name = f"Preference ({method})"
        if status_tracker is not None:
            status_tracker.record_method_start(dataset_name, method_display_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Preference Method: {method.upper()}")
        logger.info(f"{'='*60}")
        
        # Initialize/clear result files at the start
        cf_path, params_path = initialize_preference_result_files(
            dataset_name, method, feature_names
        )
        logger.info(f"✓ Initialized result files (cleared old data)")
        
        # Track statistics for summary
        n_cfs = 0
        n_successful = 0
        distances = []
        
        # Collect CF data for metrics
        method_cf_data = []
        method_run_stats = []
        expected_cfs = get_configured_num_cfs(config)
        
        for i, (idx, x_original) in enumerate(zip(test_indices, X_test_samples)):
            sample_id = int(idx)
            
            if config['output']['verbose']:
                logger.info(f"\n  Sample {i+1}/{len(X_test_samples)} (ID: {sample_id})")

            calc_start = time.perf_counter()
            
            results, params = generate_preference_cfs_for_sample(
                sample=x_original,
                sample_id=sample_id,
                model=model,
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                config=config,
                method=method,
                target_sample=target_sample
            )
            calc_duration = time.perf_counter() - calc_start

            results = enforce_expected_cf_count(results, expected_cfs, method_display_name, sample_id)

            # Ensure stored rank reflects final per-sample ordering after enforcement.
            for rank, result in enumerate(results, start=1):
                result['cf_rank'] = rank

            sample_total_cfs = 0
            sample_valid_cfs = 0
            max_iteration_found = 0
            
            # Save priority parameters immediately
            if params:
                save_preference_params(params, sample_id, dataset_name, method)
            
            # Save each counterfactual immediately
            for result in results:
                save_preference_cf(result, dataset_name, method, feature_names)
                n_cfs += 1
                sample_total_cfs += 1
                if result['target_achieved']:
                    n_successful += 1
                    sample_valid_cfs += 1
                distances.append(result['l2_distance'])
                max_iteration_found = max(max_iteration_found, int(result.get('iteration_found', 0)))
                
                # Collect for metrics
                method_cf_data.append({
                    'x_original': x_original.copy(),
                    'x_cf': np.array(result['cf_values']),
                    'pred_cf': result['prediction'],
                    'class_cf': result['predicted_class'],
                    'preference_score': result.get('preference_score')
                })

            run_iterations = float(max_iteration_found) if max_iteration_found > 0 else float(
                config['preference_method']['max_iterations']
            )

            is_valid_expected = (
                sample_total_cfs >= expected_cfs and sample_valid_cfs >= expected_cfs
            )
            method_run_stats.append({
                'duration_sec': float(calc_duration),
                'iterations': run_iterations,
                'is_valid_expected': bool(is_valid_expected)
            })
        
        # Store CF data for this method
        all_preference_cf_data[method_display_name] = method_cf_data
        all_preference_run_stats[method_display_name] = method_run_stats
        
        # Calculate summary statistics
        n_samples = len(X_test_samples)
        success_rate = (n_successful / n_cfs * 100) if n_cfs > 0 else 0
        avg_distance = np.mean(distances) if distances else 0
        
        logger.info(f"\n  Summary: {n_successful}/{n_cfs} successful ({success_rate:.1f}%), "
                   f"avg distance: {avg_distance:.4f}")
        logger.info(f"✓ All results saved to: {cf_path}")
        logger.info(f"✓ All parameters saved to: {params_path}")
        
        method_results[method] = {
            'n_samples': n_samples,
            'n_cfs': n_cfs,
            'n_successful': n_successful,
            'success_rate': success_rate,
            'avg_distance': avg_distance
        }
    
    return {
        'summary': method_results,
        'cf_data': all_preference_cf_data,
        'run_stats': all_preference_run_stats
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_dataset(dataset_key, config, status_tracker=None):
    """
    Process a single dataset: load data, train model, run all CF methods.
    
    Returns:
        Dictionary with results for standard and preference methods
    """
    dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
    
    logger.info("\n" + "="*80)
    logger.info(f"PROCESSING DATASET: {dataset_name.upper()}")
    logger.info("="*80)
    
    # Load or prepare dataset
    data = load_dataset(dataset_key)
    
    if data is None:
        logger.info("Dataset not cached, loading from source...")
        if dataset_key == 'communities_crime':
            X_train, X_test, y_train, y_test, feature_names, scaler = load_communities_and_crime()
        elif dataset_key == 'german_credit':
            X_train, X_test, y_train, y_test, feature_names, scaler = load_german_credit()
        elif dataset_key == 'lending_club':
            X_train, X_test, y_train, y_test, feature_names, scaler = load_lending_club_selected_features(str(DATA_DIR / "LoanStats3a.csv"))
            logger.info(f"Lending Club: Number of features loaded: {X_train.shape[1]}")
            if X_train.shape[1] != 8:
                logger.error(f"Lending Club dataset should have 8 features, but got {X_train.shape[1]}.")
                raise ValueError("Lending Club dataset feature count mismatch.")
        elif dataset_key == 'credit_card_default':
            X_train, X_test, y_train, y_test, feature_names, scaler = load_credit_card_default()
            logger.info(f"Credit Card Default: {X_train.shape[0]} train samples, {X_train.shape[1]} features")
        else:
            logger.error(f"Unknown dataset: {dataset_key}")
            return None
        # Save for future use
        save_dataset(X_train, X_test, y_train, y_test, feature_names, scaler, dataset_key)
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Convert feature_names to list
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    elif not isinstance(feature_names, list):
        feature_names = list(feature_names)
    
    # Train or load model
    model, history, model_info = train_simple_model(
        X_train, y_train, X_test, y_test, dataset_name, config
    )
    
    # Select test samples
    test_indices, X_test_samples = select_test_samples(X_test, y_test, model, config)
    
    if test_indices is None:
        logger.error("Failed to select test samples")
        return None
    
    # Generate global priorities and feature bounds (used by all methods)
    feature_bounds, target_sample, priority_params, _ = generate_global_priorities_and_bounds(
        X_train, y_train, model, feature_names, config
    )
    
    if feature_bounds is None:
        logger.error("Failed to generate global priorities and bounds")
        return None
    
    # Save feature bounds statistics
    save_feature_bounds_statistics(dataset_name, X_train, feature_names, feature_bounds)
    
    # Test standard methods (using feature bounds)
    # For Lending Club the 4 target-encoded score columns may have zero MAD;
    # OfficialDiceCounterfactual will automatically exclude them from features_to_vary.
    standard_results = test_standard_methods(
        model, X_train, y_train, test_indices, X_test_samples,
        dataset_name, feature_names, config, feature_bounds, status_tracker,
        target_sample=target_sample
    )
    
    # Test preference-based methods (using same target sample)
    preference_results = test_preference_methods(
        model, X_train, y_train, test_indices, X_test_samples,
        dataset_name, feature_names, config, target_sample, status_tracker
    )
    
    # Combine CF data from both methods
    all_cf_data = {}
    if 'cf_data' in standard_results:
        all_cf_data.update(standard_results['cf_data'])
    if 'cf_data' in preference_results:
        all_cf_data.update(preference_results['cf_data'])

    # Combine run stats from both methods
    all_run_stats = {}
    if 'run_stats' in standard_results:
        all_run_stats.update(standard_results['run_stats'])
    if 'run_stats' in preference_results:
        all_run_stats.update(preference_results['run_stats'])
    
    # Compute and save metrics
    target_class = config['datasets']['target_class']
    compute_and_save_method_metrics(
        dataset_name=dataset_name,
        all_method_cfs=all_cf_data,
        X_train=X_train,
        target_class=target_class,
        config=config,
        all_method_run_stats=all_run_stats,
        feature_names=feature_names
    )
    
    return {
        'dataset_name': dataset_name,
        'standard_methods': standard_results.get('summary', standard_results),
        'preference_methods': preference_results.get('summary', preference_results)
    }


def _worker_init(cfg):
    """
    Initialise a worker process: set the global config and configure logging.
    Called once per child process before any tasks are submitted to it.
    """
    global config
    config = cfg
    # Spawned child processes don't inherit parent logging handlers.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if cfg.get('random_seed') is not None:
        np.random.seed(cfg['random_seed'])
        tf.random.set_seed(cfg['random_seed'])


def _run_dataset_worker(dataset_key):
    """
    Worker entry point for running a single dataset in a child process.
    Sets up per-dataset file logging, creates a per-dataset status tracker,
    then delegates to process_dataset().
    """
    log_path = setup_dataset_logging(dataset_key)
    logger.info(f"=== Dataset worker started: {dataset_key} — log: {log_path} ===")

    standard_method_names = get_selected_standard_method_display_names(config)
    preference_method_names = get_selected_preference_method_display_names(config)
    total_steps = len(standard_method_names) + len(preference_method_names)

    status_file = SCRIPT_DIR / f"status_{dataset_key}.txt"
    status_tracker = RunStatusTracker(status_file, total_steps)
    status_tracker.reset_file()

    return process_dataset(dataset_key, config, status_tracker)


def main():
    """Main execution function."""
    # Load configuration
    config_path = SCRIPT_DIR / 'config.yaml'
    global config
    config = load_config(config_path)

    # Set random seed in the main process
    if config['random_seed'] is not None:
        np.random.seed(config['random_seed'])
        tf.random.set_seed(config['random_seed'])

    logger.info("\n" + "="*80)
    logger.info("COMBINED COUNTERFACTUAL METHODS TEST")
    logger.info("="*80)
    logger.info(f"Random seed: {config['random_seed']}")
    logger.info(
        "Expected CFs per sample for all methods: %s",
        get_configured_num_cfs(config)
    )

    # Determine datasets to process
    dataset_selection = config['datasets']['selection']
    if dataset_selection == 'all':
        datasets = ['credit_card_default', 'communities_crime', 'german_credit']
    elif isinstance(dataset_selection, list):
        datasets = dataset_selection
    else:
        datasets = [dataset_selection]

    logger.info(f"Datasets to process in parallel: {datasets}")

    # Run all datasets simultaneously in separate worker processes
    all_results = {}
    n_workers = len(datasets)
    logger.info(f"Launching {n_workers} parallel worker process(es)...")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(config,)
    ) as executor:
        futures = {executor.submit(_run_dataset_worker, dk): dk for dk in datasets}
        for future in concurrent.futures.as_completed(futures):
            dataset_key = futures[future]
            try:
                result = future.result()
                if result:
                    all_results[result['dataset_name']] = result
                    logger.info(f"✓ Completed dataset: {dataset_key}")
            except Exception as exc:
                logger.error(
                    f"✗ Dataset '{dataset_key}' raised an exception: {exc}",
                    exc_info=True
                )

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("OVERALL SUMMARY")
    logger.info("="*80)

    for dataset_name, result in all_results.items():
        logger.info(f"\n{dataset_name}:")

        # Standard methods
        if 'summary' in result['standard_methods']:
            logger.info("  Standard Methods:")
            for method_result in result['standard_methods']['summary']:
                logger.info(f"    {method_result['method']:<30s} "
                          f"Success: {method_result['n_successful']}/{method_result['n_cfs']} "
                          f"({method_result['success_rate']:.1f}%) "
                          f"Avg Dist: {method_result['avg_distance']:.4f}")

        # Preference methods
        if result['preference_methods']:
            logger.info("  Preference-Based Methods:")
            for method_name, method_result in result['preference_methods'].items():
                if isinstance(method_result, dict) and 'n_successful' in method_result:
                    logger.info(f"    {method_name.capitalize():<30s} "
                              f"Success: {method_result['n_successful']}/{method_result['n_cfs']} "
                              f"({method_result['success_rate']:.1f}%) "
                              f"Avg Dist: {method_result['avg_distance']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)

    # Display Mothilal et al. 2020 metrics from all per-dataset files
    display_mothilal_metrics()


def display_mothilal_metrics():
    """Display Mothilal et al. 2020 metrics from per-dataset CSV files."""
    metrics_files = sorted(RESULTS_DIR.glob('metrics_*.csv'))

    if not metrics_files:
        logger.warning("\nNo metrics_*.csv files found in results directory")
        return

    logger.info("\n" + "="*80)
    logger.info("MOTHILAL ET AL. 2020 METRICS")
    logger.info("="*80)

    try:
        all_dfs = []
        for path in metrics_files:
            try:
                all_dfs.append(pd.read_csv(path))
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
        if not all_dfs:
            logger.warning("  No readable metrics files found")
            return
        metrics_df = pd.concat(all_dfs, ignore_index=True)
        
        # Group by dataset
        for dataset in metrics_df['dataset'].unique():
            dataset_df = metrics_df[metrics_df['dataset'] == dataset]
            logger.info(f"\n{dataset}:")

            def _fmt_float(val, decimals=4):
                if pd.isna(val):
                    return 'N/A'
                return f"{float(val):.{decimals}f}"

            logger.info(
                f"{'Method':<30s} {'Valid%':<8s} {'Prox':<8s} {'Spars':<8s} "
                f"{'DistDiv':<8s} {'CountDiv':<9s} {'AvgPref':<9s} {'TimeV':<8s} {'TimeAll':<8s} {'IterV':<8s} {'IterAll':<8s}"
            )
            logger.info("-" * 160)
            
            for _, row in dataset_df.iterrows():
                method_name = row['method']
                valid_pct = row['pct_valid_cfs'] * 100
                proximity = row['continuous_proximity']
                sparsity = row['continuous_sparsity']
                dist_diversity = row.get('continuous_diversity', 0.0)
                count_diversity = row.get('cont_count_diversity', 0.0)
                avg_preference = row.get('avg_preference_valid_cfs', np.nan)
                time_valid = row.get('avg_time_valid_expected', np.nan)
                time_all = row.get('avg_time_all', np.nan)
                iter_valid = row.get('avg_iterations_valid_expected', np.nan)
                iter_all = row.get('avg_iterations_all', np.nan)

                logger.info(
                    f"{method_name:<30s} {valid_pct:>6.2f}% {proximity:>7.4f} {sparsity:>7.4f} "
                    f"{dist_diversity:>7.4f} {count_diversity:>8.4f} "
                    f"{_fmt_float(avg_preference, 4):>8s} "
                    f"{_fmt_float(time_valid, 4):>8s} {_fmt_float(time_all, 4):>8s} "
                    f"{_fmt_float(iter_valid, 2):>8s} {_fmt_float(iter_all, 2):>8s}"
                )
        
        logger.info("\n" + "="*80)
        logger.info(f"Full metrics loaded from: {len(metrics_files)} file(s) in {RESULTS_DIR}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error reading metrics file: {e}")


if __name__ == "__main__":
    main()
