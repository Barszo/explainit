"""
Preference-Based Counterfactual Method Test

This script tests the preference-based random search counterfactual method
on the same datasets and samples used in dice_test.py.

The method:
1. Loads the same datasets (Communities and Crime, German Credit)
2. Loads the same trained models
3. Reads the original samples from existing CSV files
4. Automatically defines preferences based on sample-target relationships
5. Generates counterfactuals using RandomSearchExplainer
6. Saves results with preference scores and priority parameters

Key differences from dice_test.py:
- Only tests the preference-based method (not standard CF methods)
- Automatically generates priorities/preferences for each sample
- Saves detailed preference parameters for reproducibility
- Returns preference scores for all counterfactuals

Output Files (OVERWRITTEN each run):
- *_counterfactuals.csv: Generated counterfactuals with metrics
- *_priority_parameters.csv: Preference function parameters for each feature
  (N_samples × N_features rows - needed for reproducibility)
- *_summary.txt: Summary statistics and configuration

Note: Old result files are explicitly deleted at the start of each run
to ensure fresh results.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import csv
import logging
from datetime import datetime
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import local modules
from data_downloader import load_communities_and_crime, load_german_credit
from explainit.priorities.nonlinear import exponential
from explainit.explainers.random_search import RandomSearchExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset Selection
    'datasets': 'all',  # ['communities_crime', 'german_credit'] or 'all'
    
    # Method Selection
    'method': 'both',  # 'binary', 'continuous', or 'both' to test both approaches
    
    # Target Settings
    'source_class': 0,  # Generate CFs from class 0
    'target_class': 1,  # To class 1
    'target_probability': 0.75,  # Target prediction probability for target_class (continuous method)
    'threshold': 0.5,  # Decision threshold for binary classification (binary method)
    
    # Preference-Based Method Settings
    'exemplar_weight': 0.75,  # Weight assigned to boundary values (0.01=very strict, 0.5=balanced, 0.9=permissive)
    'expected_counterfactuals': 2,  # Number of counterfactuals to find
    'max_iterations': 1000,  # Maximum number of iterations to try
    'epsilon': 0.25,  # Acceptable deviation from target probability (continuous method only)
                       # Note: Smaller epsilon = stricter requirement, may find fewer CFs
                       # Recommended: 0.10-0.25 for classification tasks
    'use_monte_carlo': True,  # Use Monte Carlo sampling
    'return_top_n': 5,  # Return top N counterfactuals by preference score
    'max_tries': 100,  # Max tries for sampling
    'random_seed': 42,  # Random seed for reproducibility
    
    # Reporting
    'verbose': True,  # Print detailed progress
}

# Dataset name mapping
DATASET_NAMES = {
    'communities_crime': 'Communities and Crime',
    'german_credit': 'German Credit'
}

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def clean_old_results(dataset_name, method='binary'):
    """
    Delete old result files for a specific dataset and method.
    This ensures fresh results for each run.
    
    Args:
        dataset_name: Name of dataset
        method: 'binary' or 'continuous'
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    
    # Define files to delete
    files_to_delete = [
        RESULTS_DIR / f"preference_based_{method}_{clean_name}_counterfactuals.csv",
        RESULTS_DIR / f"preference_based_{method}_{clean_name}_priority_parameters.csv",
        RESULTS_DIR / f"preference_based_{method}_{clean_name}_summary.txt"
    ]
    
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            deleted_count += 1
            logger.debug(f"  Deleted old file: {file_path.name}")
    
    if deleted_count > 0:
        logger.info(f"✓ Cleaned {deleted_count} old result file(s) for {method} method")
    
    return deleted_count


# ============================================================================
# DATA AND MODEL LOADING
# ============================================================================

def load_dataset(dataset_name):
    """Load dataset from disk (saved by dice_test.py)."""
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    data_path = DATA_DIR / f"{clean_name}_data.pkl"
    
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        logger.error("Please run dice_test.py first to generate the dataset!")
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"✓ Dataset loaded from: {data_path}")
    logger.info(f"  Train shape: {data['train_shape']}, Test shape: {data['test_shape']}")
    
    return data


def load_model(dataset_name):
    """Load trained model from disk (saved by dice_test.py)."""
    import tensorflow as tf
    
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    model_path = MODELS_DIR / f"{clean_name}_model.keras"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run dice_test.py first to train the model!")
        return None
    
    model = tf.keras.models.load_model(model_path)
    logger.info(f"✓ Model loaded from: {model_path}")
    
    return model


def load_original_samples(dataset_name, source_class=0):
    """
    Load original samples from CSV files saved by dice_test.py.
    
    Args:
        dataset_name: Name of dataset
        source_class: Class to load samples from
        
    Returns:
        DataFrame with original samples
    """
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    
    # Try to find any originals file for this dataset (from any method)
    originals_files = list(RESULTS_DIR.glob(f"*_{clean_name}_originals.csv"))
    
    if not originals_files:
        logger.error(f"No original samples found for {dataset_name}")
        logger.error("Please run dice_test.py first to generate samples!")
        return None
    
    # Load from first available file
    originals_path = originals_files[0]
    df = pd.read_csv(originals_path)
    
    # Filter by source class
    df_filtered = df[df['predicted_class'] == source_class].copy()
    
    logger.info(f"✓ Loaded {len(df_filtered)} original samples from class {source_class}")
    logger.info(f"  Source file: {originals_path.name}")
    
    return df_filtered


# ============================================================================
# PREFERENCE FUNCTION GENERATION
# ============================================================================

def create_numerical_preference_function(sample_value, target_value, 
                                        dataset_min, dataset_max,
                                        exemplar_weight=0.5):
    """
    Create an exponential preference function for a numerical feature.
    
    The function creates a preference where:
    - f(sample_value) = 1.0 (most preferred - current value)
    - f(target_value) = exemplar_weight (boundary of acceptability)
    - f(x0) = 0.0 (completely unacceptable)
    
    Args:
        sample_value: Current value in the sample
        target_value: Value in the target/exemplar
        dataset_min: Minimum value in dataset
        dataset_max: Maximum value in dataset
        exemplar_weight: Weight at target_value (0-1)
        
    Returns:
        Tuple of (preference_function, x0, x1, direction_info)
    """
    a = 5  # Steepness parameter for exponential
    
    # Calculate t value where function equals exemplar_weight
    t_target = np.log(1 + exemplar_weight * (np.exp(a) - 1)) / a
    
    # Determine direction (increasing or decreasing preference)
    if sample_value < target_value:
        # Moving towards higher values - prefer lower (decreasing)
        x1 = (target_value - t_target * sample_value) / (1 - t_target)
        x0 = sample_value
        increasing = False
        direction = "decreasing"
    else:
        # Moving towards lower values - prefer lower (increasing from low)
        x0 = (target_value - t_target * sample_value) / (1 - t_target)
        x1 = sample_value
        increasing = True
        direction = "increasing"
    
    # Determine acceptable range
    if increasing:
        acceptable_min = max(dataset_min, x0)
        acceptable_max = dataset_max
    else:
        acceptable_min = dataset_min
        acceptable_max = min(dataset_max, x0)
    
    def preference_func(x):
        return exponential(x, x0=x0, x1=x1, increasing=increasing, a=a)
    
    info = {
        'sample_value': float(sample_value),
        'target_value': float(target_value),
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


def define_preferences(sample, target_sample, X_train, feature_names, exemplar_weight=0.5):
    """
    Automatically define preferences based on sample and target.
    
    Creates preference functions that guide the search from the sample
    towards the target while respecting feature ranges.
    
    Args:
        sample: Original sample (numpy array or list)
        target_sample: Target sample to guide preferences (numpy array or list)
        X_train: Training data for extracting feature ranges (numpy array or DataFrame)
        feature_names: List of feature names
        exemplar_weight: Weight assigned to target values
        
    Returns:
        Tuple of (preferences dict, params list for saving)
    """
    # Convert to numpy arrays
    sample = np.array(sample).flatten()
    target_sample = np.array(target_sample).flatten()
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    
    numerical_preferences = {}
    params_list = []
    
    for idx in range(len(sample)):
        sample_val = sample[idx]
        target_val = target_sample[idx]
        
        # Get dataset range for this feature
        dataset_min = float(X_train_np[:, idx].min())
        dataset_max = float(X_train_np[:, idx].max())
        
        # Create preference function
        pref_func, acceptable_min, acceptable_max, info = create_numerical_preference_function(
            sample_value=sample_val,
            target_value=target_val,
            dataset_min=dataset_min,
            dataset_max=dataset_max,
            exemplar_weight=exemplar_weight
        )
        
        numerical_preferences[idx] = {
            'function': pref_func,
            'min': acceptable_min,
            'max': acceptable_max
        }
        
        # Store parameters for CSV
        params_list.append({
            'feature_index': idx,
            'feature_name': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
            **info
        })
    
    preferences = {
        'numerical': numerical_preferences,
        'categorical': {}  # No categorical features in these datasets
    }
    
    return preferences, params_list


def find_target_sample(X_train, y_train, model, target_class=1, target_probability=0.75):
    """
    Find a sample from training data close to the target probability.
    
    Args:
        X_train: Training features (numpy array or DataFrame)
        y_train: Training labels (numpy array or Series)
        model: Trained model
        target_class: Target class
        target_probability: Desired probability for target class
        
    Returns:
        Target sample (numpy array)
    """
    # Convert to numpy arrays if needed
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
    
    logger.debug(f"  Target sample found: prediction={actual_prob:.4f} (target={target_probability})")
    
    return target_sample


# ============================================================================
# COUNTERFACTUAL GENERATION
# ============================================================================

def generate_counterfactuals_for_sample(sample, sample_id, model, X_train, y_train,
                                        feature_names, config, method='binary'):
    """
    Generate counterfactuals for a single sample using preference-based method.
    
    Args:
        sample: Original sample (numpy array without sample_id, prediction, class)
        sample_id: Sample identifier
        model: Trained model
        X_train: Training data
        y_train: Training labels
        feature_names: Feature names
        config: Configuration dict
        method: 'binary' or 'continuous' to select generation method
        
    Returns:
        List of counterfactual dictionaries
    """
    # Find target sample for preference generation
    target_sample = find_target_sample(
        X_train, y_train, model,
        target_class=config['target_class'],
        target_probability=config['target_probability']
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
        exemplar_weight=config['exemplar_weight']
    )
    
    # Define model prediction function
    model_pred = lambda x: model.predict(np.array(x), verbose=0).flatten()
    
    # Calculate target value
    target_value = config['target_probability']
    
    # Initialize explainer
    explainer = RandomSearchExplainer(
        model_pred=model_pred,
        priorities=preferences,
        sample=sample.tolist(),
        target=target_value
    )
    
    # Generate counterfactuals based on method
    logger.debug(f"  Generating counterfactuals for sample {sample_id} using {method} method...")
    
    try:
        if method == 'binary':
            logger.debug(f"  Target class: {config['target_class']} (threshold={config['threshold']:.2f})")
            cf_samples, cf_predictions, cf_scores, cf_iterations = explainer.generate_for_binary(
                expected_counterfactuals=config['expected_counterfactuals'],
                max_iterations=config['max_iterations'],
                target_class=config['target_class'],
                threshold=config['threshold'],
                random_seed=config['random_seed'],
                use_monte_carlo=config['use_monte_carlo'],
                max_tries=config['max_tries'],
                return_top_n=config['return_top_n']
            )
        else:  # continuous method
            logger.debug(f"  Target: {target_value:.4f} +/- {config['epsilon']:.4f} (range: [{target_value - config['epsilon']:.4f}, {target_value + config['epsilon']:.4f}])")
            cf_samples, cf_predictions, cf_scores, cf_iterations = explainer.generate_random_samples(
                expected_counterfactuals=config['expected_counterfactuals'],
                max_iterations=config['max_iterations'],
                epsilon=config['epsilon'],
                random_seed=config['random_seed'],
                use_monte_carlo=config['use_monte_carlo'],
                max_tries=config['max_tries'],
                return_top_n=config['return_top_n']
            )
        
        # Diagnostic: log prediction statistics if no CFs found
        if len(cf_samples) == 0:
            logger.warning(f"  No counterfactuals found. Checking why...")
            logger.warning(f"  Original sample prediction: {model_pred([sample])[0]:.4f}")
            if method == 'binary':
                logger.warning(f"  Target class: {config['target_class']} (threshold={config['threshold']})")
            else:
                logger.warning(f"  Target range: [{target_value - config['epsilon']:.4f}, {target_value + config['epsilon']:.4f}]")
            
    except Exception as e:
        logger.error(f"  Error generating counterfactuals for sample {sample_id}: {e}")
        return [], params_list
    
    # Package results
    results = []
    for i, (cf_sample, cf_pred, cf_score, cf_iter) in enumerate(zip(cf_samples, cf_predictions, cf_scores, cf_iterations)):
        cf_array = np.array(cf_sample)
        sample_array = np.array(sample)
        
        # Calculate metrics
        l1_distance = float(np.sum(np.abs(cf_array - sample_array)))
        l2_distance = float(np.linalg.norm(cf_array - sample_array))
        
        # Sparsity: count features that changed significantly (threshold: 0.01)
        sparsity = int(np.sum(np.abs(cf_array - sample_array) > 0.01))
        
        # Predicted class (threshold at 0.5)
        cf_class = int(cf_pred >= 0.5)
        
        result = {
            'sample_id': sample_id,
            'cf_rank': i + 1,
            'cf_values': cf_sample,
            'prediction': float(cf_pred),
            'predicted_class': cf_class,
            'preference_score': float(cf_score),
            'iteration_found': int(cf_iter),
            'l1_distance': l1_distance,
            'l2_distance': l2_distance,
            'sparsity': sparsity,
            'target_achieved': cf_class == config['target_class']
        }
        
        results.append(result)
    
    # Log results
    if len(results) > 0:
        max_pref = max([r['preference_score'] for r in results])
        logger.info(f"  Sample {sample_id}: Generated {len(results)} counterfactuals "
                    f"(max preference={max_pref:.4f})")
    else:
        logger.warning(f"  Sample {sample_id}: No counterfactuals found within epsilon={config['epsilon']} "
                      f"of target={target_value:.4f}")
    
    return results, params_list


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_counterfactuals_csv(results_list, feature_names, dataset_name, method='binary'):
    """Save counterfactuals to CSV file."""
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    filename = RESULTS_DIR / f"preference_based_{method}_{clean_name}_counterfactuals.csv"
    
    if not results_list:
        logger.warning(f"No counterfactuals to save for {dataset_name}")
        return None
    
    with open(filename, 'w', newline='') as f:
        # Create header
        header = ['sample_id', 'cf_rank'] + feature_names + [
            'prediction', 'predicted_class', 'preference_score', 'iteration_found',
            'l1_distance', 'l2_distance', 'sparsity', 'target_achieved'
        ]
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Write data
        for result in results_list:
            # Convert cf_values to list if it's a numpy array
            cf_values_list = result['cf_values'].tolist() if hasattr(result['cf_values'], 'tolist') else list(result['cf_values'])
            row = [
                result['sample_id'],
                result['cf_rank']
            ] + cf_values_list + [
                result['prediction'],
                result['predicted_class'],
                result['preference_score'],
                result['iteration_found'],
                result['l1_distance'],
                result['l2_distance'],
                result['sparsity'],
                result['target_achieved']
            ]
            writer.writerow(row)
    
    logger.info(f"✓ Saved {len(results_list)} counterfactuals to: {filename}")
    return filename


def save_priority_parameters_csv(all_params, dataset_name, method='binary'):
    """
    Save priority/preference parameters to CSV for reproducibility.
    
    This saves one row per (sample_id, feature_index) combination.
    For N samples with M features each, this will create N*M rows.
    
    Each row contains:
    - sample_id: which sample was processed
    - feature_index: which feature this preference is for
    - feature_name: name of the feature
    - All preference function parameters (x0, x1, direction, weight, etc.)
    
    Args:
        all_params: List of (sample_id, params_list) tuples
        dataset_name: Name of dataset
        method: 'binary' or 'continuous'
        
    Returns:
        Path to saved file, or None if no data to save
    """
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    filename = RESULTS_DIR / f"preference_based_{method}_{clean_name}_priority_parameters.csv"
    
    if not all_params:
        logger.warning(f"No priority parameters to save for {dataset_name}")
        return None
    
    # Flatten the nested list structure
    flat_params = []
    for sample_id, params_list in all_params:
        for params in params_list:
            flat_params.append({
                'sample_id': sample_id,
                **params
            })
    
    if not flat_params:
        logger.warning("No priority parameters to save!")
        return None
    
    # Create DataFrame and save
    df = pd.DataFrame(flat_params)
    df.to_csv(filename, index=False)
    
    n_samples = len(set(p['sample_id'] for p in flat_params))
    n_features = len(flat_params) // n_samples if n_samples > 0 else 0
    logger.info(f"✓ Saved priority parameters to: {filename}")
    logger.info(f"  ({n_samples} samples × {n_features} features = {len(flat_params)} rows)")
    return filename


def save_summary_statistics(results_list, dataset_name, config, method='binary'):
    """Save summary statistics to text file."""
    # Clean dataset name to match dice_test.py naming convention
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    filename = RESULTS_DIR / f"preference_based_{method}_{clean_name}_summary.txt"
    
    # Calculate statistics
    n_total_samples = len(set(r['sample_id'] for r in results_list)) if results_list else 0
    n_total_cfs = len(results_list)
    n_successful = sum(1 for r in results_list if r['target_achieved'])
    success_rate = (n_successful / n_total_cfs * 100) if n_total_cfs > 0 else 0
    
    # Get score lists (handle empty case)
    preference_scores = [r['preference_score'] for r in results_list] if results_list else []
    l2_distances = [r['l2_distance'] for r in results_list] if results_list else []
    sparsities = [r['sparsity'] for r in results_list] if results_list else []
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREFERENCE-BASED COUNTERFACTUAL METHOD - SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Results:\n")
        f.write(f"  Total original samples: {n_total_samples}\n")
        f.write(f"  Total counterfactuals generated: {n_total_cfs}\n")
        f.write(f"  Successful (target class achieved): {n_successful} ({success_rate:.1f}%)\n\n")
        
        if preference_scores:
            f.write("Preference Scores:\n")
            f.write(f"  Mean: {np.mean(preference_scores):.4f}\n")
            f.write(f"  Median: {np.median(preference_scores):.4f}\n")
            f.write(f"  Min: {np.min(preference_scores):.4f}\n")
            f.write(f"  Max: {np.max(preference_scores):.4f}\n")
            f.write(f"  Std: {np.std(preference_scores):.4f}\n\n")
            
            f.write("L2 Distances:\n")
            f.write(f"  Mean: {np.mean(l2_distances):.4f}\n")
            f.write(f"  Median: {np.median(l2_distances):.4f}\n")
            f.write(f"  Min: {np.min(l2_distances):.4f}\n")
            f.write(f"  Max: {np.max(l2_distances):.4f}\n\n")
            
            f.write("Sparsity (# changed features):\n")
            f.write(f"  Mean: {np.mean(sparsities):.2f}\n")
            f.write(f"  Median: {np.median(sparsities):.1f}\n")
            f.write(f"  Min: {int(np.min(sparsities))}\n")
            f.write(f"  Max: {int(np.max(sparsities))}\n\n")
        else:
            f.write("No counterfactuals were generated.\n")
            f.write("Consider adjusting parameters:\n")
            f.write("  - Increase epsilon (currently: {:.4f})\n".format(config.get('epsilon', 0.05)))
            f.write("  - Increase n_samples (currently: {})\n".format(config.get('n_samples', 10000)))
            f.write("  - Adjust exemplar_weight (currently: {:.4f})\n\n".format(config.get('exemplar_weight', 0.01)))
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"✓ Saved summary statistics to: {filename}")
    return filename


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_preference_method_on_dataset(dataset_name, config):
    """
    Test preference-based method on a single dataset.
    
    Args:
        dataset_name: Name of dataset
        config: Configuration dictionary
        
    Returns:
        Dictionary with detailed results for each method tested
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TESTING PREFERENCE-BASED METHOD ON {dataset_name.upper()}")
    logger.info("=" * 80)
    
    # Load dataset and model
    data = load_dataset(dataset_name)
    if data is None:
        return False
    
    model = load_model(dataset_name)
    if model is None:
        return False
    
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']
    
    # Convert feature_names to list if it's an Index or similar
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    elif not isinstance(feature_names, list):
        feature_names = list(feature_names)
    
    # Load original samples
    originals_df = load_original_samples(dataset_name, source_class=config['source_class'])
    if originals_df is None or len(originals_df) == 0:
        logger.error(f"No original samples found for {dataset_name}")
        return False
    
    logger.info(f"\nProcessing {len(originals_df)} samples...")
    
    # Get feature column names from the CSV (excluding metadata)
    metadata_cols = ['sample_id', 'prediction', 'predicted_class']
    csv_feature_names = [col for col in originals_df.columns if col not in metadata_cols]
    
    # Determine which methods to test
    methods_to_test = []
    if config['method'] == 'both':
        methods_to_test = ['binary', 'continuous']
    else:
        methods_to_test = [config['method']]
    
    # Track results for each method
    method_results = {}
    
    # Test each method
    for method in methods_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING {method.upper()} METHOD")
        logger.info(f"{'='*80}")
        
        # Clean old result files for this dataset and method
        logger.info(f"\nCleaning old results for {method} method...")
        clean_old_results(dataset_name, method)
        
        # Generate counterfactuals for each sample
        all_results = []
        all_params = []
    
        for idx, row in originals_df.iterrows():
            sample_id = int(row['sample_id'])
            
            # Extract feature values (exclude sample_id, prediction, predicted_class)
            sample_features = row[csv_feature_names].values
            
            logger.info(f"\nSample {sample_id}/{len(originals_df)}:")
            logger.info(f"  Original prediction: {row['prediction']:.4f} (class {int(row['predicted_class'])})")
            
            # Generate counterfactuals
            results, params_list = generate_counterfactuals_for_sample(
                sample=sample_features,
                sample_id=sample_id,
                model=model,
                X_train=X_train,
                y_train=y_train,
                feature_names=csv_feature_names,  # Use CSV column names
                config=config,
                method=method
            )
            
            all_results.extend(results)
            all_params.append((sample_id, params_list))
    
        # Save results
        logger.info("\n" + "=" * 80)
        logger.info(f"SAVING RESULTS FOR {method.upper()} METHOD")
        logger.info("=" * 80)
        
        # Save priority parameters (always save these, even if no CFs found)
        if all_params:
            save_priority_parameters_csv(all_params, dataset_name, method)
        
        # Save counterfactuals and summary
        if all_results:
            save_counterfactuals_csv(all_results, csv_feature_names, dataset_name, method)
            save_summary_statistics(all_results, dataset_name, config, method)
        else:
            logger.warning("No counterfactuals generated. Saving summary with diagnostic info.")
            save_summary_statistics(all_results, dataset_name, config, method)
        
        # Print summary
        n_samples = len(originals_df)
        n_cfs = len(all_results)
        n_successful = sum(1 for r in all_results if r['target_achieved'])
        success_rate = (n_successful / n_cfs * 100) if n_cfs > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info(f"RESULTS SUMMARY - {method.upper()} METHOD")
        logger.info("=" * 80)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Method: {method}")
        logger.info(f"Samples processed: {n_samples}")
        logger.info(f"Counterfactuals generated: {n_cfs}")
        if n_cfs > 0:
            logger.info(f"Successful (target achieved): {n_successful} ({success_rate:.1f}%)")
        else:
            logger.warning("No counterfactuals found. Consider:")
            if method == 'continuous':
                logger.warning(f"  - Increasing epsilon (current: {config['epsilon']:.4f})")
            logger.warning(f"  - Increasing max_iterations (current: {config['max_iterations']})")
            logger.warning(f"  - Adjusting exemplar_weight (current: {config['exemplar_weight']:.4f})")
        logger.info("=" * 80)
        
        # Calculate median iterations per sample
        if all_results:
            # Group by sample_id and get median iterations
            sample_iterations = {}
            for result in all_results:
                sid = result['sample_id']
                if sid not in sample_iterations:
                    sample_iterations[sid] = []
                sample_iterations[sid].append(result['iteration_found'])
            
            # Calculate median for each sample
            sample_medians = [np.median(iters) for iters in sample_iterations.values()]
            overall_median = np.median(sample_medians) if sample_medians else 0
        else:
            overall_median = 0
        
        # Store method results
        method_results[method] = {
            'n_samples': n_samples,
            'n_cfs': n_cfs,
            'n_successful': n_successful,
            'success_rate': success_rate,
            'median_iterations': overall_median
        }
    
    # Return detailed results
    return {
        'success': any(m['n_cfs'] > 0 for m in method_results.values()),
        'methods': method_results
    }


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("PREFERENCE-BASED COUNTERFACTUAL METHOD TEST")
    logger.info("=" * 80)
    
    # Display configuration
    logger.info("\nConfiguration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Determine datasets to test
    if CONFIG['datasets'] == 'all':
        datasets = ['communities_crime', 'german_credit']
    elif isinstance(CONFIG['datasets'], list):
        datasets = CONFIG['datasets']
    else:
        datasets = [CONFIG['datasets']]
    
    # Test each dataset
    results = {}
    for dataset in datasets:
        dataset_name = DATASET_NAMES.get(dataset, dataset)
        result = test_preference_method_on_dataset(dataset_name, CONFIG)
        results[dataset_name] = result
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETE")
    logger.info("=" * 80 + "\n")
    
    for dataset_name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        logger.info(f"Dataset: {dataset_name} - {status}")
        
        for method, stats in result['methods'].items():
            logger.info(f"  {method.capitalize()} Method:")
            logger.info(f"    Samples: {stats['n_samples']} | "
                       f"CFs Found: {stats['n_cfs']} | "
                       f"Target Achieved: {stats['n_successful']} ({stats['success_rate']:.1f}%)")
            logger.info(f"    Median Iterations/Sample: {stats['median_iterations']:.0f} | "
                       f"Max Iterations: {CONFIG['max_iterations']}")
        logger.info("")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
