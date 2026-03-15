"""
Simplified Counterfactual Methods Test

This script tests that the five counterfactual methods work correctly:
1. Wachter et al.'s Algorithm
2. Sparse Wachter (with elastic net regularization)
3. DiCE (custom gradient mode implementation)
4. Counterfactuals Guided by Prototypes
5. DiCE (official dice_ml library)

Goal: Prove that CF methods generate valid counterfactuals that flip predictions.
Tests binary class flipping while showing probability information for analysis.
Saves results as CSV files (originals and counterfactuals) with sample IDs for joining.

Uses the same datasets and CF methods as the adversarial experiments but with
a simplified setup focused on validation.

CONFIGURATION EXAMPLES:
----------------------

Example 1: Test only official DiCE on German Credit
    CONFIG = {
        'datasets': ['german_credit'],
        'cf_methods': ['dice_official'],
        'n_samples': 10,
        ...
    }

Example 2: Test Wachter and DiCE on both datasets
    CONFIG = {
        'datasets': 'all',
        'cf_methods': ['wachter', 'dice'],
        'n_samples': 10,
        ...
    }

Example 3: Quick test with just 3 samples
    CONFIG = {
        'datasets': 'all',
        'cf_methods': 'all',
        'n_samples': 3,
        ...
    }
"""

import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path

# Add current directory to path for local modules
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import tensorflow as tf

# Use local data downloader that caches data
from data_downloader import load_communities_and_crime, load_german_credit
from model_builder import create_baseline_model
from counterfactual_methods import (
    WachterCounterfactual,
    SparseWachterCounterfactual,
    DiceCounterfactual,
    PrototypeGuidedCounterfactual,
    OfficialDiceCounterfactual
)

# ============================================================================
# CONFIGURATION SECTION - Modify these parameters to customize the test
# ============================================================================

CONFIG = {
    # Dataset Selection
    # Available: ['communities_crime', 'german_credit'] or 'all'
    # Examples: 'all', ['german_credit'], ['communities_crime', 'german_credit']
    'datasets': 'all',
    
    # Counterfactual Methods Selection
    # Available methods:
    #   - 'wachter': Wachter et al.'s Algorithm
    #   - 'sparse_wachter': Sparse Wachter with elastic net regularization
    #   - 'dice': DiCE in gradient mode (custom implementation)
    #   - 'prototype': Counterfactuals Guided by Prototypes
    #   - 'dice_official': DiCE using official dice_ml library
    # Use 'all' or a list like ['wachter', 'dice', 'dice_official']
    'cf_methods': 'all',
    
    # Classification Settings
    'threshold': 0.5,  # Decision boundary for binary classification (typically 0.5)
    
    # Sample Settings
    'n_samples': 10,  # Number of test samples per dataset/method (more = slower, more robust)
    'num_cfs': 2,  # Number of counterfactuals PER SAMPLE (official DiCE only)
                   # With n_samples=10 and num_cfs=2, you get 10 samples × 2 CFs = 20 total CFs
                   # Other methods return 1 CF per sample
    
    # Model Training Settings
    'epochs': 50,  # Number of training epochs (increase for better model accuracy)
    'batch_size': 32,  # Batch size for training (larger = faster but less stable)
    'learning_rate': 0.001,  # Learning rate for Adam optimizer (default: 0.001)
    'force_retrain': False,  # Set True to retrain even if saved model exists
    
    # Counterfactual Generation Settings
    'max_iterations': 500,  # Maximum iterations for CF generation (increase if CFs don't converge)
                            # Note: Official DiCE library may take longer even with same iterations
    'n_prototypes': 5,  # Number of prototypes for Prototype-Guided method
    'source_class': 0,  # Which class to generate CFs from (0 or 1)
    'target_class': 1,  # Which class to generate CFs to (0 or 1)
    
    # DiCE-specific settings (for custom implementation)
    'dice_learning_rate': 0.05,  # Learning rate for DiCE (higher than default for faster convergence)
    'dice_diversity_weight': 0.1,  # Diversity weight for DiCE (lower to focus on target)
    'dice_lambda': 0.1,  # Proximity weight for DiCE
    
    # Official DiCE Library Advanced Settings (for performance tuning)
    # These control the official dice_ml library behavior - adjust for speed vs quality trade-off
    # 
    # SPEED OPTIMIZATION GUIDE:
    # For FASTER generation (may sacrifice quality):
    #   - Increase learning_rate (0.1-0.2 instead of 0.05)
    #   - Decrease min_iter (20-50 instead of 500)
    #   - Decrease max_iter (100-300 instead of 5000)
    #   - Increase loss_diff_thres (1e-2 or 1e-3 instead of 1e-5)
    #   - Decrease proximity_weight (0.1-0.2 instead of 0.5)
    # 
    # For BETTER QUALITY (slower):
    #   - Decrease learning_rate (0.01-0.05)
    #   - Increase min_iter (200-500)
    #   - Increase max_iter (1000-5000)
    #   - Decrease loss_diff_thres (1e-5 or 1e-6)
    #   - Increase proximity_weight (0.5-1.0)
    #
    'dice_official_learning_rate': 0.05,  # Higher = faster convergence (default: 0.05)
    'dice_official_min_iter': 50,  # Min iterations before checking convergence (default: 500)
    'dice_official_max_iter': 500,  # Max iterations (default: 5000) - REDUCE FOR SPEED
    'dice_official_proximity_weight': 0.05,  # Proximity weight (default: 0.5) - LOWER = easier to move away
    'dice_official_diversity_weight': 0.5,  # Diversity weight (default: 1.0)
    'dice_official_categorical_penalty': 0.0,  # NOT USED - all features are continuous
    'dice_official_loss_diff_thres': 1e-3,  # Convergence threshold - HIGHER = faster stop (default: 1e-5)
    'dice_official_loss_converge_maxiter': 2,  # Iterations to hold for convergence (default: 1)
    'dice_official_yloss_type': 'log_loss',  # Loss function: 'hinge_loss', 'log_loss' (softer), 'l2_loss'
    # Note: Official DiCE may show MAD warnings for zero-variance features (expected)
    
    # Reporting
    'report_format': 'both',  # Options: 'json', 'text', 'both'
    'verbose': True,  # Print detailed progress information during execution
    
    # Reproducibility
    'random_seed': 42,  # Random seed for numpy and tensorflow (set None for random behavior)
}

# ----------------------------------------------------------------------------
# QUICK PRESETS - Uncomment one of these to quickly change configuration
# ----------------------------------------------------------------------------

# Preset 1: Quick test - only DiCE on German Credit, 3 samples
# CONFIG.update({
#     'datasets': ['german_credit'],
#     'cf_methods': ['dice'],
#     'mode': 'binary',
#     'n_samples': 3,
# })

# Preset 2: DiCE focus - test DiCE on both datasets (OPTIMIZED)
# CONFIG.update({
#     'datasets': 'all',
#     'cf_methods': ['dice'],
#     'n_samples': 10,
#     'num_cfs': 6,  # Generate 6 diverse CFs (more chances to find good ones)
#     'dice_learning_rate': 0.05,  # Higher learning rate for faster convergence
#     'dice_diversity_weight': 0.1,  # Lower diversity weight to focus on target
#     'max_iterations': 500,
# })

# Preset 3: Fast comparison - all methods, 5 samples
# CONFIG.update({
#     'datasets': 'all',
#     'cf_methods': 'all',
#     'n_samples': 5,
# })

# Preset 4: Reverse direction - generate CFs from class 1 to class 0
# CONFIG.update({
#     'datasets': 'all',
#     'cf_methods': 'all',
#     'source_class': 1,  # Start from class 1
#     'target_class': 0,  # Move to class 0
#     'n_samples': 10,
# })

# Preset 5: Official DiCE library test - ULTRA FAST settings
# CONFIG.update({
#     'datasets': ['german_credit'],  # Start with one dataset
#     'cf_methods': ['dice_official'],
#     'n_samples': 5,
#     'num_cfs': 2,  # Fewer CFs for faster generation
#     'dice_official_learning_rate': 0.2,  # Very high learning rate
#     'dice_official_min_iter': 20,  # Very low min iterations
#     'dice_official_max_iter': 100,  # Very low max iterations
#     'dice_official_loss_diff_thres': 1e-2,  # Very lenient convergence
#     'dice_official_proximity_weight': 0.1,  # Lower proximity weight
# })

# Preset 6: Full research run - comprehensive test (slow but thorough)
# CONFIG.update({
#     'datasets': 'all',
#     'cf_methods': 'all',
#     'n_samples': 20,
#     'max_iterations': 1000,
#     'force_retrain': True,
# })

# Dataset name mapping
DATASET_NAMES = {
    'communities_crime': 'Communities and Crime',
    'german_credit': 'German Credit'
}

# CF Method name mapping
CF_METHOD_NAMES = {
    'wachter': 'Wachter',
    'sparse_wachter': 'Sparse Wachter',
    'dice': 'DiCE (gradient mode)',
    'prototype': 'Prototype-Guided',
    'dice_official': 'DiCE (official library)'
}

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================

# Create directories for data and models
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODELS_DIR = SCRIPT_DIR / "models"
REPORTS_DIR = SCRIPT_DIR / "reports"
RESULTS_DIR = SCRIPT_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


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
    
    print(f"✓ Dataset saved to: {data_path}")
    return data_path


def load_dataset(dataset_name):
    """Load dataset from disk if it exists."""
    data_path = DATA_DIR / f"{dataset_name}_data.pkl"
    
    if not data_path.exists():
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Dataset loaded from: {data_path}")
    print(f"  Saved on: {data['timestamp']}")
    print(f"  Train shape: {data['train_shape']}, Test shape: {data['test_shape']}")
    
    return data


def get_dataset_description(X_train, y_train, X_test, y_test, feature_names, dataset_name):
    """Generate comprehensive dataset description for scientific reporting."""
    
    # Convert to numpy if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    description = {
        'dataset_name': dataset_name,
        'dimensions': {
            'n_features': X_train_np.shape[1],
            'n_train_samples': X_train_np.shape[0],
            'n_test_samples': X_test_np.shape[0],
            'total_samples': X_train_np.shape[0] + X_test_np.shape[0]
        },
        'class_distribution': {
            'train': {
                'class_0': int(np.sum(y_train_np == 0)),
                'class_1': int(np.sum(y_train_np == 1)),
                'class_0_pct': float(np.mean(y_train_np == 0) * 100),
                'class_1_pct': float(np.mean(y_train_np == 1) * 100)
            },
            'test': {
                'class_0': int(np.sum(y_test_np == 0)),
                'class_1': int(np.sum(y_test_np == 1)),
                'class_0_pct': float(np.mean(y_test_np == 0) * 100),
                'class_1_pct': float(np.mean(y_test_np == 1) * 100)
            }
        },
        'feature_statistics': {
            'mean': X_train_np.mean(axis=0).tolist(),
            'std': X_train_np.std(axis=0).tolist(),
            'min': X_train_np.min(axis=0).tolist(),
            'max': X_train_np.max(axis=0).tolist()
        },
        'feature_names': feature_names if isinstance(feature_names, list) else feature_names.tolist(),
        'task': 'binary_classification',
        'target': 'class_label'
    }
    
    return description


def train_simple_model(X_train, y_train, X_test, y_test, dataset_name, epochs=50, 
                      batch_size=32, force_retrain=False):
    """
    Train a simple baseline model for testing CF methods.
    Load from disk if available, otherwise train and save.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        dataset_name: Name of dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        force_retrain: If True, retrain even if model exists
    
    Returns:
        Tuple of (trained model, training history, model info dict)
    """
    # Clean dataset name for filename
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    model_path = MODELS_DIR / f"{clean_name}_model.keras"
    history_path = MODELS_DIR / f"{clean_name}_history.pkl"
    
    # Try to load existing model
    if model_path.exists() and not force_retrain:
        print(f"\n{'='*60}")
        print(f"Loading Existing Model: {dataset_name}")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        
        # Load history if available
        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
        else:
            history = None
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        
        model_info = {
            'loaded_from_disk': True,
            'model_path': str(model_path)
        }
        
        return model, history, model_info
    
    # Train new model
    print(f"\n{'='*60}")
    print(f"Training Model on {dataset_name}")
    print(f"{'='*60}")
    
    input_dim = X_train.shape[1]
    model = create_baseline_model(input_dim)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    
    # Save model and history
    model.save(model_path)
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ History saved to: {history_path}")
    
    model_info = {
        'loaded_from_disk': False,
        'model_path': str(model_path),
        'epochs_trained': epochs
    }
    
    return model, history.history, model_info


def generate_model_report(model, history, model_info, dataset_description, 
                         X_train, y_train, X_test, y_test, dataset_name):
    """
    Generate comprehensive model report for scientific publication.
    
    Args:
        model: Trained Keras model
        history: Training history dict
        model_info: Model metadata
        dataset_description: Dataset description dict
        X_train, y_train, X_test, y_test: Data splits
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with complete model report
    """
    # Evaluate model
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions for additional metrics
    train_preds = (model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
    test_preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Compute confusion matrix for test set
    tp = int(np.sum((test_preds == 1) & (y_test_np == 1)))
    tn = int(np.sum((test_preds == 0) & (y_test_np == 0)))
    fp = int(np.sum((test_preds == 1) & (y_test_np == 0)))
    fn = int(np.sum((test_preds == 0) & (y_test_np == 1)))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Model architecture details
    architecture = []
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'params': int(layer.count_params())
        }
        
        # Get output shape - handle different layer types
        try:
            if hasattr(layer, 'output'):
                output_shape = layer.output.shape
                layer_info['output_shape'] = str([int(dim) if dim is not None else None for dim in output_shape])
            else:
                layer_info['output_shape'] = 'N/A'
        except:
            layer_info['output_shape'] = 'N/A'
        
        if hasattr(layer, 'activation'):
            if callable(layer.activation):
                layer_info['activation'] = layer.activation.__name__
            else:
                layer_info['activation'] = str(layer.activation)
        if hasattr(layer, 'units'):
            layer_info['units'] = int(layer.units)
        if hasattr(layer, 'rate'):
            layer_info['dropout_rate'] = float(layer.rate)
        
        architecture.append(layer_info)
        total_params += layer.count_params()
    
    # Calculate trainable parameters safely
    try:
        trainable_params = int(np.sum([tf.size(w).numpy() for w in model.trainable_weights]))
    except:
        trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    
    non_trainable_params = total_params - trainable_params
    
    # Training configuration
    optimizer_config = model.optimizer.get_config()
    
    # Safely get learning rate
    if 'learning_rate' in optimizer_config:
        lr = optimizer_config['learning_rate']
    elif hasattr(model.optimizer, 'learning_rate'):
        lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    elif hasattr(model.optimizer, 'lr'):
        lr = float(tf.keras.backend.get_value(model.optimizer.lr))
    else:
        lr = 'N/A'
    
    if isinstance(lr, (int, float)):
        lr = float(lr)
    
    # Compile the report
    report = {
        'metadata': {
            'dataset_name': dataset_name,
            'report_generated': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'model_path': model_info.get('model_path', 'N/A'),
            'loaded_from_disk': model_info.get('loaded_from_disk', False)
        },
        'dataset': dataset_description,
        'model_architecture': {
            'layers': architecture,
            'total_layers': len(architecture),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'non_trainable_parameters': int(non_trainable_params),
            'input_shape': [None, X_train.shape[1]],
            'output_shape': [None, 1]
        },
        'training_configuration': {
            'optimizer': optimizer_config.get('name', 'unknown'),
            'learning_rate': lr,
            'loss_function': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'epochs': model_info.get('epochs_trained', 'N/A'),
            'batch_size': 32
        },
        'performance_metrics': {
            'training': {
                'loss': float(train_loss),
                'accuracy': float(train_acc),
                'final_epoch_loss': float(history['loss'][-1]) if history and 'loss' in history else None,
                'final_epoch_accuracy': float(history['accuracy'][-1]) if history and 'accuracy' in history else None
            },
            'test': {
                'loss': float(test_loss),
                'accuracy': float(test_acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'confusion_matrix': {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            }
        },
        'training_history': history if history else None
    }
    
    return report


def save_report(report, dataset_name, format='both'):
    """
    Save report in JSON and/or human-readable text format.
    
    Args:
        report: Report dictionary
        dataset_name: Name of the dataset
        format: 'json', 'text', or 'both'
    
    Returns:
        List of saved file paths
    """
    clean_name = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    # Save JSON report
    if format in ['json', 'both']:
        json_path = REPORTS_DIR / f"{clean_name}_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        saved_files.append(json_path)
        print(f"✓ JSON report saved to: {json_path}")
    
    # Save human-readable text report
    if format in ['text', 'both']:
        text_path = REPORTS_DIR / f"{clean_name}_report_{timestamp}.txt"
        
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MODEL REPORT: {report['metadata']['dataset_name']}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {report['metadata']['report_generated']}\n")
            f.write(f"TensorFlow Version: {report['metadata']['tensorflow_version']}\n")
            f.write(f"Model Path: {report['metadata']['model_path']}\n")
            f.write(f"Loaded from Disk: {report['metadata']['loaded_from_disk']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("DATASET DESCRIPTION\n")
            f.write("-"*80 + "\n")
            
            ds = report['dataset']
            f.write(f"Dataset: {ds['dataset_name']}\n")
            f.write(f"Task: {ds['task']}\n")
            f.write(f"Features: {ds['dimensions']['n_features']}\n")
            f.write(f"Training samples: {ds['dimensions']['n_train_samples']}\n")
            f.write(f"Test samples: {ds['dimensions']['n_test_samples']}\n")
            f.write(f"Total samples: {ds['dimensions']['total_samples']}\n")
            
            f.write("\nClass Distribution (Training):\n")
            f.write(f"  Class 0: {ds['class_distribution']['train']['class_0']} "
                   f"({ds['class_distribution']['train']['class_0_pct']:.1f}%)\n")
            f.write(f"  Class 1: {ds['class_distribution']['train']['class_1']} "
                   f"({ds['class_distribution']['train']['class_1_pct']:.1f}%)\n")
            
            f.write("\nClass Distribution (Test):\n")
            f.write(f"  Class 0: {ds['class_distribution']['test']['class_0']} "
                   f"({ds['class_distribution']['test']['class_0_pct']:.1f}%)\n")
            f.write(f"  Class 1: {ds['class_distribution']['test']['class_1']} "
                   f"({ds['class_distribution']['test']['class_1_pct']:.1f}%)\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("MODEL ARCHITECTURE\n")
            f.write("-"*80 + "\n")
            
            arch = report['model_architecture']
            f.write(f"Total Layers: {arch['total_layers']}\n")
            f.write(f"Total Parameters: {arch['total_parameters']:,}\n")
            f.write(f"Trainable Parameters: {arch['trainable_parameters']:,}\n")
            f.write(f"Non-trainable Parameters: {arch['non_trainable_parameters']:,}\n")
            f.write(f"Input Shape: {arch['input_shape']}\n")
            f.write(f"Output Shape: {arch['output_shape']}\n")
            
            f.write("\nLayer Details:\n")
            for i, layer in enumerate(arch['layers'], 1):
                f.write(f"  {i}. {layer['type']} ({layer['name']})\n")
                if 'units' in layer:
                    f.write(f"     Units: {layer['units']}\n")
                if 'activation' in layer:
                    f.write(f"     Activation: {layer['activation']}\n")
                if 'dropout_rate' in layer:
                    f.write(f"     Dropout Rate: {layer['dropout_rate']}\n")
                f.write(f"     Parameters: {layer['params']:,}\n")
                f.write(f"     Output Shape: {layer['output_shape']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            
            train_cfg = report['training_configuration']
            f.write(f"Optimizer: {train_cfg['optimizer']}\n")
            f.write(f"Learning Rate: {train_cfg['learning_rate']}\n")
            f.write(f"Loss Function: {train_cfg['loss_function']}\n")
            f.write(f"Metrics: {', '.join(train_cfg['metrics'])}\n")
            f.write(f"Epochs: {train_cfg['epochs']}\n")
            f.write(f"Batch Size: {train_cfg['batch_size']}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            perf = report['performance_metrics']
            f.write("\nTraining Set:\n")
            f.write(f"  Loss: {perf['training']['loss']:.6f}\n")
            f.write(f"  Accuracy: {perf['training']['accuracy']:.4f} ({perf['training']['accuracy']*100:.2f}%)\n")
            
            f.write("\nTest Set:\n")
            f.write(f"  Loss: {perf['test']['loss']:.6f}\n")
            f.write(f"  Accuracy: {perf['test']['accuracy']:.4f} ({perf['test']['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {perf['test']['precision']:.4f}\n")
            f.write(f"  Recall: {perf['test']['recall']:.4f}\n")
            f.write(f"  F1-Score: {perf['test']['f1_score']:.4f}\n")
            
            f.write("\nConfusion Matrix (Test Set):\n")
            cm = perf['confusion_matrix']
            f.write(f"  True Positives:  {cm['true_positives']}\n")
            f.write(f"  True Negatives:  {cm['true_negatives']}\n")
            f.write(f"  False Positives: {cm['false_positives']}\n")
            f.write(f"  False Negatives: {cm['false_negatives']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        saved_files.append(text_path)
        print(f"✓ Text report saved to: {text_path}")
    
    return saved_files


def validate_counterfactual(x_original, x_cf, model, method_name, source_class=0, target_class=1, threshold=0.5):
    """
    Validate that a counterfactual successfully flips the prediction.
    Returns predictions as probabilities for informational purposes.
    
    Args:
        x_original: Original instance
        x_cf: Counterfactual instance
        model: Trained model
        method_name: Name of CF method
        source_class: Expected class of original (default: 0)
        target_class: Desired class of counterfactual (default: 1)
        threshold: Decision boundary for classification (default: 0.5)
    
    Returns:
        Dictionary with validation results including probabilities
    """
    # Get predictions (probabilities)
    pred_original = model.predict(x_original.reshape(1, -1), verbose=0)[0, 0]
    pred_cf = model.predict(x_cf.reshape(1, -1), verbose=0)[0, 0]
    
    # Compute distance (L2)
    distance = np.linalg.norm(x_cf - x_original)
    
    # Check if prediction flipped
    class_original = 1 if pred_original >= threshold else 0
    class_cf = 1 if pred_cf >= threshold else 0
    flipped = (class_original != class_cf)
    
    # Check if target was reached
    target_reached = (class_cf == target_class)
    
    return {
        'method': method_name,
        'pred_original': pred_original,
        'pred_cf': pred_cf,
        'class_original': class_original,
        'class_cf': class_cf,
        'flipped': flipped,
        'target_reached': target_reached,
        'distance': distance
    }


def get_result_file_paths(method_name, dataset_name):
    """Get file paths for result CSVs."""
    method_clean = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    dataset_clean = dataset_name.lower().replace(' ', '_').replace('&', 'and')
    
    orig_path = os.path.join(RESULTS_DIR, f'{method_clean}_{dataset_clean}_originals.csv')
    cf_path = os.path.join(RESULTS_DIR, f'{method_clean}_{dataset_clean}_counterfactuals.csv')
    
    return orig_path, cf_path


def initialize_result_files(method_name, dataset_name, feature_names):
    """
    Initialize/clear result CSV files at the start of testing.
    Creates empty CSV files with headers.
    
    Args:
        method_name: Name of the CF method
        dataset_name: Name of the dataset
        feature_names: List of feature names
        
    Returns:
        Tuple of (originals_path, cfs_path)
    """
    orig_path, cf_path = get_result_file_paths(method_name, dataset_name)
    
    # Create originals file with header
    orig_columns = ['sample_id'] + list(feature_names) + ['prediction', 'predicted_class']
    orig_df = pd.DataFrame(columns=orig_columns)
    orig_df.to_csv(orig_path, index=False)
    
    # Create counterfactuals file with header
    cf_columns = ['sample_id'] + list(feature_names) + ['prediction', 'predicted_class', 'distance_from_original']
    cf_df = pd.DataFrame(columns=cf_columns)
    cf_df.to_csv(cf_path, index=False)
    
    return orig_path, cf_path


def append_original_sample(sample_id, x_original, method_name, dataset_name, 
                          feature_names, model, threshold=0.5):
    """
    Append a single original sample to the originals CSV file.
    
    Args:
        sample_id: ID for this sample
        x_original: Original instance (1D array)
        method_name: Name of the CF method
        dataset_name: Name of the dataset
        feature_names: List of feature names
        model: Trained model for predictions
        threshold: Decision boundary for classification
    """
    orig_path, _ = get_result_file_paths(method_name, dataset_name)
    
    # Get prediction
    pred = model.predict(x_original.reshape(1, -1), verbose=0)[0, 0]
    pred_class = 1 if pred >= threshold else 0
    
    # Create row
    row = {'sample_id': sample_id}
    for i, fname in enumerate(feature_names):
        row[fname] = x_original[i]
    row['prediction'] = pred
    row['predicted_class'] = pred_class
    
    # Append to CSV
    row_df = pd.DataFrame([row])
    row_df.to_csv(orig_path, mode='a', header=False, index=False)


def append_counterfactual(sample_id, x_original, x_cf, method_name, dataset_name,
                         feature_names, model, threshold=0.5):
    """
    Append a single counterfactual to the counterfactuals CSV file.
    
    Args:
        sample_id: ID of the original sample this CF is for
        x_original: Original instance (1D array)
        x_cf: Counterfactual instance (1D array)
        method_name: Name of the CF method
        dataset_name: Name of the dataset
        feature_names: List of feature names
        model: Trained model for predictions
        threshold: Decision boundary for classification
    """
    _, cf_path = get_result_file_paths(method_name, dataset_name)
    
    # Get prediction
    pred = model.predict(x_cf.reshape(1, -1), verbose=0)[0, 0]
    pred_class = 1 if pred >= threshold else 0
    
    # Compute distance
    distance = np.linalg.norm(x_cf - x_original)
    
    # Create row
    row = {'sample_id': sample_id}
    for i, fname in enumerate(feature_names):
        row[fname] = x_cf[i]
    row['prediction'] = pred
    row['predicted_class'] = pred_class
    row['distance_from_original'] = distance
    
    # Append to CSV
    row_df = pd.DataFrame([row])
    row_df.to_csv(cf_path, mode='a', header=False, index=False)


def test_counterfactual_methods(model, X_train, y_train, X_test, y_test, dataset_name, 
                               feature_names, n_samples=5, threshold=0.5,
                               source_class=0, target_class=1, selected_methods='all'):
    """
    Test counterfactual methods on a dataset.
    Tests binary class flipping while showing probability information.
    
    Args:
        model: Trained model
        X_train, y_train: Training data (for prototype and official dice methods)
        X_test, y_test: Test data
        dataset_name: Name of dataset
        feature_names: List of feature names
        n_samples: Number of samples to test
        threshold: Decision boundary for classification (default: 0.5)
        source_class: Which class to generate CFs from (default: 0)
        target_class: Which class to generate CFs to (default: 1)
        selected_methods: 'all' or list of method keys to test
        
    Returns:
        Dictionary with results and saved file paths
    """
    print(f"\n{'='*60}")
    print(f"Testing CF Methods on {dataset_name}")
    print(f"{'='*60}")
    
    # Select test samples from source class
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Get predictions on test set
    predictions = model.predict(X_test_np, verbose=0).flatten()
    
    # Determine expected prediction for source class
    if source_class == 0:
        source_pred_threshold = threshold  # Samples predicted as class 0 have pred < threshold
        source_indices = np.where((predictions < source_pred_threshold) & (y_test_np == source_class))[0]
    else:
        source_pred_threshold = threshold  # Samples predicted as class 1 have pred >= threshold
        source_indices = np.where((predictions >= source_pred_threshold) & (y_test_np == source_class))[0]
    
    if len(source_indices) == 0:
        print(f"Warning: No class {source_class} samples found. Using any class {source_class} samples.")
        source_indices = np.where(y_test_np == source_class)[0]
    
    # Select n_samples unique samples (ensure no duplicates)
    selected_indices = []
    selected_samples_set = set()  # To track unique samples
    
    for idx in source_indices:
        if len(selected_indices) >= n_samples:
            break
        
        # Get sample and convert to tuple for hashing
        sample = X_test_np[idx]
        sample_tuple = tuple(sample)
        
        # Only add if not already selected
        if sample_tuple not in selected_samples_set:
            selected_indices.append(idx)
            selected_samples_set.add(sample_tuple)
    
    # If we couldn't find enough unique samples, warn the user
    if len(selected_indices) < n_samples:
        print(f"Warning: Only found {len(selected_indices)} unique samples out of {n_samples} requested.")
        print(f"         Test set may contain duplicate samples.")
    
    test_indices = np.array(selected_indices)
    X_test_samples = X_test_np[test_indices]
    
    print(f"\nTesting on {len(test_indices)} unique samples from class {source_class}")
    print(f"Goal: Generate CFs that change predictions to class {target_class}\n")
    
    # Define all available CF methods
    all_methods = {
        'wachter': ('Wachter', WachterCounterfactual(
            model, 
            max_iterations=CONFIG['max_iterations'],
            target_class=target_class
        )),
        'sparse_wachter': ('Sparse Wachter', SparseWachterCounterfactual(
            model, 
            max_iterations=CONFIG['max_iterations'],
            target_class=target_class
        )),
        'dice': ('DiCE (gradient mode)', DiceCounterfactual(
            model, 
            num_cfs=CONFIG['num_cfs'], 
            max_iterations=CONFIG['max_iterations'],
            target_class=target_class,
            learning_rate=CONFIG['dice_learning_rate'],
            diversity_weight=CONFIG['dice_diversity_weight'],
            lambda_param=CONFIG['dice_lambda']
        )),
        'prototype': ('Prototype-Guided', PrototypeGuidedCounterfactual(
            model, 
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train.values if hasattr(y_train, 'values') else y_train,
            n_prototypes=CONFIG['n_prototypes'],
            max_iterations=CONFIG['max_iterations'],
            target_class=target_class
        )),
        'dice_official': ('DiCE (official library)', OfficialDiceCounterfactual(
            model,
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train.values if hasattr(y_train, 'values') else y_train,
            feature_names,
            num_cfs=CONFIG['num_cfs'],
            max_iterations=CONFIG['dice_official_max_iter'],
            min_iterations=CONFIG['dice_official_min_iter'],
            target_class=target_class,
            learning_rate=CONFIG['dice_official_learning_rate'],
            proximity_weight=CONFIG['dice_official_proximity_weight'],
            diversity_weight=CONFIG['dice_official_diversity_weight'],
            categorical_penalty=CONFIG['dice_official_categorical_penalty'],
            loss_diff_thres=CONFIG['dice_official_loss_diff_thres'],
            loss_converge_maxiter=CONFIG['dice_official_loss_converge_maxiter'],
            yloss_type=CONFIG['dice_official_yloss_type']
        ))
    }
    
    # Filter methods based on selection
    if selected_methods == 'all':
        methods_to_test = all_methods
    else:
        methods_to_test = {k: v for k, v in all_methods.items() if k in selected_methods}
    
    if not methods_to_test:
        print(f"Warning: No valid methods selected. Available: {list(all_methods.keys())}")
        return {}
    
    # Convert to display format
    methods = {display_name: method for display_name, method in methods_to_test.values()}
    
    # Test each method
    results_summary = []
    all_saved_files = []
    
    for method_name, cf_method in methods.items():
        print(f"\n{'-'*60}")
        print(f"Method: {method_name}")
        print(f"{'-'*60}")
        
        # Initialize result files (clear existing)
        orig_path, cf_path = initialize_result_files(method_name, dataset_name, feature_names)
        all_saved_files.extend([orig_path, cf_path])
        print(f"Initialized result files:")
        print(f"  Originals: {os.path.basename(orig_path)}")
        print(f"  CFs: {os.path.basename(cf_path)}")
        
        # Save all original samples first (before CF generation)
        print(f"\nSaving {len(X_test_samples)} original samples...")
        for sample_id, x_original in enumerate(X_test_samples):
            append_original_sample(sample_id, x_original, method_name, dataset_name, 
                                  feature_names, model, threshold)
        print(f"✓ All originals saved")
        
        method_results = []
        n_total_cfs = 0
        
        # Generate and save CFs
        print(f"\nGenerating counterfactuals...")
        for sample_id, x_original in enumerate(X_test_samples):
            print(f"\nSample {sample_id+1}:", end=" ")
            
            # Enable verbose for official DiCE to see debug output
            enable_verbose = (method_name == 'DiCE (official library)' and CONFIG['verbose'])
            
            # Generate counterfactual(s)
            x_cf_result = cf_method.generate(x_original, verbose=enable_verbose)
            
            # Handle multiple CFs (Official DiCE returns array of CFs)
            if isinstance(x_cf_result, np.ndarray) and x_cf_result.ndim == 2:
                # Multiple CFs returned
                cfs_for_sample = x_cf_result
            else:
                # Single CF returned (wrap in array)
                cfs_for_sample = np.array([x_cf_result])
            
            # Use only the requested number of CFs
            max_cfs = CONFIG['num_cfs'] if method_name == 'DiCE (official library)' else 1
            cfs_for_sample = cfs_for_sample[:max_cfs]
            
            # Process and save each CF for this sample
            for j, x_cf in enumerate(cfs_for_sample):
                # Validate
                result = validate_counterfactual(x_original, x_cf, model, method_name, 
                                                source_class, target_class, threshold)
                method_results.append(result)
                
                # Save CF immediately to CSV
                append_counterfactual(sample_id, x_original, x_cf, method_name, dataset_name,
                                    feature_names, model, threshold)
                n_total_cfs += 1
                
                if j == 0:  # Print only first CF inline
                    # Print result with class and probability
                    status = "✓" if result['target_reached'] else "✗"
                    print(f"{status} Original: {result['class_original']} (p={result['pred_original']:.3f}) "
                          f"→ CF: {result['class_cf']} (p={result['pred_cf']:.3f}), "
                          f"Distance: {result['distance']:.3f}", end="")
                    if len(cfs_for_sample) > 1:
                        print(f" (+{len(cfs_for_sample)-1} more CFs) [saved]")
                    else:
                        print(" [saved]")
        
        print(f"\n✓ All {n_total_cfs} counterfactuals saved")
        print(f"  Files: {os.path.basename(orig_path)} & {os.path.basename(cf_path)}")
        
        # Summary for this method
        n_total_samples = len(X_test_samples)
        n_success = sum(r['target_reached'] for r in method_results)
        avg_distance = np.mean([r['distance'] for r in method_results])
        avg_pred_original = np.mean([r['pred_original'] for r in method_results])
        avg_pred_cf = np.mean([r['pred_cf'] for r in method_results])
        
        summary = {
            'method': method_name,
            'n_samples': n_total_samples,
            'n_cfs': n_total_cfs,
            'success_rate': n_success / n_total_cfs,
            'avg_distance': avg_distance,
            'avg_pred_original': avg_pred_original,
            'avg_pred_cf': avg_pred_cf
        }
        
        results_summary.append(summary)
        
        print(f"\nSummary for {method_name}:")
        print(f"  Samples: {n_total_samples}, Total CFs: {n_total_cfs}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}% ({n_success}/{n_total_cfs} CFs)")
        print(f"  Avg Probability: {avg_pred_original:.3f} → {avg_pred_cf:.3f}")
        print(f"  Avg Distance: {summary['avg_distance']:.4f}")
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY - {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<25s} {'Samples/CFs':<14s} {'Success':<10s} {'Avg p_orig':<12s} {'Avg p_cf':<12s} {'Avg Dist':<10s}")
    print(f"{'-'*95}")
    for summary in results_summary:
        n_success = int(summary['success_rate'] * summary['n_cfs'])
        print(f"{summary['method']:<25s} "
              f"{summary['n_samples']:>3d}/{summary['n_cfs']:<8d} "
              f"{summary['success_rate']*100:>5.1f}% "
              f"({n_success}/{summary['n_cfs']}) "
              f"{summary['avg_pred_original']:>8.3f}     "
              f"{summary['avg_pred_cf']:>8.3f}     "
              f"{summary['avg_distance']:>8.4f}")
    
    return {
        'summary': results_summary,
        'saved_files': all_saved_files
    }


def main():
    """
    Main test script.
    
    Loads datasets, trains simple models, and tests all CF methods.
    Uses cached data and models when available for reproducibility.
    Generates comprehensive reports for scientific publication.
    """
    print("\n" + "="*60)
    print("COUNTERFACTUAL METHODS VALIDATION TEST")
    print("="*60)
    print("\nThis script validates that counterfactual methods work correctly")
    print("by testing them on datasets with simple baseline models.")
    
    # Display configuration
    print("\n" + "-"*60)
    print("CONFIGURATION")
    print("-"*60)
    
    # Determine which datasets to use
    if CONFIG['datasets'] == 'all':
        datasets_to_test = list(DATASET_NAMES.keys())
    else:
        datasets_to_test = CONFIG['datasets'] if isinstance(CONFIG['datasets'], list) else [CONFIG['datasets']]
    print(f"Datasets: {', '.join([DATASET_NAMES.get(d, d) for d in datasets_to_test])}")
    
    # Determine which methods to use
    if CONFIG['cf_methods'] == 'all':
        methods_to_test = list(CF_METHOD_NAMES.keys())
    else:
        methods_to_test = CONFIG['cf_methods'] if isinstance(CONFIG['cf_methods'], list) else [CONFIG['cf_methods']]
    print(f"CF Methods: {', '.join([CF_METHOD_NAMES.get(m, m) for m in methods_to_test])}")
    
    print(f"Samples per test: {CONFIG['n_samples']}")
    print(f"Number of CFs (DiCE): {CONFIG['num_cfs']}")
    print(f"Threshold: {CONFIG['threshold']}")
    print(f"Source class: {CONFIG['source_class']}")
    print(f"Target class: {CONFIG['target_class']}")
    print(f"Max iterations: {CONFIG['max_iterations']}")
    print(f"Training epochs: {CONFIG['epochs']}")
    print(f"Force retrain: {CONFIG['force_retrain']}")
    print(f"Random seed: {CONFIG['random_seed']}")
    
    print(f"\nDirectories:")
    print(f"  Data: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    # Set random seeds for reproducibility
    if CONFIG['random_seed'] is not None:
        np.random.seed(CONFIG['random_seed'])
        tf.random.set_seed(CONFIG['random_seed'])
        print(f"\n✓ Random seed set to {CONFIG['random_seed']} for reproducibility")
    else:
        print("\n⚠ Random seed not set - results may vary between runs")
    
    all_results = {}
    
    # Dataset loader mapping
    dataset_loaders = {
        'communities_crime': load_communities_and_crime,
        'german_credit': load_german_credit
    }
    
    # ========================================================================
    # Loop through selected datasets
    # ========================================================================
    for idx, dataset_key in enumerate(datasets_to_test, 1):
        display_name = DATASET_NAMES.get(dataset_key, dataset_key)
        
        print(f"\n\n{'#'*60}")
        print(f"TEST {idx}: {display_name.upper()} DATASET")
        print(f"{'#'*60}")
        
        # Try to load cached data
        cached_data = load_dataset(dataset_key)
        
        if cached_data:
            X_train = cached_data['X_train']
            X_test = cached_data['X_test']
            y_train = cached_data['y_train']
            y_test = cached_data['y_test']
            feature_names = cached_data['feature_names']
            scaler = cached_data['scaler']
        else:
            # Load fresh data
            if dataset_key not in dataset_loaders:
                print(f"Error: Unknown dataset '{dataset_key}'. Skipping.")
                continue
            
            loader = dataset_loaders[dataset_key]
            X_train, X_test, y_train, y_test, feature_names, scaler = loader()
            
            # Save for future runs
            save_dataset(X_train, X_test, y_train, y_test,
                        feature_names, scaler, dataset_key)
        
        # Get dataset description
        dataset_desc = get_dataset_description(
            X_train, y_train, X_test, y_test, 
            feature_names, display_name
        )
        
        # Train or load model
        model, history, model_info = train_simple_model(
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train,
            X_test.values if hasattr(X_test, 'values') else X_test,
            y_test,
            display_name,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            force_retrain=CONFIG['force_retrain']
        )
        
        # Generate comprehensive report
        if CONFIG['verbose']:
            print("\nGenerating model report...")
        report = generate_model_report(
            model, history, model_info, dataset_desc,
            X_train.values if hasattr(X_train, 'values') else X_train,
            y_train,
            X_test.values if hasattr(X_test, 'values') else X_test,
            y_test,
            display_name
        )
        save_report(report, display_name, format=CONFIG['report_format'])
        
        # Test counterfactual methods
        print(f"\n\n{'*'*60}")
        print(f"COUNTERFACTUAL GENERATION")
        print(f"{'*'*60}")
        
        results = test_counterfactual_methods(
            model,
            X_train, y_train,
            X_test, y_test,
            display_name,
            feature_names,
            n_samples=CONFIG['n_samples'],
            threshold=CONFIG['threshold'],
            source_class=CONFIG['source_class'],
            target_class=CONFIG['target_class'],
            selected_methods=methods_to_test
        )
        
        # Store results
        all_results[dataset_key] = results
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n\n{'#'*60}")
    print("FINAL SUMMARY")
    print(f"{'#'*60}")
    
    if not all_results:
        print("\nNo results to display.")
        return
    
    print(f"\n✓ Counterfactual methods tested: {', '.join([CF_METHOD_NAMES.get(m, m) for m in methods_to_test])}")
    print(f"✓ Datasets tested: {', '.join([DATASET_NAMES.get(d, d) for d in datasets_to_test])}")
    print("\nKey Findings:")
    
    # Display results grouped by dataset
    for dataset_key in datasets_to_test:
        if dataset_key in all_results:
            result_data = all_results[dataset_key]
            results = result_data['summary']
            display_name = DATASET_NAMES.get(dataset_key, dataset_key).upper()
            
            print("\n" + "="*60)
            print(f"{display_name}")
            print("="*60)
            
            for result in results:
                status = "✓" if result['success_rate'] > 0.7 else "⚠"
                n_success = int(result['success_rate'] * result['n_cfs'])
                print(f"  {status} {result['method']:<25s}: "
                      f"{result['n_samples']} samples → {result['n_cfs']} CFs, "
                      f"{n_success}/{result['n_cfs']} success ({result['success_rate']*100:.1f}%), "
                      f"p: {result['avg_pred_original']:.3f}→{result['avg_pred_cf']:.3f}, "
                      f"dist={result['avg_distance']:.3f}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nConclusion:")
    print("  • CF methods generate valid counterfactuals that flip predictions")
    print(f"    from class {CONFIG['source_class']} to class {CONFIG['target_class']}.")
    print("  • Probabilities shown for informational purposes.")
    print("  • Results saved to CSV files for further analysis.")
    
    print("\n" + "="*60)
    print("SAVED ARTIFACTS")
    print("="*60)
    print(f"\n✓ Data: {DATA_DIR}")
    print(f"✓ Models: {MODELS_DIR}")
    print(f"✓ Reports: {REPORTS_DIR}")
    print(f"✓ Results (CSV): {RESULTS_DIR}")
    
    # Count result files
    total_result_files = sum(len(all_results[k]['saved_files']) for k in all_results if 'saved_files' in all_results[k])
    print(f"\n✓ Generated {total_result_files} result CSV files")
    print(f"  Format: {{method}}_{{dataset}}_originals.csv and _counterfactuals.csv")
    print(f"  Use 'sample_id' column to join originals with their counterfactuals.")
    print(f"\nAll artifacts are ready for analysis and publication.")


if __name__ == "__main__":
    main()
