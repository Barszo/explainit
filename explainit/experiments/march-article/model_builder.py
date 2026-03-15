"""
Model Builder for Adversarial Counterfactual Experiments

Implements the neural network architecture as specified in the paper:
- Feed-forward neural network
- 4 hidden layers with 200 nodes each
- tanh activation function
- Adam optimizer
- Binary cross-entropy loss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


def create_adversarial_model(input_dim, hidden_layers=4, hidden_units=200, 
                            activation='tanh', learning_rate=0.001):
    """
    Create the feed-forward neural network as specified in the paper.
    
    Architecture from paper:
    - 4 layers of 200 nodes
    - tanh activation function
    - Adam optimizer
    - Binary cross-entropy loss
    
    Args:
        input_dim: Number of input features
        hidden_layers: Number of hidden layers (default: 4)
        hidden_units: Number of units per hidden layer (default: 200)
        activation: Activation function (default: 'tanh')
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers (4 layers of 200 nodes with tanh)
    for i in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation=activation, 
                              name=f'hidden_{i+1}'))
    
    # Output layer (sigmoid for binary classification)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    # Compile with Adam optimizer and binary cross-entropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_baseline_model(input_dim):
    """
    Create baseline (unmodified) model with same architecture.
    
    Returns:
        Compiled Keras model
    """
    return create_adversarial_model(
        input_dim=input_dim,
        hidden_layers=4,
        hidden_units=200,
        activation='tanh',
        learning_rate=0.001
    )


def train_baseline_model(model, X_train, y_train, X_val, y_val, 
                        steps=50, batch_size=32, verbose=1):
    """
    Train baseline model for 50 optimization steps as per paper.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        steps: Number of training steps (default: 50 as per paper)
        batch_size: Batch size
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    # Calculate epochs from steps
    # steps = (num_samples / batch_size) * epochs
    # epochs = steps * batch_size / num_samples
    num_samples = len(X_train)
    epochs = max(1, int(steps * batch_size / num_samples))
    
    print(f"Training baseline model for {steps} steps (~{epochs} epochs)")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )
    
    return history


def get_model_summary(model):
    """Print and return model summary statistics."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    return total_params


def evaluate_model(model, X_test, y_test, dataset_name=""):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of dataset for display
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"MODEL EVALUATION{' - ' + dataset_name if dataset_name else ''}")
    print(f"{'='*80}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    return results


def save_model(model, filepath):
    """Save model to file."""
    model.save(filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """Load model from file."""
    model = keras.models.load_model(filepath)
    print(f"Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    print("Testing model builder...")
    
    # Create a test model
    input_dim = 99  # Communities and Crime has 99 features
    model = create_adversarial_model(input_dim)
    
    # Print summary
    get_model_summary(model)
    
    # Test with random data
    X_test = np.random.randn(100, input_dim)
    y_test = np.random.randint(0, 2, 100)
    
    print("\nTesting prediction...")
    predictions = model.predict(X_test[:5], verbose=0)
    print(f"Sample predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.flatten()}")
    
    print("\nModel builder working correctly!")
