# --- Lending Club Preprocessing ---
def preprocess_lending_club_data(csv_path):
    """
    Preprocess Lending Club dataset into 8 continuous features and a binary target.

    Features (columns 1-8):
        annual_inc, open_acc, emp_length_years, credit_history_years,
        grade_score, home_score, purpose_score, state_score
    Target (column 9):
        0 = Fully Paid, 1 = Charged Off / Default

    Categorical features (grade, home_ownership, purpose, addr_state) are
    converted to continuous risk scores via target encoding.

    Args:
        csv_path: Path to LoanStats3a.csv (first row is a notes row; skiprows=1 is applied)

    Returns:
        X: DataFrame with 8 continuous features
        y: Series with binary target
    """
    df = pd.read_csv(csv_path, low_memory=False)

    cols = [
        "emp_length",
        "annual_inc",
        "open_acc",
        "earliest_cr_line",
        "grade",
        "home_ownership",
        "purpose",
        "addr_state",
        "loan_status",
    ]
    df = df[cols]

    # Keep only loans with a final outcome
    valid_status = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(valid_status)].copy()

    # Binary target: 0 = repaid, 1 = defaulted
    df["target"] = df["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)

    # Convert emp_length to numeric years
    def emp_to_years(emp):
        if pd.isna(emp):
            return np.nan
        emp = str(emp)
        if "<" in emp:
            return 0
        if "+" in emp:
            return 10
        return int(emp.split()[0])

    df["emp_length_years"] = df["emp_length"].apply(emp_to_years)

    # Convert earliest_cr_line to credit history in years (reference: end of dataset 2011)
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_years"] = 2011 - df["earliest_cr_line"].dt.year

    # Remove unrealistic credit histories
    df = df[df["credit_history_years"] <= 50]

    # Drop rows with any remaining missing values
    df = df.dropna()

    # Target-encode categorical features into continuous risk scores
    def target_encode(data, col, target):
        mapping = data.groupby(col)[target].mean()
        return data[col].map(mapping)

    df["grade_score"] = target_encode(df, "grade", "target")
    df["home_score"] = target_encode(df, "home_ownership", "target")
    df["purpose_score"] = target_encode(df, "purpose", "target")
    df["state_score"] = target_encode(df, "addr_state", "target")

    features = [
        "annual_inc",
        "open_acc",
        "emp_length_years",
        "credit_history_years",
        "grade_score",
        "home_score",
        "purpose_score",
        "state_score",
    ]

    X = df[features]
    y = df["target"]

    return X, y


# --- Lending Club Model Training ---
def train_lending_club_model(X_train, y_train, X_val, y_val, steps=50, batch_size=32, verbose=1):
    """
    Train baseline model for Lending Club dataset.
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        steps: Number of training steps (default: 50)
        batch_size: Batch size
        verbose: Verbosity level
    Returns:
        Trained model, training history
    """
    input_dim = X_train.shape[1]
    model = create_baseline_model(input_dim)
    num_samples = len(X_train)
    epochs = max(1, int(steps * batch_size / num_samples))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )
    return model, history
"""
Model Builder for Adversarial Counterfactual Experiments

Implements the neural network architecture as specified in the paper:
- Feed-forward neural network
- 4 hidden layers with 200 nodes each
- tanh activation function
- Adam optimizer
- Binary cross-entropy loss
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


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
