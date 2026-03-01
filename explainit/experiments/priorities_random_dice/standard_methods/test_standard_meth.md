# Testing Standard Counterfactual Methods - DiCE Focus

This document describes the test scripts for evaluating standard counterfactual explanation methods in the `priorities_random_dice` experiment. Currently focused on **DiCE (Diverse Counterfactual Explanations)** with framework for additional methods.

---

## **SUMMARY: What is standard_methods.py?**

✅ **BOTH a runnable script AND a library module**

**What you CAN do:**
1. **Run it to test ALL methods:** `python standard_methods.py`
2. **Import it in your Python code:** `from standard_methods import dice_counterfactual`
3. **Run individual method tests:** `python standard_methods/test_dice.py`

**Quick Commands:**
```bash
# ✅ Run ALL standard method tests (orchestrator mode)
cd explainit/experiments/priorities_random_dice
python standard_methods.py

# ✅ Run ONLY DiCE tests
cd explainit/experiments/priorities_random_dice/standard_methods
python test_dice.py

# ✅ Import functions in your own code
from standard_methods import dice_counterfactual, compute_metrics, run_all_methods
```

**What it does when you run it:**
1. Loads Auto MPG dataset and trained model
2. Selects 3 test samples (low, medium, high predictions)
3. Runs ALL methods on all sample→target combinations (6 experiments per method)
4. Saves results to `results/all_standard_methods_results.csv`
5. Prints summary statistics

---

## **IMPORTANT: Understanding the File Structure**

**`standard_methods.py` is BOTH a library AND a test orchestrator!**

**File Roles:**
- **`standard_methods.py`** = Library with counterfactual methods + Main test orchestrator
  - As library: Import functions like `dice_counterfactual()`
  - As script: Run `python standard_methods.py` to test ALL methods
- **`standard_methods/test_dice.py`** = Standalone DiCE test (runs only DiCE)
- **`standard_methods/test_standard_meth.md`** = Documentation (this file)

**Two Ways to Run Tests:**

**1. Run ALL methods (orchestrator mode):**
```bash
cd explainit/experiments/priorities_random_dice
python standard_methods.py
```
- Tests all available methods (currently: DiCE)
- Saves to `results/all_standard_methods_results.csv`

**2. Run ONE method (individual test):**
```bash
cd explainit/experiments/priorities_random_dice/standard_methods
python test_dice.py
```
- Tests only DiCE
- Saves to `results/dice_test_results.csv`

---

## **QUICK START: Using standard_methods.py in Your Code**

The `standard_methods.py` module provides reusable functions for generating counterfactual explanations. Here's how to use it in your own scripts:

### **Basic Usage**

```python
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import standard_methods
sys.path.insert(0, str(Path(__file__).parent.parent))
from standard_methods import dice_counterfactual, compute_metrics, run_all_methods

# Assuming you have:
# - X_original: numpy array of the instance to explain
# - model: TensorFlow/Keras model
# - target_value: desired prediction value
# - X_train_df: training data as DataFrame (must include outcome column)
# - feature_names: list of feature names

# Generate a single counterfactual with DiCE
cf, prediction, info = dice_counterfactual(
    X_original=X_original,
    model=model,
    target_value=target_value,
    X_train_df=X_train_df,
    feature_names=feature_names,
    epsilon=2.0,          # ±2 MPG tolerance
    total_CFs=5,          # Generate 5 diverse CFs
    method='random'       # Use random sampling method
)

# Check if successful
if cf is not None and info['valid']:
    print(f"✓ Valid counterfactual found!")
    print(f"  Prediction: {prediction:.2f}")
    print(f"  Distance: {info['distance']:.4f}")
    print(f"  Valid CFs: {info['n_valid']}/{info['n_generated']}")
    
    # Compute detailed metrics
    metrics = compute_metrics(X_original, cf, prediction, target_value, epsilon=2.0)
    print(f"  L2 Distance: {metrics['l2_distance']:.4f}")
    print(f"  Sparsity: {metrics['sparsity']} features changed")
else:
    print(f"✗ Failed: {info.get('reason', info.get('error', 'unknown'))}")
```

### **Running All Available Methods**

```python
# Run all implemented methods and compare
results = run_all_methods(
    X_original=X_original,
    model=model,
    target_value=target_value,
    X_train_df=X_train_df,
    feature_names=feature_names,
    epsilon=2.0,
    methods_to_run=['dice'],  # Currently only DiCE implemented
    dice_params={
        'total_CFs': 5,
        'method': 'random',
        'diversity_weight': 1.0,
        'proximity_weight': 0.5
    }
)

# Access results for each method
if results['dice']['counterfactual'] is not None:
    dice_cf = results['dice']['counterfactual']
    dice_metrics = results['dice']['metrics']
    print(f"DiCE: distance={dice_metrics['l2_distance']:.4f}, "
          f"sparsity={dice_metrics['sparsity']}")
```

### **Required Data Format**

**X_train_df (DataFrame):**
- Must contain all feature columns (matching `feature_names`)
- Must contain outcome column (e.g., 'mpg', 'outcome', or target variable name)
- Example:
  ```python
  # If your features are scaled
  X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
  X_train_df['mpg'] = y_train  # Add outcome column
  ```

**X_original (numpy array):**
- 1D array with same number of features as training data
- Should be scaled the same way as training data
- Example: `X_original = X_test_scaled[0]`

**Model:**
- Must be a TensorFlow/Keras model for DiCE
- Should accept numpy arrays and return predictions

### **Function Reference**

**`dice_counterfactual()`**
- **Returns:** `(counterfactual, prediction, info)`
  - `counterfactual`: numpy array (or None if failed)
  - `prediction`: float prediction value
  - `info`: dict with 'valid', 'distance', 'n_generated', 'n_valid', etc.

**`compute_metrics()`**
- **Returns:** dict with 'l1_distance', 'l2_distance', 'sparsity', 'validity', 'prediction_error'

**`run_all_methods()`**
- **Returns:** dict with results for each method
  - Keys: method names ('dice', etc.)
  - Values: dict with 'counterfactual', 'prediction', 'metrics', 'info'

### **Location**

```bash
# File location
explainit/experiments/priorities_random_dice/standard_methods.py

# Import in scripts from standard_methods/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from standard_methods import dice_counterfactual, compute_metrics

# Or use relative import
from ..standard_methods import dice_counterfactual
```

---

## **HOW TO RUN TEST SCRIPTS**

### **Option 1: Run ALL Methods at Once (Recommended)**

Use the orchestrator in `standard_methods.py` to run all available methods:

```bash
# From priorities_random_dice directory
cd explainit/experiments/priorities_random_dice
python standard_methods.py
```

```bash
# From project root
cd /Users/bartosz/projects/explainit/explainit_project
python explainit/experiments/priorities_random_dice/standard_methods.py
```

**What happens:**
- Loads Auto MPG data and model
- Selects 3 test samples (low, medium, high MPG)
- Tests ALL methods on all 6 sample→target combinations
- Saves results to `results/all_standard_methods_results.csv`
- Prints summary for each method

**Output location:** `explainit/experiments/priorities_random_dice/results/all_standard_methods_results.csv`

### **Option 2: Run Individual Method Tests**

Run specific method tests from the `standard_methods/` directory:

**Running DiCE Tests Only:**

From the standard_methods directory:
```bash
cd explainit/experiments/priorities_random_dice/standard_methods
python test_dice.py
```

From project root:
```bash
cd /Users/bartosz/projects/explainit/explainit_project
python explainit/experiments/priorities_random_dice/standard_methods/test_dice.py
```

**Output location:** `explainit/experiments/priorities_random_dice/standard_methods/results/dice_test_results.csv`

### **Which Option Should You Use?**

| Use Case | Command | Output | Best For |
|----------|---------|--------|----------|
| Compare all methods | `python standard_methods.py` | `results/all_standard_methods_results.csv` | Benchmarking, paper results |
| Debug/test specific method | `python standard_methods/test_dice.py` | `standard_methods/results/dice_test_results.csv` | Parameter tuning, debugging |
| Custom experiments | Import and use functions | Your own script | Research, custom analysis |

---

### **Expected Output**

The script will:
1. Load Auto MPG dataset and trained model
2. Select 3 test samples (low, medium, high predictions)
3. Run 6 experiments (all sample→target combinations)
4. Display results for each experiment in console
5. Save results to `standard_methods/results/dice_test_results.csv`

**Console output example:**
```
2026-03-01 10:30:45 - INFO - Loading Auto MPG dataset...
2026-03-01 10:30:46 - INFO - Data loaded: Train=313, Test=79
2026-03-01 10:30:46 - INFO - Selected test samples:
2026-03-01 10:30:46 - INFO -   Sample 1: Prediction=15.50 MPG, Actual=15.00 MPG
2026-03-01 10:30:46 - INFO -   Sample 2: Prediction=24.30 MPG, Actual=24.00 MPG
2026-03-01 10:30:46 - INFO -   Sample 3: Prediction=37.30 MPG, Actual=38.00 MPG

2026-03-01 10:30:46 - INFO - Experiment: Sample 1 → Target 2
2026-03-01 10:30:46 - INFO -   Sample prediction: 15.50 MPG
2026-03-01 10:30:46 - INFO -   Target prediction: 24.30 MPG
2026-03-01 10:30:50 - INFO -   Result: VALID
2026-03-01 10:30:50 - INFO -     CF Prediction: 24.10 MPG
2026-03-01 10:30:50 - INFO -     L2 Distance: 0.4523
...

2026-03-01 10:32:15 - INFO - Results saved to: .../results/dice_test_results.csv
```

### **Viewing Results**

**CSV file location:**
```bash
explainit/experiments/priorities_random_dice/standard_methods/results/dice_test_results.csv
```

**Quick view from terminal:**
```bash
# View entire CSV
cat explainit/experiments/priorities_random_dice/standard_methods/results/dice_test_results.csv

# View with column formatting
column -s, -t < explainit/experiments/priorities_random_dice/standard_methods/results/dice_test_results.csv

# Open in default CSV viewer (macOS)
open explainit/experiments/priorities_random_dice/standard_methods/results/dice_test_results.csv
```

### **Troubleshooting**

**Error: No module named 'dice_ml'**
```bash
pip install dice-ml
```

**Error: Cannot find model or data files**
- Ensure you're running from correct directory
- Check that ML_models_for_tests/ML_regression_results/Auto_MPG/ exists
- Verify the model file exists: `NN_Residual_model.h5`

**Error: TensorFlow/CUDA warnings**
- Ignore TensorFlow CPU/GPU warnings (script forces CPU mode)
- If concerned, set: `export CUDA_VISIBLE_DEVICES=-1`

**Exit Code 2 (ImportError or similar)**
```bash
# Activate virtual environment first
source /Users/bartosz/projects/explainit/explainit_project/.venv/bin/activate

# Install dependencies
pip install tensorflow dice-ml pandas numpy scikit-learn

# Then run again
python explainit/experiments/priorities_random_dice/standard_methods/test_dice.py
```

---

## **USING WITH YOUR OWN DATASET AND MODEL**

This section provides step-by-step instructions for adapting `standard_methods.py` to work with your own dataset and trained model.

### **Prerequisites**

Before starting, ensure you have:
1. ✅ A trained model (TensorFlow/Keras for DiCE, or compatible with other methods)
2. ✅ Training data (X_train, y_train) as CSV files or pandas DataFrames
3. ✅ Test data (X_test, y_test)
4. ✅ A feature scaler (if features were scaled during training)
5. ✅ List of feature names
6. ✅ Knowledge of which features are categorical vs continuous

---

### **Step 1: Prepare Your Data Files**

Organize your data following this structure:

```
your_project/
├── data/
│   ├── X_data.csv          # All features
│   ├── y_data.csv          # Target variable
│   ├── scaler.pkl          # Feature scaler (StandardScaler, MinMaxScaler, etc.)
│   └── trained_model.h5    # Your trained model (TensorFlow/Keras)
```

**Required file formats:**
- **X_data.csv**: Feature matrix (one row per sample, columns = features)
- **y_data.csv**: Target values (one column)
- **scaler.pkl**: Pickled sklearn scaler object
- **trained_model.h5**: Keras model file (or .keras, .pb for TensorFlow)

---

### **Step 2: Identify Feature Types**

**Crucial step:** You must manually specify which features are categorical.

**Method 1: By inspection**
```python
# Example: Adult Income dataset
feature_names = ['age', 'workclass', 'education', 'marital_status', 'occupation', 
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country']

# Indices of categorical features (workclass, education, marital_status, etc.)
categorical_features = [1, 2, 3, 4, 5, 6, 7, 11]  # Indices of categorical columns

# Continuous features are automatically inferred as all others
# continuous_features = [0, 8, 9, 10]  # age, capital_gain, capital_loss, hours_per_week
```

**Method 2: Check your data**
```python
import pandas as pd

X_data = pd.read_csv('data/X_data.csv')

# Print data types
print(X_data.dtypes)

# Identify categorical features
categorical_indices = []
for i, col in enumerate(X_data.columns):
    if X_data[col].dtype == 'object' or X_data[col].nunique() < 20:
        categorical_indices.append(i)
        print(f"Feature {i} ({col}) is likely categorical")
```

**Current Auto MPG example (all continuous):**
```python
feature_names = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model_year']
categorical_features = None  # or []  # No categorical features
```

---

### **Step 3: Create Data Loading Function**

Create a function similar to `load_auto_mpg_data()` for your dataset:

```python
def load_your_dataset():
    """
    Load your dataset and trained model for testing.
    
    Returns:
        X_train_df: Training data as DataFrame (with outcome column)
        X_test_df: Test data as DataFrame  
        y_train: Training predictions
        y_test: Test predictions
        model: Trained model
        scaler: Feature scaler
        feature_names: List of feature names
        categorical_features: List of categorical feature indices
    """
    import pandas as pd
    import pickle
    import tensorflow as tf
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    # Navigate to your data directory
    base_dir = Path(__file__).parent / "your_data_folder"
    
    logger.info(f"Loading dataset from {base_dir}...")
    
    # Load data
    X_data = pd.read_csv(base_dir / "X_data.csv")
    y_data = pd.read_csv(base_dir / "y_data.csv")
    
    # Load scaler
    with open(base_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Load trained model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (optional)
    model = tf.keras.models.load_model(
        base_dir / "trained_model.h5",
        compile=False
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Build the model (optional: depends on your model architecture)
    sample_input = tf.constant([[0.0] * len(X_data.columns)], dtype=tf.float32)
    _ = model(sample_input)
    
    # Split data (use same random_state as original training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    # Scale data
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Get feature names
    feature_names = list(X_data.columns)
    
    # IMPORTANT: Define categorical features for your dataset
    # Example for mixed dataset:
    categorical_features = [1, 2, 3, 5]  # Adjust to your dataset!
    # For all-continuous dataset:
    # categorical_features = None
    
    # Create dataframes (DiCE requires outcome column in training data)
    y_train_ravel = y_train.values.ravel()
    y_test_ravel = y_test.values.ravel()
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_train_df['outcome'] = y_train_ravel  # Add outcome column
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    logger.info(f"Data loaded: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Features: {len(feature_names)} total")
    if categorical_features:
        logger.info(f"  Categorical: {len(categorical_features)} features")
        logger.info(f"  Continuous: {len(feature_names) - len(categorical_features)} features")
    else:
        logger.info(f"  All features are continuous")
    
    return X_train_df, X_test_df, y_train_ravel, y_test_ravel, model, scaler, feature_names, categorical_features
```

---

### **Step 4: Update the Main Orchestrator Function**

Modify `run_all_standard_methods_tests()` to use your data:

**Find this section in `standard_methods.py`:**
```python
def run_all_standard_methods_tests():
    """Main orchestrator function to run all standard method tests."""
    
    # CHANGE THIS LINE:
    # X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names = load_auto_mpg_data()
    
    # TO THIS:
    X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names, categorical_features = load_your_dataset()
```

**Update epsilon and other parameters for your task:**
```python
    # Test parameters
    epsilon = 2.0  # ±2 MPG tolerance for Auto MPG
    
    # CHANGE TO:
    epsilon = 0.1  # For classification: ±0.1 probability
    # OR
    epsilon = 5.0  # For different regression scale
```

**Update article metrics computation:**
```python
    # Around line 775, find:
    article_metrics = compute_article_metrics(
        X_original=sample,
        all_cfs=all_cfs,
        all_predictions=all_predictions,
        target_value=target_pred,
        epsilon=epsilon,
        mad_values=mad_values,
        k=k_requested,
        categorical_features=None  # CHANGE THIS!
    )
    
    # TO:
    article_metrics = compute_article_metrics(
        X_original=sample,
        all_cfs=all_cfs,
        all_predictions=all_predictions,
        target_value=target_pred,
        epsilon=epsilon,
        mad_values=mad_values,
        k=k_requested,
        categorical_features=categorical_features  # Use your list!
    )
```

---

### **Step 5: Adjust DiCE Configuration (if using categorical features)**

If your dataset has categorical features, update the DiCE data preparation:

**Find in `dice_counterfactual()` function (around line 98):**
```python
d = dice_ml.Data(
    dataframe=X_train_df,
    continuous_features=feature_names,  # WRONG if you have categorical features!
    outcome_name=outcome_name
)
```

**Change to:**
```python
# Separate continuous and categorical feature names
if categorical_features:
    continuous_feature_names = [feature_names[i] for i in range(len(feature_names)) 
                                if i not in categorical_features]
    categorical_feature_names = [feature_names[i] for i in categorical_features]
else:
    continuous_feature_names = feature_names
    categorical_feature_names = []

d = dice_ml.Data(
    dataframe=X_train_df,
    continuous_features=continuous_feature_names,
    categorical_features=categorical_feature_names,  # Add this!
    outcome_name=outcome_name
)
```

---

### **Step 6: Test Your Configuration**

Before running the full orchestrator, test with a simple script:

```python
# test_your_config.py
from standard_methods import load_your_dataset, compute_article_metrics, dice_counterfactual
import numpy as np

# Load data
X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names, categorical_features = load_your_dataset()

print(f"✓ Data loaded successfully")
print(f"  Features: {feature_names}")
print(f"  Categorical indices: {categorical_features}")
print(f"  Train shape: {X_train_df.shape}")
print(f"  Test shape: {X_test_df.shape}")

# Test prediction
X_test = X_test_df.values
sample = X_test[0]
prediction = model.predict(np.array([sample]), verbose=0)[0]
print(f"\n✓ Model prediction works")
print(f"  Sample prediction: {prediction}")

# Test MAD calculation
from standard_methods import calculate_mad_from_training
X_train = X_train_df[feature_names].values
mad_values = calculate_mad_from_training(X_train)
print(f"\n✓ MAD calculation works")
print(f"  MAD values: {mad_values}")

# Test DiCE (optional)
print(f"\n✓ Configuration test complete!")
```

Run: `python test_your_config.py`

---

### **Step 7: Run the Full Test**

Once configuration is verified:

```bash
cd explainit/experiments/priorities_random_dice
python standard_methods.py
```

**Expected output:**
- Loads your dataset
- Selects test samples
- Runs DiCE on all sample→target pairs
- Computes both basic and article-based metrics
- Saves results to CSV files

---

### **Common Issues and Solutions**

#### **Issue 1: "No outcome column in training data"**
**Solution:** Ensure `X_train_df` has the outcome column:
```python
X_train_df['outcome'] = y_train_ravel  # or 'target', 'label', etc.
```

#### **Issue 2: "Categorical features have continuous values"**
**Solution:** If you encoded categorical features as integers but they're still categorical:
```python
# In your data loading, explicitly mark them
categorical_features = [1, 2, 3]  # Even if they're encoded as 0,1,2...
```

#### **Issue 3: "DiCE fails to generate counterfactuals"**
**Solution:** Check epsilon tolerance and adjust parameters:
```python
epsilon = 5.0  # Increase tolerance
dice_params = {
    'total_CFs': 10,        # Generate more CFs
    'method': 'genetic',    # Try different method
}
```

#### **Issue 4: "Model expects different input shape"**
**Solution:** Verify scaler and feature order match training:
```python
print(f"Expected features: {model.input_shape}")
print(f"Provided features: {X_train_scaled.shape}")
```

#### **Issue 5: "Metrics all zero or Infinite"**
**Solution:** Check if features were scaled correctly:
```python
# MAD values should be reasonable (0.1-10 for scaled features)
print(f"MAD values: {mad_values}")
print(f"Feature ranges: {X_train.min(axis=0)} to {X_train.max(axis=0)}")
```

---

### **Complete Example: Adult Income Dataset**

Here's a complete example adapting for Adult Income (classification):

```python
# In standard_methods.py, add this function:

def load_adult_income_data():
    """Load Adult Income dataset for testing."""
    base_dir = Path(__file__).parent.parent.parent / "ML_models_for_tests" / "binary_classifiers_results" / "Adult_Income"
    
    X_data = pd.read_csv(base_dir / "X_data.csv")
    y_data = pd.read_csv(base_dir / "y_data.csv")
    
    with open(base_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    model = tf.keras.models.load_model(base_dir / "NN_model.h5", compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    feature_names = list(X_data.columns)
    
    # Adult Income has these categorical features (after one-hot encoding):
    # workclass, education, marital_status, occupation, relationship, race, sex, native_country
    categorical_features = [1, 2, 3, 4, 5, 6, 7, 11]  # Adjust based on your encoding
    
    y_train_ravel = y_train.values.ravel()
    y_test_ravel = y_test.values.ravel()
    
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_train_df['income'] = y_train_ravel
    
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_df, X_test_df, y_train_ravel, y_test_ravel, model, scaler, feature_names, categorical_features

# In run_all_standard_methods_tests(), change to:
# X_train_df, X_test_df, y_train, y_test, model, scaler, feature_names, categorical_features = load_adult_income_data()
# epsilon = 0.1  # For classification probability
```

---

### **Checklist: Before Running**

- [ ] Data files prepared (X_data.csv, y_data.csv, scaler.pkl, model file)
- [ ] Feature names list defined
- [ ] Categorical feature indices identified and specified
- [ ] Data loading function created and tested
- [ ] Epsilon value adjusted for your task
- [ ] DiCE configuration updated (if using categorical features)
- [ ] Test configuration script runs successfully
- [ ] Main orchestrator function updated to use your data loader

Once all items are checked, you're ready to run `python standard_methods.py`!

---

## **1. DiCE METHOD (Diverse Counterfactual Explanations)**

**Test Script:** `test_dice.py`

### **Purpose**
Evaluate Microsoft's DiCE library for generating diverse counterfactual explanations. DiCE aims to provide multiple, diverse paths to achieve a desired prediction, helping users understand different ways to reach their goal.

**Key Feature:** Unlike single-CF methods, DiCE generates multiple counterfactuals with built-in diversity optimization.

### **Test Input**

**Data Selection:**
```python
# Automatically selects 3 quantile points from test set:
Sample 1: Low MPG prediction  (e.g., 15.50 MPG)
Sample 2: Medium MPG prediction (e.g., 24.30 MPG)
Sample 3: High MPG prediction (e.g., 37.30 MPG)

# Tests all possible sample → target combinations (6 scenarios):
Sample 1 → Target 2 (Low → Medium)
Sample 1 → Target 3 (Low → High)
Sample 2 → Target 1 (Medium → Low)
Sample 2 → Target 3 (Medium → High)
Sample 3 → Target 1 (High → Low)
Sample 3 → Target 2 (High → Medium)
```

**Why Test All Combinations?**

DiCE's performance varies based on:
- **Direction of change:** Some targets are easier to reach than others
- **Magnitude of change:** Larger prediction gaps may be harder
- **Feature space density:** Some regions may have more diverse solutions

Testing all combinations reveals which scenarios DiCE handles well and which are challenging.

### **Parameters Tested**

The current test uses fixed parameters (can be extended):

| Parameter | Current Value | Description |
|-----------|---------------|-------------|
| **epsilon** | 2.0 MPG | Tolerance for target prediction |
| **total_CFs** | 5 | Number of diverse counterfactuals to generate |
| **method** | 'random' | DiCE generation method |

**DiCE Method Options:**
- **'random'**: Randomly samples feature space - fast, reliable, good default
- **'genetic'**: Uses genetic algorithm - explores diverse solutions, slower
- **'gradient'**: Uses gradient-based optimization - best for neural networks, may be sensitive

**Parameters for Extended Testing** (can be added):

| Parameter | Suggested Values | Description |
|-----------|------------------|-------------|
| **epsilon** | [0.5, 1.0, 2.0, 3.0, 5.0] MPG | Tolerance for validity |
| **total_CFs** | [3, 5, 10, 15] | Number of CFs to generate |
| **method** | ['random', 'genetic', 'gradient'] | Generation algorithm |
| **diversity_weight** | [0.5, 1.0, 2.0, 5.0] | Weight for diversity vs other objectives |
| **proximity_weight** | [0.1, 0.5, 1.0, 2.0] | Weight for staying close to original |

### **Test Workflow**

1. **Data Loading**
   - Load Auto MPG dataset and trained neural network
   - Split into train/test sets (80/20, random_state=42)
   - Scale features using training data scaler

2. **Sample Selection**
   - Select 3 representative samples across prediction quantiles
   - Display prediction and actual MPG values

3. **Feature Range Extraction**
   - Compute min/max for each feature from training data
   - Used for constraining counterfactual generation

4. **DiCE Experiments**
   - For each sample-target pair (6 total):
     - Generate counterfactuals using DiCE
     - Select best CF (closest to original, within epsilon)
     - Compute quality metrics
     - Record results

5. **Results Aggregation**
   - Save detailed results to CSV
   - Compute summary statistics
   - Display success rates and averages

### **Output Format**

**Per Experiment:**
```
Experiment: Sample 1 → Target 2
  Sample prediction: 15.50 MPG
  Target prediction: 24.30 MPG
  Distance: 8.80 MPG
  Result: VALID
    CF Prediction: 24.10 MPG
    L2 Distance: 0.4523
    L1 Distance: 1.2341
    Sparsity: 3 features changed
    Prediction Error: 0.2000
    Generated CFs: 5
    Valid CFs: 3
    Diversity Score: 0.7234
```

**Summary Statistics:**
```
SUMMARY
════════════════════════════════════════════════════════════════
Total experiments: 6
Valid counterfactuals: 6
Success rate: 100.0%

Valid CF Statistics:
  Average L2 Distance: 0.4125
  Average L1 Distance: 1.1432
  Average Sparsity: 3.50
  Average Prediction Error: 0.3256
  Average Generated CFs: 5.00
  Average Valid CFs: 3.17

Results saved to: .../results/dice_test_results.csv
```

**CSV Output Columns:**
- `sample_idx`: Index of original sample (1-3)
- `target_idx`: Index of target sample (1-3)
- `sample_pred`: Original prediction
- `target_pred`: Target prediction
- `valid`: Whether CF is valid (within epsilon)
- `cf_pred`: Prediction of counterfactual
- `l2_distance`: Euclidean distance
- `l1_distance`: Manhattan distance
- `sparsity`: Number of features changed
- `pred_error`: |cf_pred - target_pred|
- `n_generated`: Total CFs generated
- `n_valid`: Number of valid CFs

### **Interpretation Guide**

**Success Indicators:**

✓ **High Success Rate (100%):** DiCE successfully finds counterfactuals for all scenarios
- Indicates robust performance across different prediction targets
- Model and data are well-suited for counterfactual generation

✓ **Multiple Valid CFs (n_valid > 1):** DiCE generates diverse alternatives
- Users have multiple options to achieve target
- Key advantage over single-CF methods

✓ **Low Prediction Error (<1.0):** Counterfactuals accurately hit target
- Precise control over outcome
- Good epsilon calibration

✓ **Reasonable Distance:** Counterfactuals are achievable changes
- Low distance = minimal changes needed
- Compare with other methods to assess competitiveness

**Quality Metrics:**

**L2 Distance (Euclidean):**
- **< 0.5:** Excellent - very close to original
- **0.5-1.0:** Good - reasonable changes
- **1.0-2.0:** Moderate - substantial changes
- **> 2.0:** Large - may be impractical

**Sparsity:**
- **1-2 features:** Excellent - simple, interpretable changes
- **3-4 features:** Good - manageable changes
- **5+ features:** Many changes - less interpretable

**Prediction Error:**
- **< 0.5:** Excellent precision
- **0.5-1.0:** Good (within half epsilon)
- **1.0-2.0:** Acceptable (within epsilon)
- **> 2.0:** Poor - exceeds tolerance

**Diversity Score (DiCE-specific):**
- **> 0.7:** High diversity - CFs are quite different
- **0.4-0.7:** Moderate diversity
- **< 0.4:** Low diversity - CFs are similar

**Problem Indicators:**

✗ **Low Success Rate (<50%):** May indicate:
- Epsilon too strict - consider relaxing
- Target unreachable for many scenarios
- Method/parameters not well-suited for dataset

✗ **Zero Valid CFs (n_valid=0):** DiCE generated CFs but none within epsilon
- Generated CFs exist but don't hit target precisely
- Consider: increase epsilon or change method parameter

✗ **Few Generated CFs (n_generated < total_CFs):** DiCE struggled to find solutions
- May indicate sparse feature space
- Try different method ('genetic' or 'gradient')

✗ **High Sparsity (>6):** Changes many features
- May indicate difficult optimization problem
- Consider: smaller prediction gaps or feature selection

### **Comparison with Other Methods**

**DiCE vs Wachter:**
- **DiCE:** Multiple diverse CFs, built-in diversity objective
- **Wachter:** Single CF, pure distance+prediction optimization
- **Use DiCE when:** User wants to see alternative paths

**DiCE vs Growing Spheres:**
- **DiCE:** Synthetic CFs through optimization, may be unrealistic
- **Growing Spheres:** Interpolates real training instances, more realistic
- **Use DiCE when:** Model is neural network and diversity is important

**DiCE vs Prototype-Based:**
- **DiCE:** Generates synthetic CFs, typically closer
- **Prototype:** Returns real training instances, always realistic
- **Use DiCE when:** Willing to accept synthetic instances for better quality

**DiCE vs Gradient-Based:**
- **DiCE:** Library-based, multiple CFs with diversity
- **Gradient-Based:** Direct optimization, single CF
- **Use DiCE when:** Want diversity and library support

### **Parameter Tuning Guidelines**

**When to increase epsilon:**
- Success rate is low (<50%)
- Many experiments generate CFs but n_valid=0
- Target precision is less critical

**When to increase total_CFs:**
- Want more diverse alternatives
- Need higher chance of finding valid CF
- Willing to accept longer computation time

**When to change method:**
- **'random' → 'genetic':** Want better diversity exploration
- **'random' → 'gradient':** Have neural network, want precision
- **'genetic' → 'random':** Too slow, need faster results

**When to adjust weights:**
- **Increase diversity_weight:** CFs are too similar
- **Increase proximity_weight:** CFs are too far from original
- **Decrease proximity_weight:** Can't reach target

### **How to Run**

**Basic Test (Current):**
```bash
cd explainit/experiments/priorities_random_dice/standard_methods
python test_dice.py
```

**Expected Runtime:** 2-5 minutes (6 experiments × ~30-60 seconds each)

**Requirements:**
- TensorFlow/Keras model (neural network)
- dice-ml library: `pip install dice-ml`
- Sufficient training data for DiCE data preparation

**Output Location:**
- CSV results: `results/dice_test_results.csv`
- Console logs: Full experiment details

---

## **2. METRICS REFERENCE GUIDE**

This section provides comprehensive documentation for all metrics used in counterfactual evaluation, including both standard metrics and article-based metrics from the DiCE paper (Mothilal et al., 2020).

### **2.1 STANDARD/BASIC METRICS**

These metrics are computed for every counterfactual and provide fundamental quality measures. They are implemented in the `compute_metrics()` function.

---

#### **L1 Distance (Manhattan Distance)**

**Mathematical Definition:**
$$\text{L1}(x_{cf}, x_{orig}) = \sum_{i=1}^{d} |x_{cf,i} - x_{orig,i}|$$

where $d$ is the number of features.

**What it measures:**
- Total absolute change across all features
- Sum of absolute differences in each dimension
- Represents "city block" distance (sum of steps in each direction)

**Interpretation:**
- **Lower is better** - smaller changes from original
- Sensitive to number of features changed AND magnitude of changes
- More interpretable than L2 for humans (each feature contributes linearly)

**Scale dependence:**
- Depends on feature scales (scaled features recommended)
- For Auto MPG with 4 features, typical scaled range: 0-10

**Practical Examples:**

**Example 1: Auto MPG Dataset (4 features, scaled)**
```
Original:     [0.5, -0.3, 1.2, 0.8]
Counterfactual: [0.7, -0.3, 1.5, 0.9]
L1 Distance = |0.7-0.5| + |-0.3-(-0.3)| + |1.5-1.2| + |0.9-0.8|
            = 0.2 + 0.0 + 0.3 + 0.1 = 0.6
```
**Interpretation:** Small change (0.6 in scaled space), only 3 features changed slightly.

**Example 2: Large Change**
```
Original:     [0.5, -0.3, 1.2, 0.8]
Counterfactual: [2.1, 1.5, -0.8, 2.3]
L1 Distance = 1.6 + 1.8 + 2.0 + 1.5 = 6.9
```
**Interpretation:** Large change (6.9), all features changed significantly - may be impractical.

**Quality Guidelines:**
- **Excellent:** < 1.0 (minimal changes)
- **Good:** 1.0-3.0 (moderate changes)
- **Acceptable:** 3.0-6.0 (substantial changes)
- **Poor:** > 6.0 (extensive changes, may be unrealistic)

---

#### **L2 Distance (Euclidean Distance)**

**Mathematical Definition:**
$$\text{L2}(x_{cf}, x_{orig}) = \sqrt{\sum_{i=1}^{d} (x_{cf,i} - x_{orig,i})^2}$$

**What it measures:**
- Straight-line distance in feature space
- "As the crow flies" distance
- Penalizes large changes more than L1 (squared differences)

**Interpretation:**
- **Lower is better** - closer to original instance
- Standard geometric distance measure
- More sensitive to outlier changes (due to squaring)

**Relationship to L1:**
- Always $\text{L2} \leq \text{L1}$ (by Cauchy-Schwarz inequality)
- L2 emphasizes large changes more than L1
- For same L1, fewer large changes → larger L2

**Practical Examples:**

**Example 1: Small, Distributed Changes**
```
Original:     [0.5, -0.3, 1.2, 0.8]
Counterfactual: [0.7, -0.3, 1.5, 0.9]
Changes:      [0.2, 0.0, 0.3, 0.1]
L2 Distance = √(0.2² + 0.0² + 0.3² + 0.1²) = √0.14 = 0.374
L1 Distance = 0.6
Ratio L2/L1 = 0.62 (indicates distributed changes)
```

**Example 2: One Large Change**
```
Original:     [0.5, -0.3, 1.2, 0.8]
Counterfactual: [0.5, 2.7, 1.2, 0.8]
Changes:      [0.0, 3.0, 0.0, 0.0]
L2 Distance = √(0.0² + 3.0² + 0.0² + 0.0²) = 3.0
L1 Distance = 3.0
Ratio L2/L1 = 1.0 (indicates concentrated change)
```
**Interpretation:** L2 = L1 when only one feature changes (worst case for L2/L1 ratio).

**Quality Guidelines (Auto MPG with 4 features):**
- **Excellent:** < 0.5 (very close)
- **Good:** 0.5-1.5 (reasonable distance)
- **Acceptable:** 1.5-3.0 (substantial distance)
- **Poor:** > 3.0 (very far from original)

**L2 vs L1 Ratio Analysis:**
- **Ratio ≈ 0.5:** Changes distributed across many features
- **Ratio ≈ 0.7:** Moderate distribution
- **Ratio ≈ 1.0:** Change concentrated in few features

---

#### **Sparsity (Number of Changed Features)**

**Mathematical Definition:**
$$\text{Sparsity}(x_{cf}, x_{orig}) = \sum_{i=1}^{d} \mathbb{1}[|x_{cf,i} - x_{orig,i}| > \tau]$$

where $\tau$ is a small threshold (typically 0.001) to account for numerical precision, and $\mathbb{1}[\cdot]$ is the indicator function.

**What it measures:**
- Count of features that changed by more than threshold
- Interpretability measure (fewer changes = easier to understand)
- Actionability measure (fewer changes = easier to implement)

**Interpretation:**
- **Lower is better** - simpler, more interpretable explanations
- High sparsity = complex explanations requiring many changes
- Trade-off: Sometimes lower sparsity requires larger magnitude changes

**Threshold Selection:**
- **0.001:** Standard choice for scaled features (filters noise)
- **0.01:** More conservative (only "meaningful" changes)
- **Custom:** Domain-specific (e.g., 0.1 for unscaled features)

**Practical Examples:**

**Example 1: Sparse Counterfactual (Auto MPG)**
```
Original:     [cylinders=8, weight=3500, acceleration=12, year=70]
Counterfactual: [cylinders=4, weight=3500, acceleration=12, year=70]
Sparsity = 1 (only cylinders changed)
```
**Interpretation:** Highly interpretable - "To increase MPG, reduce cylinders from 8 to 4."

**Example 2: Dense Counterfactual**
```
Original:     [cylinders=8, weight=3500, acceleration=12, year=70]
Counterfactual: [cylinders=4, weight=2800, acceleration=15, year=78]
Sparsity = 4 (all features changed)
```
**Interpretation:** Complex - "To increase MPG, reduce cylinders, reduce weight, increase acceleration, AND use newer model year."

**Example 3: Feature Magnitude Trade-off**
```
Sparse CF:  [cylinders: 8→4, others unchanged]  Sparsity=1, L1=4.0
Dense CF:   [cylinders: 8→6, weight: 3500→3300, year: 70→75]  Sparsity=3, L1=2.5
```
**Interpretation:** Dense CF has lower distance but higher sparsity - trade-off between simplicity and proximity.

**Quality Guidelines:**
- **Excellent:** 1-2 features (highly interpretable)
- **Good:** 3-4 features (reasonably interpretable)
- **Acceptable:** 5-6 features (complex but manageable)
- **Poor:** > 6 features (too complex for practical use)

**Domain Considerations:**
- **Financial loans:** Low sparsity preferred (users want simple actions)
- **Medical diagnosis:** Sparsity less critical (multiple interventions common)
- **Marketing:** Low sparsity preferred (focused campaigns)

---

#### **Validity (Target Achievement)**

**Mathematical Definition:**
$$\text{Validity} = \mathbb{1}[|f(x_{cf}) - y_{target}| \leq \epsilon]$$

where $f(\cdot)$ is the model prediction, $y_{target}$ is the target prediction, and $\epsilon$ is the tolerance.

**What it measures:**
- Whether the counterfactual achieves the desired target prediction
- Binary success/failure indicator
- Fundamental requirement for a useful counterfactual

**Interpretation:**
- **Valid (True):** CF successfully achieves target within tolerance
- **Invalid (False):** CF fails to reach target - not actionable
- Invalid CFs may still be useful for debugging/understanding

**Epsilon Selection:**
- **Tight (small ε):** More precise but harder to achieve
- **Loose (large ε):** Easier to find but less controlled
- **Domain-specific:** Based on acceptable prediction error

**Practical Examples:**

**Example 1: Regression (Auto MPG)**
```
Target: 30.0 MPG
Epsilon: 2.0 MPG
CF Prediction: 29.5 MPG
Prediction Error: |29.5 - 30.0| = 0.5 MPG
Validity: 0.5 ≤ 2.0 → Valid ✓
```

**Example 2: Tight Tolerance**
```
Target: 30.0 MPG
Epsilon: 0.5 MPG
CF Prediction: 29.0 MPG
Prediction Error: |29.0 - 30.0| = 1.0 MPG
Validity: 1.0 > 0.5 → Invalid ✗
```
**Interpretation:** CF gets close but misses tight tolerance - may still be useful.

**Example 3: Classification (Binary)**
```
Target: Class 1 (favorable outcome)
CF Prediction: Class 1 with confidence 0.95
Validity: Class matches → Valid ✓
```

**Validity Rate in Experiments:**
- **100%:** All CFs valid - method very successful
- **80-99%:** Most CFs valid - good performance
- **50-79%:** Moderate success - may need parameter tuning
- **< 50%:** Poor performance - method/parameters need adjustment

**Epsilon Trade-offs:**

| Epsilon | Pros | Cons | Use Case |
|---------|------|------|----------|
| **Small (0.5)** | Precise control | Hard to achieve | Critical applications (medical, finance) |
| **Medium (2.0)** | Balanced | Standard choice | General research, Auto MPG |
| **Large (5.0)** | Easy to achieve | Low precision | Exploratory analysis |

---

#### **Prediction Error**

**Mathematical Definition:**
$$\text{PredError}(x_{cf}) = |f(x_{cf}) - y_{target}|$$

**What it measures:**
- Continuous measure of how close CF is to target
- More informative than binary validity
- Useful for ranking multiple counterfactuals

**Interpretation:**
- **0:** Perfect hit (prediction exactly equals target)
- **< ε:** Valid CF (within tolerance)
- **> ε:** Invalid CF (outside tolerance)
- Lower values indicate better target achievement

**Relationship to Validity:**
- Validity = (Prediction Error ≤ ε)
- Prediction Error is the continuous version of binary Validity
- Use Prediction Error to compare valid CFs or rank invalid ones

**Practical Examples:**

**Example 1: Comparing Valid CFs**
```
CF1: pred_error = 0.2 MPG, l2_distance = 1.5
CF2: pred_error = 1.8 MPG, l2_distance = 0.8
```
**Interpretation:** CF1 has better target achievement but requires more change. Choose based on priorities.

**Example 2: Ranking Invalid CFs**
```
Target: 30.0 MPG, Epsilon: 2.0 MPG
CF1: prediction = 27.0 MPG, pred_error = 3.0 (invalid)
CF2: prediction = 25.0 MPG, pred_error = 5.0 (invalid)
CF3: prediction = 35.0 MPG, pred_error = 5.0 (invalid)
```
**Interpretation:** CF1 is closest to valid - might work with relaxed epsilon or slight adjustment.

**Quality Guidelines:**
- **Excellent:** < 0.5 (very precise)
- **Good:** 0.5-1.0 (good precision)
- **Acceptable:** 1.0-2.0 (within typical epsilon)
- **Poor:** > 2.0 (far from target)

**Uses:**
- **Optimization objective:** Minimize prediction error
- **Method comparison:** Lower average error = better method
- **Debugging:** High error indicates optimization failure

---

### **2.2 ARTICLE-BASED METRICS (Mothilal et al., 2020)**

These metrics are defined in the DiCE paper: "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations" (Mothilal et al., FAT* 2020). They provide a comprehensive evaluation framework specifically designed for diverse counterfactual generation methods.

**Key Innovation:** Metrics designed for evaluating **sets of counterfactuals** (not just individual CFs).

---

#### **MAD (Median Absolute Deviation)**

**Mathematical Definition:**
$$\text{MAD}_j = \text{median}_i(|X_{i,j} - \text{median}_i(X_{i,j})|)$$

where $X_{i,j}$ is the value of feature $j$ in training sample $i$.

**What it measures:**
- Robust measure of feature variability (alternative to standard deviation)
- Used for normalizing distances across features with different scales
- Less sensitive to outliers than standard deviation

**Why MAD instead of Standard Deviation?**
- **Robust to outliers:** Median-based, not affected by extreme values
- **Better for mixed distributions:** Works well with skewed data
- **Interpretable:** Represents "typical" deviation from median

**Computation:**
1. Compute median of feature across training data
2. Compute absolute deviations from median
3. Take median of those deviations

**Practical Example:**

**Example: Weight feature in Auto MPG**
```
Training weights (scaled): [-1.2, -0.8, 0.0, 0.3, 0.5, 0.8, 1.5, 2.1, 3.0]
Step 1: Median = 0.5
Step 2: Absolute deviations = [1.7, 1.3, 0.5, 0.2, 0.0, 0.3, 1.0, 1.6, 2.5]
Step 3: MAD = median([1.7, 1.3, 0.5, 0.2, 0.0, 0.3, 1.0, 1.6, 2.5]) = 1.0
```

**Using MAD for Normalization:**
```
Original feature value: 0.5
CF feature value: 2.5
Raw difference: 2.5 - 0.5 = 2.0
MAD-normalized difference: 2.0 / 1.0 = 2.0
Interpretation: Change is 2.0 times the typical variation in training data
```

**Handling Edge Cases:**
- **Zero MAD:** Feature is constant in training data → replace with 1.0 to avoid division by zero
- **Very small MAD:** Feature has little variation → small changes appear large when normalized

---

#### **Validity (Article Version - Set-based)**

**Mathematical Definition:**
$$\text{Validity} = \frac{\#\{c_i : |f(c_i) - y_{target}| \leq \epsilon\}}{k}$$

where $k$ is the number of counterfactuals requested, and $c_i$ are generated CFs.

**Difference from Basic Validity:**
- **Basic Validity:** Binary indicator for single CF (True/False)
- **Article Validity:** Fraction of valid CFs in set (0.0 to 1.0)
- **Article Validity** evaluates method's success rate, not individual CF

**What it measures:**
- Success rate of counterfactual generation method
- Fraction of generated CFs that achieve target
- Method robustness indicator

**Interpretation:**
- **1.0 (100%):** All requested CFs are valid - best case
- **0.8 (80%):** Most CFs valid - good performance
- **0.5 (50%):** Half valid - moderate performance
- **0.0 (0%):** No valid CFs - method failed completely

**Practical Examples:**

**Example 1: Perfect Success**
```
Requested: k = 5 CFs
Generated: 5 CFs with predictions [29.5, 30.1, 29.8, 30.5, 29.2] MPG
Target: 30.0 MPG, Epsilon: 2.0 MPG
Valid CFs: All 5 within [28.0, 32.0] range
Validity = 5/5 = 1.0 (100%)
```

**Example 2: Partial Success**
```
Requested: k = 5 CFs
Generated: 5 CFs with predictions [29.5, 35.0, 30.1, 25.0, 29.8] MPG
Target: 30.0 MPG, Epsilon: 2.0 MPG
Valid CFs: 3 CFs within range (29.5, 30.1, 29.8)
Validity = 3/5 = 0.6 (60%)
```

**Example 3: Generated Fewer CFs**
```
Requested: k = 10 CFs
Generated: 5 CFs (method struggled), 4 are valid
Validity = 4/10 = 0.4 (40%)
```
**Note:** Denominator is $k$ (requested), not number generated. Penalizes methods that can't generate enough CFs.

**Method Comparison:**
```
DiCE (random):    Validity = 0.95 (very reliable)
DiCE (gradient):  Validity = 0.70 (less reliable but may have other benefits)
Growing Spheres:  Validity = 0.85 (good balance)
```

---

#### **Continuous Proximity (MAD-normalized)**

**Mathematical Definition:**
$$\text{ContinuousProximity} = -\frac{1}{k} \sum_{i=1}^{k} \text{dist}_{cont}(c_i, x)$$

where the continuous distance is:
$$\text{dist}_{cont}(c, x) = \frac{1}{d_{cont}} \sum_{j \in \text{continuous}} \frac{|c_j - x_j|}{\text{MAD}_j}$$

and $d_{cont}$ is the number of continuous features.

**What it measures:**
- Average similarity between CFs and original instance
- MAD-normalized to account for feature scales
- Negative sign converts to minimization objective (higher is better)

**Interpretation:**
- **Closer to 0 (e.g., -0.1):** CFs very close to original (better)
- **More negative (e.g., -2.0):** CFs far from original (worse)
- Represents "cost" of counterfactual in standardized units

**Why Negative?**
- DiCE paper uses minimization framework
- Negative proximity → higher (less negative) is better
- Allows combining with other objectives in optimization

**Practical Examples:**

**Example 1: Close CF (Auto MPG with 4 features)**
```
Original:    [0.5, -0.3, 1.2, 0.8]
CF:          [0.7, -0.3, 1.5, 0.9]
Changes:     [0.2, 0.0, 0.3, 0.1]
MAD values:  [1.0, 0.8, 1.2, 0.9]
MAD-normalized: [0.2/1.0, 0.0/0.8, 0.3/1.2, 0.1/0.9] = [0.20, 0.00, 0.25, 0.11]
dist_cont = mean([0.20, 0.00, 0.25, 0.11]) = 0.14
For k=1: ContinuousProximity = -0.14
```
**Interpretation:** Small change (0.14 MADs on average per feature) → -0.14 is good.

**Example 2: Far CF**
```
Original:    [0.5, -0.3, 1.2, 0.8]
CF:          [2.5, 2.0, -1.0, 3.5]
Changes:     [2.0, 2.3, 2.2, 2.7]
MAD values:  [1.0, 0.8, 1.2, 0.9]
MAD-normalized: [2.0, 2.875, 1.833, 3.0]
dist_cont = mean([2.0, 2.875, 1.833, 3.0]) = 2.43
For k=1: ContinuousProximity = -2.43
```
**Interpretation:** Large change (2.43 MADs per feature) → -2.43 is poor.

**Example 3: Set of CFs**
```
5 CFs with distances: [0.14, 0.20, 0.18, 0.25, 0.30]
Average distance: 0.214
ContinuousProximity = -0.214
```
**Interpretation:** On average, CFs are 0.214 MADs away from original.

**Quality Guidelines:**
- **Excellent:** -0.0 to -0.3 (very close, < 0.3 MADs per feature)
- **Good:** -0.3 to -0.7 (reasonable distance)
- **Acceptable:** -0.7 to -1.5 (substantial distance)
- **Poor:** < -1.5 (very far, > 1.5 MADs per feature)

**Method Comparison:**
```
Method A: ContinuousProximity = -0.25 (closer CFs)
Method B: ContinuousProximity = -0.80 (farther CFs)
→ Method A provides more actionable CFs (if validity is similar)
```

---

#### **Diversity (MAD-normalized Pairwise Distance)**

**Mathematical Definition:**
$$\text{Diversity} = \frac{1}{\binom{k}{2}} \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} \text{dist}_{cont}(c_i, c_j)$$

where $\binom{k}{2} = \frac{k(k-1)}{2}$ is the number of pairs, and $\text{dist}_{cont}$ is the MAD-normalized distance.

**What it measures:**
- How different the generated CFs are from each other
- Average dissimilarity between all pairs of CFs
- Key innovation of DiCE method (diversity objective)

**Why Diversity Matters:**
- **Single path problem:** One CF shows only one way to achieve target
- **User choice:** Multiple diverse CFs let users pick preferred option
- **Robustness:** Diverse CFs may work under different constraints
- **Understanding:** Shows multiple mechanisms to achieve outcome

**Interpretation:**
- **Higher is better** - more diverse alternatives
- **0.0:** All CFs are identical (no diversity)
- **> 1.0:** CFs differ by > 1 MAD on average (good diversity)
- **> 2.0:** CFs are very different (excellent diversity)

**Practical Examples:**

**Example 1: Low Diversity (Similar CFs)**
```
CF1: [cylinders: 8→6, weight: 3500→3400, acc: 12→12, year: 70→72]
CF2: [cylinders: 8→6, weight: 3500→3300, acc: 12→12, year: 70→73]
CF3: [cylinders: 8→6, weight: 3500→3450, acc: 12→12, year: 70→71]

All CFs use same strategy: reduce cylinders to 6, slightly adjust weight/year.
MAD-normalized pairwise distances: [0.12, 0.08, 0.15]
Diversity = mean([0.12, 0.08, 0.15]) = 0.117
```
**Interpretation:** Low diversity (0.117) - CFs are very similar, limited alternatives for user.

**Example 2: High Diversity (Diverse Strategies)**
```
CF1: [cylinders: 8→4, weight: 3500→3500, acc: 12→12, year: 70→70]
     Strategy: Reduce cylinders significantly

CF2: [cylinders: 8→8, weight: 3500→2500, acc: 12→12, year: 70→70]
     Strategy: Reduce weight significantly

CF3: [cylinders: 8→8, weight: 3500→3500, acc: 12→18, year: 70→82]
     Strategy: Improve acceleration and use newer model

MAD-normalized pairwise distances:
  dist(CF1, CF2) = 1.8
  dist(CF1, CF3) = 2.3
  dist(CF2, CF3) = 2.1
Diversity = mean([1.8, 2.3, 2.1]) = 2.07
```
**Interpretation:** High diversity (2.07) - three distinct strategies, user can choose based on feasibility.

**Example 3: Computing Diversity for 5 CFs**
```
5 CFs → C(5,2) = 10 pairwise distances
Pairwise distances: [0.8, 1.2, 1.5, 1.0, 1.8, 1.3, 1.6, 1.1, 1.4, 1.7]
Diversity = mean(10 distances) = 1.34
```

**Quality Guidelines:**
- **Excellent:** > 2.0 (very diverse alternatives)
- **Good:** 1.0-2.0 (reasonably diverse)
- **Acceptable:** 0.5-1.0 (moderate diversity)
- **Poor:** < 0.5 (similar CFs, limited alternatives)

**Trade-offs:**
- **High diversity + High proximity:** Difficult - diverse CFs often require exploration
- **High diversity + Low validity:** Diversity at cost of hitting target
- **Low diversity + High validity:** All CFs hit target but similar strategies

**Method Comparison:**
```
DiCE (diversity_weight=1.0): Diversity = 1.85 (explicitly optimizes diversity)
DiCE (diversity_weight=0.5): Diversity = 1.20 (less emphasis on diversity)
Gradient-based (no diversity): Diversity = 0.30 (finds similar local optima)
```

---

#### **Count Diversity (Feature-based)**

**Mathematical Definition:**
$$\text{CountDiversity} = \frac{1}{\binom{k}{2}} \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} \#\{f : |c_{i,f} - c_{j,f}| > \tau\}$$

where $\tau$ is a small threshold (e.g., $10^{-6}$) to determine if features differ.

**What it measures:**
- Average number of differing features between CF pairs
- Discrete/count version of continuous diversity
- Emphasizes feature-level differences (not magnitudes)

**Difference from Continuous Diversity:**
- **Continuous Diversity:** Considers magnitude of differences (MAD-normalized)
- **Count Diversity:** Only counts which features differ (binary)
- Count diversity can be high even if differences are small

**Interpretation:**
- **Higher is better** - CFs differ in more features
- **0:** All CFs identical across all features
- **d (max):** CF pairs differ in all $d$ features

**Practical Examples:**

**Example 1: Same Features Changed (Low Count Diversity)**
```
Auto MPG (4 features): [cylinders, weight, acceleration, year]

CF1: Changes: [cylinders, weight] → 2 features
CF2: Changes: [cylinders, weight] → 2 features  (same as CF1)
CF3: Changes: [cylinders, weight] → 2 features  (same as CF1)

All CFs change same features, but with different magnitudes.

Pairwise differences:
  CF1 vs CF2: 2 features differ (cylinders and weight have different values)
  CF1 vs CF3: 2 features differ
  CF2 vs CF3: 2 features differ

CountDiversity = mean([2, 2, 2]) = 2.0
```
**Interpretation:** Moderate count diversity (2.0 out of 4 possible features).

**Example 2: Different Features Changed (High Count Diversity)**
```
CF1: Changes [cylinders] only        → 1 feature changed
CF2: Changes [weight] only           → 1 feature changed
CF3: Changes [year] only             → 1 feature changed

Pairwise differences:
  CF1 vs CF2: All 4 features differ (CF1 changes cyl, CF2 changes weight, etc.)
  CF1 vs CF3: All 4 features differ
  CF2 vs CF3: All 4 features differ

CountDiversity = mean([4, 4, 4]) = 4.0
```
**Interpretation:** Maximum count diversity (4.0 out of 4 features) - CFs use completely different features.

**Example 3: Mixed Strategies**
```
CF1: Changes [cylinders, weight]
CF2: Changes [cylinders, acceleration]
CF3: Changes [weight, year]

Pairwise differences:
  CF1 vs CF2: Differ in [weight, acceleration] → 2 features
  CF1 vs CF3: Differ in [cylinders, year] → 2 features
  CF2 vs CF3: Differ in [cylinders, acceleration, weight, year] → all 4 features

CountDiversity = mean([2, 2, 4]) = 2.67
```

**Quality Guidelines (for 4 features):**
- **Excellent:** 3-4 (CFs use very different features)
- **Good:** 2-3 (reasonably different feature sets)
- **Acceptable:** 1-2 (moderate overlap)
- **Poor:** < 1 (CFs mostly change same features)

**Use Cases:**
- **Feature selection:** Shows which features provide alternative paths
- **Interpretability:** Different feature sets → different explanations
- **Robustness:** If one feature is immutable, alternatives exist

**Relationship to Continuous Diversity:**

| Scenario | Continuous Diversity | Count Diversity | Interpretation |
|----------|---------------------|-----------------|----------------|
| **A:** All CFs change same features, large magnitudes | High | Low | Same strategy, different degrees |
| **B:** All CFs change different features, small magnitudes | Low | High | Different strategies, subtle changes |
| **C:** Different features, large magnitudes | High | High | Very diverse (ideal) |
| **D:** Same features, small magnitudes | Low | Low | Poor diversity |

---

### **2.3 PRACTICAL INTERPRETATION EXAMPLES**

This section provides concrete scenarios showing how to interpret metrics in realistic situations.

---

#### **Scenario 1: Comparing Two Counterfactual Methods**

**Context:** Testing DiCE vs Simple Gradient-based method on Auto MPG dataset

**Results:**

| Metric | DiCE (Random) | Gradient-Based | Winner |
|--------|---------------|----------------|--------|
| **Validity** | 0.95 (95%) | 0.80 (80%) | DiCE |
| **L2 Distance** | 0.65 | 0.45 | Gradient |
| **Sparsity** | 2.8 | 3.5 | DiCE |
| **Prediction Error** | 0.8 MPG | 1.2 MPG | DiCE |
| **Continuous Proximity** | -0.42 | -0.28 | Gradient |
| **Diversity** | 1.85 | 0.35 | DiCE |
| **Count Diversity** | 2.4 | 1.1 | DiCE |

**Interpretation:**

**DiCE Advantages:**
- ✓ **Higher validity** (95% vs 80%): More reliable at finding valid CFs
- ✓ **Lower sparsity** (2.8 vs 3.5): Simpler explanations (fewer features changed)
- ✓ **Better target achievement** (0.8 vs 1.2 MPG error): More precise
- ✓ **Much higher diversity** (1.85 vs 0.35): Provides diverse alternatives
- ✓ **Higher count diversity** (2.4 vs 1.1): Uses different feature combinations

**Gradient-Based Advantages:**
- ✓ **Closer to original** (0.45 vs 0.65 L2, -0.28 vs -0.42 proximity): Smaller changes needed

**Decision:**
- **Choose DiCE if:** User wants multiple options, reliability is critical, or interpretability matters
- **Choose Gradient if:** Minimizing distance is top priority, diversity not needed, single CF sufficient

---

#### **Scenario 2: Parameter Tuning (Epsilon Sensitivity)**

**Context:** Testing different epsilon values on same dataset/method

**Results:**

| Epsilon | Validity | Avg L2 | Avg Sparsity | Avg Pred Error | Diversity |
|---------|----------|--------|--------------|----------------|-----------|
| **0.5 MPG** | 0.40 | 0.45 | 2.2 | 0.3 | 0.85 |
| **1.0 MPG** | 0.75 | 0.58 | 2.8 | 0.6 | 1.20 |
| **2.0 MPG** | 0.95 | 0.72 | 3.1 | 0.9 | 1.65 |
| **5.0 MPG** | 1.00 | 1.20 | 4.2 | 2.5 | 2.40 |

**Interpretation:**

**Epsilon = 0.5 MPG (Very Strict):**
- Low validity (40%): Hard to achieve tight tolerance
- Low distance (0.45): When successful, CFs are close
- Low sparsity (2.2): Simple changes
- **Problem:** Only 40% success rate - too restrictive

**Epsilon = 1.0 MPG (Strict):**
- Moderate validity (75%): Most CFs valid
- Moderate distance (0.58): Reasonable changes
- **Balanced** but could be better

**Epsilon = 2.0 MPG (Moderate - RECOMMENDED):**
- High validity (95%): Very reliable
- Moderate distance (0.72): Acceptable changes
- Good diversity (1.65): Multiple alternatives
- **Sweet spot** for Auto MPG dataset

**Epsilon = 5.0 MPG (Loose):**
- Perfect validity (100%): All CFs valid
- High distance (1.20): Larger changes needed
- High sparsity (4.2): Complex changes
- High prediction error (2.5): Poor precision
- **Too loose:** CFs imprecise and far from original

**Recommendation:** Use epsilon = 2.0 MPG for Auto MPG dataset (balances validity, distance, and precision).

---

#### **Scenario 3: Detecting Method Failure**

**Context:** Method generates CFs but metrics reveal problems

**Case A: High Distance, Good Validity**
```
Validity: 0.90 (good)
L2 Distance: 3.5 (very high)
L1 Distance: 8.2 (very high)
Sparsity: 4 (all features changed)
Prediction Error: 0.5 (good)
```

**Diagnosis:** Method finds valid CFs but they are far from original.
**Problem:** CFs may be impractical/unrealistic for users.
**Solution:** 
- Add proximity constraint to optimization
- Increase proximity_weight parameter
- Try different method (e.g., Growing Spheres for realistic CFs)

---

**Case B: Low Distance, Poor Validity**
```
Validity: 0.20 (poor)
L2 Distance: 0.35 (good)
Sparsity: 2 (good)
Prediction Error: 3.5 (high)
```

**Diagnosis:** Method finds nearby points but they don't achieve target.
**Problem:** Method getting stuck in local optima near original.
**Solution:**
- Increase total_CFs (more exploration)
- Try 'genetic' method instead of 'random'
- Relax epsilon slightly
- Check if target is reachable (may be in sparse region)

---

**Case C: Zero Diversity**
```
Validity: 1.00 (perfect)
L2 Distance: 0.50 (good)
Diversity: 0.05 (very low)
Count Diversity: 0.2 (very low)
```

**Diagnosis:** All CFs are nearly identical despite being requested to generate multiple.
**Problem:** Method converging to same solution repeatedly.
**Solution:**
- Increase diversity_weight parameter
- Use 'genetic' or 'random' method (better exploration)
- Increase total_CFs substantially
- Add randomization/noise to initialization

---

#### **Scenario 4: Real-World Application Guidance**

**Use Case: Loan Application**

**User:** "My loan was denied (predicted default probability = 0.7). What can I do to get approved (target < 0.4)?"

**Method Results:**
```
Generated 5 CFs with following strategies:

CF1: L2=0.4, Sparsity=1, Valid=Yes
     Change: Increase income by $15,000

CF2: L2=0.6, Sparsity=2, Valid=Yes
     Changes: Increase income by $8,000 AND reduce debt by $5,000

CF3: L2=0.5, Sparsity=1, Valid=Yes
     Change: Increase credit score by 50 points

CF4: L2=0.8, Sparsity=3, Valid=Yes
     Changes: Reduce debt, increase savings, longer employment

CF5: L2=1.2, Sparsity=4, Valid=No
     Many changes, doesn't achieve target

Metrics:
- Validity: 4/5 = 0.80
- Avg L2 (valid): 0.575
- Avg Sparsity (valid): 1.75
- Diversity: 1.45
- Count Diversity: 2.1
```

**Interpretation for User:**

**Primary Options (low sparsity):**
1. **CF1 (Simplest):** Only increase income → feasible if expecting raise/promotion
2. **CF3 (Alternative):** Only improve credit score → feasible if can pay down credit cards

**Balanced Option:**
- **CF2 (Moderate):** Combination of income increase + debt reduction → may be most realistic

**Complex Option:**
- **CF4 (Many changes):** Multiple actions required → longer-term strategy

**Metrics Tell Us:**
- **Good diversity (1.45):** User has multiple distinct paths
- **High validity (80%):** Most strategies work
- **Low sparsity (1.75 avg):** Simple, actionable advice
- **Moderate distance (0.575):** Achievable changes

**User Actionability:**
- CF1 or CF3: Single action, high feasibility
- CF2: Two actions, moderate feasibility
- CF4: Many actions, low short-term feasibility but good long-term plan

---

#### **Scenario 5: Debugging Model vs Method Issues**

**Problem:** Low validity rate (30%) across all experiments

**Investigation:**

**Step 1: Check Prediction Errors**
```
All invalid CFs have pred_error > 5.0 (way outside epsilon=2.0)
```
**Diagnosis:** CFs are far from target, not just slightly outside epsilon.

**Step 2: Check Distances**
```
L2 Distance: Small to moderate (0.5-1.0)
Proximity: Good (-0.4 to -0.7)
```
**Diagnosis:** CFs are close to original, but still can't reach target.

**Step 3: Check Training Data**
```
Original: prediction = 15 MPG
Target: prediction = 35 MPG
Training data: No samples with > 32 MPG in nearby feature space
```

**Root Cause:** **Model issue, not method issue**
- Target (35 MPG) is in sparse/extrapolation region
- Model may not be accurate in that region
- Method correctly stays close to original, but can't reach unrealistic target

**Solution:**
- Choose more realistic targets (within training data range)
- Increase tolerance epsilon
- Use targets from actual test set (known to be achievable)
- Check model coverage and retrain if needed

---

### **2.4 METRIC SELECTION GUIDE**

**Which metrics should you report?**

**For Single CF Evaluation:**
- ✓ **L2 Distance:** Primary proximity measure
- ✓ **Sparsity:** Interpretability measure
- ✓ **Validity:** Success indicator
- ✓ **Prediction Error:** Precision measure
- Optional: L1 Distance (if interpretability matters)

**For Multiple CF Evaluation (DiCE, etc.):**
- ✓ **Validity (article version):** Success rate
- ✓ **Continuous Proximity:** Average distance (MAD-normalized)
- ✓ **Diversity:** Key advantage of multiple CFs
- ✓ **Count Diversity:** Feature-level diversity
- Optional: All single-CF metrics for best CF

**For Method Comparison:**
- ✓ **Validity rate:** Reliability comparison
- ✓ **Average L2 distance:** Proximity comparison
- ✓ **Average sparsity:** Interpretability comparison
- ✓ **Diversity:** Only for multi-CF methods
- ✓ **Runtime:** Efficiency comparison

**For Paper/Publication:**
- ✓ All article-based metrics (if using DiCE paper framework)
- ✓ Statistical significance tests
- ✓ Confidence intervals
- ✓ Multiple datasets and models

---

## **3. EXTENDED TESTING (Future Implementation)**

### **Parameter Sensitivity Testing**

The current test uses fixed parameters. To perform comprehensive parameter sensitivity analysis, create additional test scripts:

**Test Script Ideas:**

**`test_dice_epsilon_sensitivity.py`:**
- Fix: total_CFs=5, method='random'
- Vary: epsilon = [0.5, 1.0, 2.0, 3.0, 5.0]
- Purpose: Find optimal tolerance for validity

**`test_dice_total_cfs_sensitivity.py`:**
- Fix: epsilon=2.0, method='random'
- Vary: total_CFs = [3, 5, 10, 15, 20]
- Purpose: Assess diversity vs computation tradeoff

**`test_dice_method_comparison.py`:**
- Fix: epsilon=2.0, total_CFs=5
- Vary: method = ['random', 'genetic', 'gradient']
- Purpose: Compare generation algorithms

**`test_dice_weights_sensitivity.py`:**
- Fix: epsilon=2.0, total_CFs=5, method='random'
- Vary: diversity_weight × proximity_weight combinations
- Purpose: Optimize diversity vs proximity tradeoff

### **Extended Metrics**

For deeper analysis, consider tracking:

**Diversity Metrics:**
- Average pairwise distance between generated CFs
- Feature-wise diversity (which features vary most)
- Prediction diversity (range of predictions)

**Efficiency Metrics:**
- Generation time per CF
- Memory usage
- Number of model calls

**Realism Metrics:**
- Distance to nearest training instance
- Plausibility score (custom heuristics)
- Feature correlation preservation

---

## **4. ADDING NEW STANDARD METHODS**

### **Framework for Extension**

The `standard_methods.py` file provides a modular framework for adding additional counterfactual methods:

**To Add a New Method:**

1. **Implement Method Function** in `standard_methods.py`:
```python
def new_method_counterfactual(
    X_original: np.ndarray,
    model,
    target_value: float,
    # Method-specific parameters
    epsilon: float = 1.0,
    # ...
) -> Tuple[Optional[np.ndarray], Optional[float], dict]:
    """
    Description of method.
    
    Args: ...
    Returns: (counterfactual, prediction, info)
    """
    # Implementation
    pass
```

2. **Update `run_all_methods`** to include new method:
```python
if 'new_method' in methods_to_run:
    logger.info("  Running New Method...")
    cf, pred, info = new_method_counterfactual(...)
    # ... add to results
```

3. **Create Test Script** `test_new_method.py`:
- Follow structure of `test_dice.py`
- Test parameter sensitivity
- Save results for comparison

4. **Update Documentation:**
- Add section to this file describing:
  - Method overview
  - Parameters tested
  - Output format
  - Interpretation guidelines

### **Candidate Methods for Addition**

Based on `priorities_with_random_search/standard_methods.py`, consider adding:

1. **Wachter's Method (2017)**
   - Classic optimization-based approach
   - Good baseline for comparison

2. **Growing Spheres**
   - Searches in expanding spheres
   - Uses real training instances

3. **Prototype-Based**
   - Returns real training instances as CFs
   - Simple, interpretable, realistic

4. **Gradient-Based**
   - Direct neural network gradient optimization
   - Fast for differentiable models

**Priority Recommendation:** Add Wachter and Prototype-Based first for comprehensive comparison with DiCE.

---

## **5. COMPARISON EXPERIMENTS**

### **Multi-Method Comparison**

Once multiple methods are implemented, create `experiment_comparison.py` to:

1. Run all methods on same sample-target pairs
2. Compare metrics side-by-side:
   - Success rates
   - Distance metrics
   - Sparsity
   - Computation time
3. Identify best method for different scenarios

**Example Comparison Table:**

| Method | Success Rate | Avg L2 Distance | Avg Sparsity | Avg Time (s) |
|--------|--------------|-----------------|--------------|--------------|
| DiCE | 100% | 0.4125 | 3.5 | 45.2 |
| Wachter | 83% | 0.5234 | 4.2 | 12.1 |
| Prototype | 100% | 0.6789 | N/A | 0.3 |
| Growing Spheres | 67% | 0.4567 | 3.8 | 8.7 |

**Insights from Comparison:**
- **DiCE:** High success, moderate distance, slowest (generates multiple CFs)
- **Wachter:** Good success, moderate quality, fast single CF
- **Prototype:** Perfect success, larger distance, very fast (lookup)
- **Growing Spheres:** Lowest success, good when works, fast

### **Statistical Analysis**

For rigorous comparison:
- Run experiments multiple times with different random seeds
- Use statistical tests (t-test, Wilcoxon) to compare methods
- Create visualization: box plots, scatter plots, radar charts

---

## **General Notes**

### **Best Practices**

1. **Always use same data splits:** Ensure fair comparison across methods
2. **Record all parameters:** Enable reproducibility
3. **Save raw results:** CSV format for post-analysis
4. **Log verbosely:** Helps debug issues
5. **Version control:** Track changes to test scripts and parameters

### **Common Issues**

**DiCE fails to install:**
```bash
pip install --upgrade pip
pip install dice-ml tensorflow
```

**DiCE generates no CFs:**
- Check epsilon isn't too strict
- Verify training data has instances near target
- Try different method ('genetic' instead of 'random')

**Memory errors:**
- Reduce total_CFs
- Use smaller training dataset for DiCE data preparation
- Enable TensorFlow memory growth

**Slow performance:**
- Use 'random' method instead of 'genetic' or 'gradient'
- Reduce total_CFs
- Use CPU instead of GPU for small models (faster startup)

### **Next Steps**

After running DiCE tests:

1. **Analyze Results:**
   - Which scenarios succeed? Which fail?
   - Are distances competitive with priorities-based method?
   - Is diversity valuable? (n_valid > 1 useful?)

2. **Compare with Priorities Method:**
   - Run same scenarios with priorities-based approach
   - Compare success rates, distances, sparsity
   - Identify strengths and weaknesses of each

3. **Extend Testing:**
   - Implement parameter sensitivity tests
   - Add more standard methods
   - Run on multiple datasets

4. **Document Findings:**
   - Create comparison report
   - Generate visualizations
   - Prepare for publication/presentation

---

## **References**

**DiCE:**
- Paper: Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *FAT* 2020.
- Documentation: https://interpret.ml/DiCE/
- GitHub: https://github.com/interpretml/DiCE

**Other Methods** (for future implementation):
- Wachter et al. (2017): Counterfactual explanations without opening the black box
- Laugel et al. (2017): Inverse classification for comparison-based interpretability in ML
- Dandl et al. (2020): Multi-objective counterfactual explanations

---

**Last Updated:** March 2026  
**Experiment:** priorities_random_dice  
**Status:** DiCE implemented, framework ready for additional methods
