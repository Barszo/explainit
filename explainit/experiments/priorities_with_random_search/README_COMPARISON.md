# Counterfactual Explanation Experiments

This directory contains a comprehensive framework for comparing different counterfactual explanation methods.

## Files Overview

### Core Files

1. **`standard_methods.py`** - Implementations of popular counterfactual methods:
   - Wachter's Method (2017)
   - Growing Spheres
   - Prototype-based
   - Gradient-based (TensorFlow/Keras)
   
2. **`experiment_final.py`** - Main experiment runner that compares:
   - Preference-based Random Search (your original method)
   - Standard methods from literature
   
3. **`test_experiment.py`** - Test scripts for quick verification

## Quick Start

### 1. Test Individual Methods (Recommended First Step)

Test on a single sample-target pair to see detailed comparison:

```bash
cd /Users/bartosz/projects/explainit/explainit_project/explainit/experiments/priorities_with_random_search
python test_experiment.py 3
```

### 2. Test Standard Methods Only

Quick test with 2 sample-target pairs (~2-5 minutes):

```bash
python test_experiment.py 1
```

This will create `test_standard_methods.csv` with results.

### 3. Test Preference-Based Method Only

Quick test with your original method:

```bash
python test_experiment.py 2
```

This will create `test_preference_based.csv` with results.

### 4. Run Full Comparison Experiment

Run both methods on all sample-target pairs:

```bash
python experiment_final.py
```

This will create two CSV files:
- `experiment_results_preference_based.csv` - Your method's results
- `experiment_results_standard_methods.csv` - Standard methods' results

## Configuration

Edit the `config` dictionary in `experiment_final.py` or `test_experiment.py`:

```python
config = {
    'dataset': 'Auto_MPG',          # Dataset to use
    'model': 'NN_Residual',         # Model to use
    'n_quantiles': 3,               # Number of test points (3 = 6 experiments)
    'return_top_n': 5,              # Top N CFs to save (preference method)
    'epsilon': 2.0,                 # Target tolerance (±2 MPG)
    'exemplar_weight': 0.01,        # Weight for exemplar in preferences
    'n_samples': 10000,             # Samples for preference method
    'use_monte_carlo': True         # Monte Carlo sampling
}
```

### Key Parameters

- **`n_quantiles`**: Number of prediction quantile points to test
  - `n_quantiles=2` → 2 samples → 2 experiments (A→B, B→A)
  - `n_quantiles=3` → 3 samples → 6 experiments (3×2)
  - `n_quantiles=5` → 5 samples → 20 experiments (5×4)

- **`epsilon`**: Target prediction tolerance
  - Larger = easier to find counterfactuals
  - Smaller = more precise but harder

- **`n_samples`**: Number of candidates for preference method
  - More = better quality but slower
  - Recommended: 5000-10000 for testing, 50000+ for final results

## Output Files

### Preference-Based Results CSV

Columns:
- `sample_idx`, `target_idx`: Sample-target pair identifiers
- `sample_prediction`, `target_prediction`: Original predictions
- `total_cf_found`: Total counterfactuals before filtering
- `max_preference_score`: Best preference score achieved
- `cf_rank`: Rank of this CF (1 = best)
- `cf_prediction`: CF prediction
- `cf_preference_score`: Preference score
- `cf_values`: Feature values

### Standard Methods Results CSV

Columns:
- `sample_idx`, `target_idx`: Sample-target pair identifiers
- `method`: Method name (wachter, growing_spheres, prototype, gradient_based)
- `valid`: Whether CF is valid (within epsilon of target)
- `cf_prediction`: CF prediction
- `l1_distance`, `l2_distance`: Distance metrics
- `sparsity`: Number of changed features
- `prediction_error`: |prediction - target|

## Interpreting Results

### Success Metrics

1. **Validity**: CF achieves target within epsilon
2. **Distance**: How much the instance changed (lower = better)
3. **Sparsity**: Fewer feature changes = more interpretable
4. **Preference Score**: Higher = better aligned with preferences (preference method only)

### Comparing Methods

```python
import pandas as pd

# Load results
pref_df = pd.read_csv('experiment_results_preference_based.csv')
std_df = pd.read_csv('experiment_results_standard_methods.csv')

# Analyze preference-based method
print(pref_df.groupby('sample_idx')['cf_preference_score'].max())

# Analyze standard methods
print(std_df.groupby('method')['valid'].mean())  # Success rate
print(std_df[std_df['valid']].groupby('method')['l2_distance'].mean())  # Avg distance
print(std_df[std_df['valid']].groupby('method')['sparsity'].mean())  # Avg sparsity
```

## Troubleshooting

### TensorFlow/Keras Issues

If you see errors related to model loading:

```python
# In experiment_final.py, the model is loaded with:
model = tf.keras.models.load_model(..., compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### No Counterfactuals Found

If methods fail to find counterfactuals:
1. Increase `epsilon` (e.g., from 2.0 to 3.0 or 5.0)
2. For preference method: increase `n_samples`
3. Check if sample and target are too far apart

### Memory Issues

If running out of memory:
1. Reduce `n_quantiles` (e.g., from 5 to 3)
2. Reduce `n_samples` for preference method
3. Run test scripts instead of full experiment

## Advanced Usage

### Add a New Dataset

1. Create a new experiment class in `experiment_final.py`:

```python
class NewDatasetExperiment:
    def __init__(self):
        self.dataset_name = "New_Dataset"
        self.model_name = "Model_Name"
        self.feature_names = ['feature1', 'feature2', ...]
        
    def load_data(self):
        # Load your data and model
        return X_train, X_test, y_train, y_test, scaler, model, X_full, y_full
    
    def define_preferences(self, sample, exemplar, X_train, exemplar_weight):
        # Define preferences for your features
        pass
```

2. Update config:

```python
config = {
    'dataset': 'New_Dataset',
    ...
}
```

### Add a New Method

1. Add function to `standard_methods.py`:

```python
def my_new_method(X_original, model_predict, target_value, epsilon, ...):
    # Your implementation
    return counterfactual, prediction, info
```

2. Update `run_all_methods()` in `standard_methods.py`:

```python
cf_new, pred_new, info_new = my_new_method(...)
if cf_new is not None:
    metrics_new = compute_metrics(...)
    results['my_method'] = {...}
```

## Performance Expectations

### Quick Tests (n_quantiles=2)
- Standard methods: ~2-5 minutes
- Preference method: ~3-10 minutes (depends on n_samples)

### Full Experiment (n_quantiles=3)
- 6 sample-target pairs total
- Standard methods: ~5-15 minutes
- Preference method: ~10-30 minutes

### Large Experiment (n_quantiles=5)
- 20 sample-target pairs total
- Standard methods: ~15-45 minutes
- Preference method: ~30-90 minutes

## Citation

If you use these implementations, please cite the original papers:

- **Wachter et al. (2017)**: "Counterfactual Explanations without Opening the Black Box"
- **Growing Spheres**: Laugel et al. (2018)
- **Prototype-based**: Van Looveren & Klaise (2021)

## Support

For issues or questions:
1. Check the test scripts work first
2. Review the CSV output files
3. Check TensorFlow/Keras model compatibility
4. Verify data shapes and types match expected format
