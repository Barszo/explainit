# Configuration Guide

## Quick Start - Method Selection

### Run BOTH methods (default)
```python
config = {
    'run_preference_method': True,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    # ... other settings
}
```

### Run ONLY your preference-based method
```python
config = {
    'run_preference_method': True,
    'run_standard_methods': False,
    # ... other settings
}
```

### Run ONLY standard methods
```python
config = {
    'run_preference_method': False,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    # ... other settings
}
```

### Run SELECTED standard methods
```python
config = {
    'run_preference_method': False,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'prototype'],  # Only these two
    # ... other settings
}
```

## Complete Configuration Template

```python
config = {
    # Dataset and model
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    # Experiment settings
    'n_quantiles': 3,               # Number of test points (2-10)
    'epsilon': 2.0,                 # Target tolerance
    
    # ===== METHOD SELECTION =====
    'run_preference_method': True,  # Enable/disable preference method
    'run_standard_methods': True,   # Enable/disable standard methods
    
    # Which standard methods to run (if run_standard_methods=True)
    'standard_methods': [
        'wachter',          # Optimization-based
        'growing_spheres',  # Geometric search
        'prototype',        # Real instances
        'gradient_based'    # Neural network gradients
    ],
    
    # ===== PREFERENCE METHOD SETTINGS =====
    # (only used if run_preference_method=True)
    'return_top_n': 5,              # Top N CFs to save
    'exemplar_weight': 0.01,        # Exemplar weight
    'n_samples': 10000,             # Candidates to generate
    'use_monte_carlo': True,        # Monte Carlo sampling
}
```

## Standard Methods - Quick Reference

| Method | Speed | Quality | Description |
|--------|-------|---------|-------------|
| `wachter` | Medium | Good | Optimization-based, balanced |
| `growing_spheres` | Fast | Good | Geometric search, uses dataset |
| `prototype` | Fast | High | Real training instances |
| `gradient_based` | Very Fast | Medium | Direct gradient descent |

## Common Configurations

### Quick Test (2-5 minutes)
```python
config = {
    'n_quantiles': 2,
    'epsilon': 3.0,
    'run_preference_method': True,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'growing_spheres'],  # Skip slower methods
    'n_samples': 1000,
}
```

### Standard Test (15-30 minutes)
```python
config = {
    'n_quantiles': 3,
    'epsilon': 2.0,
    'run_preference_method': True,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    'n_samples': 10000,
}
```

### Comprehensive Test (1-2 hours)
```python
config = {
    'n_quantiles': 5,
    'epsilon': 2.0,
    'run_preference_method': True,
    'run_standard_methods': True,
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    'n_samples': 50000,
}
```

## How to Apply Configuration

### Option 1: Edit experiment_final.py directly
Open [experiment_final.py](experiment_final.py) and modify the `config` dictionary in the `main()` function.

### Option 2: Use config_examples.py
```python
# In experiment_final.py main() function:
from config_examples import config_quick_test
config = config_quick_test
```

### Option 3: Command line (if you create wrapper)
```python
import sys
config['run_preference_method'] = '--no-preference' not in sys.argv
```

## Output Files

Based on configuration:

| Setting | Output File |
|---------|-------------|
| `run_preference_method=True` | `experiment_results_preference_based.csv` |
| `run_standard_methods=True` | `experiment_results_standard_methods.csv` |
| Both enabled | Both CSV files |

## Timing Estimates

Experiment time depends on `n_quantiles`:

| n_quantiles | Experiments | Est. Time (both methods) |
|-------------|-------------|--------------------------|
| 2 | 2 | 5-15 min |
| 3 | 6 | 15-45 min |
| 4 | 12 | 30-90 min |
| 5 | 20 | 45-120 min |

Add ~50% time if `n_samples` > 10000 for preference method.

Reduce time by:
- Lowering `n_quantiles`
- Reducing `standard_methods` list
- Lowering `n_samples` for preference method
- Disabling one method type
