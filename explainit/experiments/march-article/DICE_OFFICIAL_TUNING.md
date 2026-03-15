# Official DiCE Library Performance Tuning Guide

This guide explains how to optimize the official DiCE library (`dice_ml`) for faster counterfactual generation or better quality results.

## Quick Summary

The official DiCE library can be **significantly faster** by adjusting parameters in the `CONFIG` dictionary in `dice_test.py`.

### Current Settings (Balanced)
```python
'dice_official_learning_rate': 0.1,
'dice_official_min_iter': 50,
'dice_official_max_iter': 300,
'dice_official_proximity_weight': 0.2,
'dice_official_diversity_weight': 0.5,
'dice_official_loss_diff_thres': 1e-3,
'dice_official_loss_converge_maxiter': 1,
'dice_official_yloss_type': 'hinge_loss',
```

## Parameter Explanations

### Learning Rate (`dice_official_learning_rate`)
- **What it does**: Controls the step size in gradient descent
- **Default**: 0.05
- **For SPEED**: 0.1 - 0.2 (larger steps = faster convergence)
- **For QUALITY**: 0.01 - 0.05 (smaller steps = more precise)
- **Recommended**: 0.1 for most cases

### Minimum Iterations (`dice_official_min_iter`)
- **What it does**: Minimum iterations before checking for convergence
- **Default**: 500
- **For SPEED**: 20 - 50 (check convergence earlier)
- **For QUALITY**: 200 - 500 (allow more optimization)
- **Recommended**: 50 for speed, 100 for balance

### Maximum Iterations (`dice_official_max_iter`)
- **What it does**: Maximum iterations allowed (hard stop)
- **Default**: 5000
- **For SPEED**: 100 - 300 (stop early)
- **For QUALITY**: 1000 - 5000 (allow full optimization)
- **Recommended**: 300 for speed, 500 for balance

### Proximity Weight (`dice_official_proximity_weight`)
- **What it does**: Penalty for changes from original instance
- **Default**: 0.5
- **For SPEED**: 0.1 - 0.2 (allow more changes = easier to find CFs)
- **For QUALITY**: 0.5 - 1.0 (enforce minimal changes)
- **Recommended**: 0.2 for speed, 0.5 for realistic CFs

### Diversity Weight (`dice_official_diversity_weight`)
- **What it does**: Encourages diverse counterfactuals
- **Default**: 1.0
- **For SPEED**: 0.5 - 1.0
- **For QUALITY**: 1.0 - 2.0
- **Recommended**: 0.5 for speed, 1.0 for diverse CFs

### Loss Convergence Threshold (`dice_official_loss_diff_thres`)
- **What it does**: Minimum loss change to declare convergence
- **Default**: 1e-5 (0.00001)
- **For SPEED**: 1e-2 to 1e-3 (0.01 to 0.001) - more lenient
- **For QUALITY**: 1e-5 to 1e-6 - strict convergence
- **Recommended**: 1e-3 for speed, 1e-4 for balance

### Loss Convergence Max Iterations (`dice_official_loss_converge_maxiter`)
- **What it does**: How many iterations loss must stay below threshold
- **Default**: 1
- **For SPEED**: 1 (declare convergence immediately)
- **For QUALITY**: 2-3 (ensure stable convergence)
- **Recommended**: 1 for speed

### Y-Loss Type (`dice_official_yloss_type`)
- **What it does**: Loss function for classification
- **Options**: 'hinge_loss', 'log_loss', 'l2_loss'
- **Recommended**: 'hinge_loss' for binary classification

## Preset Configurations

### Ultra Fast (Lowest Quality)
```python
CONFIG.update({
    'dice_official_learning_rate': 0.2,
    'dice_official_min_iter': 20,
    'dice_official_max_iter': 100,
    'dice_official_proximity_weight': 0.1,
    'dice_official_loss_diff_thres': 1e-2,
    'num_cfs': 2,  # Also reduce number of CFs
})
```
**Speed**: ~10-20 seconds per sample
**Quality**: May not achieve target class, CFs may be far from original

### Fast (Good Balance) ⭐ RECOMMENDED
```python
CONFIG.update({
    'dice_official_learning_rate': 0.1,
    'dice_official_min_iter': 50,
    'dice_official_max_iter': 300,
    'dice_official_proximity_weight': 0.2,
    'dice_official_loss_diff_thres': 1e-3,
    'num_cfs': 3,
})
```
**Speed**: ~30-60 seconds per sample
**Quality**: Good CFs that usually achieve target

### High Quality (Slow)
```python
CONFIG.update({
    'dice_official_learning_rate': 0.05,
    'dice_official_min_iter': 200,
    'dice_official_max_iter': 1000,
    'dice_official_proximity_weight': 0.5,
    'dice_official_loss_diff_thres': 1e-4,
    'num_cfs': 5,
})
```
**Speed**: ~2-5 minutes per sample
**Quality**: Very high quality, diverse CFs

## How to Apply

### Method 1: Directly in CONFIG
Edit the `CONFIG` dictionary at the top of `dice_test.py`:

```python
CONFIG = {
    # ... other settings ...
    'dice_official_learning_rate': 0.2,  # Change this
    'dice_official_max_iter': 100,      # And this
    # ... etc ...
}
```

### Method 2: Use Preset 6
Uncomment and modify Preset 6 in `dice_test.py`:

```python
# Preset 6: Official DiCE library test - ULTRA FAST settings
CONFIG.update({
    'datasets': ['german_credit'],
    'cf_methods': ['dice_official'],
    'mode': 'binary',
    'n_samples': 5,
    'num_cfs': 2,
    'dice_official_learning_rate': 0.2,
    'dice_official_min_iter': 20,
    'dice_official_max_iter': 100,
    'dice_official_loss_diff_thres': 1e-2,
    'dice_official_proximity_weight': 0.1,
})
```

## Expected Performance

With default settings (max_iter=5000):
- **Time per sample**: 5-10 minutes
- **Total for 10 samples**: 50-100 minutes

With FAST settings (max_iter=300):
- **Time per sample**: 30-60 seconds
- **Total for 10 samples**: 5-10 minutes

With ULTRA FAST settings (max_iter=100):
- **Time per sample**: 10-20 seconds
- **Total for 10 samples**: 2-3 minutes

## Why Official DiCE is Slower

The official `dice_ml` library is slower than custom implementations because:

1. **More sophisticated optimization**: Uses advanced loss calculations
2. **Diversity enforcement**: Ensures generated CFs are diverse from each other
3. **Post-processing**: Performs sparsity enhancement and constraint checks
4. **Feature encoding**: Handles categorical variables properly
5. **Default settings**: Conservative defaults (5000 iterations) for research quality

The custom DiCE implementation in `counterfactual_methods.py` is faster because it's simpler and optimized for speed over comprehensive diversity.

## Recommendations

1. **For quick testing**: Use ULTRA FAST preset
2. **For paper/research**: Use FAST preset (good balance)
3. **For final results**: Use HIGH QUALITY preset
4. **For comparison**: Use same settings across all methods

## Troubleshooting

### Still too slow?
- Reduce `num_cfs` to 2 or even 1
- Reduce `n_samples` in testing
- Test on one dataset first (`'datasets': ['german_credit']`)

### Not achieving target class?
- Increase `max_iter` (allow more optimization)
- Decrease `proximity_weight` (allow more changes)
- Increase `loss_diff_thres` might be too high

### CFs too far from original?
- Increase `proximity_weight` (0.5 - 1.0)
- Adjust `learning_rate` down (0.05)

## Example: Complete Fast Configuration

```python
CONFIG.update({
    # Test settings
    'datasets': ['german_credit'],
    'cf_methods': ['dice_official'],
    'mode': 'binary',
    'n_samples': 5,
    'num_cfs': 3,
    
    # Official DiCE speed optimization
    'dice_official_learning_rate': 0.1,
    'dice_official_min_iter': 50,
    'dice_official_max_iter': 300,
    'dice_official_proximity_weight': 0.2,
    'dice_official_diversity_weight': 0.5,
    'dice_official_loss_diff_thres': 1e-3,
})
```

Run with:
```bash
python dice_test.py
```

This should complete in approximately 5-10 minutes for 5 samples.
