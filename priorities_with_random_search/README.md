# Counterfactual Explanations with Random Search

Generate preference-based counterfactual explanations for binary classification models.

## Quick Start

```bash
python counterfactual_example.py
```

This will:
- Train a model on German Credit Dataset
- Run 90 experiments (10 samples × 9 targets each)
- Save results to `experiment_parameters.csv` and `experiment_results.csv`

## Example Use Case

**Scenario:** Person with 25% credit approval chance wants to reach 70%

**Sample Person:**
- Credit Amount: €5,000
- Loan Duration: 24 months
- Prediction: 0.25 (likely REJECTED)

**Target Person:**
- Credit Amount: €2,000
- Loan Duration: 12 months  
- Prediction: 0.70 (likely APPROVED)

**Result:** System generates counterfactuals like:
```
Rank 1: Reduce credit to €2,100, duration to 11 months
→ Prediction: 0.69 ✓
→ Preference Score: 8.5 (most realistic change)
```

## Configuration

Edit `config` dictionary in `main()`:
- `n_quantiles`: Number of sample points (default: 10)
- `n_samples`: Counterfactuals to generate per experiment (default: 10,000)
- `epsilon`: Target tolerance (default: 0.05)
- `exemplar_weight`: Preference parameter (default: 0.01)
- `run_experiment`: True for batch mode, False for single example with plots

## View Results

```python
import pandas as pd

# All counterfactuals
df = pd.read_csv('experiment_results.csv')

# Filter specific experiment (sample 1 → target 5)
df[(df['sample_idx'] == 1) & (df['target_idx'] == 5)].nsmallest(5, 'cf_rank')
```

Use `random_search_analysis.ipynb` for analysis.

## Output Files

- **experiment_parameters.csv**: Feature statistics and preference settings
- **experiment_results.csv**: All generated counterfactuals (one row per counterfactual)
- **images/**: Preference plots (single-example mode only)
