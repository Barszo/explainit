# Metrics Implementation Documentation

This document describes the metrics calculated in `experiment_random_search.py` for evaluating counterfactual explanations.

## Dataset Context

**Auto MPG Dataset**: 4 continuous features (no categorical features)
- Cylinders
- Displacement  
- Horsepower
- Weight

All metrics treat these features as continuous and use MAD (Median Absolute Deviation) normalization.

---

## Metrics Calculated Per Experiment

Each experiment generates k counterfactuals for a single (original, target) pair. The following metrics are calculated:

### 1️⃣ % Valid CFs

**Formula:**
```python
valid_mask = |predictions - target| <= epsilon
pct_valid_cfs = n_valid / k_requested
```

**Interpretation:**
- Percentage of generated samples that achieve the target prediction within epsilon tolerance
- Example: 0.18% (18/10000) means 18 out of 10,000 samples were valid CFs

---

### 2️⃣ % Valid CFs after eps

**Formula:**
```python
n_valid_after_eps = sum(1 for pred in cf_predictions if |pred - target| <= epsilon)
pct_valid_after_eps = (n_valid_after_eps / total_generated) * 100
```

**Interpretation:**
- Same as "% Valid CFs" - both count CFs within epsilon threshold
- Displayed for consistency with expected output format

---

### 3️⃣ Continuous-Proximity (Mothilal et al. 2020)

**Formula:**
```
Continuous-Proximity = -(1/k) * Σ_(i=1 to k) dist_cont(c_i, x)

where:
dist_cont(c, x) = (1/d) * Σ_p |c_p - x_p| / MAD_p
```

**Implementation:**
```python
for cf in all_cf_samples:
    dist = np.mean(np.abs(np.array(cf) - np.array(sample)) / mad_values)
    proximities.append(dist)
continuous_proximity = -np.mean(proximities)
```

**Interpretation:**
- Negative average MAD-normalized distance from original
- More negative = CFs are farther from original
- Less negative (closer to 0) = CFs are closer to original

---

### 4️⃣ Categorical-Proximity

**Value:** Hardcoded to 1.0

**Reason:** Auto MPG has no categorical features

---

### 5️⃣ Continuous-Sparsity (Mothilal et al. 2020)

**Formula:**
```
Continuous-Sparsity = 1 - (1/(k*d)) * Σ_i Σ_p 1[c_i,p ≠ x_p]
```

**Implementation:**
```python
total_changes = 0
for cf in all_cf_samples:
    n_changes = np.sum(np.abs(np.array(cf) - np.array(sample)) > 1e-6)
    total_changes += n_changes
continuous_sparsity = 1.0 - (total_changes / (n_generated * d))
```

**Interpretation:**
- 1.0 = sparse (no features changed)
- 0.0 = all features changed
- Measures how few features are modified on average

---

### 6️⃣ Continuous-Diversity (Mothilal et al. 2020)

**Formula:**
```
Continuous-Diversity = (1/C(k,2)) * Σ_(i<j) dist_cont(c_i, c_j)

where:
dist_cont(a, b) = (1/d) * Σ_p |a_p - b_p| / MAD_p
```

**Implementation:**
```python
if n_generated > 1:
    pairwise_distances = []
    for i in range(n_generated):
        for j in range(i+1, n_generated):
            dist = np.mean(np.abs(np.array(cf_i) - np.array(cf_j)) / mad_values)
            pairwise_distances.append(dist)
    continuous_diversity = np.mean(pairwise_distances)
```

**Interpretation:**
- Average MAD-normalized distance between all CF pairs
- Higher = more diverse counterfactuals
- Lower = more similar counterfactuals

---

### 7️⃣ Categorical-Diversity

**Value:** Hardcoded to 0.0

**Reason:** Auto MPG has no categorical features

---

### 8️⃣ Cont-Count-Diversity (Mothilal et al. 2020)

**Formula:**
```
Cont-Count-Diversity = (1/(C(k,2) * d)) * Σ_(i<j) Σ_p 1[c_i,p ≠ c_j,p]
```

**Implementation:**
```python
if n_generated > 1:
    count_diffs = []
    for i in range(n_generated):
        for j in range(i+1, n_generated):
            n_diff = np.sum(np.abs(np.array(cf_i) - np.array(cf_j)) > 1e-6)
            count_diffs.append(n_diff)
    cont_count_diversity = np.mean(count_diffs) / d
```

**Interpretation:**
- Average number of different features between CF pairs, normalized by d
- 1.0 = all features differ between CFs
- 0.0 = all CFs are identical

---

### 9️⃣ Highest Priority Value (NEW)

**Formula:**
```python
valid_cf_scores = [score for i, score in enumerate(all_cf_scores) 
                   if |all_cf_predictions[i] - target| <= epsilon]
highest_priority = max(valid_cf_scores)
```

**Interpretation:**
- Maximum priority/preference score among valid CFs
- Shows the best priority value achieved by valid counterfactuals

---

### 🔟 Number of Highest CFs (NEW)

**Formula:**
```python
n_highest = sum(1 for score in valid_cf_scores if score == highest_priority)
pct_highest = (n_highest / len(valid_cf_scores)) * 100
```

**Interpretation:**
- How many valid CFs achieved the highest priority value
- Example: "30.0% (300/1000)" means 300 out of 1000 valid CFs have the max priority
- Indicates how concentrated valid solutions are at the top priority level

---

## Aggregated Metrics (Across All Experiments)

After running all experiments, metrics are aggregated:

- **% Valid CFs**: Total valid / Total requested (across all experiments)
- **% Valid CFs after eps**: Total valid / Total generated (across all experiments)
- **Highest priority value**: Maximum priority among ALL valid CFs
- **Number of highest CFs**: Count and % of valid CFs with max priority
- **Other metrics**: Averaged across experiments

---

## Example Output

```
ARTICLE-BASED METRICS (Mothilal et al. 2020):
  % Valid CFs: 0.18% (18/10000)
  Continuous-Proximity: -1.6261
  Categorical-Proximity: 1.0000
  Continuous-Sparsity: 0.0000 (1.0 = sparse)
  Continuous-Diversity: 1.1740
  Categorical-Diversity: 0.0000
  Cont-Count-Diversity: 1.0000
  % Valid CFs after eps: 100.00% (18/18)
  Highest priority value: 3.0000
  Number of highest CFs: 100.00% (18/18)
```

---

## Key Implementation Details

1. **Per-experiment calculation**: Metrics are calculated for k CFs generated for the same original point, then averaged across experiments

2. **MAD normalization**: All distance calculations use Median Absolute Deviation for scale-invariant comparison

3. **Epsilon threshold**: Used to determine validity (|prediction - target| <= epsilon)

4. **No filtering**: ALL generated counterfactuals are kept (no top-N filtering)

5. **Priority scores**: Based on exponential preference functions that reward proximity to exemplars

---

## Reference

Metrics 1-8 are based on:
**Mothilal, R. K., Sharma, A., & Tan, C. (2020).** Explaining machine learning classifiers through diverse counterfactual explanations. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607-617.

Metrics 9-10 are custom additions for evaluating priority-based counterfactual generation.
