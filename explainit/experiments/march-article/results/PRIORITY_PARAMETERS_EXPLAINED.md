# Priority Parameters CSV - Complete Guide

## Table of Contents
1. [What is saved in priority_parameters.csv?](#what-is-saved)
2. [The Preference Creation Workflow](#workflow)
3. [Mathematical Details](#mathematics)
4. [Worked Examples](#examples)
5. [File Structure](#file-structure)

---

## What is saved in `priority_parameters.csv`? {#what-is-saved}

The `preference_based_*_priority_parameters.csv` files save **one row per feature for each sample processed**.

### Row Count Calculation

**Number of rows = Number of samples × Number of features**

For example:
- **Communities and Crime dataset**: 10 samples × 99 features = **990 rows** (+ 1 header row = 991 lines total)
- **German Credit dataset**: 10 samples × 20 features = **200 rows** (+ 1 header row = 201 lines total)

---

## The Preference Creation Workflow {#workflow}

The preference-based method generates counterfactuals by creating **feature-specific preference functions** that guide the search. Here's the complete workflow:

### Step 1: Find a Target Sample (Exemplar)

For each original sample to explain, the algorithm first finds a **target sample** from the training data:

```python
target_sample = find_target_sample(
    X_train, y_train, model,
    target_class=1,           # Want to reach class 1
    target_probability=0.75   # With ~75% confidence
)
```

**What happens:**
- Filters training data for samples in `target_class` (e.g., class 1)
- Finds the sample whose prediction is **closest to target_probability** (e.g., 0.75)
- This target sample serves as a "guide" showing reasonable feature values for the target class

**Example:**
- Original sample: `[age=25, income=30k, ...]` → predicted class 0 (loan denied)
- Target sample: `[age=35, income=50k, ...]` → predicted class 1 at 0.74 probability (loan approved)

### Step 2: Create Preferences for Each Feature

For **every feature** in the sample, the algorithm creates a preference function by comparing:
- **sample_value**: Original value in the sample to explain
- **target_value**: Corresponding value in the target sample
- **dataset_min/max**: Valid range for this feature in the dataset

```python
for each feature:
    preference_func = create_numerical_preference_function(
        sample_value=original_sample[feature],
        target_value=target_sample[feature],
        dataset_min=X_train[:, feature].min(),
        dataset_max=X_train[:, feature].max(),
        exemplar_weight=0.9  # How strictly to follow target
    )
```

### Step 3: Generate Counterfactuals

The `RandomSearchExplainer` uses these preference functions to:
1. **Sample candidate solutions** (random modifications to the original sample)
2. **Evaluate each candidate** by:
   - Checking if it meets the prediction target (e.g., class 1)
   - Calculating preference score (how well it satisfies all preferences)
3. **Return top-N candidates** with highest preference scores

**Key insight:** Candidates that:
- Stay closer to the original sample → higher preference
- Move toward the target values → still acceptable (based on exemplar_weight)
- Move away from both → lower preference

---

## Mathematical Details {#mathematics}

### The Exponential Preference Function

Each feature's preference is modeled using an exponential function:

```
f(x) = (exp(a*t) - 1) / (exp(a) - 1)

where: t = (x - x0) / (x1 - x0)
       t is clipped to [0, 1]
```

**Parameters:**
- `a = 5`: Steepness parameter (higher = steeper curve)
- `x0, x1`: Boundary points defining the preference range
- `increasing`: Direction of preference (True/False)

**Preference values:**
- `f(sample_value) = 1.0` → **Most preferred** (current value)
- `f(target_value) = exemplar_weight` → **Acceptability boundary**
- `f(extreme_value) = 0.0` → **Unacceptable**

### Computing x0 and x1

The algorithm calculates `x0` and `x1` such that the preference function passes through the desired points:

**Step 1:** Find parameter `t` where function equals `exemplar_weight`:
```
t_target = log(1 + exemplar_weight × (exp(a) - 1)) / a
```

**Step 2:** Determine direction and calculate boundaries:

**Case A: sample_value < target_value** (need to increase)
```
Direction: DECREASING (prefer lower = current value)
x0 = sample_value          # Most preferred (returns 1.0)
x1 = (target_value - t_target × sample_value) / (1 - t_target)
increasing = False
```

**Case B: sample_value > target_value** (need to decrease)
```
Direction: INCREASING (prefer lower values)
x0 = (target_value - t_target × sample_value) / (1 - t_target)
x1 = sample_value          # Most preferred (returns 1.0)
increasing = True
```

### Acceptable Ranges

After computing `x0` and `x1`, the acceptable range for sampling is:

```
if increasing:
    acceptable_min = max(dataset_min, x0)
    acceptable_max = dataset_max
else:
    acceptable_min = dataset_min
    acceptable_max = min(dataset_max, x0)
```

---

## Worked Examples {#examples}

### Example 1: Income Feature (Decrease Needed)

**Scenario:**
- Original sample: income = **0.92** (standardized, high income)
- Target sample: income = **0.25** (standardized, moderate income)
- Dataset range: [-1.18, 2.29]
- exemplar_weight = 0.9

**Need to decrease income**, so we want to prefer lower values (increasing preference from low to high).

**Calculations:**
```
a = 5
t_target = log(1 + 0.9 × (exp(5) - 1)) / 5 = 0.9618

Since sample_value (0.92) > target_value (0.25):
  x0 = (0.25 - 0.9618 × 0.92) / (1 - 0.9618) = -30.88
  x1 = 0.92
  increasing = True
  direction = "increasing"

acceptable_min = max(-1.18, -30.88) = -1.18
acceptable_max = 2.29
```

**Interpretation:**
- `f(0.92) = 1.0` → Original value most preferred
- `f(0.25) ≈ 0.9` → Target value still highly acceptable
- `f(-30.88) = 0.0` → Very low values unacceptable
- Values between 0.25 and 0.92 are highly preferred
- Values below 0.25 gradually become less acceptable

**Saved row:**
```csv
0,5,income,0.92,0.25,-30.88,0.92,increasing,True,5,0.9,-1.18,2.29,-1.18,2.29
```

### Example 2: Age Feature (Increase Needed)

**Scenario:**
- Original sample: age = **-0.44** (standardized, young)
- Target sample: age = **0.22** (standardized, slightly older)
- Dataset range: [-1.48, 1.92]
- exemplar_weight = 0.9

**Need to increase age**, so we want to prefer lower values (closer to current).

**Calculations:**
```
a = 5
t_target = 0.9618

Since sample_value (-0.44) < target_value (0.22):
  x0 = -0.44
  x1 = (0.22 - 0.9618 × (-0.44)) / (1 - 0.9618) = 16.89
  increasing = False
  direction = "decreasing"

acceptable_min = -1.48
acceptable_max = min(1.92, -0.44) = -0.44
```

**Interpretation:**
- `f(-0.44) = 1.0` → Original value most preferred
- `f(0.22) ≈ 0.9` → Target value acceptable
- `f(16.89) = 0.0` → Very high values unacceptable
- Values between -0.44 and 0.22 are acceptable
- Moving toward 0.22 gradually reduces preference but stays acceptable

**Saved row:**
```csv
0,3,age,-0.44,0.22,-0.44,16.89,decreasing,False,5,0.9,-1.48,-0.44,-1.48,1.92
```

### Example 3: Unchanged Feature

**Scenario:**
- Original sample: education = **-0.70**
- Target sample: education = **-0.70** (same value!)
- Dataset range: [-0.70, 3.46]
- exemplar_weight = 0.9

**No change needed**, but still needs a preference function.

**Calculations:**
```
Since sample_value == target_value:
  x0 = -0.70
  x1 = -0.70
  increasing = True
  direction = "increasing"

acceptable_min = -0.70
acceptable_max = 3.46
```

**Interpretation:**
- `f(-0.70) = 1.0` → Original value most preferred
- Since x0 = x1, function returns 0 below -0.70, and 1 at or above -0.70
- Effectively says: "keep this value or any higher value is equally acceptable"

**Saved row:**
```csv
0,22,education,-0.70,-0.70,-0.70,-0.70,increasing,True,5,0.9,-0.70,3.46,-0.70,3.46
```

### Example 4: Effect of exemplar_weight

Let's see how `exemplar_weight` affects the preference function:

**Same scenario: income = 0.92 → 0.25**

| exemplar_weight | Interpretation | x0 | x1 |
|----------------|----------------|-----|-----|
| 0.01 | Very strict - must stay very close to original | -0.92 | 0.92 |
| 0.5 | Balanced - moderate flexibility | -7.46 | 0.92 |
| 0.9 | Permissive - allows significant change | -30.88 | 0.92 |

**Effect on counterfactual generation:**
- **Low exemplar_weight (0.01)**: Counterfactuals stay very close to original sample
- **Medium exemplar_weight (0.5)**: Balanced trade-off between proximity and target
- **High exemplar_weight (0.9)**: More freedom to change features toward target

---

## File Structure {#file-structure}

### CSV Columns

Each row contains these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `sample_id` | Which sample is being processed | 0 |
| `feature_index` | Which feature (0 to N-1) | 5 |
| `feature_name` | Human-readable feature name | "income" |
| `sample_value` | Original value in sample | 0.92 |
| `target_value` | Value in target sample | 0.25 |
| `x0` | Lower boundary of preference range | -30.88 |
| `x1` | Upper boundary of preference range | 0.92 |
| `direction` | "increasing" or "decreasing" | "increasing" |
| `increasing` | Boolean flag for direction | True |
| `a` | Steepness parameter (always 5) | 5 |
| `exemplar_weight` | Weight at target value | 0.9 |
| `acceptable_min` | Minimum acceptable value for sampling | -1.18 |
| `acceptable_max` | Maximum acceptable value for sampling | 2.29 |
| `dataset_min` | Minimum value in training data | -1.18 |
| `dataset_max` | Maximum value in training data | 2.29 |

### Why These Values Matter

**For reproducibility:**
- You can recreate the exact preference function: `f(x) = exponential(x, x0, x1, increasing, a)`
- You know the acceptable sampling range: `[acceptable_min, acceptable_max]`

**For understanding:**
- `sample_value` vs `target_value` shows the desired direction of change
- `x0` and `x1` show how much flexibility the method has
- `direction` indicates whether the preference increases or decreases
- `exemplar_weight` shows how strictly the method follows the target

### File Lifecycle

**The file is OVERWRITTEN (not appended) each time you run `explainit_meth_test.py`.**

- Old files are explicitly deleted when you run the script (via `clean_old_results()`)
- Fresh parameters are saved for the current run
- Each method (binary/continuous) has its own separate file

### Example Complete Row

```csv
sample_id,feature_index,feature_name,sample_value,target_value,x0,x1,direction,increasing,a,exemplar_weight,acceptable_min,acceptable_max,dataset_min,dataset_max
0,5,income,0.92,0.25,-30.88,0.92,increasing,True,5,0.9,-1.18,2.29,-1.18,2.29
```

**Reading this row:**
- Sample 0, feature 5 (income)
- Original: 0.92, Target: 0.25 → need to decrease
- Preference function: exponential(x, x0=-30.88, x1=0.92, increasing=True, a=5)
- Can sample from [-1.18, 2.29]
- Prefers values close to 0.92, accepts down to 0.25 (weight=0.9), rejects extreme low values
