# Testing Standard Counterfactual Methods

This document describes the test scripts for evaluating standard counterfactual explanation methods with different parameter configurations.

---

## **1. WACHTER'S METHOD**

**Test Script:** `test_wachter_method.py`

### **Purpose**
Analyze how lambda (prediction vs distance weight) and epsilon (tolerance) parameters affect counterfactual quality and success rate.

### **Test Input**

**Data Selection:**
```python
# Automatically selects 3 quantile points from test set:
Sample 1: Low MPG prediction  (e.g., 8.85 MPG)
Sample 2: Medium MPG prediction (e.g., 24.30 MPG)
Sample 3: High MPG prediction (e.g., 37.30 MPG)

# Test scenario: Low → High (most challenging)
Original: 8.85 MPG
Target:   37.30 MPG
Change needed: +28.45 MPG
```

### **Parameters Tested**

| Parameter | Values | Description |
|-----------|--------|-------------|
| **lambda** | [0.01, 0.1, 0.5, 1.0, 5.0, 10.0] | Weight: prediction loss vs distance |
| **epsilon** | [0.5, 1.0, 2.0, 3.0, 5.0] MPG | Tolerance for target (±epsilon) |

**Total combinations:** 6 × 5 = 30

### **Test Workflow**

1. **Gradient Sensitivity Test**
   - Tests if loss function responds to feature changes
   - Identifies if optimization can proceed

2. **Parameter Sweep**
   - For each epsilon value:
     - For each lambda value:
       - Run optimization (tries multiple optimizers: L-BFGS-B, SLSQP, Powell)
       - Record: validity, prediction, distance, sparsity, iterations

3. **Summary Analysis**
   - Best by distance
   - Best by sparsity
   - Parameter effect insights

### **Output Format**

**Per Configuration:**
```
Lambda =  0.50 (weight: prediction vs distance)
──────────────────────────────────────────────────────────────────────────
✓ VALID counterfactual found
  Optimizer: L-BFGS-B
  Prediction: 36.80 MPG (target: 37.30, error: 0.50)
  Distance: L2=0.4523, L1=1.2341
  Sparsity: 3 features changed
  Optimization: 47 iters, 235 func evals
  Loss: Initial=809.546326 → Final=0.458231 (reduction: 99.9%)
  
  Feature Changes:
    displacement   : 0.8234 → 0.3456 (Δ=-0.4778, -58.0%)
    horsepower     : 0.7123 → 0.4567 (Δ=-0.2556, -35.9%)
    weight         : 0.8901 → 0.5234 (Δ=-0.3667, -41.2%)
```

**Summary Table:**
```
Epsilon   Lambda    Valid     Prediction  Pred Error  L2 Distance    Sparsity    Iterations  
────────────────────────────────────────────────────────────────────────────────────────────
0.5       0.01      ✗         8.85        28.45       0.0000         0           0           
0.5       0.10      ✗         8.85        28.45       0.0000         0           0           
0.5       0.50      ✓         37.10       0.20        0.4523         3           47          
1.0       0.50      ✓         37.25       0.05        0.4812         3           52          
2.0       0.50      ✓         36.90       0.40        0.4234         3           41          
```

### **Interpretation Guide**

**Valid (✓) vs Invalid (✗):**
- ✓ = Found counterfactual within epsilon of target
- ✗ = Failed to reach target or optimizer didn't move

**Lambda Effect:**
- **Low (0.01-0.1):** Prioritizes proximity → smaller distance, may not reach target
- **Medium (0.5-1.0):** Balanced → good tradeoff
- **High (5.0-10.0):** Prioritizes target → reaches target, larger distance

**Epsilon Effect:**
- **Strict (0.5-1.0):** Fewer valid solutions, more precise
- **Relaxed (3.0-5.0):** More valid solutions, less precise

**Common Issues:**
- **0 iterations:** Gradient computation failed, model insensitive
- **High sparsity:** Changes many features (not sparse)
- **No change (distance=0):** Optimization completely failed

### **How to Run**
```bash
python explainit/experiments/priorities_with_random_search/test_wachter_method.py
```

---

## **2. GROWING SPHERES**

**Test Script:** `test_growing_spheres_method.py`

### **Purpose**
Evaluate how epsilon, n_search_samples, and n_top_candidates parameters affect counterfactual quality and success rate.

### **Test Input**

**Data Selection:**
```python
# Same as Wachter: 3 quantile points
Sample 1: Low MPG (8.85 MPG)
Sample 2: Medium MPG (24.30 MPG)
Sample 3: High MPG (37.30 MPG)

# Uses training data predictions
Training samples: 313 instances
Training predictions: Pre-computed for filtering
```

### **Parameters Tested**

| Parameter | Values | Description |
|-----------|--------|-------------|
| **epsilon** | [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] MPG | Tolerance for target |
| **n_search_samples** | [5, 10, 20, 50, 100] | Interpolation granularity |
| **n_top_candidates** | [5, 10, 20] | Number of prototypes to try |

### **Test Workflow**

1. **Training Data Analysis**
   - For each epsilon: Count available training instances near target
   - Shows distance range of available prototypes

2. **Test 1: Epsilon Sensitivity**
   - Fixed: n_search_samples=20, n_top_candidates=10
   - Vary epsilon to find which values work

3. **Test 2: n_search_samples Sensitivity** (only if Test 1 succeeds)
   - Fixed: epsilon from Test 1, n_top_candidates=10
   - Vary n_search_samples to see interpolation effect

4. **Test 3: n_top_candidates Sensitivity** (only if Test 1 succeeds)
   - Fixed: epsilon from Test 1, n_search_samples=20
   - Vary n_top_candidates to see breadth effect

### **Output Format**

**Training Data Analysis:**
```
TRAINING DATA ANALYSIS
════════════════════════════════════════════════════════════════════════════
Epsilon= 0.5:   12 training instances found | Distance range: [0.3456, 0.8923], mean=0.5234
Epsilon= 1.0:   28 training instances found | Distance range: [0.2341, 0.9456], mean=0.4892
Epsilon= 2.0:   45 training instances found | Distance range: [0.1234, 1.0234], mean=0.4567
Epsilon= 5.0:   89 training instances found | Distance range: [0.0567, 1.2341], mean=0.4123
```

**Per Configuration:**
```
Epsilon =  2.0 MPG
──────────────────────────────────────────────────────────────────────────
✓ VALID counterfactual found
  Prediction: 36.80 MPG (target: 37.30, error: 0.50)
  Distance: L2=0.3245, L1=0.8923
  Sparsity: 4 features changed
  Training instances available: 45
  Candidates tried: 10, Valid found: 3
  
  Feature Changes:
    displacement   : 0.8234 → 0.4567 (Δ=-0.3667, -44.5%)
    horsepower     : 0.7123 → 0.3890 (Δ=-0.3233, -45.4%)
    weight         : 0.8901 → 0.5123 (Δ=-0.3778, -42.4%)
    acceleration   : 0.2345 → 0.4567 (Δ=+0.2222, +94.7%)
```

**Summary Table:**
```
Epsilon   Valid     Prediction  Pred Error  L2 Distance    Sparsity    Candidates
──────────────────────────────────────────────────────────────────────────────────
0.5       ✗         N/A         N/A         N/A            N/A         12          
1.0       ✓         37.10       0.20        0.3456         4           28          
2.0       ✓         36.80       0.50        0.3245         4           45          
3.0       ✓         36.50       0.80        0.2987         4           67          
5.0       ✓         35.90       1.40        0.2456         4           89          
```

### **Interpretation Guide**

**Epsilon Effect:**
- **Too strict (0.5):** No training instances → fails
- **Optimal (1.0-2.0):** Enough instances, precise target
- **Too relaxed (5.0+):** Many instances, less precise, finds closer counterfactual

**n_search_samples Effect:**
- **5 vs 20:** Typically 5-10% distance improvement
- **20 vs 100:** Minimal improvement (<2%)
- **Recommendation:** 20-50 is sweet spot

**n_top_candidates Effect:**
- **5 vs 10:** May find 10-20% better counterfactual if better paths exist
- **10 vs 20:** Usually minimal improvement
- **Recommendation:** 10 is typically sufficient

**Success Indicators:**
- ✓ Valid + Low distance = Excellent
- ✓ Valid + High candidates tried = Had to search far
- ✗ Failed + 0 candidates = No training data near target (increase epsilon)
- ✗ Failed + Many candidates = Target unreachable with this method

### **How to Run**
```bash
python explainit/experiments/priorities_with_random_search/test_growing_spheres_method.py
```

---

## **3. PROTOTYPE-BASED**

**Test Script:** `test_prototype_based_method.py`

### **Purpose**
Evaluate how epsilon and top_k parameters affect prototype selection quality. Unlike Growing Spheres which interpolates between points, this method returns actual training instances as counterfactuals.

### **Test Input**

**Data Selection:**

Uses the same three quantile-based samples (low, medium, high MPG predictions) as other methods, but tests ALL possible sample-target combinations to evaluate the method across different scenarios.

**All Scenarios Tested:**
- Sample 1 → Target 2 (Low → Medium)
- Sample 1 → Target 3 (Low → High)
- Sample 2 → Target 1 (Medium → Low)
- Sample 2 → Target 3 (Medium → High)
- Sample 3 → Target 1 (High → Low)
- Sample 3 → Target 2 (High → Medium)

**Total:** 6 scenarios × multiple parameter combinations per scenario

**Why Test All Combinations?**

Prototype-based methods depend heavily on training data distribution. Different target directions may have very different prototype availability:
- Some targets have many nearby training instances (easy)
- Some targets are in sparse regions (hard or impossible)
- Testing all combinations reveals which scenarios work best

**Key Difference from Other Methods:**

Prototype-based returns REAL training instances, not synthetic counterfactuals. This guarantees realistic feature combinations but may result in larger distances than interpolation-based methods like Growing Spheres.

### **Parameters Tested**

| Parameter | Values | Description |
|-----------|--------|-------------|
| **epsilon** | [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] MPG | Tolerance for target prediction |
| **top_k** | [1, 2, 3, 5, 10] | Which k-th nearest prototype to return |

**top_k Explanation:**
- top_k=1: Returns the closest prototype to original (minimum distance)
- top_k=2: Returns the 2nd closest prototype (offers diversity)
- top_k=5: Returns the 5th closest prototype (more distant alternative)

### **Test Workflow**

For each of the 6 sample-target scenarios:

1. **Training Data Analysis**
   - For each epsilon: Count available prototypes near target
   - Shows distance range of available prototypes

2. **Test 1: Epsilon Sensitivity**
   - Fixed: top_k=1 (closest prototype)
   - Vary epsilon to find which values provide valid prototypes

3. **Test 2: top_k Sensitivity** (only if Test 1 succeeds)
   - Fixed: epsilon from Test 1
   - Vary top_k to see how distance increases with more distant prototypes

4. **Aggregate Summary**
   - Success rate across all scenarios
   - Best configurations by scenario
   - Overall statistics

### **Output Format**

**Training Data Analysis (per scenario):**
```
TRAINING DATA ANALYSIS
════════════════════════════════════════════════════════════════════════════
Epsilon= 0.5:   12 prototypes available | Distance range: [0.3456, 0.8923], mean=0.5234
Epsilon= 1.0:   28 prototypes available | Distance range: [0.2341, 0.9456], mean=0.4892
Epsilon= 2.0:   45 prototypes available | Distance range: [0.1234, 1.0234], mean=0.4567
```

**Per Configuration:**
```
Epsilon =  2.0 MPG
──────────────────────────────────────────────────────────────────────────
✓ VALID prototype found
  Prediction: 36.80 MPG (target: 37.30, error: 0.50)
  Distance: L2=0.4567, L1=1.2345
  Sparsity: 4 features changed
  Available prototypes: 45
  Prototype rank: 1 (closest)
  Real training instance: Yes
  
  Feature Changes (original → prototype):
    displacement   : 0.8234 → 0.4567 (Δ=-0.3667, -44.5%)
    horsepower     : 0.7123 → 0.3890 (Δ=-0.3233, -45.4%)
    weight         : 0.8901 → 0.5123 (Δ=-0.3778, -42.4%)
    acceleration   : 0.2345 → 0.4567 (Δ=+0.2222, +94.7%)
```

**Overall Summary Across All Scenarios:**
```
OVERALL SUMMARY - ALL SCENARIOS
════════════════════════════════════════════════════════════════════════════

SUCCESS RATE BY SCENARIO
────────────────────────────────────────────────────────────────────────────
Sample 1 → Target 2: 15.50 → 24.30 MPG (+8.80)
  Valid configurations: 5/6 epsilon values tested
  Best: epsilon=2.0, distance=0.3245, sparsity=3

Sample 1 → Target 3: 15.50 → 37.30 MPG (+21.80)
  Valid configurations: 4/6 epsilon values tested
  Best: epsilon=2.0, distance=0.4567, sparsity=4

OVERALL STATISTICS
────────────────────────────────────────────────────────────────────────────
Total scenarios tested: 6
Scenarios with valid prototypes: 6/6 (100.0%)

Average metrics (best solutions):
  Average L2 distance: 0.3892
  Average sparsity: 3.5 features
```

### **Interpretation Guide**

**Epsilon Effect:**
- **Too strict (0.5-1.0):** Few or no prototypes available → often fails
- **Optimal (2.0-3.0):** Good balance of availability and precision
- **Too relaxed (5.0+):** Many prototypes but less precise prediction match

**top_k Effect:**
- **k=1:** Closest prototype (minimum distance)
- **k>1:** More distant prototypes
  - May offer diversity in feature changes
  - Distance increases linearly with k
  - Use when closest prototype is undesirable for some reason

**Comparison with Growing Spheres:**
- **Prototype-Based:** Returns REAL training instances
  - Guarantees realistic feature combinations (actually observed)
  - Typically LARGER distances (can't interpolate)
  - May fail if no training instance near target
- **Growing Spheres:** Interpolates between training instances
  - Creates SYNTHETIC counterfactuals (may be unrealistic)
  - Typically SMALLER distances (can find points between instances)
  - More flexible in finding solutions

**Success Indicators:**
- ✓ Valid + Many prototypes = Good scenario, target well-represented in training
- ✓ Valid + Few prototypes = Marginal scenario, sensitive to epsilon
- ✗ Failed + 0 prototypes = Target outside training data range (increase epsilon)
- All scenarios succeed = Method is robust for this dataset

**Scenario-Specific Insights:**

Look at which sample→target combinations succeed:
- **High success (5-6/6 epsilon values):** Target well-represented in training data
- **Medium success (3-4/6 epsilon values):** Target in moderate density region
- **Low success (0-2/6 epsilon values):** Target in sparse region or outside range

### **How to Run**
```bash
python explainit/experiments/priorities_with_random_search/test_prototype_based_method.py
```

**Note:** This test runs significantly longer than Wachter or Growing Spheres because it tests all 6 sample-target combinations. Expected runtime: 2-5 minutes.

---

## **4. GRADIENT-BASED**

**Test Script:** `test_gradient_based_method.py`

### **Purpose**
Test learning_rate, lambda, and epsilon parameters for gradient descent optimization through neural networks. This method exploits differentiability to compute exact gradients and optimize counterfactuals efficiently.

**CRITICAL REQUIREMENT:** Only works with TensorFlow/Keras neural networks. Cannot be used with tree-based models, traditional ML, or non-differentiable models.

### **Test Input**

**Data Selection:**

Uses the same three quantile-based samples as other methods, and tests ALL 6 possible sample-target combinations to evaluate performance across different scenarios (same as Prototype-Based method).

**All Scenarios Tested:**
- Sample 1 → Target 2 (Low → Medium)
- Sample 1 → Target 3 (Low → High)
- Sample 2 → Target 1 (Medium → Low)
- Sample 2 → Target 3 (Medium → High)
- Sample 3 → Target 1 (High → Low)
- Sample 3 → Target 2 (High → Medium)

**Model Verification:**

The script first verifies that the model is a neural network before proceeding. If not, it exits with an error message.

**Key Difference from Other Methods:**

Gradient-based creates SYNTHETIC counterfactuals through iterative optimization using backpropagation. Unlike Growing Spheres (which interpolates) or Prototype-Based (which returns real instances), this method directly modifies feature values using gradient information from the neural network.

### **Parameters Tested**

| Parameter | Values | Description |
|-----------|--------|-------------|
| **epsilon** | [0.5, 1.0, 2.0, 3.0, 5.0] MPG | Tolerance for target prediction |
| **learning_rate** | [0.001, 0.01, 0.05, 0.1, 0.5] | Gradient descent step size |
| **lambda** | [0.01, 0.1, 0.5, 1.0, 5.0] | Weight: prediction loss vs distance |

**Fixed parameter:** max_iter = 500 iterations

**Parameter Explanations:**
- **learning_rate**: Controls how aggressively features are updated each iteration
  - Too low → Slow convergence, may not reach target
  - Too high → Unstable, may overshoot or diverge
- **lambda**: Balances achieving target prediction vs staying close to original
  - Low → Prioritizes proximity (smaller changes)
  - High → Prioritizes reaching target (may make larger changes)
- **epsilon**: Defines success threshold for target match

### **Test Workflow**

For each of the 6 sample-target scenarios:

1. **Model Type Verification**
   - Check if model is TensorFlow/Keras neural network
   - Exit if not compatible (cannot use gradient-based with other model types)

2. **Test 1: Epsilon Sensitivity**
   - Fixed: learning_rate=0.01, lambda=1.0
   - Vary epsilon to find tolerance levels that work

3. **Test 2: Learning Rate Sensitivity** (only if Test 1 succeeds)
   - Fixed: epsilon from Test 1, lambda=1.0
   - Vary learning_rate to find optimal step size

4. **Test 3: Lambda Sensitivity** (only if Test 1 succeeds)
   - Fixed: epsilon from Test 1, learning_rate=0.01
   - Vary lambda to see prediction vs distance tradeoff

5. **Aggregate Summary**
   - Success rate across all scenarios
   - Best configurations by scenario
   - Overall statistics and comparison insights

### **Output Format**

**Model Verification:**
```
GRADIENT-BASED METHOD - PARAMETER SENSITIVITY ANALYSIS
════════════════════════════════════════════════════════════════════════════
✓ Model is a neural network (TensorFlow/Keras)
  Model type: <class 'keras.src.models.functional.Functional'>
  Model architecture: 12345 parameters
```

**Per Configuration:**
```
Epsilon =  2.0 MPG
──────────────────────────────────────────────────────────────────────────
✓ VALID counterfactual found
  Prediction: 36.85 MPG (target: 37.30, error: 0.45)
  Distance: L2=0.3892, L1=1.0234
  Sparsity: 4 features changed
  Iterations: 87
  Final loss: 0.000234
  
  Feature Changes:
    displacement   : 0.8234 → 0.4123 (Δ=-0.4111, -49.9%)
    horsepower     : 0.7123 → 0.3456 (Δ=-0.3667, -51.5%)
    weight         : 0.8901 → 0.4789 (Δ=-0.4112, -46.2%)
    acceleration   : 0.2345 → 0.5678 (Δ=+0.3333, +142.1%)
```

**Overall Summary Across All Scenarios:**
```
OVERALL SUMMARY - ALL SCENARIOS
════════════════════════════════════════════════════════════════════════════

SUCCESS RATE BY SCENARIO
────────────────────────────────────────────────────────────────────────────
Sample 1 → Target 2: 15.50 → 24.30 MPG (+8.80)
  Valid configurations: 5/5 epsilon values tested
  Best: epsilon=1.0, distance=0.2987, sparsity=3

Sample 1 → Target 3: 15.50 → 37.30 MPG (+21.80)
  Valid configurations: 4/5 epsilon values tested
  Best: epsilon=2.0, distance=0.3892, sparsity=4

OVERALL STATISTICS
────────────────────────────────────────────────────────────────────────────
Total scenarios tested: 6
Scenarios with valid counterfactuals: 6/6 (100.0%)

Average metrics (best solutions):
  Average L2 distance: 0.3245
  Average sparsity: 3.7 features
```

### **Interpretation Guide**

**Epsilon Effect:**
- **Strict (0.5-1.0):** Harder to satisfy, may require many iterations
- **Optimal (1.0-2.0):** Good balance of precision and convergence speed
- **Relaxed (3.0-5.0):** Easier to satisfy, converges faster

**Learning Rate Effect:**
- **Too low (0.001):** Very slow convergence, may not reach target in 500 iterations
- **Optimal (0.01-0.05):** Steady convergence, reliable results
- **Too high (0.1-0.5):** Fast but unstable, may overshoot or oscillate

**Lambda Effect:**
- **Low (0.01-0.1):** Prioritizes small changes → may not reach target
- **Medium (0.5-1.0):** Balanced approach → good tradeoff
- **High (5.0+):** Prioritizes target → larger changes, but reaches target reliably

**Comparison with Other Methods:**

| Aspect | Gradient-Based | Wachter | Growing Spheres | Prototype |
|--------|----------------|---------|-----------------|-----------|
| **Model Type** | Neural networks ONLY | Any black-box | Any black-box | Any black-box |
| **Result Type** | Synthetic (optimized) | Synthetic (optimized) | Synthetic (interpolated) | Real instance |
| **Efficiency** | Fast (uses gradients) | Slow (numeric gradients) | Medium (searches) | Very fast (lookup) |
| **Typical Distance** | Small-Medium | Medium | Small | Medium-Large |
| **Sparsity** | Low (changes many) | Low (changes many) | Low (changes many) | N/A (real instance) |
| **Convergence** | Good (with right params) | Variable (local minima) | Deterministic | Always succeeds if data exists |

**Success Indicators:**
- ✓ Valid + Low iterations (<100) = Good learning rate and lambda
- ✓ Valid + Many iterations (>300) = Sub-optimal parameters, but still works
- ✗ Failed + Low iterations = Learning rate too low or lambda imbalanced
- ✗ Failed + Many iterations = Target unreachable or epsilon too strict

**Scenario-Specific Insights:**

Success patterns indicate:
- **All succeed with low iterations:** Well-conditioned optimization problem
- **Some succeed, some fail:** Target difficulty varies by direction
- **High success but many iterations:** Parameters need tuning (increase learning_rate or adjust lambda)

### **How to Run**
```bash
python explainit/experiments/priorities_with_random_search/test_gradient_based_method.py
```

**Requirements:**
- Model MUST be a TensorFlow/Keras neural network
- Script will check model type and exit if incompatible
- Expected runtime: 3-7 minutes (longer than other methods due to iterative optimization)

**Note:** If using with different models, ensure they are TensorFlow/Keras neural networks. The method will not work with scikit-learn models, XGBoost, LightGBM, or other non-differentiable models.

---

## **General Notes**

### **Why These Tests?**

1. **Parameter Sensitivity:** Understand which parameters matter most
2. **Success Rate:** See which configurations reliably find counterfactuals
3. **Quality Metrics:** Compare distance, sparsity, prediction accuracy
4. **Practical Guidance:** Determine optimal parameter ranges

### **Common Output Patterns**

**All methods fail:**
- Target may be unreachable for this instance/model
- Try: Larger epsilon, different target, or different method

**Some configs work:**
- Identify working parameter ranges
- Use insights to tune experiment_final.py

**All configs work:**
- Good scenario! Compare quality metrics
- Choose based on distance vs sparsity tradeoff

### **Next Steps**

After running tests:
1. Identify best parameter configurations
2. Update `experiment_final.py` config with optimal values
3. Run full experiments with multiple sample-target pairs
4. Analyze aggregated results across dataset
