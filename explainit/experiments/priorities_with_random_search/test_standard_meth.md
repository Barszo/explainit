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

**Test Script:** _Not yet implemented_

### **Purpose**
To be added: Test epsilon and top_k parameters for prototype selection.

---

## **4. GRADIENT-BASED**

**Test Script:** _Not yet implemented_

### **Purpose**
To be added: Test learning rate, lambda, and max_iter parameters for gradient descent optimization.

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
