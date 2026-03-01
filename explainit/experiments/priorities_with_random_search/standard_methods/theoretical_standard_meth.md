# **Standard Counterfactual Explanation Methods - Detailed Analysis**

## **1. WACHTER'S METHOD (2017)**

### **Theoretical Description**
Wachter's method formulates counterfactual generation as a constrained optimization problem. It balances two competing objectives: achieving the desired prediction (prediction loss) and staying close to the original instance (distance penalty). The optimization minimizes:

**Loss Function:** `λ × (f(x') - target)² + ||x' - x||²`

Where:
- `f(x')` = model prediction for counterfactual x'
- `target` = desired prediction value
- `λ` = weight parameter balancing prediction vs distance
- `||x' - x||²` = Euclidean distance between counterfactual and original

### **Workflow**
```
1. Start: Original instance x, target value t
         ↓
2. Initialize: x' = x (counterfactual candidate)
         ↓
3. Optimization Loop:
   - Calculate: prediction loss + distance penalty
   - Compute gradients using L-BFGS-B optimizer
   - Update x' to minimize total loss
   - Apply feature bounds/constraints
         ↓
4. Check Convergence:
   - Max iterations reached? → Exit
   - Loss improvement < threshold? → Exit
   - Otherwise → Continue loop
         ↓
5. Validate: Is |f(x') - target| ≤ ε?
         ↓
6. Output: Counterfactual x' (if valid) or None
```

### **Practical Example**
```python
# Loan applicant denied (prediction = 0.3, needs 0.6 to approve)
original_instance = [0.45, 0.2, 0.35, ...]  # [income, age, debt, ...]
target_value = 0.6
epsilon = 0.05

# Wachter finds: "If income increases from 0.45 to 0.67 
# and debt decreases from 0.35 to 0.20, approval probability 
# would be 0.62"
counterfactual = [0.67, 0.2, 0.20, ...]
```

### **Requirements**
- **Data Type:** Continuous numerical features (can handle mixed with proper encoding)
- **Feature Space:** Must be differentiable or optimization-friendly
- **Model Type:** Black-box (only requires prediction function)
- **Additional Needs:** Feature ranges/bounds for realistic constraints

### **Limitations**
1. **No guarantee of validity** - May not reach target within epsilon
2. **Computationally intensive** - Requires many prediction calls during optimization - same as random search
3. **Local minima** - Can get stuck, especially with complex loss landscapes
4. **Hyperparameter sensitivity** - λ value significantly affects results
5. **Unrealistic changes** - May suggest impractical feature modifications
6. **No feature sparsity** - Often changes many features simultaneously

## **Why random search is better**
1. If solution exists it will eventually find it
2. Does not get stuck, unless solution does not exist
3. Considers individual priorities

### **Determinism**
**NON-DETERMINISTIC** ✗

**Variability factors:**
- **Optimizer initialization** - L-BFGS-B may use different starting points
- **Numerical precision** - Floating-point arithmetic introduces minor variations
- **Random seeds** - If optimization uses stochastic components

**How different can results be?**
- **Minor variance:** With fixed random seed, results typically vary by < 1% in feature values
- **Moderate variance:** Different λ values produce significantly different tradeoffs (10-50% feature changes)
- **Major variance:** Different initializations can find different local minima (completely different solutions)

**What decides variability:** Primary factor is λ parameter and optimization algorithm's random state

---

## **2. GROWING SPHERES**

### **Theoretical Description**
Growing Spheres uses a geometric search strategy inspired by the intuition that counterfactuals exist in "expanding spheres" around the original instance. It leverages the training dataset to find real instances with target predictions, then uses binary search along interpolation paths to find the closest valid counterfactual on the decision boundary.

**Key Concept:** Instead of optimizing in feature space, search in "distance spheres" using actual data distribution.

### **Workflow**
```
1. Start: Original instance x, target t, training data D
         ↓
2. Filter Training Data:
   Find all instances in D where |f(x_i) - t| ≤ ε
   → Target instances: {x₁, x₂, ..., xₙ}
         ↓
3. Sort by Distance:
   Calculate Euclidean distance from x to each xᵢ
   Sort: nearest → farthest
         ↓
4. For each target instance xᵢ (try top 10):
   ├─ Binary Search on line segment [x, xᵢ]:
   │  ├─ Generate interpolations: x + α(xᵢ - x)
   │  │  where α ∈ [0, 1] with 20 steps
   │  ├─ For each interpolated point:
   │  │  └─ Check if |f(point) - t| ≤ ε
   │  └─ Save if valid and closer than current best
   └─ Continue to next instance
         ↓
5. Output: Closest valid counterfactual found
```

### **Practical Example**
```python
# Student wants to improve grade prediction from 65 to 80
original_student = [study_hours=2, attendance=70%, ...]
target_grade = 80

# Growing Spheres finds:
# - 50 students in training data with grade ≈ 80
# - Closest one: [study_hours=6, attendance=95%, ...]
# - Binary search between original and closest finds:
#   [study_hours=4.5, attendance=87%, ...] → predicts 79.8
```

### **Requirements**
- **Data Type:** Any (numerical, categorical, mixed)
- **Training Data:** **REQUIRED** - Must have access to original training dataset
- **Model Type:** Black-box (only needs predictions)
- **Feature Space:** Works with any metric space (Euclidean distance standard)

### **Limitations**
1. **Training data dependency** - Cannot work without access to training data
2. **Sparse target regions** - Fails if no training instances near target value
3. **Interpolation validity** - Linear interpolation may create invalid feature combinations
4. **Coarse search** - Limited to 20 interpolation samples (may miss optimal solution)
5. **Computational cost** - Evaluates multiple candidates × interpolation points
6. **Distance metric bias** - Euclidean distance may not reflect feature importance

### **Determinism**
**DETERMINISTIC** ✓ (with caveats)

**Variability factors:**
- **With fixed data ordering:** Always produces same result
- **With randomized data:** Results depend on tie-breaking in distance sorting
- **Numerical precision:** Minor floating-point variations possible

**How different can results be?**
- **No variance:** With sorted data and fixed epsilon, produces identical counterfactual
- **Minimal variance:** Tie-breaking among equidistant instances creates small differences
- **No major variance:** Algorithm structure ensures consistency

**What decides variability:** Data ordering and tie-breaking rules for equidistant instances

### **Important Note: Hybrid Approach**

**Growing Spheres is a hybrid method:**

1. **Uses training data** to find prototypes (real instances with target prediction)
2. **Creates NEW synthetic points** by **linear interpolation** between original and prototypes
3. **Returns interpolated point** (NOT a real training instance)

**Example:**
```python
# Real training data:
Original:  [0.2, 0.3, 0.4, 0.5] → 10 MPG (this is your instance)
Prototype: [0.8, 0.7, 0.6, 0.9] → 35 MPG (real training instance)

# n_search_samples creates SYNTHETIC points along the line:
α=0.5: [0.5, 0.5, 0.5, 0.7] → 22 MPG ← NEW synthetic point!
α=0.6: [0.56, 0.54, 0.52, 0.74] → 25 MPG ← NEW synthetic point!
                                    ↑
                            This is returned (not in training data!)
```

**Key Distinction:**

- **Growing Spheres**: Uses training data to find direction, returns **synthetic interpolated point**
- **Prototype Method**: Returns **actual training instance** (nothing synthetic)

So n_search_samples controls how many synthetic interpolation points are created between original and prototype!

### **Role of Parameters**

#### **Input Values:**
```python
X_original: [0.2, 0.3, 0.4, 0.5]    # Original instance (scaled features)
target_value: 30.0                   # Target prediction (e.g., 30 MPG)
epsilon: 2.0                         # Tolerance (±2 MPG acceptable)
X_train: [[...], [...], ...]        # All training data (313 samples)
y_train: [10.5, 25.3, 35.2, ...]    # Predictions for training data
n_search_samples: 20                 # Interpolation granularity
n_top_candidates: 10                 # How many prototypes to try
```

#### **Complete Workflow with Examples:**

**STEP 1: Filter Training Data**

**Goal:** Find training instances with predictions close to target

```python
# Find instances within epsilon of target
target_mask = |y_train - target_value| <= epsilon

# If target=30, epsilon=2:
# Keep training samples with predictions in [28, 32] MPG

Example results:
  Training sample #45:  features=[0.7, 0.6, 0.5, 0.8], prediction=28.5 ✓
  Training sample #102: features=[0.8, 0.7, 0.6, 0.9], prediction=31.2 ✓
  Training sample #205: features=[0.6, 0.5, 0.7, 0.8], prediction=29.8 ✓
  ...
  Total found: 45 instances
```

**If no instances found → FAIL (epsilon too strict)**

**STEP 2: Calculate Distances & Sort**

**Goal:** Rank prototypes by proximity to original instance

```python
# Calculate Euclidean distance from original to each prototype
distances = []
for prototype in target_instances:
    dist = √((prototype - X_original)²)
    distances.append(dist)

# Sort by distance (closest first)
sorted_indices = argsort(distances)
```

**Example:**
```
X_original = [0.2, 0.3, 0.4, 0.5]

Prototype #102: [0.6, 0.5, 0.6, 0.7] → distance = 0.4359  ← Closest
Prototype #205: [0.7, 0.6, 0.5, 0.8] → distance = 0.5477
Prototype #45:  [0.8, 0.7, 0.6, 0.9] → distance = 0.6928
Prototype #330: [0.9, 0.8, 0.7, 0.9] → distance = 0.8944
...
```

**STEP 3: Try Top Candidates (ROLE OF n_top_candidates)**

**Goal:** Try multiple prototypes to find best interpolation path

```python
best_cf = None
best_distance = infinity

# Try only the n_top_candidates closest prototypes
for i in range(min(n_top_candidates, len(sorted_indices))):
    prototype = target_instances[sorted_indices[i]]
    
    # For each prototype, do binary search (STEP 4)
    cf, pred = binary_search_along_line(X_original, prototype)
    
    if cf is valid and closer than best:
        best_cf = cf
        best_distance = distance(cf, X_original)
```

**Example with n_top_candidates=3:**
```
Try Prototype 1 (closest):
  └─> Interpolation → counterfactual at distance 0.35

Try Prototype 2 (2nd closest):
  └─> Interpolation → counterfactual at distance 0.28 ← BETTER!

Try Prototype 3 (3rd closest):
  └─> Interpolation → counterfactual at distance 0.42

Result: Use counterfactual from Prototype 2 (distance 0.28)
```

**If n_top_candidates=1:**
```
Try Prototype 1 (closest):
  └─> Interpolation → counterfactual at distance 0.35

Stop here! (might miss better solution from Prototype 2)
```

**STEP 4: Binary Search Along Line (ROLE OF n_search_samples)**

**Goal:** Find point along line from original to prototype where prediction crosses target

**For each prototype tried in Step 3:**

```python
# Create interpolation points along line
alphas = [0.00, 0.05, 0.10, ..., 0.95, 1.00]  # n_search_samples points

for alpha in alphas:
    # Linear interpolation
    interpolated = (1 - alpha) * X_original + alpha * prototype
    
    # Get prediction
    prediction = model.predict(interpolated)
    
    # Check if within epsilon of target
    if |prediction - target_value| <= epsilon:
        return interpolated  # Found valid counterfactual!
```

#### **Detailed Example with n_search_samples:**

**Setup:**
```
X_original = [0.2, 0.3, 0.4, 0.5] → prediction = 10 MPG
Prototype  = [0.8, 0.7, 0.6, 0.9] → prediction = 35 MPG
Target = 25 MPG, epsilon = 2 MPG
```

**With n_search_samples=5 (coarse):**
```
α=0.00: (1-0.00)*[0.2,0.3,0.4,0.5] + 0.00*[0.8,0.7,0.6,0.9] = [0.20, 0.30, 0.40, 0.50]
        → Prediction: 10.0 MPG (too low)

α=0.25: (1-0.25)*[0.2,0.3,0.4,0.5] + 0.25*[0.8,0.7,0.6,0.9] = [0.35, 0.40, 0.45, 0.60]
        → Prediction: 16.5 MPG (too low)

α=0.50: (1-0.50)*[0.2,0.3,0.4,0.5] + 0.50*[0.8,0.7,0.6,0.9] = [0.50, 0.50, 0.50, 0.70]
        → Prediction: 22.8 MPG (close but still low)

α=0.75: (1-0.75)*[0.2,0.3,0.4,0.5] + 0.75*[0.8,0.7,0.6,0.9] = [0.65, 0.60, 0.55, 0.80]
        → Prediction: 29.1 MPG ✓ VALID! (within [23, 27])
        → Distance from original: 0.528

Result: Return [0.65, 0.60, 0.55, 0.80]
```

**With n_search_samples=20 (fine):**
```
α=0.00: → 10.0 MPG (too low)
α=0.05: → 11.2 MPG (too low)
...
α=0.50: → 22.8 MPG (too low)
α=0.55: → 24.1 MPG ✓ VALID!
        → [0.53, 0.52, 0.51, 0.72]
        → Distance from original: 0.412 ← CLOSER!

Result: Return [0.53, 0.52, 0.51, 0.72]
```

**Key Difference:**
- **5 samples**: Jumps in steps of 0.25 → finds α=0.75 (distance=0.528)
- **20 samples**: Jumps in steps of 0.05 → finds α=0.55 (distance=0.412)
- **Finer search finds point closer to threshold, thus closer to original**

#### **Visual Representation:**

```
Original                                    Prototype
    ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●
  10 MPG                                    35 MPG

Target zone: [23-27 MPG]
                    ▼━━━━━━━━▼
    ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●

n_search_samples = 5 (coarse):
    ●----●----●----●----●
              ↑
           Found here (α=0.75, distance=0.528)

n_search_samples = 20 (fine):
    ●-●-●-●-●-●-●-●-●-●-●-●-●-●-●-●-●-●-●-●
           ↑
        Found here (α=0.55, distance=0.412) ← Better!
```

#### **Complete Algorithm Summary:**

```
INPUT: X_original, target, epsilon, X_train, y_train, n_search_samples, n_top_candidates

1. Filter: Find training instances where |y_train - target| ≤ epsilon
   → Result: List of candidate prototypes
   
2. Sort: Calculate distances, sort by proximity to X_original
   → Result: Ordered list (closest first)
   
3. Loop over n_top_candidates closest prototypes:
   
   For each prototype:
     4. Generate n_search_samples interpolation points:
        points = [(1-α)*X_original + α*prototype for α in linspace(0,1,n_search_samples)]
     
     5. For each interpolated point:
        - Get prediction
        - If |prediction - target| ≤ epsilon:
            → Found valid counterfactual on this path!
            → Check if closer than current best
            → If yes, update best
   
6. Return best counterfactual found across all prototypes

OUTPUT: Best counterfactual, its prediction, distance
```

#### **Parameter Effects Summary:**

| Parameter | Low Value | High Value | Sweet Spot |
|-----------|-----------|------------|------------|
| **n_search_samples** | Fast but coarse (might miss closer points) | Slow but precise (finds exact crossing point) | 20-50 |
| **n_top_candidates** | Fast but might miss better paths | Slow but explores more options | 10-20 |

**Computational Cost:**
- Time complexity: O(n_top_candidates × n_search_samples × model_prediction_time)
- Example: 10 candidates × 20 samples = 200 model predictions per original instance

**Typical Results:**
- **n_search_samples**: 5 vs 50 might reduce distance by **5-15%**
- **n_top_candidates**: 5 vs 20 might reduce distance by **10-30%** (if better prototypes exist)

**But**: If the nearest prototype is already optimal, increasing n_top_candidates won't help!

### **Measure**

- Find sample form those that meet the target that is the most wanted or create it and check which one in turn it is indicated by the method. In other words: how many other samples is indicated before returning the one that is the most expected
- 

---

## **3. PROTOTYPE-BASED**

### **Theoretical Description**
The simplest and most interpretable method. Prototype-based counterfactuals are **real instances from the training data** that achieve the target prediction and are closest to the original instance. This guarantees realistic, actionable explanations since they represent actual observed cases.

**Philosophy:** "Show me someone like me who achieved the desired outcome"


### **Workflow**
```
1. Start: Original instance x, target t, training data D
         ↓
2. Filter Training Data:
   Find all instances where |f(x_i) - t| ≤ ε
   → Candidate prototypes: P = {x₁, x₂, ..., xₙ}
         ↓
3. Check Availability:
   ├─ If P is empty:
   │  └─ FAIL: No valid prototypes
   └─ If P has instances: Continue
         ↓
4. Calculate Distances:
   For each xᵢ ∈ P:
   dist(xᵢ) = ||xᵢ - x||₂
         ↓
5. Sort & Select:
   Sort by distance: d₁ ≤ d₂ ≤ ... ≤ dₙ
   Select k-th closest (typically k=1)
         ↓
6. Verify:
   Confirm: |f(selected) - t| ≤ ε
         ↓
7. Output: Selected prototype
```

### **Practical Example**
```python
# Employee salary prediction = $45K, wants to understand $65K
original_employee = [experience=2, degree=bachelor, ...]
target_salary = 65000

# Prototype method finds:
# Real employee in training data:
# [experience=5, degree=master, ...] with salary = $66,000
# This is an actual person who achieved the target!

# Interpretation: "Someone like you with 3 more years 
# experience and a master's degree earns $66K"
```

### **Requirements**
- **Data Type:** Any (works with all feature types)
- **Training Data:** **REQUIRED** - Must have complete training dataset with predictions
- **Model Type:** Black-box (only needs predictions)
- **Distance Metric:** Needs meaningful distance function (Euclidean, Manhattan, custom)

### **Limitations**
1. **Data coverage dependency** - Can only suggest changes observed in training data
2. **Curse of dimensionality** - In high dimensions, "closest" instances may still be very different
3. **Outlier sensitivity** - May select unusual/rare instances
4. **No optimization** - Just finds nearest, doesn't minimize distance
5. **Privacy concerns** - Returns actual training data (may expose sensitive information)
6. **Distribution bias** - Limited by training data distribution
7. **Multiple changes** - Often changes many features simultaneously (can't control sparsity)

### **Determinism**
**DETERMINISTIC** ✓

**Variability factors:**
- **None** - Given the same data and target, always returns the same instance

**How different can results be?**
- **Zero variance:** Completely deterministic
- **Exception:** If `top_k` parameter changes, returns different prototype

**What decides variability:** Only the `top_k` parameter (which k-th nearest to return)

---

## **4. GRADIENT-BASED (for Neural Networks)**

### **Theoretical Description**
Gradient-based methods directly optimize the input features using backpropagation through the neural network. Instead of treating the model as a black box, they exploit the differentiable nature of neural networks to compute exact gradients of the prediction with respect to input features, enabling efficient optimization.

**Mathematical Foundation:**
```
Minimize: Loss(x') = (f(x') - target)² + λ||x' - x||²

Using gradient descent:
x'ₜ₊₁ = x'ₜ - η∇ₓ′Loss(x')
```

Where gradients are computed via automatic differentiation (backpropagation).

### **Workflow**
```
1. Start: Original x, target t, neural network model
         ↓
2. Initialize: 
   x' = x (as TensorFlow/PyTorch variable)
   Setup optimizer (Adam, SGD, etc.)
         ↓
3. Optimization Loop (for max_iter iterations):
   ├─ Forward Pass:
   │  pred = model(x')
   │  
   ├─ Compute Loss:
   │  loss = (pred - target)² + λ||x' - x||²
   │  
   ├─ Backward Pass:
   │  gradients = ∂loss/∂x' (via autograd)
   │  
   ├─ Update:
   │  x' ← x' - learning_rate × gradients
   │  
   ├─ Apply Constraints:
   │  x' = clip(x', feature_mins, feature_maxs)
   │  
   ├─ Check Validity:
   │  If |pred - target| ≤ ε and closer than best:
   │  └─ Save as best counterfactual
   │  
   └─ Early stopping if good solution found
         ↓
4. Output: Best valid counterfactual found
```

### **Practical Example**
```python
# Neural network predicts house price = $200K, want $300K
original_house = [size=1200sqft, bedrooms=2, age=30, ...]
target_price = 300000

# Gradient method computes:
# ∂price/∂size = +150 per sqft
# ∂price/∂bedrooms = +25000 per bedroom
# ∂price/∂age = -500 per year

# Iteratively adjusts:
# Iteration 1: size → 1350, bedrooms → 2, age → 25 (pred=$235K)
# Iteration 10: size → 1550, bedrooms → 3, age → 20 (pred=$285K)
# Iteration 45: size → 1650, bedrooms → 3, age → 15 (pred=$299K) ✓
```

### **Requirements**
- **Data Type:** Continuous numerical features (primary); categorical requires embedding
- **Model Type:** **MUST be differentiable** (Neural Networks: TensorFlow, PyTorch, Keras)
- **Framework:** Requires automatic differentiation framework (TensorFlow, PyTorch)
- **Feature Space:** Must be differentiable (no discrete/categorical without special handling)

### **Limitations**
1. **Model restriction** - **ONLY works with differentiable models** (no tree-based, SVM, traditional ML)
2. **Categorical features** - Difficult to handle (requires embedding or relaxation)
3. **Local minima** - Can get stuck in local optima
4. **Adversarial-like** - May find adversarial examples rather than meaningful counterfactuals
5. **Hyperparameter sensitive** - Learning rate, λ, max_iter significantly affect results
6. **Unrealistic solutions** - May suggest impossible feature combinations
7. **No sparsity control** - Changes many features (unless explicitly regularized)

### **Determinism**
**NON-DETERMINISTIC** ✗

**Variability factors:**
- **Optimizer randomness** - Adam optimizer has momentum buffers that may vary
- **Floating-point operations** - GPU parallelism introduces non-deterministic rounding
- **Weight initialization** - If model re-initialized between runs
- **TensorFlow/PyTorch settings** - Deterministic ops must be explicitly enabled

**How different can results be?**
- **Minor variance:** With deterministic settings, < 5% variation in feature values
- **Moderate variance:** Different learning rates produce 20-40% different solutions
- **Major variance:** Different random seeds can find completely different local minima

**What decides variability:** 
1. Learning rate (most impact)
2. Optimizer type (Adam vs SGD vs RMSprop)
3. Random seed / deterministic settings
4. λ parameter value

---

## **GROWING SPHERES VS PROTOTYPE-BASED: KEY DIFFERENCES**

### **Core Difference:**

**Prototype-Based:**
- Finds a **real training instance** with the target prediction
- Returns it **as-is** (unchanged)
- "Here's someone like you who achieved the desired outcome"

**Growing Spheres:**
- Also finds real training instances with target prediction (prototypes)
- But then **interpolates** between your instance and the prototype
- Creates a **synthetic point** along that path
- "Here's a path from you toward someone successful - you don't need to go all the way"

---

### **Practical Example:**

**Scenario:** You earn $30K, want to understand $80K salary

#### **Prototype Method:**
```
Searches training data...
Found: Real employee earning $82K
Features: [experience=10yrs, degree=PhD, skills=expert]

Returns: This exact person
Distance from you: Very far (huge gap)
```

#### **Growing Spheres:**
```
Searches training data...
Found: Real employee earning $82K  
Features: [experience=10yrs, degree=PhD, skills=expert]

Interpolates along path from you to them...
Creates synthetic point: [experience=6yrs, degree=Master, skills=advanced]
Predicts: $79K

Returns: This synthetic interpolated point
Distance from you: Moderate (more realistic change)
```

---

### **Key Distinctions:**

| Aspect | Prototype | Growing Spheres |
|--------|-----------|-----------------|
| **Output** | Real training instance | Synthetic interpolated point |
| **Realism** | 100% real (existed) | May suggest impossible combinations |
| **Distance** | Often far from original | Usually closer (stops along path) |
| **Actionability** | Shows what worked for someone | Shows minimal sufficient change |
| **Privacy** | Exposes actual training data | No privacy issue (synthetic) |
| **Precision** | Takes you "all the way" to prototype | Finds "just enough" change |

---

### **Geometric Intuition:**

```
Prototype:
  You ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● Real person (target)
      Returns this ────────────────────^

Growing Spheres:
  You ●━━━━━━━━━━━●─────────────────● Real person (target)
      Returns this ─^
      (stops when target is reached)
```

---

### **When to Use Each:**

**Use Prototype when:**
- You want to see **real examples**
- Interpretability is critical ("this actually happened")
- You're okay with larger changes
- Privacy concerns allow showing training data

**Use Growing Spheres when:**
- You want **minimal changes**
- Closer counterfactuals matter
- Synthetic examples are acceptable
- You want more precision in reaching target

---

**Bottom line:** Prototype is like showing a destination. Growing Spheres is like showing the shortest path to that destination, stopping as soon as you reach your goal.

---

## **COMPARATIVE SUMMARY TABLE**

| Aspect | Wachter | Growing Spheres | Prototype | Gradient-Based |
|--------|---------|-----------------|-----------|----------------|
| **Speed** | Medium (100-1000 iterations) | Fast (vectorized) | Very Fast (single lookup) | Very Fast (50-200 iterations) |
| **Quality** | Good | Good | High (real data) | Medium-Good |
| **Deterministic** | No ⚠️ | Yes ✓ (mostly) | Yes ✓ | No ⚠️ |
| **Requires Training Data** | No | **Yes** ✓ | **Yes** ✓ | No |
| **Model Type** | Any black-box | Any black-box | Any black-box | **Neural Networks ONLY** |
| **Feature Types** | Continuous preferred | Any | Any | **Continuous only** |
| **Realism** | Medium | High | **Very High** | Low-Medium |
| **Sparsity** | Low | Medium | Low | Low |
| **Computational Cost** | High | Medium | **Very Low** | Medium |
| **Interpretability** | Medium | High | **Very High** | Medium |

---

## **WHEN TO USE EACH METHOD**

### **Use Wachter when:**
- No access to training data
- Need flexible optimization
- Working with any model type
- Computational resources available

### **Use Growing Spheres when:**
- Have training data
- Want realistic counterfactuals
- Need decent speed
- Any model type

### **Use Prototype when:**
- **Interpretability is critical**
- **Must use real examples only**
- Have training data
- Need fastest method
- Privacy/realism is priority

### **Use Gradient-Based when:**
- Working with neural networks
- Need fastest optimization
- Have differentiable model
- Continuous features only
- Want to exploit model structure

---

This detailed analysis should give you a comprehensive understanding of each method's mechanics, strengths, and weaknesses. Each method represents different trade-offs between realism, speed, flexibility, and requirements.
