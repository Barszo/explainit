# **Standard Counterfactual Explanation Methods - Detailed Analysis**

## **1. DiCE (Diverse Counterfactual Explanations)**

### **Theoretical Description**

DiCE (Diverse Counterfactual Explanations), developed by Microsoft Research, is a counterfactual generation method that addresses a critical limitation of traditional approaches: they typically provide only a single counterfactual explanation. DiCE generates **multiple diverse counterfactuals** simultaneously, giving users a range of actionable options rather than a single path.

**Core Philosophy:** "Show me **multiple different ways** I can achieve the desired outcome"

**Key Innovation:** DiCE optimizes for two competing objectives:
1. **Proximity** - Counterfactuals should be close to the original instance
2. **Diversity** - Counterfactuals should be different from each other

**Mathematical Formulation:**

DiCE minimizes a composite loss function:

```
Loss = Σᵢ [λ₁ × validity_loss(xᵢ', target) + λ₂ × proximity_loss(xᵢ', x) + λ₃ × sparsity_loss(xᵢ', x)]
       + λ₄ × diversity_loss({x₁', x₂', ..., xₙ'})
```

Where:
- `xᵢ'` = i-th counterfactual candidate (generates n counterfactuals)
- `validity_loss` = How far prediction is from target (typically MSE for regression)
- `proximity_loss` = Distance from original instance (L1 or L2 norm)
- `sparsity_loss` = Penalty for changing many features
- `diversity_loss` = Encourages counterfactuals to differ from each other (typically determinantal point processes or pairwise distances)
- `λ₁, λ₂, λ₃, λ₄` = Weighting hyperparameters

**Diversity Mechanism:**

The diversity loss term is what distinguishes DiCE:

```
diversity_loss = -Σᵢ Σⱼ>ᵢ ||xᵢ' - xⱼ'||²
```

This negative term **rewards** counterfactuals that are far apart from each other, ensuring the set provides diverse options.

---

### **Workflow**

DiCE supports multiple optimization strategies. The most common are:

#### **A) Random Search Method (Default in Implementation)**

```
1. Start: Original instance x, target t, training data D
         ↓
2. Initialize: 
   - Extract feature ranges from training data D
   - Set number of CFs to generate: n (default: 5)
   - Set desired prediction range: [target - ε, target + ε]
         ↓
3. Random Sampling Loop (for k iterations):
   ├─ For each of n counterfactuals:
   │  ├─ Randomly perturb features within valid ranges
   │  ├─ Start from original x or previous best CF
   │  └─ Apply feature constraints (immutable, bounds, etc.)
   │
   ├─ Forward Pass:
   │  └─ Get predictions: ŷᵢ = f(xᵢ')
   │
   ├─ Compute Loss for each CF:
   │  ├─ Validity: |ŷᵢ - target|²
   │  ├─ Proximity: ||xᵢ' - x||₁
   │  └─ Sparsity: Σⱼ 𝟙(xᵢⱼ' ≠ xⱼ)
   │
   ├─ Compute Diversity Loss:
   │  └─ Pairwise distances: Σᵢ Σⱼ>ᵢ ||xᵢ' - xⱼ'||²
   │
   ├─ Evaluate Total Loss:
   │  └─ Weighted sum of all components
   │
   ├─ Update Best CFs:
   │  └─ Keep n CFs with lowest losses
   │
   └─ Check Convergence:
      - k iterations reached? → Exit
      - All CFs valid? → Exit early
         ↓
4. Post-processing:
   - Filter CFs by validity: |ŷᵢ' - target| ≤ ε
   - Sort by proximity to original
   - Remove duplicates (too similar CFs)
         ↓
5. Output: Set of n diverse counterfactuals {x₁', x₂', ..., xₙ'}
```

#### **B) Gradient-Based Method (for Neural Networks)**

```
1. Start: Original x, target t, neural network model
         ↓
2. Initialize: 
   - Create n counterfactual variables: {x₁', x₂', ..., xₙ'}
   - Initialize each xᵢ' = x + small_random_noise
   - Setup optimizer (Adam, SGD)
         ↓
3. Optimization Loop:
   ├─ Forward Pass:
   │  └─ Predictions: {ŷ₁, ŷ₂, ..., ŷₙ}
   │
   ├─ Compute Loss:
   │  ├─ Validity loss for each CF
   │  ├─ Proximity loss for each CF
   │  ├─ Sparsity loss for each CF
   │  └─ Diversity loss across all CFs
   │
   ├─ Backward Pass:
   │  └─ ∇_xᵢ' Loss (via autograd)
   │
   ├─ Update:
   │  └─ xᵢ' ← xᵢ' - η × ∇_xᵢ' Loss
   │
   ├─ Apply Constraints:
   │  └─ Clip to feature bounds
   │
   └─ Check Convergence
         ↓
4. Output: Set of diverse counterfactuals
```

---

### **Practical Example**

**Scenario:** Loan Application - Current prediction: 0.35 (denied), Target: 0.70 (approved)

**Original Applicant:**
```
[Annual Income: $35K, Debt-to-Income: 45%, Credit Score: 620, Employment Years: 1.5]
→ Approval Probability: 0.35 (DENIED)
```

**DiCE generates 3 diverse counterfactuals:**

**Counterfactual #1 (Income-focused):**
```
[Annual Income: $52K, Debt-to-Income: 45%, Credit Score: 620, Employment Years: 1.5]
→ Approval Probability: 0.72
→ Changes: +$17K income
→ Interpretation: "Focus on increasing income"
```

**Counterfactual #2 (Debt-focused):**
```
[Annual Income: $35K, Debt-to-Income: 28%, Credit Score: 620, Employment Years: 1.5]
→ Approval Probability: 0.68
→ Changes: -17% debt ratio
→ Interpretation: "Focus on reducing debt"
```

**Counterfactual #3 (Credit-focused):**
```
[Annual Income: $35K, Debt-to-Income: 45%, Credit Score: 710, Employment Years: 1.5]
→ Approval Probability: 0.71
→ Changes: +90 credit score
→ Interpretation: "Focus on improving credit score"
```

**Key Advantage:** User sees **three different strategies**, can choose based on what's most actionable for them.

---

### **Requirements**

#### **Library Requirements:**
- **DiCE Library:** `pip install dice-ml`
- **Dependencies:** pandas, numpy, scikit-learn
- **Model Backend:** TensorFlow, PyTorch, or scikit-learn models

#### **Data Requirements:**
- **Training Data:** Required for establishing feature ranges and distributions
- **Feature Types:** Supports continuous, categorical, and mixed features
- **Outcome Column:** Training data must include the prediction target

#### **Model Requirements:**
- **Model Type:** 
  - Black-box (for random search method) ✓
  - Differentiable (for gradient method) - Neural Networks only
- **Task Type:** Both classification and regression

#### **Computational Requirements:**
- **Medium Computational Cost:** Generates multiple CFs (typically 3-5)
- **Scales with:** Number of CFs (total_CFs parameter) and iterations
- **Random Method:** ~100-1000 samples per CF
- **Gradient Method:** ~50-200 iterations

---

### **Limitations**

#### **1. Computational Overhead**
- Generates multiple CFs instead of one → slower than single-CF methods
- Computational cost = O(n × iterations) where n = number of CFs
- Random search can take 5-10x longer than single-CF methods

#### **2. Hyperparameter Sensitivity**
- **λ weights:** Difficult to balance validity, proximity, and diversity
- **Total_CFs:** Too few → limited options; too many → redundant CFs
- **Diversity weight:** Too high → unrealistic distant CFs; too low → similar CFs
- Requires manual tuning for each dataset/model combination

#### **3. Diversity-Proximity Trade-off**
- Encouraging diversity can push CFs **farther** from original
- May suggest less realistic changes to achieve diversity
- No guarantee that diverse CFs are all equally actionable

#### **4. No Validity Guarantee (Random Method)**
- Random search may not find valid CFs within desired range
- Success depends on model landscape and feature space
- Can return best-effort CFs that don't meet target

#### **5. Feature Actionability Not Considered**
- Treats all features as equally modifiable
- Doesn't inherently respect feature priorities or user preferences
- May suggest changing difficult-to-modify features (age, historical data)

#### **6. Training Data Dependency**
- Needs training data to establish feature ranges
- Generated CFs may fall outside observed data distribution
- Can suggest unrealistic feature combinations not seen in training

#### **7. Model-Specific Issues**
- **Gradient method:** Only works with differentiable models (neural networks)
- **Black-box method:** Slower, less precise
- **Tree-based models:** Can only use random search (no gradients)

#### **8. Scalability with Features**
- High-dimensional spaces → harder to find diverse valid CFs
- Curse of dimensionality affects both proximity and diversity
- May need more iterations/samples to cover feature space

#### **9. Post-hoc Filtering Required**
- Often generates invalid CFs that need filtering
- Final set may have fewer than requested total_CFs
- Need to check validity manually: `|f(x') - target| ≤ ε`

#### **10. Interpretability Paradox**
- Multiple diverse CFs can confuse users ("which path should I take?")
- Users may not understand why different strategies work
- Requires post-processing to explain why CFs differ

---

### **Determinism**

**NON-DETERMINISTIC** ✗ (for Random Search method)

**DEPENDS ON METHOD** ⚠️

#### **Random Search Method (Default):**

**Non-Deterministic Factors:**
- **Random sampling** - Core algorithm uses random perturbations
- **Random initialization** - CFs start from different random points
- **Stochastic selection** - Random exploration of feature space
- **No fixed seed** - DiCE library doesn't expose full control over randomness

**Variability Level:**
- **High variance:** Different runs produce completely different CF sets
- **Diversity adds variance:** Optimizing for diversity increases randomness
- **Feature values:** Can vary 20-50% between runs
- **CF selection:** Different CFs selected from random search space

**What decides variability:**
1. Random seed (if settable - not always exposed)
2. Number of iterations (more → more exploration)
3. Diversity weight λ₄ (higher → more variance)
4. total_CFs parameter (more CFs → more combinations)

**Example Variance:**
```
Run 1:
  CF1: [income=$52K, debt=45%, ...]  → 0.72
  CF2: [income=$35K, debt=28%, ...]  → 0.68
  CF3: [income=$38K, debt=35%, ...]  → 0.71

Run 2:
  CF1: [income=$48K, debt=38%, ...]  → 0.69
  CF2: [income=$42K, debt=25%, ...]  → 0.73
  CF3: [income=$35K, debt=30%, ...]  → 0.67
```

---

#### **Gradient-Based Method (for Neural Networks):**

**Non-Deterministic Factors:**
- **Random initialization** - Each CF starts with random noise
- **Optimizer state** - Adam/SGD have stochastic components
- **Floating-point precision** - GPU operations introduce variance
- **Local minima** - Different initializations → different solutions

**Variability Level:**
- **Moderate variance:** ~10-30% variation in feature values
- **Convergence dependent:** Different local minima possible
- **More stable than random** - Gradient guidance reduces variance

**Can be made more deterministic:**
- Set TensorFlow/PyTorch random seeds
- Use deterministic GPU operations
- Fix optimizer initialization
- Still not perfectly reproducible due to numerical precision

---

### **Key DiCE Parameters and Their Effects**

| Parameter | Effect | Typical Values | Impact on Results |
|-----------|--------|----------------|-------------------|
| **total_CFs** | Number of CFs to generate | 3-10 | More CFs → more options but slower |
| **desired_range** | Target prediction range | [target-ε, target+ε] | Wider → easier to find but less precise |
| **method** | Optimization strategy | 'random', 'gradient', 'genetic' | Random: slower, any model; Gradient: faster, NN only |
| **features_to_vary** | Which features can change | All or subset | Fewer → more realistic but harder to find |
| **proximity_weight** | Balance proximity vs diversity | 0.1 - 1.0 | Higher → CFs closer to original |
| **diversity_weight** | How different CFs should be | 0.1 - 1.0 | Higher → more diverse but farther CFs |
| **sparsity_weight** | Prefer few feature changes | 0.0 - 1.0 | Higher → fewer features changed |
| **stopping_threshold** | Early stop when valid | 0.5 - 0.95 | Lower → faster but may miss better CFs |

---

### **Comparison with Single-CF Methods**

| Aspect | DiCE | Wachter/Growing Spheres/Prototype |
|--------|------|-----------------------------------|
| **Number of CFs** | Multiple (3-10) | Single (1) |
| **Diversity** | Explicitly optimized | N/A |
| **Interpretability** | High (multiple strategies) | Medium (one path) |
| **Speed** | Slower (generates many) | Faster |
| **User Choice** | Multiple options | No choice |
| **Computational Cost** | High (n × base cost) | Low |
| **Complexity** | High (multiple objectives) | Simple |

---

### **When to Use DiCE**

#### **Use DiCE when:**
- Users need **multiple actionable options**
- Different stakeholders prefer different strategies
- Want to explore **diverse paths** to same outcome
- Interpretability and user agency are critical
- Computational resources available (can afford slower method)
- Working with high-stakes decisions (loan, medical, hiring)

#### **Don't use DiCE when:**
- Only need a single counterfactual
- Speed is critical (real-time applications)
- Limited computational resources
- Diversity not important
- Single optimal path is sufficient
- Feature space is very high-dimensional (slow)

---

### **DiCE vs Priority-Based Methods**

**Key Philosophical Difference:**

- **DiCE:** Generates multiple diverse CFs, assumes user will pick the most actionable
- **Priority-Based (explainit):** Directly incorporates feature priorities into search, finds CF respecting priorities

**Trade-offs:**

| Aspect | DiCE | Priority-Based Methods |
|--------|------|------------------------|
| **User Input** | None (post-hoc selection) | Requires priority specification |
| **Optimization** | Multi-objective (proximity + diversity) | Single-objective (proximity with priority weights) |
| **Results** | Multiple options | Single optimal CF for given priorities |
| **Actionability** | User decides post-hoc | Built into optimization |
| **Efficiency** | Lower (generates many) | Higher (targeted search) |

**Example:**

**DiCE:** "Here are 5 ways to get approved - pick your favorite"
**Priority-Based:** "Here's the best way given you want to change income more than debt"

---

## **Future Methods to Add**

This document is designed to be extensible. Future methods to document:

### **Potential Additions:**
1. **FACE (Feasible and Actionable Counterfactual Explanations)**
   - Focuses on actionability and feasibility constraints
   
2. **Counterfactual Prototypes**
   - Combines prototype-based with optimization
   
3. **Model-Agnostic Counterfactuals (MOC)**
   - Works with any black-box model
   
4. **CERTIFAI**
   - Ensures counterfactuals satisfy domain constraints
   
5. **Causal Counterfactuals**
   - Uses causal models to ensure realistic changes

---

## **References**

**DiCE Paper:**
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). "Explaining machine learning classifiers through diverse counterfactual explanations." *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAT\* '20)*.

**Official Documentation:**
- https://interpret.ml/DiCE/
- https://github.com/interpretml/DiCE

**Implementation:**
- Microsoft Research InterpretML
- PyPI: `pip install dice-ml`

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Maintained for:** explainit project - priorities_random_dice experiment
