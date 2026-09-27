# Experimental framework for MINLP-Search — what was built, why, and what it shows

**Status: 2026-09-21. Proof-of-concept stage, second full pass complete.** All studies have been
re-run under the hyperparameters chosen in Badania I, with a budget-matched comparison, a harder
target level, a combinatorial dataset and replicated factor variants. Every number below is final
for this pass. An interactive companion notebook (`results_explorer.ipynb`) renders these results
and the priority definitions as figures.

This document is written for a reader who has never seen the project. It starts with what is being
tested and how quality is measured, then describes each study, its results, and what should happen
next. It deliberately avoids code detail and focuses on experimental design and findings.

---

## 0. The object under test, in plain terms

### 0.1 The problem

We have a trained regression model `f` (a neural network) and one instance `x` — say, a car whose
predicted fuel consumption is 0.62 on a normalised scale. We want a **counterfactual**: a modified
instance `x'` whose prediction hits a requested **target** `t` (say 0.43) within a tolerance `ε`.

What makes this project different from standard counterfactual search is the second objective.
Classical methods look for the counterfactual that is *closest* to the original (smallest change).
Here, each feature carries a user-declared **priority function** — a curve saying how acceptable each
value of that feature is for this user (1.0 = ideal, 0 = forbidden). The method must return a
counterfactual that hits the target **and** maximises the sum of those priorities. That sum is the
**priority score**, the quantity the method claims to optimise.

So there are two hard requirements and one objective:

| | |
|---|---|
| Hard constraint 1 | `\|f(x') − t\| ≤ ε` — the counterfactual really reaches the target |
| Hard constraint 2 | every feature value lies in its allowed region (priority > 0), pinned features unchanged |
| Objective | maximise the total priority score |

### 0.2 The method (MINLP-Search)

Enough of the mechanics to read the results:

1. **Stage 0–2** — derive, from the priority functions, the allowed interval(s) of every feature.
2. **Stage 1 (anchor)** — find a reference instance ("target exemplar") whose prediction is already
   close to the target. First it looks for a *real training row* that satisfies all priorities. If
   none exists, it falls back to **randomly generating** an in-range point. This anchor matters a
   lot: everything downstream is linearised against it.
3. **Stage 3 (Shapley)** — compute Shapley values between the current instance and the anchor to get
   a *linear surrogate* `h(x)` of the model.
4. **Stage 4 (warm start)** — solve a linear program: find feature values that, according to `h`,
   hit the target.
5. **Stage 5 (SLSQP)** — nonlinear optimisation: maximise the priority score subject to
   `h(x) ∈ [t−ε, t+ε]`.
6. **Stage 6** — pick the best candidate, verify it with the real model, iterate (re-linearise).
7. **Stage 7 (peak lock-in)** — a post-processing sweep that moves single features onto their
   priority peaks and pays for the resulting prediction change with the cheapest other feature.

Two facts drive most of the results below: the optimiser works on a *linear approximation* of the
model, and the anchor may be a *random point* rather than a data point.

### 0.3 What we measure

Every counterfactual produced by any method is scored identically:

| metric | meaning | why it is there |
|---|---|---|
| `validity` | `\|f(x') − t\| ≤ ε` | did it actually solve the task |
| `priority_score` | sum of per-feature priorities | the objective |
| **`priority_score_normalised`** | `priority_score / max attainable` | **main quality metric** — comparable across datasets and priority sets, where the denominator is the best score any point in the allowed region could reach |
| `l1`, `l2`, `n_changed`, `sparsity_fraction` | distance from the original | classical counterfactual quality |
| **`model_rows`** | number of instances the method sent to the model | **main cost metric** — hardware-independent, unlike wall time |
| `time_seconds` | wall time | secondary cost |
| `cf_source` | `optimiser` / `anchor` (+ `peak_lock_in`) | **provenance**: did the optimiser produce the answer, or did the random anchor? |
| `exemplar_source` | `dataset_priority_filtered` / `random_in_range` | did Stage 1 find a real data point? |
| `warm_start_best_model_gap` | `\|f(x₀) − t\|` at the LP warm start | how wrong the linear surrogate is where the optimiser starts |

The normalised priority score and the provenance columns are the two additions that make the
diagnosis in Study 0 possible; without them a run that returned a random point looks like a success.

---

## 1. Experimental infrastructure

### 1.1 Data and models

Ten public regression datasets were preprocessed identically: numerical features standard-scaled,
categorical features one-hot encoded, the target min-max scaled to `[0, 1]` (so `ε = 0.05` means the
same thing everywhere). Each dataset gets one Keras MLP (default 64–32; diabetes uses 192–96–48 with
light dropout/L2).

Datasets actually used in the studies:

| dataset | target | train / test | columns | numerical | categorical groups | model test R² |
|---|---|---|---|---|---|---|
| `auto_mpg` | mpg | 313 / 79 | 25 | 4 | 3 | 0.909 |
| `carseats` | sales | 320 / 80 | 22 | 6 | 4 | 0.820 |
| `insurance_charges` | medical charges | 1070 / 268 | 16 | 2 | 4 | 0.841 |
| `diabetes` | disease progression | 353 / 89 | 11 | 9 | 1 | 0.508 |

Prepared but not yet used in studies: `diamonds` (R² 0.981), `bike_sharing_hourly` (0.943),
`brazilian_houses_to_rent` (0.995). Excluded on quality grounds: `wage` (0.413), `cps85` (0.228),
`forest_fires` (0.015) — a counterfactual against a model that explains nothing is not interpretable.

**Why these four.** `diabetes` is the historical example the method was developed on, and it is kept
deliberately as a *stress case* (see §1.3). `auto_mpg` and `carseats` are small, well-modelled, and
have both numerical and categorical features — good main testbeds. `insurance_charges` has only
**one** actionable numerical feature under its priority set, which makes it a natural "constrained
search space" case.

### 1.2 Priority sets — the user preferences under which the method is tested

This is the most consequential design choice in the whole framework: the priorities *are* the
problem the method is asked to solve. A different set of curves changes the optimum, the feasible
region, and therefore every number reported later. This section explains what a priority is, how
each one was chosen, and exactly what was declared for every dataset.

#### 1.2.1 What a priority actually is

For a **numerical** feature, a priority is a function `value → [0, 1]`: 1.0 = the most desirable
value, 0 = unacceptable (the search treats 0 as forbidden, not merely as "expensive"). For a
**categorical** feature it is a mapping `category → weight`, with the same interpretation. The
counterfactual's **priority score** is the sum over all features, so with *n* features the maximum
conceivable score is *n*.

Almost every curve used here is a **single-peak** shape built from four choices: where the peak is,
how high it is, and where the curve reaches zero on each side (linear or exponential decay, or a
*hard cut-off* meaning "nothing beyond the peak is acceptable"). The peak is usually anchored
**relative to the instance being explained** — e.g. "the best value is the one you have now, and
desirability falls the further you move away". This is what makes the priority a model of *effort or
acceptability of change* rather than a static ranking of values, and it is why the allowed region is
different for every instance.

Two units appear in the definitions:

* **fraction of the feature's dataset range** (`pct`): used for decay widths, so "20 %" means the
  same thing for a feature measured in kilograms and one measured in dollars;
* **standardised units**: features are standard-scaled, so an absolute offset of `0.5` means half a
  standard deviation.

#### 1.2.2 How the priorities were selected — the procedure

Two different provenances, deliberately kept apart:

**(a) `diabetes` (`set1`, `set2`) — inherited, hand-written, strict.** These existed before this
work, were written feature by feature with medical semantics in mind, and are kept **unchanged** so
that results remain comparable with the earlier analysis of the method. They are used in this
framework as the *historical example* and as a **stress case**, not as a model of good practice.

**(b) All other datasets — derived from a declared decision scenario.** Hand-writing one curve per
feature does not scale and makes "how strict are the priorities?" an incomparable, per-dataset
judgement call. The procedure used instead was:

1. **State the decision scenario in one sentence** — who is asking for the counterfactual and what
   they can actually do. (E.g. carseats: *"a retailer wants higher sales; they control their own
   price and their advertising budget, not the competitor's price or the local income".*)
2. **Classify every feature with one of five semantic rules** (below). This is the only step that
   requires domain judgement, and it is recorded explicitly per dataset in §1.2.4, so a reviewer can
   disagree with a single line rather than with an opaque curve.
3. **Materialise the curves mechanically** from that classification at three strictness levels. The
   shape is identical across features and datasets; only the rule and the level differ.
4. **Validate with the pre-flight feasibility check** (§1.3) before any study is run: does the
   resulting allowed region still contain points that reach the requested targets?

The rule vocabulary:

| rule | resulting curve | typical justification |
|---|---|---|
| `free` | peak at the instance's current value, decaying both ways | the value can move in either direction, but moving costs preference |
| `increase_only` | hard cut-off below the current value; decay above it | the quantity can only be added to (advertising spend) |
| `decrease_only` | hard cut-off above the current value; decay below it | the quantity is only worth reducing (vehicle weight) |
| `immutable` | pinned to the current value (`NON_ACTIONABLE`) | not under the decision maker's control (age, model year, competitor's price) |
| categorical `free` | every category weight 1.0 | any category is acceptable |
| categorical `immutable` | the instance's own category pinned | protected or fixed attribute (sex) |
| categorical `preferred: c` | weight 1.0 for `c`, lower for the rest | one category is the desirable end state (non-smoker) |

And the three strictness levels, which are the *same* on every dataset:

| level | decay reaches zero at | exponential steepness `a` | weight of non-preferred categories |
|---|---|---|---|
| `loose` | 100 % of the feature's dataset range away from the peak | 2.0 | 0.80 |
| `medium` | 50 % | 4.0 | 0.50 |
| `strict` | 20 % | 6.0 | 0.20 |

**Why this design matters scientifically:** strictness becomes a *controlled experimental factor*
with an identical meaning across datasets, instead of an artefact of how one particular curve was
drawn; and because the peaks are anchored at the instance, the difficulty of the problem scales with
the instance rather than being fixed by a global box.

#### 1.2.3 The hand-written diabetes sets, feature by feature

All diabetes features are standard-scaled. `set1` and `set2` share everything except the five blood
serum measurements, and both pin `sex`.

| feature | priority | reading |
|---|---|---|
| `age` | peak 1.0 at *(sample + 0.5 SD)*; **0 everywhere below**; exponential decay to 0 at the dataset maximum (`a = 5`) | age may only increase, is best just above the current value, and becomes rapidly less acceptable the further it rises |
| `bmi` | peak 0.5 at the sample; exponential decay to 0 at *20 % of the range below* the sample; **0 above the sample** | BMI may only be reduced, and only by a limited amount; its maximum contribution is half that of `age` |
| `bp` | peak 0.5 at the sample; exponential decay to 0 at the dataset minimum on the left and at the maximum on the right (`a = 5`) | blood pressure may move either way, but staying near the current value is preferred |
| `s6` | constant 0.1 everywhere | the feature is free to take any value and contributes almost nothing — deliberately "don't care" |
| `s1`–`s5` (`set1`) | peak **0.7** at *(sample − 20 % of range)*; exponential decay to 0 at the dataset minimum; **0 above the peak** | the ideal serum value is moderately below the current one; anything higher is unacceptable |
| `s1`–`s5` (`set2`) | peak **0.5** at *(sample − 40 % of range)*; otherwise identical | a stricter, more distant ideal: the target value is further from the patient's current state and worth less |
| `sex` | `NON_ACTIONABLE` | pinned to the patient's own value |

Maximum attainable score: ≈ 6.6 (`set1`) and ≈ 5.6 (`set2`), which is why the **normalised** score is
used everywhere in the results.

The property that makes these sets a stress case: `age` is zero below *sample + 0.5*, `bmi` is zero
above the sample, and `s1`–`s5` are zero above their peaks. Intersecting nine such one-sided
constraints leaves a region that **contains no training row at all** (§1.3). This is a realistic way
for a user to over-constrain the problem, and it is kept in the study for exactly that reason.

#### 1.2.4 The declared scenarios and rules per dataset

**`insurance_charges`** — *"I want to lower my medical insurance charges."*

| feature | rule | justification |
|---|---|---|
| `age` | immutable | nobody gets younger |
| `bmi` | free | the one thing the person can actually change |
| `sex` | immutable | protected attribute |
| `children` | immutable | not a lever for this decision |
| `smoker` | preferred `0` | quitting is the desirable end state; still smoking is allowed but worth less |
| `region` | free | relocating is allowed, no region is preferred |

This leaves **one actionable numerical feature**, which is why this dataset is the "narrow search
space" case and why only 20 % of its targets turned out to be reachable (§1.3).

**`carseats`** — *"a retailer wants higher sales; they control their own price and advertising."*

| feature | rule | justification |
|---|---|---|
| `Price` | free | our own price, can go either way |
| `Advertising` | increase_only | the budget can be raised; "spending less" is not the intervention being studied |
| `CompPrice`, `Income`, `Population`, `Age` | immutable | competitor's price and local market facts |
| `ShelveLoc` | free | shelf location is negotiable |
| `Education`, `Urban`, `US` | immutable | properties of the location, not decisions |

**`auto_mpg`** — *"design a more fuel-efficient car."*

| feature | rule | justification |
|---|---|---|
| `displacement`, `horsepower`, `acceleration` | free | design parameters, tradeable in both directions |
| `weight` | decrease_only | the engineering goal is to make the car lighter |
| `cylinders` | free | a design choice |
| `model_year`, `origin` | immutable | not design decisions |

**`diamonds`** — *"find a stone at a different price point."* Every physical property
(`carat`, `depth`, `table`, `x`, `y`, `z`) and every grade (`cut`, `color`, `clarity`) is `free`:
the user is choosing a different stone, not modifying one. This is the "everything is actionable"
extreme of the spectrum.

**`bike_sharing_hourly`** — *"operational planning: when should I plan for this demand level?"*

| feature | rule | justification |
|---|---|---|
| `hr`, `weekday`, `workingday` | free | the calendar slot is the decision |
| `temp`, `atemp`, `hum`, `windspeed` | immutable | weather is given, not chosen |
| `season`, `yr`, `mnth`, `holiday`, `weathersit` | immutable | context, not decisions |

Deliberately kept with **8 categorical groups** (24 hours × 7 weekdays × …) as the stress case for
the combinatorial part of the search.

**`brazilian_houses_to_rent`** — *"what property would rent at the target price?"* All four numerical
features (`area`, `hoa`, `property tax`, `fire insurance`) and all seven categorical groups
(`city`, `rooms`, `bathroom`, `parking spaces`, `floor`, `animal`, `furniture`) are `free` — again a
search over properties rather than a modification of one, with a large categorical space.

#### 1.2.5 Consequences to keep in mind when reading the results

* The number of **actionable** features differs enormously between datasets (1 for
  `insurance_charges`, 2 for `carseats`, 4 for `auto_mpg`, 9 for `diabetes`), which drives both how
  hard the problem is and how much priority is attainable.
* `immutable` features contribute **nothing** to the priority score (they are excluded from both the
  score and its maximum), whereas categorical `free`/`preferred` groups contribute priority that is
  usually easy to attain. This inflates the normalised score on categorical-rich datasets — the
  reason why the "numerical only" factor in Badania II must be read against the baseline rather than
  in absolute terms (§4.5).
* Priorities are **never** tuned to make the method look good. They were fixed from the scenario
  before any study was run, and the only post-hoc check applied was feasibility (§1.3).

### 1.3 Pre-flight feasibility check — separating "impossible" from "the method failed"

Before any expensive study, every `(dataset, priority set, sample)` cell is checked: can *any* point
in the allowed region reach the target at all? The check samples the allowed region randomly and
reports the smallest `|prediction − target|` found. It also reports **coverage**: the fraction of
training rows that satisfy the priority set.

Result (10 samples per cell, target = 1 prediction standard deviation below the current prediction):

| dataset | set | reachable targets | coverage of the training set | actionable numerical features |
|---|---|---|---|---|
| `auto_mpg` | loose / medium / strict | 1.00 / 1.00 / 0.90 | 0.017 / 0.016 / 0.006 | 4 |
| `carseats` | loose / medium / strict | 0.90 / 0.90 / 0.80 | 0.026 / 0.024 / 0.012 | 2 |
| `insurance_charges` | loose / medium / strict | 0.20 / 0.20 / 0.20 | 0.106 / 0.104 / 0.064 | 1 |
| `diabetes` | set1 / set2 | 0.30 / 0.30 | **0.0000** | 9 |

**This is already a result.** Diabetes's own priority sets exclude **every single training row**
(coverage 0.0000). That is the mechanical reason the method's Stage 1 can never find a real data
point on diabetes and always falls back to a random anchor. Tuning hyperparameters on diabetes would
mean tuning on a degenerate cell — which is why Study I was moved to `auto_mpg` and `carseats`.

### 1.4 Targets: scale-free by construction

Originally the target was `prediction − 0.3` in scaled units. That means something different on
every dataset (it depends on how spread out the model's predictions are). The framework now supports
`target_offset_pred_std: −1.0` = "one standard deviation of the model's own prediction distribution
below the current prediction", so one configuration is comparable across datasets. All new studies
use it.

### 1.5 The run harness

Each study is a YAML file listing *arms*; each arm is expanded into a grid over datasets, priority
sets, parameter values and random seeds. Every cell becomes one immutable **run** with a hashed
identifier, stored in its own directory with the exact configuration, the per-counterfactual table
and a summary. Consequences that matter for research work:

* **nothing is overwritten** — a new sweep never destroys a previous one;
* **resumable** — re-issuing the command executes only the missing cells;
* **seeded** — every stochastic element (the random anchor, Shapley sub-sampling, the baselines) is
  driven by a per-run seed recorded in the run's config;
* **budget-accounted** — every method sees the model only through a counting proxy;
* **isolated** — each method run gets its own copy of the priority dictionary. (This fixed a real
  fairness bug: the MINLP preprocessing modifies the priority dictionary in place, and the baseline
  was previously receiving the already-modified version.)

Progress is inspectable at any time:

```bash
python -m explainit.experiments.paper.status          # done/expected, throughput, ETA, what is running
python -m explainit.experiments.paper.run_study --config <study>/config.yaml   # run or resume
python -m explainit.experiments.paper.analyse_study --study <name> --by arm --reference <arm>
```

### 1.6 Statistics

The unit of analysis is the **paired cell**: the same dataset, priority set, sample, target and seed
solved by two methods (or by two configurations). Method-level averages are not comparable because
methods succeed on different subsets of cells, so the framework reports:

* **Wilcoxon signed-rank test** on the paired differences (non-parametric, no normality assumption);
* **sign test** (robust to the magnitude distribution);
* **Cliff's δ** as a non-parametric effect size (negligible / small / medium / large);
* **percentile bootstrap 95 % CI** of the mean difference (10 000 resamples);
* **Holm–Bonferroni correction** when several arms are compared against one reference.

---

## 2. Study 0 — diagnosis of the existing method (FINISHED)

*Not requested explicitly, but it is the baseline every other study is interpreted against: what
does the method do today, and how does it fail?*

### 2.1 Design

* 4 datasets × their priority sets (diabetes set1/set2; the other three at loose/medium/strict)
* 10 test instances per cell, selected at random with a fixed selection seed
* target = one prediction standard deviation below the current prediction, `ε = 0.05`
* 3 random seeds per cell
* 3 arms: MINLP search only (Stage 7 disabled), MINLP + Stage 7, and the priority-weighted random
  search baseline
* 99 runs, 990 counterfactual attempts

Stage 7 is separated from the search because it is a *post-hoc repair*: the diagnosis must show what
the optimiser itself produces before the polish.

### 2.2 Results — quality

Paired comparison against the priority-weighted random search baseline, best counterfactual per
sample, metric = normalised priority score:

| comparison | pairs | mean Δ | wins / losses | Wilcoxon p | Cliff's δ |
|---|---|---|---|---|---|
| MINLP (search only) — all datasets | 246 | **+0.087** | 204 / 42 | < 1e-4 | 0.43 medium |
| — auto_mpg | 90 | +0.124 | 74 / 16 | < 1e-4 | 0.56 large |
| — carseats | 74 | +0.082 | 63 / 11 | < 1e-4 | 0.57 large |
| — insurance_charges | 54 | +0.045 | 52 / 2 | < 1e-4 | 0.28 small |
| — diabetes | 28 | +0.063 | 15 / 13 | **0.22 (n.s.)** | negligible |
| MINLP + Stage 7 — all datasets | 246 | **+0.137** | 230 / 16 | < 1e-4 | 0.63 large |
| — diabetes | 28 | +0.317 | 28 / 0 | < 1e-4 | 1.00 large |

**Reading:** the method beats a strong baseline on preference satisfaction, with a medium-to-large
effect, everywhere except diabetes, where the search alone is statistically indistinguishable from
random search and only the Stage 7 post-processing rescues it.

### 2.3 Results — cost

Mean model evaluations per returned counterfactual:

| dataset | MINLP | random search | ratio |
|---|---|---|---|
| auto_mpg | 28 060 | 61 | 460× |
| carseats | 36 056 | 1 514 | 24× |
| insurance_charges | 18 428 | 4 074 | 4.5× |
| diabetes | 29 194 | 5 485 | 5.3× |

**Reading:** the quality advantage is bought with one to two orders of magnitude more model calls.
Any claim of superiority must be stated together with this budget.

### 2.4 Results — failure provenance

| | diabetes | auto_mpg / carseats / insurance (pooled) |
|---|---|---|
| no counterfactual returned | 36.7 % | 4.4 % |
| returned counterfactual is the **random anchor**, not the optimiser's output | **65.8 %** | 31.0 % |
| Stage 1 found a real data point as anchor | **0.0 %** | 42.2 % |
| linear surrogate already outside the tolerance band at the LP warm start | **89.5 %** (median gap 0.158 vs ε = 0.05) | 43.8 % (median 0.025) |
| random-search baseline returned nothing | 53.3 % | 18.2 % |

**Reading — three concrete, measured problems:**

1. **The anchor problem.** In 31–66 % of cells the answer the method returns is the *randomly
   generated* anchor, i.e. the optimisation contributed nothing to the final result. Previously this
   was invisible because nothing recorded the provenance.
2. **The exemplar problem.** On diabetes the "find a real data point" path never succeeds, because
   the priority sets exclude the whole training set (§1.3). The surrogate is then linearised against
   a random point.
3. **The surrogate problem.** Where the linear program starts, the linear approximation is already
   2.6× outside the tolerance band (diabetes median 0.158 vs ε = 0.05). The optimiser then optimises
   on a feasible set that the real model does not agree with.

---

## 3. Badania I — dobór hiperparametrów (FINISHED, 180/180 runs)

> *"eksperymenty - dobór hiperparametrów, pokazać jak to dobrałem, że to jest z głową"*
> *"2 eksperymenty dobór hiperparametrów - wyciśnięcie z metody maximum, pokazanie kosztem czego się
> to odbywa"*

### 3.1 What is being tuned and why those knobs

The method's tunable surface, with the rationale for including each:

| hyperparameter | what it controls | levels tested | why this range |
|---|---|---|---|
| `shap_num_samples` | Monte-Carlo sample count for Shapley values | 50, 100, 200, 400, 800 | dominant cost driver; 200 was the undocumented default, so the range brackets it 4× in both directions |
| `max_iterations` | refinement passes (re-linearisations) | 1, 3, 5, 10, 20 | 1 = "one-shot method", 20 = twice the default; tests whether iterating pays |
| `patience` | stop after N passes without improvement | 1, 3, 5, 10 | interacts with `max_iterations`, so tested at `max_iterations = 20` |
| `target_exemplar_epsilon` | how close a training row must be to the target to serve as anchor | 0.05, 0.10, 0.20, 0.40 | directly addresses failure mode 2 (§2.4): does a looser tolerance recover the data-point anchor? |
| `shap_approx` | approximate vs exact Shapley | true / false | exact is exponential in the number of features; is the approximation costing quality? |
| `peak_lock_in` | Stage 7 on/off | false / true | **ablation**: how much of the result comes from the post-processing |
| `peak_lock_in_max_sweeps` | Stage 7 effort | 1, 3, 6 | cheapest quality lever found so far |
| baseline `max_iterations` | random-search budget | 1 000, 5 000, 10 000 | gives the baseline a *budget axis* so both methods can be plotted on the same cost–quality plane |

Deliberately **not** tuned: `ε` (it defines the task, not the method) and `n_cfs` (fixed to 1 so
"best of n" cannot confound the comparison).

### 3.2 Experimental design

* **One factor at a time (OFAT)** around a documented base configuration: each arm sweeps a single
  hyperparameter and keeps the base value inside its own list, so every curve is self-contained and
  the base point is measured in each arm.
* **Where:** `auto_mpg` + `carseats` at `medium` strictness — the cells the pre-flight check shows to
  be healthy (80–100 % reachable targets). Diabetes is *excluded from tuning* and used only in a
  `verify_on_stress_case` arm, because its priority sets have 0 % coverage; defaults fitted there
  would be fitted to a degenerate case.
* 10 instances × 3 seeds per cell, 180 runs total.
* **Two experiments in one grid**, as requested: E1 = sensitivity (is each default justified?),
  E2 = "squeeze the maximum and show the price" (budget-monotone sweeps + the Stage 7 ablation, read
  as a Pareto front of normalised priority against model evaluations, wall time and L1 distance).

### 3.3 Results (final — all 180 runs; 60 counterfactual attempts per cell)

Columns: `valid` = fraction of attempts that reached the target band; `priority` = mean normalised
priority over valid counterfactuals (± standard deviation across instances and seeds); `rows` = mean
model evaluations; `L1` = mean distance from the original instance; `anchor` = fraction of answers
that were the *random anchor* rather than the optimiser's output.

**E1a — Shapley budget** (`shap_num_samples`):

| level | valid | priority | rows | time (s) | L1 | anchor |
|---|---|---|---|---|---|---|
| 50 | 0.95 | 0.890 ± 0.080 | 8 563 | 1.50 | 2.35 | 0.11 |
| 100 | 0.95 | 0.892 ± 0.078 | 16 563 | 1.48 | 2.30 | 0.09 |
| **200 (default)** | 0.95 | 0.896 ± 0.072 | 32 564 | 1.46 | 2.23 | 0.09 |
| 400 | 0.95 | 0.897 ± 0.071 | 64 564 | 1.50 | 2.26 | 0.14 |
| 800 | 0.95 | 0.890 ± 0.080 | 128 564 | 1.78 | 2.31 | 0.07 |

→ **Completely flat.** A 15× increase in Shapley sampling changes nothing (0.890 → 0.897 → 0.890,
well inside one standard deviation). The default of 200 costs 4× the cheapest setting for no
measurable gain.

**E1b — Refinement passes** (`max_iterations`):

| level | valid | priority | rows | time (s) | L1 | anchor |
|---|---|---|---|---|---|---|
| 1 | 0.90 | 0.886 ± 0.073 | 3 731 | 0.90 | 2.44 | **0.51** |
| 3 | 0.90 | 0.892 ± 0.076 | 10 138 | 1.16 | 2.35 | 0.30 |
| 5 | 0.90 | 0.890 ± 0.083 | 16 545 | 1.29 | 2.40 | 0.21 |
| **10 (default)** | 0.95 | 0.896 ± 0.072 | 32 564 | 1.55 | 2.23 | 0.09 |
| 20 | 0.95 | 0.895 ± 0.075 | 64 414 | 1.86 | 2.28 | 0.07 |

→ The priority gain from iterating is negligible (+0.010 over 20×  the cost), **but the provenance
changes dramatically**: with a single pass, 51 % of answers are the random anchor; at 10 passes, 9 %.
Validity also rises from 0.90 to 0.95. So the refinement loop earns its cost by *replacing the random
anchor with an optimised point*, not by improving the priority of that point.

**E1c — Patience** (at `max_iterations = 20`):

| level | valid | priority | rows | anchor |
|---|---|---|---|---|
| 1 | 0.95 | 0.890 | 22 959 | 0.32 |
| **3** | 0.95 | 0.895 | 62 385 | 0.07 |
| 5 (default) | 0.95 | 0.895 | 64 414 | 0.07 |
| 10 | 0.95 | 0.895 | 64 601 | 0.07 |

→ `patience = 3` is indistinguishable from 5 and 10; `patience = 1` stops too early and triples the
anchor rate. Nothing is gained above 3.

**E1d — Exemplar tolerance** (`target_exemplar_epsilon`, how close a training row must be to the
target to be usable as the anchor):

| level | valid | priority | rows | time (s) | L1 | anchor |
|---|---|---|---|---|---|---|
| **0.05** | 0.95 | 0.896 | 32 566 | 1.36 | 2.24 | 0.09 |
| 0.10 (default) | 0.95 | 0.896 | 32 564 | 1.38 | 2.23 | 0.09 |
| 0.20 | 0.90 | 0.885 | 32 560 | 1.45 | 2.16 | 0.07 |
| 0.40 | **0.80** | 0.882 | 34 057 | 0.71 | 1.96 | 0.03 |

→ Counter-intuitive but important: **loosening this tolerance hurts**. Accepting an exemplar further
from the target lets the method linearise against a worse reference, and validity drops from 0.95 to
0.80. The tightest setting is the best, so this is *not* the cheap fix for the exemplar failure mode
identified in §2.4.

**E1e — Exact vs approximate Shapley:**

| level | valid | priority | rows |
|---|---|---|---|
| approximate (default) | 0.95 | 0.896 ± 0.072 | 32 564 |
| exact | 0.95 | 0.884 ± 0.080 | 51 125 |

→ Exact Shapley is **worse and 1.6× more expensive**. The approximation is not the limiting factor.

**E2a — Stage 7 ablation** (peak lock-in on/off, everything else identical):

| level | valid | priority | rows | L1 |
|---|---|---|---|---|
| off | 0.95 | 0.867 ± 0.118 | 32 559 | 2.55 |
| **on (default)** | 0.95 | 0.896 ± 0.072 | 32 564 | 2.23 |

→ Stage 7 adds **+0.029 normalised priority, cuts the variance by ~40 %, improves L1 — at zero
additional model cost** (32 559 vs 32 564 rows). It is the single best value-for-money component of
the method. Sweeps 1 / 3 / 6 give **identical** results, so one sweep is enough.

**E2b — Budget Pareto front** (the "squeeze the maximum, show the price" experiment):

| configuration | normalised priority | model rows | rows per priority point over the baseline |
|---|---|---|---|
| random search, 1 000 / 5 000 / 10 000 iterations | 0.770 (identical at all three) | 74 / 274 / 524 | — |
| MINLP, `max_iterations = 1` | 0.886 | 3 731 | ≈ 31 000 |
| MINLP, `shap_num_samples = 50` | 0.890 | 8 563 | ≈ 71 000 |
| MINLP, base (200 samples, 10 passes) | 0.896 | 32 564 | ≈ 257 000 |
| MINLP, `shap_num_samples = 400` | 0.897 | 64 564 | ≈ 508 000 |

→ The random-search baseline **saturates immediately**: it finds a valid counterfactual within ~74
model calls and more budget changes nothing (0.770 at 1 000 and at 10 000 iterations). MINLP's
advantage over it is large and statistically unambiguous — paired on 57 comparable instances:
**Δ = +0.126, 52 wins / 5 losses, Wilcoxon p < 1e-4, Cliff's δ = 0.71 (large), bootstrap 95 % CI
[0.103, 0.149]** — but it is bought at 50–440× the model evaluations, and within MINLP the last
+0.006 of priority costs 4× the budget.

**Stress-case verification** (the tuned base configuration on diabetes, which was excluded from
tuning):

| priority set | no counterfactual | valid | priority | anchor rate | rows |
|---|---|---|---|---|---|
| set1 | 33.3 % | 0.67 | 0.854 | 0.65 | 30 337 |
| set2 | 40.0 % | 0.60 | 0.862 | 0.67 | 28 081 |

→ The configuration transfers in terms of *priority quality* (0.85–0.86 vs 0.90 on the healthy
datasets) but not in terms of *reliability*: a third to 40 % of instances get no counterfactual at
all, and two thirds of the answers are still the random anchor. This confirms that the diabetes
weakness is a property of the priority set and the anchor mechanism, not of the hyperparameters.

### 3.4 What the study establishes

1. **The defaults were not justified.** `shap_num_samples = 200` and `max_iterations = 10` cost ~9×
   more model evaluations than `50 / 1` and buy ~0.01 of normalised priority.
2. **Accuracy of the surrogate is not the bottleneck.** More Shapley samples, exact Shapley, and more
   refinement passes all fail to move quality. This corroborates the §2.4 diagnosis: the linear
   surrogate is *systematically* wrong where the optimiser operates, and sampling it more finely
   cannot fix that.
3. **What the refinement loop actually buys is provenance and validity** (anchor rate 51 % → 9 %,
   validity 0.90 → 0.95), which is a different justification from the one usually given for
   iterating, and should be stated that way in the article.
4. **Stage 7 is free quality.** Largest single quality contribution, zero extra model calls, lower
   variance, better proximity — and one sweep suffices.
5. **The tolerance of the exemplar search must stay tight**; loosening it degrades validity.

**Recommended defaults from this study:** `shap_num_samples = 50–100`, `max_iterations = 10`
(keep, for the anchor-rate benefit) or `3` if budget-constrained, `patience = 3`,
`target_exemplar_epsilon = 0.05–0.10`, `shap_approx = true`, `peak_lock_in = true` with
`peak_lock_in_max_sweeps = 1`. This is roughly **4× cheaper than the current defaults at equal
quality**.

### 3.5 Recommendations for the next iteration

1. Re-run Study 0 and Badania II/III under the recommended defaults to confirm the conclusions are
   not default-dependent, and to cut runtime by ~4×.
2. Add a second, harder target level (2σ instead of 1σ) — all conclusions here are for one target
   difficulty.
3. Plot the Pareto front with error bars across seeds (the data is stored; only the figure is
   missing).
4. The anchor rate deserves to be a first-class reported metric in the article, since it is the
   quantity that actually responds to hyperparameters.

---

## 4. Badania II — wpływ czynników, z analizą statystyczną (FINISHED, 258/258 runs)

> *"kilka takich eksperymentów z analizą statystyczną"*
> *"wpływ czegoś na coś np. wielkości zbioru uczącego, wielkości danych, typ danych (numeryczne,
> nienumeryczne), zaszumianie, niezbalansowane dane"*

### 4.1 Design principle

Each factor is studied by **rebuilding the dataset with exactly one property changed, retraining the
model on it, and re-running the identical method configuration**. Everything else — preprocessing,
architecture, priority spec, sample selection procedure, seeds — is held fixed, so the difference is
attributable to the factor. Variants are registered as first-class datasets, so the whole pipeline
(priority sets, selection, metrics, statistics) works on them unchanged.

### 4.2 The factors and how the variants were built

| factor | levels | how the variant is made | interpretation |
|---|---|---|---|
| **training-set size** | 1.0 (base), 0.5, 0.25, 0.10 | random subsample of the training rows | how much does the method depend on the amount of data the *model* saw |
| **number of features** | all (base), top 75 %, top 50 % | keep the most important logical features, ranked by permutation importance of the baseline model; whole one-hot groups are kept or dropped together | data width / search-space dimensionality |
| **feature noise** | 0 (base), 0.10, 0.25, 0.50 | Gaussian noise added to the scaled numerical columns of both splits | measurement noise in the inputs |
| **label noise** | 0 (base), 0.05, 0.10, 0.20 | Gaussian noise added to the training target | a model that has learned a noisier function |
| **target skew (imbalance)** | none (base), low-heavy, high-heavy | training rows resampled with weights proportional to `1−y` or `y` | the regression analogue of class imbalance |
| **feature type** | with categoricals (base) vs numerical only | all one-hot groups removed | numerical vs mixed data, and the combinatorial part of the search |
| **priority strictness** | loose / medium / strict | no new data — the same instance under looser or tighter preferences | how much strictness costs |

28 dataset variants were materialised for `auto_mpg` and `carseats` and their models retrained.
The retrained models' quality is itself a useful sanity curve — the manipulation clearly bites:

| variant | auto_mpg R² | carseats R² |
|---|---|---|
| baseline | 0.909 | 0.820 |
| training set 50 % / 25 % / 10 % | 0.808 / 0.738 / 0.470 | 0.740 / 0.396 / 0.420 |
| label noise 0.05 / 0.10 / 0.20 | 0.880 / 0.806 / 0.542 | 0.817 / 0.725 / 0.435 |
| feature noise 0.10 / 0.25 / 0.50 | 0.860 / 0.868 / 0.798 | 0.796 / 0.682 / 0.471 |
| top 75 % / 50 % of features | 0.889 / 0.866 | 0.795 / 0.773 |
| numerical only | 0.685 | 0.447 |

### 4.3 What the study runs and what is reported

204 runs: every factor level × 2 datasets × 3 seeds × 10 instances, with **both** MINLP and the
random-search baseline on every variant. Running the baseline everywhere is essential: it separates
"this factor hurts *the method*" from "this variant is simply a harder problem for anybody".

Each variant level is paired against its own baseline level on the same test instance and seed, and
three deltas are reported, because a factor can hurt in three different ways: lower priority, lower
validity (the method returns nothing), or higher cost. Statistics: Wilcoxon signed-rank, sign test,
Cliff's δ, bootstrap CI, Holm correction across the levels of one factor.

### 4.4 Results (final, 258 runs under the tuned defaults)

Effects on MINLP, normalised priority, negative Δ = the factor hurts. Only levels surviving Holm
correction within their factor family are listed; everything omitted was not significant.

| dataset | factor | Δ priority | better/worse | Holm p | Cliff's δ | Δ validity |
|---|---|---|---|---|---|---|
| carseats | **feature type** (numerical only) | **−0.188** | 3 / 24 | <0.0001 | −0.75 large | +0.10 |
| carseats | **number of features** (top 50 %) | **−0.088** | 5 / 22 | 0.0004 | −0.38 medium | 0.00 |
| auto_mpg | **feature type** (numerical only) | **−0.088** | 7 / 20 | 0.0003 | −0.35 medium | 0.00 |
| carseats | **label noise** 0.20 | **−0.054** | 4 / 17 | 0.0022 | −0.58 large | +0.13 |
| carseats | **target skew** (low-heavy) | **−0.024** | 6 / 21 | 0.0050 | −0.22 small | 0.00 |
| carseats | training-set size 10 % | **+0.027** | 19 / 5 | 0.0216 | +0.33 small | −0.10 |
| both | feature noise, mild label noise, other train-size levels, target skew (high) | −0.03 … +0.02 | — | n.s. | — | 0 … −0.20 |

**Replication with a second construction seed** (the check that turned out to matter most):

| dataset | factor | Δ, seed 0 | Δ, seed 1 | replicates? |
|---|---|---|---|---|
| auto_mpg | numerical only | −0.088 (δ −0.35) | −0.094 (δ −0.39) | **yes** |
| carseats | numerical only | −0.188 (δ −0.75) | −0.354 (δ −1.00) | **yes**, even stronger |
| carseats | top 50 % of features | −0.088 (δ −0.38) | −0.059 (δ −0.32) | **yes** |
| carseats | label noise 0.20 | −0.054 (δ −0.58) | not significant after correction | **partly** |

**The same factors measured on the random-search baseline** — this is what makes the reading
controlled rather than absolute:

| dataset | factor | Δ — MINLP | Δ — random search |
|---|---|---|---|
| auto_mpg | numerical only | −0.088 | **−0.196** |
| carseats | numerical only | −0.188 | **−0.409** (0 wins / 24, δ = −1.00) |
| carseats | top 50 % of features | −0.088 | **−0.176** |

**Priority strictness across three datasets** (mean normalised priority over valid CFs):

| dataset | loose | medium | strict | MINLP change | random search change |
|---|---|---|---|---|---|
| auto_mpg | 0.924 | 0.874 | 0.879 | −0.045 | 0.797 → 0.652 (**−0.145**) |
| carseats | 0.960 | 0.917 | 0.933 | −0.027 | 0.874 → 0.818 (−0.056) |
| bike_sharing_hourly | 0.988 | 0.978 | 0.990 | +0.002 | 0.880 → 0.838 (−0.042) |

**The combinatorial dataset** (`bike_sharing_hourly`, 3 multi-category groups plus 4 actionable
numerical features) is the best cell for MINLP in the whole framework: 0.978–0.990 normalised
priority at 97–100 % validity, against 0.833–0.880 for the baseline, at ~24 000 model evaluations
versus ~4.

### 4.5 What the factors tell us

1. **Data type is the dominant factor, and it is the only one that replicates everywhere.**
   Removing the categorical features costs MINLP 0.09–0.19 normalised priority on both datasets and
   under both construction seeds. *The baseline loses two to three times more* (up to −0.41,
   δ = −1.00, losing every pair). So the purely numerical problem is intrinsically harder for
   everyone — the continuous part of the search is where preference satisfaction is genuinely
   difficult — and MINLP degrades roughly half as much as priority-weighted sampling does.
   *Interpretation caveat:* categorical groups contribute priority that is easy to attain, so the
   normalised denominator is "easier" when categoricals are present; the comparison against the
   baseline on the same variant is what makes the conclusion safe.
2. **Replication changed the picture.** Of the effects that looked significant in the first pass,
   feature type and feature count reproduce under a second construction seed, while label noise only
   partly does. **Single-seed factor effects in this design are not trustworthy** — any factor claim
   in the article should be backed by at least two variant constructions.
3. **Narrowing the feature set hurts in proportion to how much is removed**, again less for MINLP
   than for the baseline.
4. **Label noise matters only when severe** (20 %, where the model's R² falls from 0.82 to 0.44).
5. **Training-set size is essentially irrelevant to preference quality** and at 10 % even improves it
   slightly (a simpler model is easier to satisfy) while costing validity. The effect is on
   reliability, not quality.
6. **Feature noise produced no significant priority effect** but moved validity (−0.10 / −0.20 on
   auto_mpg): noise makes the method fail to reach the target rather than return a worse preference.
7. **Strictness costs the baseline far more than it costs MINLP** (auto_mpg −0.145 vs −0.045).
   This is the clearest evidence that the optimisation earns its keep exactly where the feasible
   region is tight — which is also the regime the method was designed for.

### 4.6 Recommendations

1. Report the quality delta **together with** the validity delta — several factors act only on
   reliability.
2. Replicate *every* factor with a second construction seed (currently done for three), since
   replication already overturned one finding.
3. Add a dataset with many *interacting* categorical groups beyond the three in
   `bike_sharing_hourly` to push the combinatorial axis further.
4. Consider reporting an "attainability-corrected" priority next to the normalised one, to remove
   the categorical-denominator artefact noted in 4.5(1).

---

## 5. Badania III — porównanie z innymi metodami (FINISHED, 198/198 runs)

> *"porównanie z innymi metodami"*

### 5.1 Methods compared

| method | family | what it optimises |
|---|---|---|
| **MINLP-Search** | ours | target + priority score |
| **priority-weighted random search** | ours (baseline) | samples proportionally to the priorities, keeps valid draws |
| **DiCE** (genetic) | reference implementation | diverse, close counterfactuals |
| **Wachter** | gradient descent | prediction error + L2 proximity |
| **sparse Wachter** | gradient descent | as above + L1 (sparser edits) |
| **prototype-guided** | gradient descent | pulled toward training points near the target |
| **growing spheres** | model-agnostic | expanding L2 shells around the instance |
| **Nelder–Mead** | gradient-free | direct search |
| **Bayesian optimisation** | surrogate-based | GP + expected improvement |

The last seven do **not** optimise preferences — that is the point. Their counterfactuals are scored
post hoc with the same priority function, which quantifies what is gained by optimising preferences
explicitly versus taking a conventional counterfactual and measuring how preferable it happens to be.

### 5.2 Fairness design

All methods solve the identical instance: same sample, same target, same `ε`, same number of
requested counterfactuals (1), and — critically — **the same feasible region**, derived from the
priority set: pinned features stay pinned, actionable features get the same `[min, max]` box.
Without this the baselines would search a larger space and the comparison would be meaningless.

Known PoC caveats, to be reported with the numbers:

* the baselines have no one-hot repair, so their categorical columns are pinned to the sample's
  category while the priority methods may switch categories — *conservative for us*;
* gradient methods reach the network through autodiff, so their `model_rows` budget is
  under-counted (Wachter reports ~1);
* budgets are not equalised; quality is compared at each method's own reported cost.

### 5.3 Results (final, 198 runs)

3 datasets (`auto_mpg`, `carseats`, `bike_sharing_hourly`) × `medium` priorities × 3 seeds × 10
instances = 90 attempts per arm. Priority is averaged over **valid** counterfactuals only; that
matters here because a method that returns the instance unchanged scores a high priority and zero
validity (Nelder–Mead does exactly this).

**(a) Free budget — each method at its natural cost**

| method | validity | normalised priority | L1 | model rows |
|---|---|---|---|---|
| **MINLP** | 0.967 | **0.923** | 3.06 | 19 030 |
| sparse Wachter | 0.200 | 0.782 | **0.84** | 303 |
| priority-weighted random search | 0.967 | 0.791 | 4.73 | 350 |
| growing spheres | 1.000 | 0.738 | 2.09 | 1 174 |
| DiCE | 1.000 | 0.735 | 2.64 | 7 180 |
| Bayesian optimisation | 1.000 | 0.724 | 2.43 | 100 |
| Wachter | 0.567 | 0.698 | 1.42 | 271 |
| prototype-guided | 0.333 | 0.641 | 3.31 | 787 |
| Nelder–Mead | **0.000** | — | — | 138 |

**(b) Budget-matched at 2 000 model evaluations — the advantage disappears**

| method @2k | validity | priority (valid) | budget failures | paired Δ vs MINLP@2k |
|---|---|---|---|---|
| MINLP | **0.567** (was 0.967) | 0.798 | 0 % | — |
| random search | 0.967 | 0.791 | 0 % | **+0.035, p = 0.009, δ = +0.27 (beats MINLP)** |
| growing spheres | 0.856 | 0.737 | 14 % | −0.017, n.s. |
| Bayesian optimisation | 1.000 | 0.723 | 0 % | −0.028, n.s. |
| Wachter | 0.567 | 0.698 | 0 % | −0.065, p = 0.071 (n.s.) |
| DiCE | 0.489 | 0.691 | 33 % | −0.062, p = 0.048 |

**(c) Harder target (−2σ) — the advantage grows**

| method | validity | priority | paired Δ vs MINLP | wins/losses | Cliff's δ |
|---|---|---|---|---|---|
| **MINLP** | 0.933 | **0.882** | — | — | — |
| random search | 1.000 | 0.766 | −0.116 | 2 / 82 | −0.58 large |
| DiCE | 0.933 | 0.690 | −0.204 | 0 / 78 | −0.86 large |
| growing spheres | 0.944 | 0.680 | −0.212 | 0 / 79 | −0.90 large |

(all p < 1e-4)

**(d) Categorical release — letting the baselines switch categories**

| method | validity, pinned → released | priority, paired Δ | p |
|---|---|---|---|
| DiCE | 1.000 → **0.744** | +0.028 | 0.023 |
| growing spheres | 1.000 → **0.133** | +0.027 | 0.38 n.s. |
| Wachter | 0.567 → **0.100** | +0.003 | 0.25 n.s. |

### 5.4 What the comparison establishes

1. **At its natural budget the method wins decisively** — 0.923 versus 0.64–0.79 for every
   baseline, at the highest validity of any method except the ones that are trivially valid.
2. **At a fixed 2 000-evaluation budget the advantage vanishes.** MINLP's validity collapses from
   0.967 to 0.567 — it simply cannot complete its Shapley + LP + SLSQP pipeline inside that budget
   on most instances — and on the instances where it does finish, the priority-weighted random
   search is *statistically better* (+0.035, p = 0.009). Every other method is a statistical tie.
   **This is the single most important result for the article**: the method's superiority is a
   statement about a high-budget regime, and must be reported as such.
3. **The harder the task, the better the method looks.** At a 2σ target the baselines lose every
   single paired comparison (0/78, 0/79) with large effect sizes, and random search falls behind by
   −0.116. Preference-aware search matters most exactly where a counterfactual is hard to find.
4. **The honest trade-off is proximity.** Gradient baselines produce much smaller edits (L1 0.84–1.42
   versus MINLP's 3.06). A preference-optimal counterfactual is a *bigger* change than a
   proximity-optimal one, by construction; both columns belong in the article.
5. **Reliability separates the field more than quality does.** MINLP, random search, growing spheres,
   DiCE and Bayesian optimisation reach the band on nearly every instance; Wachter 57 %, prototype
   33 %, sparse Wachter 20 %, Nelder–Mead **0 %**. Methods that cannot hit the target cannot be
   compared on priority at all, which is why the pair counts differ (18–90).
6. **Post-hoc one-hot repair does not work.** Releasing the categorical columns buys the baselines a
   small priority gain (DiCE +0.028) but destroys validity (DiCE 1.00 → 0.74, growing spheres
   1.00 → 0.13, Wachter 0.57 → 0.10), because snapping the one-hot state after optimisation moves
   the prediction out of the band. Categorical handling has to be inside the search.

### 5.5 Recommendations

1. **Report the budget-quality curve, not a single point.** The result reverses between "free" and
   2 000 evaluations, so the whole curve (500 / 2 000 / 10 000 / 50 000) is the object of interest
   and should be the central figure of this study.
2. Give the baselines in-search categorical handling instead of post-hoc repair, then re-run (d).
3. Investigate Nelder–Mead's 0 % validity before reporting it — a completely failing method is a
   weak comparison point and may simply need a different parameterisation.
4. Report a proximity-normalised priority (priority gained per unit of L1) as the fairest
   single-number summary of the trade-off in 5.4(4).

---

## 6. Consolidated state

| study | what it answers | runs | state |
|---|---|---|---|
| Pre-flight feasibility | are the priorities solvable at all | 110 cells | done |
| Study 0 — diagnosis | what the method does today and how it fails | 99 | done, analysed |
| Extension ablation | do three optional extensions help (kept off by default) | 42 | done, analysed |
| Badania I — hyperparameters | are the defaults justified, what does maximum quality cost | 180 | done, analysed |
| Badania II — factors | what affects the method and by how much | 258 | done, analysed |
| Badania III — comparison | how does it compare to established methods | 198 | done, analysed |

**Total: 777 runs, ~7 800 counterfactual attempts**, across 4 base datasets, 40 dataset variants,
9 methods and 3 priority-strictness levels.

### 6.1 The six findings a reader should take away

1. **At its natural budget the method wins decisively.** 0.923 normalised priority versus 0.64–0.79
   for every established baseline, with large effect sizes and near-total win counts.
2. **That advantage is budget-conditional.** Capped at 2 000 model evaluations, MINLP's validity
   falls from 0.967 to 0.567 and the priority-weighted random baseline *beats* it (+0.035,
   p = 0.009). The claim must always be stated together with the evaluation budget.
3. **The harder the target, the larger the advantage** (−2σ: baselines lose 0/78 and 0/79 paired
   comparisons, δ up to −0.90).
4. **The defaults were not justified by data.** Quality is flat across a 15× Shapley budget and a
   17× refinement budget; a ~4× cheaper configuration matches them. What the refinement loop buys is
   provenance (anchor rate 51 % → 9 %) and validity (0.90 → 0.95), not priority.
5. **Stage 7 (peak lock-in) is the best value component**: +0.029 priority, −40 % variance, better
   L1, zero extra model calls, one sweep sufficient.
6. **Weakness is concentrated in over-constrained configurations.** On diabetes — whose priority
   sets exclude 100 % of training rows — a third of instances get no counterfactual, two thirds of
   answers are a random anchor, and the method is statistically indistinguishable from random
   search. On the healthy datasets the anchor rate is ~9 % and the advantage is large.

All results live under `explainit/experiments/paper/results/<study>/`, one immutable directory per
run plus an `index.csv`, with `analysis/{summary,paired,diagnostics}.csv` written by the analysis
command.

---

## 7. Threats to validity (to state in the article)

1. **Coverage confound.** Priority sets differ enormously in how much of the data space they allow
   (0 % on diabetes, 0.6–2.6 % on auto_mpg/carseats, 6–11 % on insurance). Results must always be
   reported per dataset, never pooled without noting this.
2. **Small instance counts per cell.** 10 instances × 3 seeds is enough for paired non-parametric
   tests but not for tight confidence intervals; the bootstrap CIs are reported for that reason.
3. **Model quality varies** (R² 0.51–0.91 in the studies). Counterfactuals against a weak model are
   less meaningful; diabetes is the weakest and also the most pathological.
4. **Budget dependence.** The headline comparison is budget-conditional (§5.4(2)); only the
   2 000-evaluation cap has been measured so far, so the shape of the quality-versus-budget curve
   between 2 000 and "free" is unknown.
5. **Unbounded baselines.** DiCE's genetic search has no evaluation or wall-clock limit and was
   observed consuming ~2 h of CPU on a single instance whose 2σ target was unreachable. Any
   comparison that does not impose an evaluation budget is therefore not reproducible in bounded
   time; all our arms are now capped.
6. **Single-seed factor effects proved unreliable** (§4.5(2)): replication with a second variant
   construction changed one of the four significant findings. Only three factors have been
   replicated so far.
7. **One model family.** All results are for MLPs; a second family (e.g. gradient boosting) would
   strengthen external validity.
8. **Target definition.** Two difficulties are now covered (−1σ and −2σ); the easy direction
   (+σ, increasing the prediction) has not been tested at all.

---

## 8. Recommended next steps, in order

**First (the result that is currently under-measured):**

1. **The budget-quality curve.** Cap every method at 500 / 2 000 / 10 000 / 50 000 evaluations and
   plot normalised priority and validity against budget. The comparison reverses between 2 000 and
   free budget, so this curve — not a single table — is the honest presentation of Badania III, and
   it is the figure a reviewer will ask for.
2. **In-search categorical handling for the baselines**, replacing the post-hoc repair that
   destroys their validity (§5.4(6)).
3. **Replicate every Badania II factor with a second construction seed**, since replication already
   overturned one finding.

**Then (external validity):**

4. A second model family (gradient boosting) and one more dataset with interacting categorical
   groups.
5. The easy-direction target (+σ) to complete the difficulty axis.

**Only after that — act on the method:**

6. The evidence now rules out the "obvious" fix (more surrogate precision does nothing) and points
   at four targets, in order of how much they cost today: **(a)** the method cannot operate inside a
   small evaluation budget — the most exposed weakness; **(b)** the random anchor still produces
   ~9 % of answers on healthy data and ~66 % on over-constrained ones; **(c)** the data-point
   exemplar path never succeeds under strict priorities, and loosening its tolerance makes validity
   *worse*; **(d)** the systematic surrogate bias where the optimiser operates. Three optional
   extensions addressing (b)–(d) are implemented but **disabled by default** and were measured in
   the ablation study; they are candidates, not conclusions.
7. **Repair or duplicate the diabetes priority sets.** A set that excludes 100 % of the training data
   is a configuration error as much as a stress test; keeping both a repaired and the original
   version would separate the two readings and remove the main confound in Study 0.
