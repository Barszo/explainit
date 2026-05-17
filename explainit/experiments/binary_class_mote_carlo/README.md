## Overview of the binary_class_mote_carlo

The **binary_class_mote_carlo** directory is a research experiment focused on **comparing different counterfactual explanation methods** for machine learning models. The aim is to test module kept in explainers/random_search.py Here's what it contains:

### **Purpose**
This is a comparative study of various counterfactual explanation algorithms (techniques that explain why a model made a particular prediction by finding hypothetical changes to the input that would change the prediction).

### **Key Components**

**Datasets (3 datasets tested):**
- **Communities & Crime**: Predicts violent crime rates based on socioeconomic factors (~2000 communities, 99 features)
- **German Credit**: Binary credit classification dataset
- **Credit Card Default**: Predicts whether credit card holders will default

**Core Scripts:**
- test_all.py - Main test runner that orchestrates the entire experiment
- counterfactual_methods.py - Implements 5 different counterfactual algorithms:
  1. **Wachter** - Classic counterfactual method
  2. **Sparse Wachter** - Wachter with elastic net regularization
  3. **DiCE** - Diverse Counterfactual Explanations (custom gradient implementation)
  4. **Prototype-Guided** - Uses prototypes to guide counterfactual generation
  5. **Official DiCE** - Uses the official dice_ml library
- model_builder.py - Builds and trains baseline neural network models
- `data_downloader.py` - Handles dataset downloading and preprocessing

**Configuration:**
- config.yaml - Central configuration file specifying:
  - Which datasets to use
  - Model training parameters (epochs, batch size, learning rate)
  - Number of samples per dataset (100 test samples)
  - Which methods to test
  - Hyperparameters for each method

**Stored Data:**
- `/data/` - Raw and processed datasets (CSV, PKL, ZIP formats)
- `/models/` - Trained Keras models and training histories
- `/reports/` - Timestamped JSON and TXT reports from experiment runs (dating March 8-11, 2026)

**Results:**
- `/results/` - Contains:
  - `cf_metrics_overview.ipynb` - notebook to present results
  - Summary CSVs comparing metrics across datasets and methods

Data in "results" are saved there automatically but they should be moved to corresponding directories to keep them separated manually. For example, if results are obtained for expected number of CF equal to 5 then new directory called "5" should be created where the data should be moved. This is how notebook `cf_metrics_overview.ipynb` will read those results.

### **Workflow**
The experiment systematically tests all counterfactual methods on the same test samples, measuring:
- Whether target class was achieved
- L2/L1 distances from original instances
- Sparsity (number of features changed)
- Method convergence time and performance

This allows for a fair, systematic comparison of how well different counterfactual explanation techniques perform on the same problems.

## How to use binary_class_mote_carlo

The experiment workflow is basically:

1. download/preprocess data <- do it only once
2. train or load baseline models <- do it only once
3. run counterfactual explanation tests <- test until satisfied
4. inspect results

The important detail is: test_all.py already does all of that in one run, using config.yaml to control behavior.

---

## Step 1: Prepare the environment

From the repository root (`/explainit/explainit_project`):

```bash
source .venv/bin/activate # activate environment of your choosing 
pip install -r requirements.txt # install dependencies in your environment
```

If the virtual environment is already activated, just ensure you are in the repo root or experiment folder.

---

## Step 2: Configure the experiment

Open config.yaml.

Key settings:

- `datasets.selection`: e.g. `['credit_card_default', 'communities_crime', 'german_credit']`
- `model.epochs`, `model.batch_size`, `model.learning_rate`
- `model.force_retrain`: set `true` to retrain models even if saved ones exist
- `samples.n_samples`: how many test samples per dataset
- `standard_methods.selection`: which counterfactual methods to run
- `standard_methods.max_iterations`: max optimization iterations for methods

If you only want to run one dataset or one method, change these values and save the YAML.

Have in mind that there are plenty of parameters to configure. They control how each CF method is used. You should have knowledge of each of those methods to understand how to use these parameters correctly.

---

## Step 3: Run the experiment

From the experiment folder:

```bash
cd explainit/experiments/binary_class_mote_carlo
python test_all.py
```

What this does:

- loads config.yaml
- for each selected dataset:
  - loads raw/cached data
  - preprocesses if needed
  - saves preprocessed data into `data/`
- trains a baseline model or loads an existing saved model from `models/`
- selects test samples
- runs all selected counterfactual methods
- writes outputs to `results/`
- writes logs to `log_<dataset>.log`
- updates `status_<dataset>.txt`

---

## What happens automatically

### Data download/preprocessing
test_all.py calls the loader functions in data_downloader.py:
- `load_communities_and_crime()`
- `load_german_credit()`
- `load_credit_card_default()`

Each loader:
- reuses an existing processed `.pkl` cache if present
- otherwise downloads raw data and preprocesses it
- saves processed data in data

So you do not need a separate manual download step unless you want to force redownload.

### Model training
`train_simple_model()` in test_all.py:
- loads a saved model from `models/` if one exists and `force_retrain` is false
- otherwise trains a new Keras model
- saves:
  - `models/<dataset>_model.keras`
  - `models/<dataset>_history.pkl`

So the script handles training automatically.

### Running tests
The script runs:
- standard CF methods from counterfactual_methods.py
- preference-based methods via `explainit.priorities` / `explainit.explainers`

It does this in parallel by dataset using `ProcessPoolExecutor`.

---

## Step 4: Inspect the results

After a successful run, inspect:

- results
  - metric CSVs like `requested_cfs_metrics_summary_by_dataset.csv`
  - subfolders like `1/`, `2/`, `3/`, etc.
- cf_metrics_overview.ipynb
  - open notebook `/results/cf_metrics_overview.ipynb` to view summary tables and charts
- reports
  - detailed per-dataset JSON/TXT reports
- `explainit/experiments/binary_class_mote_carlo/log_<dataset>.log`
  - training and method execution logs
- `explainit/experiments/binary_class_mote_carlo/status_<dataset>.txt`
  - progress/status markers

---

## If you want to force a fresh run

### Force model retraining
In config.yaml:

```yaml
model:
  force_retrain: true
```

### Force data re-download / reprocessing
There is no direct CLI for that in test_all.py, but the loader function accepts a `force_download` argument if you call it manually from a notebook or custom script.

---
