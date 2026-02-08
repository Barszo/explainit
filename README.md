# explainit

Preference-based counterfactual explanation library for machine learning models. Generates actionable "what-if" scenarios using user-defined priority functions.

## Project Structure

```
explainit_project/
├── explainit/                        # Main package
│   ├── explainers/                   # Counterfactual generation methods
│   │   ├── random_search.py          # Preference-based random search
│   │   ├── minlp_search.py           # MINLP optimization approach
│   │   └── basic.py                  # Helper functions
│   ├── priorities/                   # User-defined preference functions
│   │   ├── linear.py                 # Linear priority functions
│   │   └── nonlinear.py              # Exponential and step functions
│   ├── utils/                        # Utilities (plotting, helpers)
│   ├── examples/                     # Tutorial notebooks and demos
│   └── experiments/                  # Research experiments
│       └── priorities_with_random_search/  # Method comparison framework
│           ├── experiment_final.py   # Main experiment runner
│           ├── standard_methods.py   # Baseline counterfactual methods
│           ├── compare_methods.py    # Results comparison analysis
│           └── *.csv                 # Experimental results
├── ML_models_for_tests/              # Pre-trained models and datasets
│   ├── ML_regression_results/        # Regression models (8 datasets)
│   ├── linear_regression_results/    # Linear models (10 datasets)
│   └── binary_classifiers_results/   # Classification models (8 datasets)
├── tests/                            # Unit tests
└── setup.py                          # Package configuration
```

## Explainers

**`random_search.py`** - Preference-based counterfactual generation using Monte Carlo sampling. Takes user-defined priority functions for each feature and generates counterfactuals ranked by preference score. Supports both numerical and categorical features with actionability constraints.

**`minlp_search.py`** - Mixed-Integer Nonlinear Programming approach. Uses Shapley values to prioritize features and optimization to find minimal-distance counterfactuals. Includes linear programming for feature selection and constraint handling.

**`basic.py`** - Helper functions for generating feature combinations based on actionability and modifiability constraints. Used by other explainers for combinatorial search.

## Examples

**`regression_tutorial.ipynb`** - Comprehensive tutorial comparing multiple regression algorithms (Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting) on California Housing and Diabetes datasets. Uses scikit-learn for data loading, preprocessing (StandardScaler), model training, and evaluation metrics (MSE, MAE, R², MAPE). Shows dataset characteristics, feature descriptions, train-test splitting, cross-validation, and performance comparisons. Teaches proper ML workflow and model selection for regression before applying explainability methods.

**`dev_notebook_random_search.ipynb`** - Demonstrates `RandomSearchExplainer` on German Credit dataset (binary classification). Uses pandas, scikit-learn (MinMaxScaler, OneHotEncoder, LogisticRegression) to preprocess mixed categorical-numerical data. Shows: (1) selecting a specific sample, (2) defining exponential priority functions via `explainit.priorities.nonlinear`, (3) generating counterfactuals with Monte Carlo sampling, (4) visualizing priority functions and results. Includes markdown explanations of priority structures and actionability constraints for features like Age, Credit amount, Duration, and categorical variables (Sex, Job, Housing, etc.).

**`dev_notebook_minl_linear.ipynb`** - Demonstrates `MINLSearchExplainer` with linear priority functions on German Credit dataset. Same preprocessing pipeline as random_search notebook but uses MINLP optimization approach instead of random sampling. Shows how to define linear priorities, compute Shapley values for feature importance, and generate counterfactuals through mixed-integer optimization. Includes detailed markdown documentation of priority specification format and explanations of the optimization process.

**`dev_notebook_minl_nonlinear.ipynb`** - Full end-to-end regression example using California Housing dataset with `MINLSearchExplainer` and nonlinear priority functions. Uses TensorFlow/Keras to build and train a deep neural network (128→64→32→16→1 with dropout). Demonstrates: (1) complete ML pipeline (data loading, StandardScaler preprocessing, model training with 100 epochs), (2) evaluation with multiple metrics (MSE, MAE, R², RMSE, MAPE), (3) training history and prediction visualizations, (4) defining exponential and step priority functions via `explainit.priorities.nonlinear.exponential` and `basic_linear_step`, (5) selecting low-prediction samples and finding counterfactuals to reach higher price targets. Shows preference visualization, Shapley value computation, and counterfactual generation with detailed markdown explanations throughout.

**`minl_nonlinear.py`** - Python script version of the nonlinear notebook. Standalone executable demonstrating California Housing regression with neural network training, evaluation, and preparation for MINLP explainer. Useful as a template for batch processing or integration into larger pipelines.

**`data/`** - Example datasets (German Credit CSV) for tutorials

## Experiments

Scientific research experiments prepared for academic papers and method validation. Contains experimental frameworks with configurable pipelines for systematic evaluation.

**`priorities_with_random_search/`** - ⚠️ **Work in Progress** - Comparative study framework evaluating the preference-based `RandomSearchExplainer` against standard counterfactual methods (Wachter 2017, Growing Spheres, Prototype-based, Gradient-based). Includes experiment runner (`experiment_final.py`) for automated testing across datasets/models, standard method implementations (`standard_methods.py`), comparison analysis tools (`compare_methods.py`), and result CSVs. Currently configured for Auto MPG dataset with NN_Residual model; expansion to additional datasets from `ML_models_for_tests/` in progress. See `README_COMPARISON.md` and `CONFIG_GUIDE.md` for usage.

## Quick Start

```python
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.priorities.nonlinear import exponential

# Define preferences and generate counterfactuals
explainer = RandomSearchExplainer(model_pred, priorities, sample, target)
counterfactuals = explainer.generate_random_samples(n_samples=10000)
```

## ML Models for Tests

Pre-trained model repository providing diverse ML models and datasets for use in `examples/` and `experiments/`. Contains automated training scripts that download datasets, train multiple model architectures, and save everything needed for counterfactual explanation research.

### Purpose

This module prepares baseline models and datasets used throughout the project:
- **For Examples**: Provides trained models and data for tutorial notebooks
- **For Experiments**: Supplies the models tested in comparative studies
- **For Research**: Offers diverse model-dataset combinations for testing explainability methods

### Directory Structure

**Three Problem Types:**

1. **`ML_regression/`** - Multi-model regression (12 algorithms × 8 datasets)
   - **Script**: `comprehensive_analysis.py`
   - **Datasets**: `regression_examples.py` (loader classes)
   - **Results**: `../ML_regression_results/`

2. **`linear_regression/`** - Simple linear models (1 algorithm × 10 datasets)
   - **Script**: `linear_regression_analysis.py`
   - **Datasets**: `linear_regression_examples.py`
   - **Results**: `../linear_regression_results/`

3. **`binary_classifiers/`** - Classification models (10 algorithms × 8 datasets)
   - **Script**: `classification_analysis.py`
   - **Datasets**: `classification_examples.py`
   - **Results**: `../binary_classifiers_results/`

### Model Types & Differences

**ML Regression (12 models):**
- **Traditional ML**: Ridge, Lasso, ElasticNet (linear with regularization), RandomForest (tree ensemble), GradientBoosting (boosted trees), XGBoost (optimized gradient boosting)
- **Neural Networks**: Basic (single layer), Shallow (2 layers, 64→32), Medium (4 layers, 128→64→32→16), Deep (6 layers with BatchNorm), Wide (512 units single layer), Regularized (heavy dropout), Residual (skip connections)
- **Key Differences**: Traditional ML = interpretable, fast, good for small data; Neural Networks = flexible, powerful for complex patterns, require more data/compute
- **Performance Range**: R² 0.65-0.99 depending on dataset complexity

**Linear Regression (1 model):**
- **Algorithm**: Standard least-squares regression
- **Best For**: Datasets with strong linear feature-target relationships
- **Advantage**: Highly interpretable coefficients, fast training
- **Performance**: R² 0.75-0.92 on suitable datasets (e.g., Yacht Hydrodynamics, Medical Insurance)

**Binary Classifiers (10 models):**
- **Traditional ML**: LogisticRegression (linear classifier), DecisionTree (single tree), RandomForest (tree ensemble), GradientBoosting (boosted trees), XGBoost (optimized boosting), SVM (support vector), NaiveBayes (probabilistic)
- **Neural Networks**: Small (1 layer, 64 units), Medium (2 layers, 128→64), Large (3 layers, 256→128→64)
- **Key Differences**: SVM/NaiveBayes = theoretical foundations; Tree methods = feature interactions; Neural nets = non-linear boundaries
- **Performance Range**: 85-98% accuracy depending on dataset

### Datasets & Their Differences

**Regression Datasets (8):**
- **California Housing** (20,640 samples, 8 features): Housing prices, geographic data, R² ~0.85
- **Diabetes** (442 samples, 10 features): Disease progression, medical measurements, R² ~0.45
- **Wine Quality** (6,497 samples, 11 features): Wine ratings, chemical properties, R² ~0.55
- **Ames Housing** (1,460 samples, 20 features): House prices, detailed property features, R² ~0.90
- **Auto MPG** (398 samples, 4 features): Fuel efficiency, vehicle specs, R² ~0.88
- **Concrete Strength** (1,030 samples, 8 features): Concrete compression, mixture ingredients, R² ~0.93
- **Energy Efficiency** (768 samples, 8 features): Building energy load, architectural features, R² ~0.99
- **Bike Sharing** (17,389 samples, 11 features): Rental demand, weather/time data, R² ~0.85

**Classification Datasets (8):**
- **Breast Cancer** (569 samples, 30 features): Tumor diagnosis, cell measurements, 98% accuracy
- **Heart Disease** (303 samples, 13 features): Disease presence, medical indicators, 86% accuracy
- **Diabetes** (768 samples, 8 features): Diabetes diagnosis, health metrics, 78% accuracy
- **Credit Card Fraud** (284,807 samples, 30 features): Fraud detection, transaction data, 99.96% accuracy
- **Spam Detection** (4,601 samples, 57 features): Email classification, word frequencies, 94% accuracy
- **Ionosphere** (351 samples, 34 features): Radar signal classification, 97% accuracy
- **Adult Income** (32,561 samples, 14 features): Income prediction, demographic data, 85% accuracy

**Key Differences:**
- **Size**: Small (300-1,000), Medium (1,000-10,000), Large (10,000+)
- **Complexity**: Linear relationships vs. non-linear patterns
- **Features**: Continuous only vs. mixed types
- **Domain**: Medical, financial, physical sciences, social sciences

### How to Train Models

**Prerequisites:**
```bash
# Activate environment
source .venv/bin/activate

# Ensure packages installed
pip install numpy pandas scikit-learn xgboost tensorflow
```

**Training Commands:**

```bash
# 1. ML Regression (12 models × 8 datasets) - 20-40 minutes
cd ML_models_for_tests/ML_regression
python comprehensive_analysis.py

# 2. Linear Regression (1 model × 10 datasets) - 5-10 minutes
cd ML_models_for_tests/linear_regression
python linear_regression_analysis.py

# 3. Binary Classification (10 models × 8 datasets) - 15-30 minutes
cd ML_models_for_tests/binary_classifiers
python classification_analysis.py
```

**What Happens:**
1. **Downloads datasets** from public sources (scikit-learn, UCI, Kaggle, OpenML)
2. **Preprocesses data**: handles missing values, scaling, encoding
3. **Trains all models** with optimized hyperparameters
4. **Evaluates performance**: cross-validation, test set metrics
5. **Saves everything**:
   - `[ModelName]_model.pkl` - Scikit-learn/XGBoost models (use `pickle.load()`)
   - `[ModelName]_model.h5` - Keras models (use `tf.keras.models.load_model()`)
   - `scaler.pkl` - StandardScaler for preprocessing
   - `X_data.csv`, `y_data.csv` - Complete dataset
   - `ANALYSIS_SUMMARY.txt` - Performance report with model comparison table

**Output Structure:**
```
ML_regression_results/
├── Auto_MPG/
│   ├── Ridge_model.pkl
│   ├── NN_Residual_model.h5
│   ├── scaler.pkl
│   ├── X_data.csv
│   ├── y_data.csv
│   └── ANALYSIS_SUMMARY.txt
├── California_Housing/
│   └── ... (same structure)
├── FINAL_SUMMARY.txt
└── data_sources.md
```

### How to Use Pre-trained Models

**Loading Models:**
```python
import pickle
import tensorflow as tf
import pandas as pd

# Load scikit-learn/XGBoost model
with open('ML_models_for_tests/ML_regression_results/Auto_MPG/Ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load neural network
nn_model = tf.keras.models.load_model(
    'ML_models_for_tests/ML_regression_results/Auto_MPG/NN_Residual_model.h5',
    compile=False
)
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Load scaler
with open('ML_models_for_tests/ML_regression_results/Auto_MPG/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data
X = pd.read_csv('ML_models_for_tests/ML_regression_results/Auto_MPG/X_data.csv')
y = pd.read_csv('ML_models_for_tests/ML_regression_results/Auto_MPG/y_data.csv')
```

**Making Predictions:**
```python
# Scale features
X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)  # For scikit-learn
predictions = nn_model.predict(X_scaled)  # For neural networks
```

### Recreating Models or Re-downloading Data

**Complete Reset:**
```bash
# Delete all results
rm -rf ML_models_for_tests/*_results/

# Re-train everything
cd ML_models_for_tests/ML_regression && python comprehensive_analysis.py
cd ../linear_regression && python linear_regression_analysis.py
cd ../binary_classifiers && python classification_analysis.py
```

**Single Dataset:**
```bash
# Edit the script to train only specific datasets
# In comprehensive_analysis.py, modify:
datasets = {
    'Auto_MPG': AutoMPGExample,  # Keep only this one
    # Comment out others
}
```

**Data Sources:**
- Scripts automatically download from public repositories
- See `data_sources.md` in each results folder for URLs
- Fallback: Scripts generate synthetic data if download fails
- All datasets are free and publicly available

### Performance Summary

**Best Models by Task:**
- **Regression**: XGBoost (traditional), Deep/Residual NN (neural) - Average R² 0.75-0.95
- **Linear Problems**: Standard Linear Regression - R² 0.80-0.92
- **Classification**: XGBoost, Random Forest - 85-98% accuracy

**Training Time (on modern CPU):**
- Traditional ML: 1-5 minutes per dataset
- Neural Networks: 5-15 minutes per dataset
- Total (all datasets): 1-2 hours for complete setup

**Storage Requirements:**
- Per dataset: 1-10 MB (models + data)
- Total (all results): ~100-200 MB

See `ML_models_for_tests/README.md` for detailed documentation including dataset descriptions, model architectures, hyperparameters, performance grading system, and troubleshooting guide.