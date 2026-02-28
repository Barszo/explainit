# ML Models for Tests

This directory contains comprehensive machine learning model training and evaluation scripts for various types of problems, along with pre-trained models and analysis results.

## 📁 Directory Structure

```
ML_models_for_tests/
├── ML_regression/                    # Multi-model regression analysis
│   ├── regression_examples.py        # Dataset loader classes
│   └── comprehensive_analysis.py     # Training script (12 models)
├── ML_regression_results/            # Pre-trained models and results
│   ├── data_sources.md              # Dataset documentation
│   ├── FINAL_SUMMARY.txt            # Overall results summary
│   └── [8 dataset folders]/         # Individual dataset results
│
├── linear_regression/                # Linear regression analysis
│   ├── linear_regression_examples.py # Dataset loader classes
│   └── linear_regression_analysis.py # Training script (1 model)
├── linear_regression_results/        # Pre-trained models and results
│   ├── data_sources.md              # Dataset documentation
│   ├── FINAL_SUMMARY.txt            # Overall results summary
│   └── [8 dataset folders]/         # Individual dataset results
│
├── binary_classifiers/               # Binary classification analysis
│   ├── classification_examples.py    # Dataset loader classes
│   └── classification_analysis.py    # Training script (10 models)
└── binary_classifiers_results/       # Pre-trained models and results
    ├── data_sources.md              # Dataset documentation
    ├── FINAL_SUMMARY.txt            # Overall results summary
    └── [7 dataset folders]/         # Individual dataset results
```

---

## 🤖 Available Model Types

### 1. **ML Regression** (12 models per dataset)

**Location:** `ML_regression/`

**Models trained:**
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Network (Basic)
- Neural Network (Shallow - 1 hidden layer)
- Neural Network (Medium - 2 hidden layers)
- Neural Network (Deep - 3 hidden layers)
- Neural Network (Wide - large single layer)
- Neural Network (Regularized - with dropout)
- Neural Network (Residual - skip connections)

**Datasets (8):**
- California Housing
- Diabetes
- Wine Quality
- Ames Housing
- Auto MPG
- Concrete Strength
- Energy Efficiency
- Bike Sharing

**Best Results:**
- Energy Efficiency: R² = 0.99+ (Neural Networks)
- Concrete Strength: R² = 0.93+ (XGBoost/Neural Networks)
- California Housing: R² = 0.85+ (Gradient Boosting)

---

### 2. **Linear Regression** (1 model per dataset)

**Location:** `linear_regression/`

**Models trained:**
- Linear Regression (sklearn)

**Datasets (8 successful):**
- Advertising Sales
- Student Performance
- Medical Insurance
- Real Estate Valuation
- Fish Market Weight
- Yacht Hydrodynamics
- Airfoil Self-Noise
- Wine Quality Red

**Best Results:**
- Yacht Hydrodynamics: R² = 0.9175 (PERFECT)
- Medical Insurance: R² = 0.9089 (PERFECT)
- Airfoil Self-Noise: R² = 0.8852 (VERY GOOD)

---

### 3. **Binary Classification** (10 models per dataset)

**Location:** `binary_classifiers/`

**Models trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- XGBoost
- Neural Network (Small - 1 layer)
- Neural Network (Medium - 2 layers)
- Neural Network (Large - 3 layers)

**Datasets (7 successful):**
- Breast Cancer
- Heart Disease
- Diabetes (Pima Indians)
- Credit Card Fraud
- Spam Detection
- Ionosphere
- Adult Income

**Best Results:**
- Breast Cancer: 98.25% accuracy (Logistic Regression)
- Ionosphere: 97.18% accuracy (Gradient Boosting)
- Credit Card Fraud: 99.96% accuracy, F1=0.87 (Random Forest)

---

## 🚀 How to Train Models

### Prerequisites

```bash
# Activate your virtual environment
source .venv/bin/activate

# Required packages are already installed in your environment:
# - numpy, pandas, scikit-learn
# - xgboost (optional but recommended)
# - tensorflow/keras (for neural networks)
```

### Training Commands

#### 1. ML Regression (Multi-Model)

```bash
# From project root
cd ML_models_for_tests/ML_regression
python comprehensive_analysis.py
```

**What it does:**
- Downloads 8 regression datasets
- Trains 12 different models on each dataset
- Saves all models, scalers, and data
- Creates comprehensive analysis summaries
- Generates performance comparison charts
- Typical runtime: 20-40 minutes

**Output:**
- `../ML_regression_results/[Dataset_Name]/` - Contains:
  - Trained models (`.pkl` and `.h5` files)
  - StandardScaler (`.pkl`)
  - Data files (`X_data.csv`, `y_data.csv`)
  - `ANALYSIS_SUMMARY.txt` - Detailed performance report
- `../ML_regression_results/FINAL_SUMMARY.txt` - Overall comparison

---

#### 2. Linear Regression

```bash
# From project root
cd ML_models_for_tests/linear_regression
python linear_regression_analysis.py
```

**What it does:**
- Downloads 8+ datasets with strong linear relationships
- Trains Linear Regression on each
- Includes feature importance analysis
- Saves models and detailed summaries
- Typical runtime: 5-10 minutes

**Output:**
- `../linear_regression_results/[Dataset_Name]/` - Contains:
  - `LinearRegression_model.pkl`
  - `scaler.pkl`
  - Data files (`X_data.csv`, `y_data.csv`)
  - `ANALYSIS_SUMMARY.txt` - With coefficients interpretation
- `../linear_regression_results/FINAL_SUMMARY.txt` - Overall results

---

#### 3. Binary Classification

```bash
# From project root
cd ML_models_for_tests/binary_classifiers
python classification_analysis.py
```

**What it does:**
- Downloads 8 binary classification datasets
- Trains 10 different models on each dataset
- Saves all models and scalers
- Creates comprehensive analysis with confusion matrices
- Typical runtime: 15-30 minutes

**Output:**
- `../binary_classifiers_results/[Dataset_Name]/` - Contains:
  - Trained models (`.pkl` files)
  - StandardScalers (`.pkl` for applicable models)
  - Data files (`X_data.csv`, `y_data.csv`)
  - `ANALYSIS_SUMMARY.txt` - With confusion matrix, precision, recall, F1, ROC AUC
- `../binary_classifiers_results/FINAL_SUMMARY.txt` - Overall comparison

---

## 📊 Understanding the Results

### Each Dataset Directory Contains:

1. **Data Files:**
   - `X_data.csv` - Feature matrix
   - `y_data.csv` - Target values/labels

2. **Model Files:**
   - `.pkl` files - Scikit-learn and XGBoost models (use `pickle.load()`)
   - `.h5` files - Keras/TensorFlow neural networks (use `tf.keras.models.load_model()`)
   - `scaler.pkl` - StandardScaler for preprocessing

3. **Analysis Files:**
   - `ANALYSIS_SUMMARY.txt` - Detailed performance metrics
     - Dataset statistics
     - Model comparison table
     - Best model details
     - Overfitting analysis
     - Feature importance (where applicable)
     - File locations

### Grading System

**For Regression (based on R² score):**
- **PERFECT:** R² ≥ 0.90
- **VERY GOOD:** R² ≥ 0.80
- **GOOD:** R² ≥ 0.65
- **NOT ACCEPTABLE:** R² ≥ 0.40
- **BAD:** R² ≥ 0.20
- **VERY BAD:** R² < 0.20

**For Classification (based on average of Accuracy + F1):**
- **EXCELLENT:** ≥ 0.95
- **VERY GOOD:** ≥ 0.90
- **GOOD:** ≥ 0.85
- **ACCEPTABLE:** ≥ 0.75
- **POOR:** ≥ 0.65
- **VERY POOR:** < 0.65

---

## 📖 Data Sources

Each results directory contains a `data_sources.md` file with:
- Dataset download URLs (verified)
- Original source organizations
- Dataset descriptions
- Feature explanations
- Citation information

All URLs have been verified against the code to ensure accuracy.

---

## 🔧 Using Pre-trained Models

### Loading a Model

```python
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load a scikit-learn/XGBoost model
with open('ML_regression_results/California_Housing/Ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load a neural network
nn_model = tf.keras.models.load_model('ML_regression_results/California_Housing/NN_Deep_model.h5')

# Load the scaler
with open('ML_regression_results/California_Housing/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### Making Predictions

```python
import pandas as pd

# Load test data
X_test = pd.read_csv('ML_regression_results/California_Housing/X_data.csv')

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = model.predict(X_test_scaled)
```

---

## 📈 Performance Summary

### Best Overall Models by Task:

**Regression:**
- **Traditional ML:** XGBoost (3/8 datasets)
- **Neural Networks:** Deep/Residual networks (5/8 datasets)
- **Average R²:** 0.75-0.95 depending on dataset complexity

**Linear Regression:**
- **Achieves PERFECT grade** on 2/8 datasets
- **Best for:** Insurance prediction, yacht hydrodynamics
- **Average R²:** 0.80+

**Binary Classification:**
- **Best Model:** XGBoost (3/7 datasets)
- **Runner-up:** Neural Networks (1/7), Random Forest (1/7)
- **Average Accuracy:** 89-92% across all models
- **Average F1 Score:** 0.78-0.83 across all models

---

## ⚠️ Important Notes

1. **Neural Networks require TensorFlow:**
   - If TensorFlow is not installed, only traditional ML models will be trained
   - Install with: `pip install tensorflow`

2. **XGBoost is highly recommended:**
   - Consistently achieves top performance
   - Install with: `pip install xgboost`

3. **Data is downloaded automatically:**
   - Scripts will attempt to download datasets from public sources
   - Fallback synthetic data is generated if downloads fail
   - Internet connection recommended for first run

4. **Training time varies:**
   - Neural networks take longer (especially with large datasets)
   - XGBoost is generally fast
   - Progress is printed during training

5. **Results are reproducible:**
   - Random seed is set to 42 for all models
   - Re-running scripts will produce consistent results

---

## 🎯 Use Cases

### For Learning:
- Study how different models perform on various datasets
- Compare traditional ML vs neural networks
- Understand hyperparameter effects
- Learn proper model evaluation techniques

### For Research:
- Baseline models for comparison
- Pre-trained models for transfer learning
- Dataset loaders for quick experiments
- Performance benchmarks

### For Production:
- Select best model for your use case
- Use pre-trained models as starting points
- Reference implementation for model pipelines
- Evaluation framework template

---

## 📝 File Naming Convention

- **Models:** `[ModelName]_model.pkl` or `.h5`
- **Scalers:** `scaler.pkl` or `[ModelName]_scaler.pkl`
- **Data:** `X_data.csv`, `y_data.csv`
- **Summaries:** `ANALYSIS_SUMMARY.txt`, `FINAL_SUMMARY.txt`
- **Documentation:** `data_sources.md`

---

## 🆘 Troubleshooting

**Import errors:**
```bash
# Make sure you're in the correct directory
cd ML_models_for_tests/[ml_type]
python [analysis_script].py
```

**Dataset download fails:**
- Scripts include fallback mechanisms
- Check internet connection
- See `data_sources.md` for alternative URLs

**Out of memory errors:**
- Reduce batch size in neural network training
- Use fewer neural network architectures
- Comment out memory-intensive models in the code

**XGBoost/TensorFlow not found:**
- These are optional but recommended
- Scripts will skip unavailable models and continue

---

**Last Updated:** December 13, 2025
