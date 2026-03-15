# Dataset Sources

This document explains the sources and characteristics of the datasets used in the counterfactual experiments.

## Datasets Overview

Both datasets are downloaded from the **UCI Machine Learning Repository**, a widely-used public repository for machine learning datasets.

---

## 1. Communities and Crime Dataset

### Source
- **Repository**: UCI Machine Learning Repository
- **URL**: http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data
- **Dataset Page**: https://archive.ics.uci.edu/ml/datasets/communities+and+crime
- **License**: Public domain
- **Citation**: U. S. Department of Commerce, Bureau of the Census, Census Of Population And Housing 1990 United States: Summary Tape File 1a & 3a (Computer Files)

### Description
The Communities and Crime dataset contains demographic and economic information about communities across the United States. The goal is to predict whether there is violent crime in a community based on various socioeconomic factors.

### Characteristics
- **Original size**: ~2000 communities (varies based on missing data handling)
- **Total attributes**: 128 (after preprocessing: 99 continuous features)
- **Target variable**: Violent crime rate (binarized at median)
  - 0 = High crime (negative outcome)
  - 1 = Low crime (positive outcome)
- **Feature types**: Continuous numerical features
- **Preprocessing**:
  - First 5 non-predictive columns removed (state, county, community, name, fold)
  - Rows with missing values removed
  - Target binarized at median crime rate
  - Features with variance < 1e-6 removed
  - Top 99 features by variance selected
  - 0 mean, unit variance scaling applied

### Usage Context
This dataset is commonly used in fairness and counterfactual explanation research. Communities assessed at higher risk for crime could face reduced funding for programs, creating incentives for favorable predictions.

### Files Generated
- `communities_crime_raw.csv` - Downloaded raw data
- `communities_crime_processed.pkl` - Preprocessed train/test splits with scaler

---

## 2. German Credit Dataset

### Source
- **Repository**: UCI Machine Learning Repository  
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- **Dataset Page**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- **License**: Public domain
- **Citation**: Professor Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg

### Description
The German Credit dataset contains financial and personal information about individuals. The task is to predict whether a person poses high or low credit risk. This dataset is a standard benchmark in credit scoring and fairness research.

### Characteristics
- **Original size**: 1000 individuals
- **Total attributes**: 21 (mixture of categorical and numerical)
- **Features used**: 7 numerical features (as per paper methodology)
  - `duration` - Duration of credit in months
  - `credit_amount` - Credit amount
  - `age` - Age in years
  - `installment_rate` - Installment rate as % of disposable income
  - `residence` - Present residence since (years)
  - `existing_credits` - Number of existing credits at this bank
  - `num_dependents` - Number of people liable for maintenance
- **Target variable**: Credit risk
  - Original: 1=good credit, 2=bad credit
  - Transformed: 1=low risk (positive outcome), 0=high risk (negative outcome)
- **Class distribution**: 70% low risk, 30% high risk
- **Preprocessing**:
  - Only 7 numerical features extracted from 21 total attributes
  - Target converted to binary (1=low risk, 0=high risk)
  - 0 mean, unit variance scaling applied
  - 80/20 train-test split with stratification

### Usage Context
This dataset is extensively used in counterfactual explanation and fairness literature. Individuals have strong incentives to receive favorable credit risk assessments to qualify for loans, making it relevant for studying adversarial manipulation.

### Files Generated
- `german_credit_raw.csv` - Downloaded raw data with all 21 columns
- `german_credit_processed.pkl` - Preprocessed train/test splits with 7 numerical features and scaler

---

## Data Processing Pipeline

Both datasets follow this pipeline:

1. **Check for cached preprocessed data** (`.pkl` files)
   - If exists: Load instantly (fastest)
   
2. **Check for raw downloaded data** (`.csv` files)
   - If exists: Load and preprocess
   
3. **Download from UCI repository**
   - Primary: Standard SSL download
   - Fallback: Unverified SSL context (for certificate issues)
   
4. **Preprocess data**
   - Select relevant features
   - Handle missing values
   - Transform target variable
   - Apply feature scaling (StandardScaler)
   - Split into train/test (80/20)
   
5. **Cache results**
   - Save raw data as CSV
   - Save preprocessed data as pickle

---

## Data Split Details

Both datasets use the same splitting strategy:
- **Split ratio**: 80% train, 20% test
- **Random seed**: 42 (for reproducibility)
- **Stratification**: Yes (maintains class balance in splits)

### After Preprocessing:

**Communities and Crime**:
- Training: 255 samples
- Test: 64 samples
- Features: 99

**German Credit**:
- Training: 800 samples
- Test: 200 samples
- Features: 7

---

## References

These datasets are standard benchmarks used in:

1. Counterfactual explanation research:
   - Wachter et al. (2017) - Counterfactual Explanations without Opening the Black Box
   - Mothilal et al. (2020) - DiCE: Diverse Counterfactual Explanations
   - Van Looveren & Klaise (2021) - Interpretable Counterfactual Explanations Guided by Prototypes

2. Fairness in machine learning:
   - Slack et al. (2020) - Fooling LIME and SHAP
   - Multiple fairness benchmarking studies

3. Adversarial robustness:
   - Research on gaming machine learning systems
   - Studies on strategic classification

---

## Notes

- Both datasets are in the **public domain** and freely available for research purposes
- The preprocessing follows the methodology described in adversarial counterfactual manipulation research papers
- The data downloader includes SSL certificate handling for macOS/Python environments that may have certificate verification issues
- Once downloaded, data is cached locally to avoid repeated downloads
