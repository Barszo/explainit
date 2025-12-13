"""
Well-known binary classification problems with practical datasets.

This module demonstrates data loading, preprocessing, training, and evaluation
for classic binary classification datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False


class BinaryClassificationExample:
    """Base class for binary classification examples."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_full = None
        self.y_full = None
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        
    def load_data(self):
        """Load the dataset. To be implemented by subclasses."""
        raise NotImplementedError
    
    def show_dataset_summary(self):
        """Display comprehensive dataset summary."""
        if self.X_full is None or self.y_full is None:
            print("Error: Dataset not loaded yet!")
            return
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        # Basic info
        print(f"\nDataset Shape: {self.X_full.shape}")
        print(f"Number of samples: {self.X_full.shape[0]}")
        print(f"Number of features: {self.X_full.shape[1]}")
        
        # Class distribution
        class_counts = pd.Series(self.y_full).value_counts().sort_index()
        print(f"\nClass Distribution:")
        for class_label, count in class_counts.items():
            percentage = (count / len(self.y_full)) * 100
            print(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        # Check for imbalance
        balance_ratio = class_counts.min() / class_counts.max()
        if balance_ratio < 0.5:
            print(f"  ⚠ Class imbalance detected (ratio: {balance_ratio:.3f})")
        else:
            print(f"  ✓ Classes reasonably balanced (ratio: {balance_ratio:.3f})")
        
        # Feature statistics
        print(f"\n{'Feature':<20} {'Type':<10} {'Unique':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 95)
        
        for col in self.X_full.columns:
            n_unique = self.X_full[col].nunique()
            min_val = self.X_full[col].min()
            max_val = self.X_full[col].max()
            mean_val = self.X_full[col].mean()
            std_val = self.X_full[col].std()
            dtype = str(self.X_full[col].dtype)
            
            print(f"{col:<20} {dtype:<10} {n_unique:<10} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} {std_val:<12.4f}")
        
        # Missing values
        print("\nMissing Values:")
        missing = self.X_full.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found ✓")
        else:
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing[col]/len(self.X_full)*100:.2f}%)")
        
        print("="*80)
        
    def preprocess(self, scale=True):
        """Preprocess the data."""
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
    def train(self, model_type='logistic'):
        """Train a classification model."""
        models = {
            'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'gbm': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(random_state=self.random_state, probability=True),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE and model_type == 'xgb':
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # Add Neural Network if available
        if NEURAL_NET_AVAILABLE and model_type in ['nn_small', 'nn_medium', 'nn_large']:
            if model_type == 'nn_small':
                hidden_layers = (50,)
            elif model_type == 'nn_medium':
                hidden_layers = (100, 50)
            else:  # nn_large
                hidden_layers = (200, 100, 50)
                
            models[model_type] = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        self.model = models.get(model_type, models['logistic'])
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        """Evaluate the model and return metrics."""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Get probability predictions for ROC AUC
        try:
            y_prob_train = self.model.predict_proba(self.X_train)[:, 1]
            y_prob_test = self.model.predict_proba(self.X_test)[:, 1]
        except:
            y_prob_train = y_pred_train
            y_prob_test = y_pred_test
        
        self.results = {
            'train': {
                'accuracy': accuracy_score(self.y_train, y_pred_train),
                'precision': precision_score(self.y_train, y_pred_train, zero_division=0),
                'recall': recall_score(self.y_train, y_pred_train, zero_division=0),
                'f1': f1_score(self.y_train, y_pred_train, zero_division=0),
                'roc_auc': roc_auc_score(self.y_train, y_prob_train),
                'confusion_matrix': confusion_matrix(self.y_train, y_pred_train)
            },
            'test': {
                'accuracy': accuracy_score(self.y_test, y_pred_test),
                'precision': precision_score(self.y_test, y_pred_test, zero_division=0),
                'recall': recall_score(self.y_test, y_pred_test, zero_division=0),
                'f1': f1_score(self.y_test, y_pred_test, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_prob_test),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred_test)
            }
        }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"{'='*80}")
        
        print(f"\nTraining Set Performance:")
        print(f"  Accuracy:  {self.results['train']['accuracy']:.4f}")
        print(f"  Precision: {self.results['train']['precision']:.4f}")
        print(f"  Recall:    {self.results['train']['recall']:.4f}")
        print(f"  F1 Score:  {self.results['train']['f1']:.4f}")
        print(f"  ROC AUC:   {self.results['train']['roc_auc']:.4f}")
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {self.results['test']['accuracy']:.4f}")
        print(f"  Precision: {self.results['test']['precision']:.4f}")
        print(f"  Recall:    {self.results['test']['recall']:.4f}")
        print(f"  F1 Score:  {self.results['test']['f1']:.4f}")
        print(f"  ROC AUC:   {self.results['test']['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix (Test):")
        cm = self.results['test']['confusion_matrix']
        print(f"  [[TN={cm[0,0]:<4} FP={cm[0,1]:<4}]")
        print(f"   [FN={cm[1,0]:<4} TP={cm[1,1]:<4}]]")
        print(f"{'='*80}\n")


class BreastCancerExample(BinaryClassificationExample):
    """
    Breast Cancer Wisconsin Dataset
    
    Features: 30 continuous features computed from digitized images
    - Mean, standard error, and "worst" values for 10 characteristics:
      * radius, texture, perimeter, area, smoothness
      * compactness, concavity, concave points, symmetry, fractal dimension
    
    Target: Binary classification (0=malignant, 1=benign)
    Samples: 569
    
    Expected Performance:
    - Logistic Regression: ~96% accuracy
    - Random Forest: ~96-97% accuracy
    - XGBoost: ~97-98% accuracy
    
    Note: Well-balanced dataset, ideal for binary classification demonstrations.
    """
    
    def load_data(self, test_size=0.2):
        """Load Breast Cancer dataset."""
        print("Loading Breast Cancer Wisconsin dataset...")
        data = load_breast_cancer(as_frame=True)
        
        self.X_full = data.data
        self.y_full = data.target
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns[:5])}... (showing first 5 of {len(self.X_full.columns)})")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class HeartDiseaseExample(BinaryClassificationExample):
    """
    Heart Disease Dataset
    
    Features: 13 clinical features
    - age, sex, chest pain type, resting blood pressure
    - cholesterol, fasting blood sugar, resting ECG
    - max heart rate, exercise induced angina
    - ST depression, slope, number of vessels, thalassemia
    
    Target: Binary (0=no disease, 1=disease)
    Samples: ~303
    
    Expected Performance:
    - Logistic Regression: ~83-85% accuracy
    - Random Forest: ~84-86% accuracy
    - XGBoost: ~85-88% accuracy
    """
    
    def load_data(self, test_size=0.2):
        """Load Heart Disease dataset."""
        print("Loading Heart Disease dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            # UCI Cleveland Heart Disease dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            
            column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            data = pd.read_csv(url, names=column_names, na_values='?')
            
            # Remove rows with missing values
            data = data.dropna()
            
            # Convert target to binary (0 = no disease, 1-4 = disease)
            data['target'] = (data['target'] > 0).astype(int)
            
            print("✓ Dataset loaded from UCI repository")
            
        except Exception as e:
            print(f"Primary source failed: {e}")
            # Fallback to alternative source
            try:
                url = "https://raw.githubusercontent.com/rashida048/Datasets/master/heart.csv"
                data = pd.read_csv(url)
                # Rename target column if needed
                if 'target' not in data.columns:
                    target_col = [col for col in data.columns if 'target' in col.lower() or 'disease' in col.lower()][0]
                    data = data.rename(columns={target_col: 'target'})
                print("✓ Dataset loaded from alternative source")
            except Exception as e2:
                raise RuntimeError(f"Could not load Heart Disease dataset: {e2}")
        
        self.X_full = data.drop('target', axis=1)
        self.y_full = data['target']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class DiabetesClassificationExample(BinaryClassificationExample):
    """
    Pima Indians Diabetes Dataset
    
    Features: 8 clinical measurements
    - pregnancies, glucose, blood pressure, skin thickness
    - insulin, BMI, diabetes pedigree function, age
    
    Target: Binary (0=no diabetes, 1=diabetes)
    Samples: 768
    
    Expected Performance:
    - Logistic Regression: ~77-78% accuracy
    - Random Forest: ~78-80% accuracy
    - XGBoost: ~79-81% accuracy
    
    Note: Imbalanced dataset with challenging medical prediction task.
    """
    
    def load_data(self, test_size=0.2):
        """Load Pima Indians Diabetes dataset."""
        print("Loading Pima Indians Diabetes dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            
            column_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                          'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
            
            data = pd.read_csv(url, names=column_names)
            print("✓ Dataset loaded successfully")
            
        except Exception as e:
            print(f"Primary source failed: {e}")
            # Create synthetic fallback
            print("⚠ Using synthetic fallback data")
            np.random.seed(self.random_state)
            n_samples = 768
            data = pd.DataFrame({
                'pregnancies': np.random.randint(0, 17, n_samples),
                'glucose': np.random.randint(70, 200, n_samples),
                'blood_pressure': np.random.randint(50, 120, n_samples),
                'skin_thickness': np.random.randint(10, 60, n_samples),
                'insulin': np.random.randint(20, 300, n_samples),
                'bmi': np.random.uniform(15, 50, n_samples),
                'diabetes_pedigree': np.random.uniform(0.1, 2.5, n_samples),
                'age': np.random.randint(21, 81, n_samples),
                'outcome': np.random.binomial(1, 0.35, n_samples)
            })
        
        self.X_full = data.drop('outcome', axis=1)
        self.y_full = data['outcome']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class BankMarketingExample(BinaryClassificationExample):
    """
    Bank Marketing Dataset
    
    Features: 16 numerical features (from 20 total after preprocessing)
    - age, duration, campaign, pdays, previous
    - emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
    - and encoded categorical features
    
    Target: Binary (0=no subscription, 1=subscription to term deposit)
    Samples: ~41,000
    
    Expected Performance:
    - Logistic Regression: ~89-90% accuracy
    - Random Forest: ~91-92% accuracy
    - XGBoost: ~92-93% accuracy
    
    Note: Large dataset with class imbalance (few positive cases).
    """
    
    def load_data(self, test_size=0.2):
        """Load Bank Marketing dataset."""
        print("Loading Bank Marketing dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
            data = pd.read_csv(url, sep=';')
            print("✓ Dataset loaded from UCI repository")
        except:
            try:
                url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv"
                data = pd.read_csv(url)
                print("✓ Dataset loaded from alternative source")
            except Exception as e:
                raise RuntimeError(f"Could not load Bank Marketing dataset: {e}")
        
        # Convert target to binary
        if 'y' in data.columns:
            data['target'] = (data['y'] == 'yes').astype(int)
            data = data.drop('y', axis=1)
        
        # Select only numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        self.X_full = data[numeric_cols]
        self.y_full = data['target']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class CreditCardFraudExample(BinaryClassificationExample):
    """
    Credit Card Fraud Detection Dataset
    
    Features: 30 numerical features
    - V1-V28: PCA transformed features (anonymized)
    - Time: seconds elapsed between transactions
    - Amount: transaction amount
    
    Target: Binary (0=legitimate, 1=fraud)
    Samples: 284,807 (highly imbalanced: ~0.17% fraud)
    
    Expected Performance:
    - Logistic Regression: ~99% accuracy (but check precision/recall)
    - Random Forest: ~99.9% accuracy
    - XGBoost: ~99.9% accuracy
    
    Note: Highly imbalanced dataset - accuracy alone is misleading!
    """
    
    def load_data(self, test_size=0.2):
        """Load Credit Card Fraud dataset."""
        print("Loading Credit Card Fraud Detection dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
            data = pd.read_csv(url)
            print("✓ Dataset loaded successfully")
        except:
            print("⚠ Could not load full dataset. Using sampled version...")
            # Create synthetic representation
            np.random.seed(self.random_state)
            n_samples = 10000
            n_fraud = int(n_samples * 0.002)  # 0.2% fraud
            
            data = pd.DataFrame({
                f'V{i}': np.random.randn(n_samples) for i in range(1, 29)
            })
            data['Time'] = np.arange(n_samples)
            data['Amount'] = np.random.gamma(2, 20, n_samples)
            data['Class'] = np.concatenate([np.ones(n_fraud), np.zeros(n_samples - n_fraud)])
            np.random.shuffle(data['Class'].values)
            print("✓ Using synthetic representative data")
        
        self.X_full = data.drop('Class', axis=1)
        self.y_full = data['Class']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns[:5])}... (showing first 5 of {len(self.X_full.columns)})")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class SpamDetectionExample(BinaryClassificationExample):
    """
    Spambase Dataset
    
    Features: 57 continuous features
    - 48 word frequency features (percentage of words in email)
    - 6 character frequency features
    - 3 capital letter features (average, longest, total)
    
    Target: Binary (0=not spam, 1=spam)
    Samples: 4,601
    
    Expected Performance:
    - Logistic Regression: ~92-93% accuracy
    - Random Forest: ~94-95% accuracy
    - XGBoost: ~95-96% accuracy
    """
    
    def load_data(self, test_size=0.2):
        """Load Spambase dataset."""
        print("Loading Spambase dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            data = pd.read_csv(url, header=None)
            
            # Create column names
            word_cols = [f'word_freq_{i}' for i in range(48)]
            char_cols = [f'char_freq_{i}' for i in range(6)]
            capital_cols = ['capital_run_avg', 'capital_run_longest', 'capital_run_total']
            data.columns = word_cols + char_cols + capital_cols + ['is_spam']
            
            print("✓ Dataset loaded from UCI repository")
        except Exception as e:
            raise RuntimeError(f"Could not load Spambase dataset: {e}")
        
        self.X_full = data.drop('is_spam', axis=1)
        self.y_full = data['is_spam']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns[:5])}... (showing first 5 of {len(self.X_full.columns)})")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class IonosphereExample(BinaryClassificationExample):
    """
    Ionosphere Dataset
    
    Features: 34 continuous features
    - Radar returns from ionosphere
    - Complex-valued data converted to real and imaginary parts
    
    Target: Binary (0=bad, 1=good)
    Samples: 351
    
    Expected Performance:
    - Logistic Regression: ~87-89% accuracy
    - Random Forest: ~92-94% accuracy
    - XGBoost: ~93-95% accuracy
    
    Note: High-dimensional relative to sample size. Classic ML benchmark.
    """
    
    def load_data(self, test_size=0.2):
        """Load Ionosphere dataset."""
        print("Loading Ionosphere dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
            data = pd.read_csv(url, header=None)
            
            # Last column is target ('g' or 'b')
            feature_cols = [f'feature_{i}' for i in range(len(data.columns) - 1)]
            data.columns = feature_cols + ['target']
            
            # Convert target to binary (b=0, g=1)
            data['target'] = (data['target'] == 'g').astype(int)
            
            print("✓ Dataset loaded from UCI repository")
        except Exception as e:
            raise RuntimeError(f"Could not load Ionosphere dataset: {e}")
        
        self.X_full = data.drop('target', axis=1)
        self.y_full = data['target']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns[:5])}... (showing first 5 of {len(self.X_full.columns)})")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class AdultIncomeExample(BinaryClassificationExample):
    """
    Adult Income Dataset (Census Income)
    
    Features: 6 continuous numerical features (from 14 total)
    - age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
    
    Target: Binary (0=<=50K, 1=>50K annual income)
    Samples: 32,561 (training) + 16,281 (test) = 48,842
    
    Expected Performance:
    - Logistic Regression: ~83-84% accuracy
    - Random Forest: ~85-86% accuracy
    - XGBoost: ~87-88% accuracy
    
    Note: Large dataset with demographic and employment data.
    """
    
    def load_data(self, test_size=0.2):
        """Load Adult Income dataset."""
        print("Loading Adult Income dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
            
            column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'sex',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
            
            data_train = pd.read_csv(url_train, names=column_names, skipinitialspace=True, na_values='?')
            data_test = pd.read_csv(url_test, names=column_names, skipinitialspace=True, skiprows=1, na_values='?')
            
            data = pd.concat([data_train, data_test], ignore_index=True)
            
            # Remove rows with missing values
            data = data.dropna()
            
            # Select only numeric features
            numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            
            # Convert target to binary
            data['target'] = data['income'].str.contains('>50K').astype(int)
            
            self.X_full = data[numeric_cols]
            self.y_full = data['target']
            
            print("✓ Dataset loaded from UCI repository")
        except Exception as e:
            raise RuntimeError(f"Could not load Adult Income dataset: {e}")
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state, stratify=self.y_full
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full
