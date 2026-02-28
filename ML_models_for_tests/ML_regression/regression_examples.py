"""
Well-known regression problems with continuous features.

This module demonstrates data loading, preprocessing, training, and evaluation
for classic regression datasets with continuous features only.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
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
    from sklearn.neural_network import MLPRegressor
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False

# For Wine Quality dataset - we'll load from URL if needed


class RegressionExample:
    """Base class for regression examples."""
    
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
        """Display comprehensive dataset summary including unique values."""
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
        
        # Feature statistics
        print(f"\n{'Feature':<15} {'Type':<10} {'Unique':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 80)
        
        for col in self.X_full.columns:
            n_unique = self.X_full[col].nunique()
            min_val = self.X_full[col].min()
            max_val = self.X_full[col].max()
            mean_val = self.X_full[col].mean()
            std_val = self.X_full[col].std()
            dtype = str(self.X_full[col].dtype)
            
            print(f"{col:<15} {dtype:<10} {n_unique:<10} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} {std_val:<12.4f}")
        
        # Target statistics
        print(f"\n{'TARGET':<15} {'-':<10} {self.y_full.nunique():<10} {self.y_full.min():<12.4f} {self.y_full.max():<12.4f} {self.y_full.mean():<12.4f} {self.y_full.std():<12.4f}")
        
        # Missing values
        print("\nMissing Values:")
        missing = self.X_full.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found ✓")
        else:
            for col in missing[missing > 0].index:
                print(f"  {col}: {missing[col]} ({missing[col]/len(self.X_full)*100:.2f}%)")
        
        # Data types
        print("\nData Types:")
        print(f"  All features: {self.X_full.dtypes.unique()}")
        
        print("="*80)
        
    def preprocess(self, scale=True):
        """Preprocess the data."""
        if scale:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
    def train(self, model_type='ridge'):
        """Train a regression model."""
        models = {
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=1.0, random_state=self.random_state),
            'elastic': ElasticNet(alpha=1.0, random_state=self.random_state),
            'rf': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE and model_type == 'xgb':
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                verbosity=0
            )
        
        # Add Neural Network if available
        if NEURAL_NET_AVAILABLE and model_type == 'nn':
            models['nn'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        self.model = models.get(model_type, models['ridge'])
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        """Evaluate the model and return metrics."""
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        self.results = {
            'train': {
                'mse': mean_squared_error(self.y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'mae': mean_absolute_error(self.y_train, y_pred_train),
                'r2': r2_score(self.y_train, y_pred_train),
                'mape': mean_absolute_percentage_error(self.y_train, y_pred_train)
            },
            'test': {
                'mse': mean_squared_error(self.y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'mae': mean_absolute_error(self.y_test, y_pred_test),
                'r2': r2_score(self.y_test, y_pred_test),
                'mape': mean_absolute_percentage_error(self.y_test, y_pred_test)
            }
        }
        
        return self.results
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"\nTraining Set Performance:")
        print(f"  R² Score:  {self.results['train']['r2']:.4f}")
        print(f"  RMSE:      {self.results['train']['rmse']:.4f}")
        print(f"  MAE:       {self.results['train']['mae']:.4f}")
        print(f"  MAPE:      {self.results['train']['mape']:.4f}")
        
        print(f"\nTest Set Performance:")
        print(f"  R² Score:  {self.results['test']['r2']:.4f}")
        print(f"  RMSE:      {self.results['test']['rmse']:.4f}")
        print(f"  MAE:       {self.results['test']['mae']:.4f}")
        print(f"  MAPE:      {self.results['test']['mape']:.4f}")
        print(f"{'='*60}\n")
        

class CaliforniaHousingExample(RegressionExample):
    """
    California Housing Dataset
    
    Features: 8 continuous features
    - MedInc: median income in block group
    - HouseAge: median house age in block group
    - AveRooms: average number of rooms per household
    - AveBedrms: average number of bedrooms per household
    - Population: block group population
    - AveOccup: average number of household members
    - Latitude: block group latitude
    - Longitude: block group longitude
    
    Target: Median house value for California districts (in $100,000s)
    Samples: 20,640
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.60
    - Random Forest: R² ~ 0.81
    - Gradient Boosting: R² ~ 0.83-0.85
    """
    
    def load_data(self, test_size=0.2):
        """Load California Housing dataset."""
        print("Loading California Housing dataset...")
        data = fetch_california_housing(as_frame=True)
        
        self.X_full = data.data
        self.y_full = data.target
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class DiabetesExample(RegressionExample):
    """
    Diabetes Dataset
    
    Features: 10 continuous features (already standardized)
    - age: age in years (standardized)
    - sex: biological sex (standardized)
    - bmi: body mass index (standardized)
    - bp: average blood pressure (standardized)
    - s1-s6: six blood serum measurements (standardized)
    
    Target: Quantitative measure of disease progression one year after baseline
    Samples: 442
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.45-0.50
    - Random Forest: R² ~ 0.40-0.45
    - Gradient Boosting: R² ~ 0.45-0.50
    
    Note: This is a harder problem with fewer samples and more noise.
    """
    
    def load_data(self, test_size=0.2):
        """Load Diabetes dataset."""
        print("Loading Diabetes dataset...")
        data = load_diabetes(as_frame=True)
        
        self.X_full = data.data
        self.y_full = data.target
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class WineQualityExample(RegressionExample):
    """
    Wine Quality Dataset (Red Wine)
    
    Features: 11 continuous physicochemical features
    - fixed acidity
    - volatile acidity
    - citric acid
    - residual sugar
    - chlorides
    - free sulfur dioxide
    - total sulfur dioxide
    - density
    - pH
    - sulphates
    - alcohol
    
    Target: Wine quality score (0-10, though typically 3-8)
    Samples: ~1,599 (red wine)
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.35-0.40
    - Random Forest: R² ~ 0.45-0.55
    - Gradient Boosting: R² ~ 0.50-0.60
    
    Note: Quality is subjective and based on sensory data, making this challenging.
    """
    
    def load_data(self, test_size=0.2):
        """Load Wine Quality dataset from UCI repository."""
        print("Loading Wine Quality dataset...")
        
        # Try multiple methods to load the dataset
        data = None
        
        # Method 1: Try with SSL context disabled
        try:
            import ssl
            import urllib.request
            
            ssl._create_default_https_context = ssl._create_unverified_context
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            data = pd.read_csv(url, sep=';')
            print("✓ Dataset loaded successfully from UCI repository")
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Try alternative URL
            try:
                url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
                data = pd.read_csv(url, sep=';')
                print("✓ Dataset loaded successfully from alternative URL")
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
                # Method 3: Try GitHub mirror
                try:
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/WineQuality-Red.csv"
                    data = pd.read_csv(url)
                    # Rename columns to match original format (remove spaces)
                    data.columns = [col.replace(' ', ' ') for col in data.columns]
                    print("✓ Dataset loaded successfully from GitHub mirror")
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    
                    # Method 4: Try another reliable source
                    try:
                        url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
                        data = pd.read_csv(url, sep=';')
                        # Ensure quality column exists
                        if 'quality' not in data.columns:
                            # Try renaming if column name is different
                            possible_names = ['Quality', 'rating', 'score']
                            for name in possible_names:
                                if name in data.columns:
                                    data = data.rename(columns={name: 'quality'})
                                    break
                        print("✓ Dataset loaded successfully from alternative source")
                    except Exception as e4:
                        print(f"Method 4 failed: {e4}")
                        print("❌ Could not load Wine Quality dataset from any source")
                        print("    The UCI repository may be temporarily down.")
                        raise RuntimeError("Could not load Wine Quality dataset")
        
        # Separate features and target
        self.X_full = data.drop('quality', axis=1)
        self.y_full = data['quality']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class AmesHousingExample(RegressionExample):
    """
    Ames Housing Dataset
    
    Modern replacement for the deprecated Boston Housing dataset.
    Contains comprehensive house price data from Ames, Iowa (2006-2010).
    
    Features: 37 numeric features (from 79 total features)
    - Property characteristics (lot size, year built, quality ratings, etc.)
    - Room counts and square footages
    - Garage and basement details
    - Location and neighborhood information
    
    Target: Sale price in dollars ($34,900 - $755,000)
    Samples: 1,460
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.75-0.80
    - Random Forest: R² ~ 0.85-0.88
    - Gradient Boosting/XGBoost: R² ~ 0.88-0.91
    
    Note: Much better documented and ethically sound compared to Boston Housing.
    """
    
    def load_data(self, test_size=0.2):
        """Load Ames Housing dataset from OpenML."""
        print("Loading Ames Housing dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        from sklearn.datasets import fetch_openml
        
        # Load from OpenML
        data = fetch_openml(name='house_prices', version=1, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # Convert target to numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Select only numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        # Remove rows with missing target
        mask = ~y.isna()
        X_numeric = X_numeric[mask]
        y = y[mask]
        
        # Handle missing values in features (fill with median)
        for col in X_numeric.columns:
            if X_numeric[col].isnull().any():
                X_numeric[col].fillna(X_numeric[col].median(), inplace=True)
        
        self.X_full = X_numeric
        self.y_full = y
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns[:10])}... (showing first 10 of {len(self.X_full.columns)})")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class AutoMPGExample(RegressionExample):
    """
    Auto MPG Dataset
    
    Classic dataset for predicting vehicle fuel efficiency.
    Data from 1970s and early 1980s cars.
    
    Features: 4 continuous features
    - displacement: Engine displacement in cubic inches
    - horsepower: Engine horsepower
    - weight: Vehicle weight in pounds
    - acceleration: Time to accelerate from 0 to 60 mph (seconds)
    
    Target: Miles per gallon (MPG) - fuel efficiency
    Samples: 392 (after removing missing values)
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.80-0.82
    - Random Forest: R² ~ 0.85-0.88
    - Gradient Boosting: R² ~ 0.87-0.90
    
    Note: Simple, interpretable dataset. Great for teaching and demonstrations.
    """
    
    def load_data(self, test_size=0.2):
        """Load Auto MPG dataset from OpenML."""
        print("Loading Auto MPG dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        from sklearn.datasets import fetch_openml
        
        # Load from OpenML
        data = fetch_openml(name='autompg', version=1, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # Convert target to numeric
        y = pd.to_numeric(y, errors='coerce')
        
        # Select only numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        # Remove rows with any missing values
        mask = ~(X_numeric.isna().any(axis=1) | y.isna())
        X_numeric = X_numeric[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        self.X_full = X_numeric
        self.y_full = y
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class ConcreteStrengthExample(RegressionExample):
    """
    Concrete Compressive Strength Dataset
    
    Predict concrete strength from mixture components.
    Real-world civil engineering dataset.
    
    Features: 8 continuous features
    - Cement: kg in m³ mixture
    - Blast Furnace Slag: kg in m³ mixture
    - Fly Ash: kg in m³ mixture
    - Water: kg in m³ mixture
    - Superplasticizer: kg in m³ mixture
    - Coarse Aggregate: kg in m³ mixture
    - Fine Aggregate: kg in m³ mixture
    - Age: days (1-365)
    
    Target: Compressive strength in MPa (2.33 - 82.60)
    Samples: 1,030
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.60-0.65
    - Random Forest: R² ~ 0.88-0.91
    - Gradient Boosting/XGBoost: R² ~ 0.90-0.93
    
    Note: Highly non-linear relationships. Complex chemistry interactions.
    """
    
    def load_data(self, test_size=0.2):
        """Load Concrete Strength dataset."""
        print("Loading Concrete Compressive Strength dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Try loading from GitHub (more reliable)
        try:
            url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"
            data = pd.read_csv(url)
            print("✓ Dataset loaded from GitHub mirror")
        except:
            # Fallback to UCI repository
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
                data = pd.read_excel(url)
                print("✓ Dataset loaded from UCI repository")
            except Exception as e:
                raise RuntimeError(f"Could not load Concrete Strength dataset: {e}")
        
        # Last column is target
        self.X_full = data.iloc[:, :-1]
        self.y_full = data.iloc[:, -1]
        
        # Simplify column names
        self.X_full.columns = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 
                                'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class EnergyEfficiencyExample(RegressionExample):
    """
    Energy Efficiency Dataset
    
    Predict heating/cooling load of buildings from design parameters.
    Important for sustainable building design.
    
    Features: 9 numeric features
    - Relative Compactness
    - Surface Area (m²)
    - Wall Area (m²)
    - Roof Area (m²)
    - Overall Height (m)
    - Orientation (2-5)
    - Glazing Area (0-0.4)
    - Glazing Area Distribution (0-5)
    
    Target: Heating Load (we use the first target, heating load)
    Samples: 768
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.88-0.90
    - Random Forest: R² ~ 0.96-0.98
    - Gradient Boosting/XGBoost: R² ~ 0.98-0.99
    
    Note: Very high accuracy achievable. Strong physics-based relationships.
    """
    
    def load_data(self, test_size=0.2):
        """Load Energy Efficiency dataset."""
        print("Loading Energy Efficiency dataset...")
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        from sklearn.datasets import fetch_openml
        
        # Try OpenML first
        try:
            data = fetch_openml(name='energy-efficiency', version=1, as_frame=True, parser='auto')
            X = data.data
            y = data.target
            
            # Handle target (could be DataFrame with 2 targets)
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]  # Use heating load (first target)
            
            # Convert to numeric and select numeric features
            y = pd.to_numeric(y, errors='coerce')
            numeric_features = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_features]
            
            print("✓ Dataset loaded from OpenML")
        except:
            # Fallback to UCI repository  
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
                data = pd.read_excel(url)
                X = data.iloc[:, :-2]  # All features
                y = data.iloc[:, -2]   # Heating Load (first target)
                y = pd.to_numeric(y, errors='coerce')
                print("✓ Dataset loaded from UCI repository")
            except Exception as e:
                raise RuntimeError(f"Could not load Energy Efficiency dataset: {e}")
        
        self.X_full = X
        self.y_full = y
        
        # Rename columns if needed
        if len(self.X_full.columns) == 9 and self.X_full.columns[0] in ['V1', 'X1', 0]:
            self.X_full.columns = ['RelativeCompactness', 'SurfaceArea', 'WallArea', 'RoofArea',
                                   'OverallHeight', 'Orientation', 'GlazingArea', 
                                   'GlazingAreaDistribution', 'Extra']
            self.X_full = self.X_full.drop('Extra', axis=1, errors='ignore')
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


class BikeSharingExample(RegressionExample):
    """
    Bike Sharing Dataset (Hourly)
    
    Predict hourly bike rental demand from weather and temporal features.
    Modern dataset from Capital Bikeshare system (Washington D.C., 2011-2012).
    
    Features: 12 numeric features
    - season: 1-4 (winter, spring, summer, fall)
    - yr: Year (0: 2011, 1: 2012)
    - mnth: Month (1-12)
    - hr: Hour (0-23)
    - holiday: Binary (0/1)
    - weekday: Day of week (0-6)
    - workingday: Binary (0/1)
    - weathersit: Weather situation (1-4)
    - temp: Normalized temperature
    - atemp: Normalized feeling temperature
    - hum: Normalized humidity
    - windspeed: Normalized wind speed
    
    Target: Total bike rental count (hourly)
    Samples: 17,379
    
    Expected Performance:
    - Ridge/Linear: R² ~ 0.40-0.45
    - Random Forest: R² ~ 0.88-0.90
    - Gradient Boosting/XGBoost: R² ~ 0.90-0.93
    
    Note: Large dataset with temporal patterns and weather interactions.
    """
    
    def load_data(self, test_size=0.2):
        """Load Bike Sharing dataset."""
        print("Loading Bike Sharing (Hourly) dataset...")
        
        import ssl
        import io
        import zipfile
        import urllib.request
        
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Try loading from UCI repository
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
            response = urllib.request.urlopen(url)
            zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
            data = pd.read_csv(zip_file.open('hour.csv'))
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback URL
            try:
                url = "https://raw.githubusercontent.com/plotly/datasets/master/bike-sharing-dataset-hour.csv"
                data = pd.read_csv(url)
                print("✓ Dataset loaded from GitHub mirror")
            except Exception as e:
                raise RuntimeError(f"Could not load Bike Sharing dataset: {e}")
        
        # Remove non-predictive columns
        drop_cols = ['instant', 'dteday', 'casual', 'registered']
        X = data.drop(columns=[col for col in drop_cols if col in data.columns] + ['cnt'])
        y = data['cnt']  # Total count is target
        
        # Select only numeric features
        X_numeric = X.select_dtypes(include=[np.number])
        
        self.X_full = X_numeric
        self.y_full = y
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {list(self.X_full.columns)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]} samples")
        print(f"Test set:  {self.X_test.shape[0]} samples")
        
        self.show_dataset_summary()
        
        return self.X_full, self.y_full


def compare_models(example_class, dataset_name):
    """Compare multiple models on a dataset."""
    print(f"\n{'#'*60}")
    print(f"# {dataset_name}")
    print(f"{'#'*60}")
    
    models_to_test = ['ridge', 'lasso', 'elastic', 'rf', 'gbm']
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        models_to_test.append('xgb')
    if NEURAL_NET_AVAILABLE:
        models_to_test.append('nn')
    
    results_summary = []
    
    # Load data once
    example = example_class(random_state=42)
    example.load_data()
    
    for model_type in models_to_test:
        # Create new instance for each model
        model_example = example_class(random_state=42)
        model_example.X_train = example.X_train.copy()
        model_example.X_test = example.X_test.copy()
        model_example.y_train = example.y_train.copy()
        model_example.y_test = example.y_test.copy()
        model_example.X_full = example.X_full.copy()
        model_example.y_full = example.y_full.copy()
        
        model_example.preprocess(scale=(model_type not in ['rf', 'gbm']))
        model_example.train(model_type=model_type)
        model_example.evaluate()
        
        results_summary.append({
            'Model': model_example.model.__class__.__name__,
            'Train R²': model_example.results['train']['r2'],
            'Test R²': model_example.results['test']['r2'],
            'Test RMSE': model_example.results['test']['rmse'],
            'Test MAE': model_example.results['test']['mae'],
            'Overfit': model_example.results['train']['r2'] - model_example.results['test']['r2']
        })
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    df_results = pd.DataFrame(results_summary)
    print(df_results.to_string(index=False))
    print("="*80)
    
    return df_results


def print_model_opinion(model_name, metrics, dataset_name):
    """Print detailed opinion about a model's performance."""
    train_r2 = metrics['Train R²']
    test_r2 = metrics['Test R²']
    overfit = metrics['Overfit']
    test_rmse = metrics['Test RMSE']
    test_mae = metrics['Test MAE']
    
    print(f"\n{'='*80}")
    print(f"MODEL ANALYSIS: {model_name} on {dataset_name}")
    print(f"{'='*80}")
    
    # Performance assessment
    if test_r2 >= 0.80:
        performance = "EXCELLENT"
    elif test_r2 >= 0.70:
        performance = "VERY GOOD"
    elif test_r2 >= 0.60:
        performance = "GOOD"
    elif test_r2 >= 0.45:
        performance = "MODERATE"
    else:
        performance = "POOR"
    
    print(f"\nPerformance Rating: {performance}")
    print(f"  • Test R²: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
    print(f"  • Test RMSE: {test_rmse:.4f}")
    print(f"  • Test MAE: {test_mae:.4f}")
    
    # Overfitting assessment
    print(f"\nOverfitting Analysis:")
    if overfit < 0.05:
        print(f"  ✓ Excellent generalization (gap: {overfit:.4f})")
        print(f"    The model generalizes very well to unseen data.")
    elif overfit < 0.10:
        print(f"  ✓ Good generalization (gap: {overfit:.4f})")
        print(f"    Minor overfitting, but still acceptable performance.")
    elif overfit < 0.20:
        print(f"  ⚠ Moderate overfitting (gap: {overfit:.4f})")
        print(f"    Consider regularization or reducing model complexity.")
    else:
        print(f"  ✗ Significant overfitting (gap: {overfit:.4f})")
        print(f"    Model memorizes training data. Needs regularization.")
    
    # Model-specific opinion
    print(f"\nExpert Opinion:")
    if model_name == "Ridge":
        print(f"  Ridge Regression uses L2 regularization, making it stable and interpretable.")
        print(f"  • Pros: Fast training, interpretable coefficients, handles multicollinearity")
        print(f"  • Cons: Linear assumption may be too restrictive for complex patterns")
        if test_r2 >= 0.55:
            print(f"  • Verdict: RECOMMENDED - Good baseline with strong linear relationships")
        else:
            print(f"  • Verdict: Consider non-linear models for better performance")
    
    elif model_name == "Lasso":
        print(f"  Lasso uses L1 regularization, performing automatic feature selection.")
        print(f"  • Pros: Feature selection, sparse solutions, interpretable")
        print(f"  • Cons: Can be unstable with correlated features")
        if test_r2 >= 0.50:
            print(f"  • Verdict: Good for identifying important features")
        else:
            print(f"  • Verdict: Poor performance - features may all be relevant")
    
    elif model_name == "ElasticNet":
        print(f"  ElasticNet combines L1 and L2 regularization.")
        print(f"  • Pros: Balanced approach, handles correlated features better than Lasso")
        print(f"  • Cons: More hyperparameters to tune")
        if test_r2 >= 0.50:
            print(f"  • Verdict: Good middle-ground between Ridge and Lasso")
        else:
            print(f"  • Verdict: Needs hyperparameter tuning for better results")
    
    elif model_name == "RandomForestRegressor":
        print(f"  Random Forest is an ensemble of decision trees.")
        print(f"  • Pros: Handles non-linearity, robust to outliers, feature importance")
        print(f"  • Cons: Can overfit with default settings, less interpretable")
        if overfit > 0.15:
            print(f"  • Verdict: OVERFITTING - Reduce max_depth or increase min_samples_leaf")
        elif test_r2 >= 0.75:
            print(f"  • Verdict: HIGHLY RECOMMENDED - Excellent balance of accuracy and robustness")
        else:
            print(f"  • Verdict: Decent performance but may need tuning")
    
    elif model_name == "GradientBoostingRegressor":
        print(f"  Gradient Boosting builds trees sequentially to minimize errors.")
        print(f"  • Pros: Often best accuracy, handles complex patterns")
        print(f"  • Cons: Slower training, more prone to overfitting, requires tuning")
        if test_r2 >= 0.75:
            print(f"  • Verdict: BEST PERFORMANCE - Recommended for production use")
        elif overfit > 0.10:
            print(f"  • Verdict: Good accuracy but tune learning_rate and n_estimators")
        else:
            print(f"  • Verdict: Solid choice with proper hyperparameter tuning")
    
    elif model_name == "XGBRegressor":
        print(f"  XGBoost is an optimized gradient boosting implementation.")
        print(f"  • Pros: State-of-the-art accuracy, fast training, handles missing values")
        print(f"  • Cons: Many hyperparameters, can overfit, needs GPU for huge datasets")
        if test_r2 >= 0.80:
            print(f"  • Verdict: OUTSTANDING - XGBoost achieves top-tier performance")
        elif test_r2 >= 0.75:
            print(f"  • Verdict: EXCELLENT - Among best models, consider for production")
        elif overfit > 0.15:
            print(f"  • Verdict: Good but reduce max_depth or increase reg_alpha/reg_lambda")
        else:
            print(f"  • Verdict: Strong performer with proper tuning")
    
    elif model_name == "MLPRegressor":
        print(f"  Neural Network (Multi-Layer Perceptron) with backpropagation.")
        print(f"  • Pros: Universal approximator, learns complex patterns")
        print(f"  • Cons: Needs scaling, slow training, many hyperparameters, black box")
        if test_r2 >= 0.80:
            print(f"  • Verdict: EXCELLENT - Neural network captures complex patterns well")
        elif test_r2 >= 0.70:
            print(f"  • Verdict: GOOD - Consider if you need complex non-linear modeling")
        elif overfit > 0.20:
            print(f"  • Verdict: OVERFITTING - Reduce layers, add regularization, or more data")
        else:
            print(f"  • Verdict: May need more training or architecture tuning")
    
    print(f"{'='*80}\n")


def select_best_model_and_params(dataset_name):
    """Select best model and parameters for each dataset."""
    print(f"\n{'#'*80}")
    print(f"# BEST MODEL SELECTION: {dataset_name}")
    print(f"{'#'*80}\n")
    
    # Try advanced models first for comparison
    if XGBOOST_AVAILABLE and "California" in dataset_name:
        print("🚀 TESTING ADVANCED MODEL: XGBoost")
        print("\nOptimized Hyperparameters:")
        print("  • n_estimators: 200")
        print("  • learning_rate: 0.05 (slower learning for better generalization)")
        print("  • max_depth: 7")
        print("  • min_child_weight: 3")
        print("  • subsample: 0.8")
        print("  • colsample_bytree: 0.8")
        print("  • gamma: 0.1 (regularization)")
        print("  • reg_alpha: 0.1 (L1 regularization)")
        print("  • reg_lambda: 1.0 (L2 regularization)")
        
        example = CaliforniaHousingExample(random_state=42)
        example.load_data()
        example.preprocess(scale=False)  # XGBoost doesn't need scaling
        
        optimized_xgb = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        
        example.model = optimized_xgb
        example.model.fit(example.X_train, example.y_train)
        example.evaluate()
        
        print("\nXGBoost Performance:")
        example.print_results()
        
        print("XGBoost Verdict:")
        if example.results['test']['r2'] > 0.82:
            print("  ✅ XGBoost WINS - Highest accuracy achieved!")
            print("  Recommended for production when maximum accuracy is needed.")
            return optimized_xgb, example
        else:
            print("  ⚠ XGBoost good but Random Forest may be more stable")
            print("  Continuing with Random Forest recommendation...\n")
    
    if "Wine" in dataset_name:
        print("SELECTED MODEL: Gradient Boosting Regressor")
        print("\nOptimal Hyperparameters:")
        print("  • n_estimators: 150 (sufficient for convergence)")
        print("  • learning_rate: 0.1 (standard for good generalization)")
        print("  • max_depth: 4 (prevent overfitting)")
        print("  • min_samples_split: 20 (avoid noise fitting)")
        print("  • min_samples_leaf: 10 (smoother predictions)")
        print("  • subsample: 0.8 (stochastic boosting for robustness)")
        print("  • random_state: 42 (reproducibility)")
        
        print("\nRationale:")
        print("  1. Medium-sized dataset (1,599 samples) suits ensemble methods")
        print("  2. Non-linear relationships between chemistry and quality")
        print("  3. Gradient Boosting handles subtle patterns well")
        print("  4. Quality is ordinal but treated as continuous")
        print("  5. Regularization prevents overfitting to subjective ratings")
        
        print("\nExpected Performance:")
        print("  • Test R²: 0.50-0.55")
        print("  • Test RMSE: 0.55-0.65")
        print("  • Training time: ~2-3 seconds")
        
        # Train the optimized model
        example = WineQualityExample(random_state=42)
        example.load_data()
        example.preprocess(scale=False)  # GBM doesn't need scaling
        
        from sklearn.ensemble import GradientBoostingRegressor
        optimized_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        example.model = optimized_model
        example.model.fit(example.X_train, example.y_train)
        example.evaluate()
        example.print_results()
        
        return optimized_model, example
        
    elif "California" in dataset_name:
        print("SELECTED MODEL: Random Forest Regressor")
        print("\nOptimal Hyperparameters:")
        print("  • n_estimators: 200 (more trees for stability)")
        print("  • max_depth: 15 (prevent overfitting)")
        print("  • min_samples_split: 10 (avoid overfitting to noise)")
        print("  • min_samples_leaf: 4 (smoother predictions)")
        print("  • max_features: 'sqrt' (reduce correlation between trees)")
        print("  • random_state: 42 (reproducibility)")
        print("  • n_jobs: -1 (use all CPU cores)")
        
        print("\nRationale:")
        print("  1. Achieves R² > 0.80 with good generalization")
        print("  2. Handles non-linear relationships between features")
        print("  3. Robust to outliers in house prices")
        print("  4. Provides feature importance for interpretability")
        print("  5. Fast prediction time for deployment")
        
        print("\nExpected Performance:")
        print("  • Test R²: 0.81-0.83")
        print("  • Test RMSE: 0.48-0.52")
        print("  • Training time: ~5-10 seconds")
        
        # Train the optimized model
        example = CaliforniaHousingExample(random_state=42)
        example.load_data()
        
        from sklearn.ensemble import RandomForestRegressor
        optimized_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        example.model = optimized_model
        example.model.fit(example.X_train, example.y_train)
        example.evaluate()
        example.print_results()
        
        return optimized_model, example
        
    elif "Diabetes" in dataset_name:
        print("SELECTED MODEL: Ridge Regression")
        print("\nOptimal Hyperparameters:")
        print("  • alpha: 0.5 (moderate regularization)")
        print("  • solver: 'auto' (let sklearn choose best)")
        print("  • random_state: 42 (reproducibility)")
        
        print("\nRationale:")
        print("  1. Small dataset (442 samples) - simpler models generalize better")
        print("  2. Features are already standardized")
        print("  3. Linear relationships are sufficient")
        print("  4. Fast training and prediction")
        print("  5. Interpretable coefficients for medical analysis")
        
        print("\nExpected Performance:")
        print("  • Test R²: 0.46-0.50")
        print("  • Test RMSE: 52-54")
        print("  • Training time: <1 second")
        
        # Train the optimized model
        example = DiabetesExample(random_state=42)
        example.load_data()
        example.preprocess(scale=False)  # Already scaled
        
        from sklearn.linear_model import Ridge
        optimized_model = Ridge(alpha=0.5, random_state=42)
        
        example.model = optimized_model
        example.model.fit(example.X_train, example.y_train)
        example.evaluate()
        example.print_results()
        
        return optimized_model, example


def main():
    """Run examples for all datasets - Original 3 + New 5."""
    
    print("\n" + "#"*80)
    print("# COMPREHENSIVE REGRESSION ANALYSIS - 8 DATASETS")
    print("# Comparing Models and Selecting Best Solutions")
    print("# Original: California Housing, Diabetes, Wine Quality")
    print("# New: Ames Housing, Auto MPG, Concrete Strength, Energy Efficiency, Bike Sharing")
    print("#"*80)
    
    # Compare all models
    print("\n" + "#"*80)
    print("# PART 1: MODEL COMPARISON - ALL 8 DATASETS")
    print("#"*80)
    
    all_results = {}
    
    # Original 3 datasets
    print("\n\n=== ORIGINAL DATASETS ===\n")
    
    print("1. California Housing Dataset:")
    try:
        all_results['California Housing'] = compare_models(CaliforniaHousingExample, "California Housing")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['California Housing'] = None
    
    print("\n\n2. Diabetes Dataset:")
    try:
        all_results['Diabetes'] = compare_models(DiabetesExample, "Diabetes")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Diabetes'] = None
    
    print("\n\n3. Wine Quality Dataset:")
    try:
        all_results['Wine Quality'] = compare_models(WineQualityExample, "Wine Quality")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Wine Quality'] = None
    
    # New 5 datasets
    print("\n\n=== NEW DATASETS ===\n")
    
    print("4. Ames Housing Dataset:")
    try:
        all_results['Ames Housing'] = compare_models(AmesHousingExample, "Ames Housing")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Ames Housing'] = None
    
    print("\n\n5. Auto MPG Dataset:")
    try:
        all_results['Auto MPG'] = compare_models(AutoMPGExample, "Auto MPG")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Auto MPG'] = None
    
    print("\n\n6. Concrete Strength Dataset:")
    try:
        all_results['Concrete Strength'] = compare_models(ConcreteStrengthExample, "Concrete Strength")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Concrete Strength'] = None
    
    print("\n\n7. Energy Efficiency Dataset:")
    try:
        all_results['Energy Efficiency'] = compare_models(EnergyEfficiencyExample, "Energy Efficiency")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Energy Efficiency'] = None
    
    print("\n\n8. Bike Sharing Dataset:")
    try:
        all_results['Bike Sharing'] = compare_models(BikeSharingExample, "Bike Sharing")
    except Exception as e:
        print(f"⚠ Failed: {e}")
        all_results['Bike Sharing'] = None
    
    # Summary table of all datasets
    print("\n" + "#"*80)
    print("# PART 2: SUMMARY TABLE - ALL DATASETS")
    print("#"*80)
    
    print("\nBest Model Performance for Each Dataset:")
    print(f"\n{'Dataset':<25} {'Best Model':<25} {'Test R²':<12} {'Samples':<10}")
    print("=" * 75)
    
    dataset_info = {
        'California Housing': 20640,
        'Diabetes': 442,
        'Wine Quality': 1599,
        'Ames Housing': 1460,
        'Auto MPG': 392,
        'Concrete Strength': 1030,
        'Energy Efficiency': 768,
        'Bike Sharing': 17379
    }
    
    for dataset_name, samples in dataset_info.items():
        if dataset_name in all_results and all_results[dataset_name] is not None:
            df = all_results[dataset_name]
            best_idx = df['Test R²'].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_r2 = df.loc[best_idx, 'Test R²']
            print(f"{dataset_name:<25} {best_model:<25} {best_r2:<12.4f} {samples:<10}")
        else:
            print(f"{dataset_name:<25} {'FAILED TO LOAD':<25} {'N/A':<12} {samples:<10}")
    
    print("=" * 75)
    
    # Detailed insights
    print("\n" + "#"*80)
    print("# PART 3: KEY INSIGHTS AND RECOMMENDATIONS")
    print("#"*80)
    
    # Dataset insights
    print("\n" + "="*80)
    print("DATASET INSIGHTS BY CATEGORY")
    print("="*80)
    
    print("\n📊 LARGE DATASETS (>10,000 samples):")
    print("   • California Housing (20,640) - Housing prices, high accuracy achievable")
    print("   • Bike Sharing (17,379) - Time-series demand prediction, temporal patterns")
    print("   → Best models: XGBoost, Random Forest, Gradient Boosting")
    print("   → Expected R²: 0.80-0.93")
    
    print("\n📊 MEDIUM DATASETS (1,000-10,000 samples):")
    print("   • Wine Quality (1,599) - Subjective ratings, moderate accuracy")
    print("   • Ames Housing (1,460) - Detailed house features, high accuracy")
    print("   • Concrete Strength (1,030) - Engineering data, highly non-linear")
    print("   • Energy Efficiency (768) - Physics-based, very high accuracy")
    print("   → Best models: Random Forest, XGBoost, Gradient Boosting")
    print("   → Expected R²: 0.50-0.98 (varies by problem difficulty)")
    
    print("\n📊 SMALL DATASETS (<1,000 samples):")
    print("   • Diabetes (442) - Medical data, noisy relationships")
    print("   • Auto MPG (392) - Simple automotive data, clear patterns")
    print("   → Best models: Ridge, Lasso, ElasticNet (avoid overfitting!)")
    print("   → Expected R²: 0.45-0.85")
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE PATTERNS")
    print("="*80)
    
    print("\n🏆 TOP PERFORMERS BY DATASET TYPE:")
    print("\n1. XGBoost:")
    print("   ✅ Best for: Large datasets with complex patterns")
    print("   ✅ Achieves: Highest accuracy (typically +2-5% over RF)")
    print("   ⚠️  Avoid: Small datasets (risk of overfitting)")
    
    print("\n2. Random Forest:")
    print("   ✅ Best for: Medium-large datasets, need interpretability")
    print("   ✅ Achieves: Excellent accuracy with robustness")
    print("   ⚠️  Watch: Can overfit on very small datasets")
    
    print("\n3. Gradient Boosting:")
    print("   ✅ Best for: Medium datasets with non-linear patterns")
    print("   ✅ Achieves: Good accuracy with sequential learning")
    print("   ⚠️  Slower: Training time longer than Random Forest")
    
    print("\n4. Ridge/Lasso Regression:")
    print("   ✅ Best for: Small datasets, need interpretability")
    print("   ✅ Achieves: Good generalization, avoids overfitting")
    print("   ⚠️  Limited: Cannot capture complex non-linear patterns")
    
    if NEURAL_NET_AVAILABLE:
        print("\n5. Neural Networks:")
        print("   ✅ Best for: Very large datasets with complex patterns")
        print("   ✅ Achieves: Competitive accuracy with proper tuning")
        print("   ⚠️  Requires: More data, longer training, careful tuning")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    print("\n🎯 MODEL SELECTION GUIDE:")
    print("\n   DATASET SIZE:")
    print("   • <500 samples    → Ridge or Lasso")
    print("   • 500-5,000       → Random Forest or Gradient Boosting")
    print("   • 5,000-50,000    → XGBoost (if available) or Random Forest")
    print("   • >50,000         → XGBoost or Neural Networks")
    
    print("\n   PROBLEM TYPE:")
    print("   • Linear relationships      → Ridge/Lasso")
    print("   • Non-linear, tabular data  → XGBoost or Random Forest")
    print("   • Physics-based features    → Any model (high accuracy expected)")
    print("   • Subjective targets        → Moderate accuracy expected")
    
    print("\n   PRIORITIES:")
    print("   • Maximum accuracy         → XGBoost")
    print("   • Interpretability         → Ridge, Random Forest")
    print("   • Fast training            → Ridge, Lasso")
    print("   • Production robustness    → Random Forest, XGBoost")
    
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE - 8 DATASETS EVALUATED")
    print("="*80)
    
    success_count = sum(1 for v in all_results.values() if v is not None)
    print(f"\n✅ Successfully analyzed: {success_count}/8 datasets")
    
    if success_count < 8:
        print(f"⚠️  Failed: {8 - success_count} dataset(s)")
        failed = [k for k, v in all_results.items() if v is None]
        print(f"   Failed datasets: {', '.join(failed)}")
    
    print("\n💡 Key Takeaway:")
    print("   Dataset size and problem complexity matter more than algorithm choice!")
    print("   Start simple (Ridge) → Try ensemble (RF) → Optimize with XGBoost")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
