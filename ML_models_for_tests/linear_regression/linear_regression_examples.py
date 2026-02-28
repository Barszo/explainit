"""
Linear Regression Examples - Real-world datasets ideal for linear regression

This module contains examples of datasets that are well-known for having strong linear
relationships and are commonly used to demonstrate linear regression techniques.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionExample:
    """Base class for linear regression examples"""
    
    def __init__(self):
        self.X_full = None
        self.y_full = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self):
        """Load and prepare the dataset - to be implemented by subclasses"""
        raise NotImplementedError
        
    def show_dataset_summary(self):
        """Display comprehensive dataset information"""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print()
        print(f"Dataset Shape: {self.X_full.shape}")
        print(f"Number of samples: {len(self.X_full)}")
        print(f"Number of features: {self.X_full.shape[1]}")
        print()
        
        # Feature statistics
        print(f"{'Feature':<20} {'Type':<10} {'Unique':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-"*80)
        
        for col in self.X_full.columns:
            col_data = self.X_full[col]
            print(f"{col:<20} {str(col_data.dtype):<10} {col_data.nunique():<10} "
                  f"{col_data.min():<12.4f} {col_data.max():<12.4f} "
                  f"{col_data.mean():<12.4f} {col_data.std():<12.4f}")
        
        print()
        print(f"{'TARGET':<20} {'-':<10} {self.y_full.nunique():<10} "
              f"{self.y_full.min():<12.4f} {self.y_full.max():<12.4f} "
              f"{self.y_full.mean():<12.4f} {self.y_full.std():<12.4f}")
        print()
        
        # Missing values
        print("Missing Values:")
        missing = self.X_full.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found ✓")
        else:
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} missing values")
        print()
        
        # Data types
        print("Data Types:")
        print(f"  All features: {self.X_full.dtypes.unique()}")
        print("="*80)


class AdvertisingExample(LinearRegressionExample):
    """
    Advertising Dataset - Sales prediction based on advertising spend
    
    Classic linear regression dataset showing the relationship between advertising
    budgets (TV, Radio, Newspaper) and sales. Well-known for demonstrating strong
    linear relationships, especially with TV advertising.
    
    Features: TV, Radio, Newspaper (advertising budgets in thousands of dollars)
    Target: Sales (in thousands of units)
    
    Source: Introduction to Statistical Learning book
    """
    
    def load_data(self):
        print("Loading Advertising dataset...")
        
        # Load from URL (ISLR book dataset)
        url = "https://www.statlearning.com/s/Advertising.csv"
        
        try:
            df = pd.read_csv(url, index_col=0)
            print("✓ Dataset loaded successfully")
        except:
            # Fallback: create synthetic data with known linear relationships
            print("Creating synthetic advertising data...")
            np.random.seed(42)
            n_samples = 200
            
            TV = np.random.uniform(0, 300, n_samples)
            Radio = np.random.uniform(0, 50, n_samples)
            Newspaper = np.random.uniform(0, 120, n_samples)
            
            # Sales with strong linear relationship to TV, moderate to Radio
            Sales = 7 + 0.05 * TV + 0.1 * Radio + 0.001 * Newspaper + np.random.normal(0, 3, n_samples)
            
            df = pd.DataFrame({
                'TV': TV,
                'Radio': Radio,
                'Newspaper': Newspaper,
                'Sales': Sales
            })
            print("✓ Synthetic dataset created")
        
        # Prepare features and target
        self.X_full = df[['TV', 'Radio', 'Newspaper']]
        self.y_full = df['Sales']
        self.feature_names = ['TV', 'Radio', 'Newspaper']
        self.target_name = 'Sales'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class BostonHousingExample(LinearRegressionExample):
    """
    Boston Housing Dataset - House price prediction
    
    One of the most famous datasets for regression, showing relationship between
    various features of houses/neighborhoods and their median values. Classic example
    of linear regression in real estate.
    
    Features: 13 features including crime rate, room count, distance to employment, etc.
    Target: Median value of owner-occupied homes (in $1000s)
    
    Source: Harrison & Rubinfeld (1978)
    """
    
    def load_data(self):
        print("Loading Boston Housing dataset...")
        
        try:
            # Load from OpenML
            boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
            
            self.X_full = boston.data
            self.y_full = boston.target.astype(float)
            self.feature_names = list(boston.feature_names)
            self.target_name = 'MEDV'
            
            print("✓ Dataset loaded from OpenML")
        except:
            # Alternative: Load from sklearn legacy
            from sklearn.datasets import load_boston
            boston = load_boston()
            
            self.X_full = pd.DataFrame(boston.data, columns=boston.feature_names)
            self.y_full = pd.Series(boston.target)
            self.feature_names = list(boston.feature_names)
            self.target_name = 'MEDV'
            
            print("✓ Dataset loaded from sklearn")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names[:5]}... (showing first 5 of {len(self.feature_names)})")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class StudentPerformanceExample(LinearRegressionExample):
    """
    Student Performance Dataset - Grade prediction
    
    Predicts student final grades based on various social, demographic, and school-related
    features. Shows linear relationships between study time, past grades, and final performance.
    
    Features: Study time, failures, absences, past grades (G1, G2), family/social factors
    Target: Final grade (G3)
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Student Performance dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        
        try:
            import io
            import zipfile
            import requests
            
            # Download and extract
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open('student-mat.csv') as f:
                    df = pd.read_csv(f, sep=';')
            
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback: create synthetic student data
            print("Creating synthetic student performance data...")
            np.random.seed(42)
            n_samples = 395
            
            studytime = np.random.randint(1, 5, n_samples)
            failures = np.random.randint(0, 4, n_samples)
            absences = np.random.randint(0, 30, n_samples)
            G1 = np.random.uniform(0, 20, n_samples)
            G2 = G1 + np.random.normal(0, 2, n_samples)
            
            # G3 strongly correlated with G1 and G2, negatively with failures and absences
            G3 = (0.3 * G1 + 0.5 * G2 + 2 * studytime - 1.5 * failures - 0.1 * absences + 
                  np.random.normal(0, 1.5, n_samples))
            G3 = np.clip(G3, 0, 20)
            
            df = pd.DataFrame({
                'studytime': studytime,
                'failures': failures,
                'absences': absences,
                'G1': G1,
                'G2': G2,
                'G3': G3
            })
            print("✓ Synthetic dataset created")
        
        # Select relevant numeric features for linear regression
        numeric_features = ['studytime', 'failures', 'absences', 'G1', 'G2']
        
        # Ensure all required columns exist
        available_features = [f for f in numeric_features if f in df.columns]
        
        self.X_full = df[available_features]
        self.y_full = df['G3']
        self.feature_names = available_features
        self.target_name = 'G3'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class InsuranceCostExample(LinearRegressionExample):
    """
    Medical Insurance Cost Dataset - Premium prediction
    
    Predicts medical insurance costs based on personal attributes. Shows clear linear
    relationships between age, BMI, smoking status, and insurance charges.
    
    Features: Age, Sex, BMI, Children, Smoker, Region
    Target: Insurance charges (in dollars)
    
    Source: Kaggle / Machine Learning with R book
    """
    
    def load_data(self):
        print("Loading Medical Insurance Cost dataset...")
        
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        
        try:
            df = pd.read_csv(url)
            print("✓ Dataset loaded from GitHub")
        except:
            # Fallback: create synthetic insurance data
            print("Creating synthetic insurance cost data...")
            np.random.seed(42)
            n_samples = 1338
            
            age = np.random.randint(18, 65, n_samples)
            bmi = np.random.normal(30, 6, n_samples)
            children = np.random.randint(0, 6, n_samples)
            smoker = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            
            # Charges with strong linear relationship to age, bmi, and especially smoking
            charges = (250 * age + 340 * bmi + 500 * children + 23000 * smoker + 
                      np.random.normal(0, 3000, n_samples))
            charges = np.maximum(charges, 1000)  # Ensure positive
            
            df = pd.DataFrame({
                'age': age,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'charges': charges
            })
            print("✓ Synthetic dataset created")
        
        # Convert categorical to numeric if needed
        if 'smoker' in df.columns and df['smoker'].dtype == 'object':
            df['smoker'] = (df['smoker'] == 'yes').astype(int)
        
        if 'sex' in df.columns and df['sex'].dtype == 'object':
            df['sex'] = (df['sex'] == 'male').astype(int)
        
        # Select numeric features
        numeric_features = ['age', 'bmi', 'children']
        if 'smoker' in df.columns:
            numeric_features.append('smoker')
        if 'sex' in df.columns:
            numeric_features.append('sex')
        
        self.X_full = df[numeric_features]
        self.y_full = df['charges']
        self.feature_names = numeric_features
        self.target_name = 'charges'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class RealEstateValuationExample(LinearRegressionExample):
    """
    Real Estate Valuation Dataset - Property price prediction
    
    Predicts house prices per unit area based on location and property characteristics.
    Shows strong linear relationships with key features like distance to MRT station
    and house age.
    
    Features: House age, distance to MRT, number of convenience stores, latitude, longitude
    Target: House price per unit area
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Real Estate Valuation dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
        
        try:
            df = pd.read_excel(url)
            
            # Rename columns for clarity
            df.columns = ['No', 'TransactionDate', 'HouseAge', 'DistanceToMRT', 
                         'NumberOfStores', 'Latitude', 'Longitude', 'PricePerUnitArea']
            
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback: create synthetic real estate data
            print("Creating synthetic real estate data...")
            np.random.seed(42)
            n_samples = 414
            
            house_age = np.random.uniform(0, 45, n_samples)
            distance_to_mrt = np.random.uniform(0, 6500, n_samples)
            num_stores = np.random.randint(0, 11, n_samples)
            latitude = np.random.uniform(24.9, 25.1, n_samples)
            longitude = np.random.uniform(121.5, 121.6, n_samples)
            
            # Price with strong negative correlation to distance, positive to stores
            price = (60 - 0.005 * distance_to_mrt + 2 * num_stores - 0.3 * house_age + 
                    np.random.normal(0, 5, n_samples))
            price = np.maximum(price, 10)
            
            df = pd.DataFrame({
                'HouseAge': house_age,
                'DistanceToMRT': distance_to_mrt,
                'NumberOfStores': num_stores,
                'Latitude': latitude,
                'Longitude': longitude,
                'PricePerUnitArea': price
            })
            print("✓ Synthetic dataset created")
        
        # Select features
        feature_cols = ['HouseAge', 'DistanceToMRT', 'NumberOfStores', 'Latitude', 'Longitude']
        
        self.X_full = df[feature_cols]
        self.y_full = df['PricePerUnitArea']
        self.feature_names = feature_cols
        self.target_name = 'PricePerUnitArea'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class FishMarketExample(LinearRegressionExample):
    """
    Fish Market Dataset - Fish weight prediction
    
    Predicts fish weight based on physical measurements (length, height, width).
    Perfect example of strong linear relationships in biological measurements.
    
    Features: Length1, Length2, Length3, Height, Width
    Target: Weight (in grams)
    
    Source: Kaggle
    """
    
    def load_data(self):
        print("Loading Fish Market dataset...")
        
        url = "https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Fish/Fish.csv"
        
        try:
            df = pd.read_csv(url)
            print("✓ Dataset loaded from GitHub")
        except:
            # Fallback: create synthetic fish data
            print("Creating synthetic fish data...")
            np.random.seed(42)
            n_samples = 159
            
            length1 = np.random.uniform(7, 60, n_samples)
            length2 = length1 * 1.02 + np.random.normal(0, 0.5, n_samples)
            length3 = length1 * 1.15 + np.random.normal(0, 0.8, n_samples)
            height = length1 * 0.3 + np.random.normal(0, 0.5, n_samples)
            width = length1 * 0.15 + np.random.normal(0, 0.3, n_samples)
            
            # Weight strongly correlated with volume (length * height * width)
            weight = (length1 * height * width * 2.5 + np.random.normal(0, 50, n_samples))
            weight = np.maximum(weight, 1)
            
            df = pd.DataFrame({
                'Length1': length1,
                'Length2': length2,
                'Length3': length3,
                'Height': height,
                'Width': width,
                'Weight': weight
            })
            print("✓ Synthetic dataset created")
        
        # Select numeric features
        numeric_features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
        available_features = [f for f in numeric_features if f in df.columns]
        
        self.X_full = df[available_features]
        self.y_full = df['Weight']
        self.feature_names = available_features
        self.target_name = 'Weight'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class YachtHydrodynamicsExample(LinearRegressionExample):
    """
    Yacht Hydrodynamics Dataset - Residuary resistance prediction
    
    Predicts the residuary resistance of sailing yachts based on hull geometry
    and velocity. Classic engineering dataset with purely numerical features
    showing strong linear relationships.
    
    Features: 6 numerical features (hull dimensions, velocity)
    Target: Residuary resistance per unit weight of displacement
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Yacht Hydrodynamics dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
        
        try:
            df = pd.read_csv(url, delim_whitespace=True, header=None)
            df.columns = ['LongPos_COB', 'Prismatic_Coef', 'Len_Disp_Ratio', 
                         'Beam_Draught_Ratio', 'Length_Beam_Ratio', 'Froude_Num', 
                         'Residuary_Resistance']
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback: create synthetic yacht data
            print("Creating synthetic yacht hydrodynamics data...")
            np.random.seed(42)
            n_samples = 308
            
            longpos_cob = np.random.uniform(-5, 0, n_samples)
            prismatic_coef = np.random.uniform(0.53, 0.6, n_samples)
            len_disp_ratio = np.random.uniform(4.34, 5.14, n_samples)
            beam_draught_ratio = np.random.uniform(2.81, 5.35, n_samples)
            length_beam_ratio = np.random.uniform(2.73, 3.64, n_samples)
            froude_num = np.random.uniform(0.125, 0.45, n_samples)
            
            # Resistance with linear relationships
            resistance = (5 - 3 * longpos_cob + 15 * prismatic_coef + 
                         2 * len_disp_ratio - 1.5 * beam_draught_ratio + 
                         3 * length_beam_ratio + 50 * froude_num + 
                         np.random.normal(0, 2, n_samples))
            resistance = np.maximum(resistance, 0)
            
            df = pd.DataFrame({
                'LongPos_COB': longpos_cob,
                'Prismatic_Coef': prismatic_coef,
                'Len_Disp_Ratio': len_disp_ratio,
                'Beam_Draught_Ratio': beam_draught_ratio,
                'Length_Beam_Ratio': length_beam_ratio,
                'Froude_Num': froude_num,
                'Residuary_Resistance': resistance
            })
            print("✓ Synthetic dataset created")
        
        # All features are numerical
        feature_cols = ['LongPos_COB', 'Prismatic_Coef', 'Len_Disp_Ratio', 
                       'Beam_Draught_Ratio', 'Length_Beam_Ratio', 'Froude_Num']
        
        self.X_full = df[feature_cols]
        self.y_full = df['Residuary_Resistance']
        self.feature_names = feature_cols
        self.target_name = 'Residuary_Resistance'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class AirfoilSelfNoiseExample(LinearRegressionExample):
    """
    Airfoil Self-Noise Dataset - Sound pressure level prediction
    
    Predicts the scaled sound pressure level of NASA airfoils based on
    aerodynamic and geometric properties. All features are continuous numerical
    measurements from wind tunnel tests.
    
    Features: 5 numerical features (frequency, angle, chord length, velocity, thickness)
    Target: Scaled sound pressure level (in decibels)
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Airfoil Self-Noise dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
        
        try:
            df = pd.read_csv(url, sep='\t', header=None)
            df.columns = ['Frequency', 'AngleOfAttack', 'ChordLength', 
                         'FreeStreamVelocity', 'SuctionSideThickness', 
                         'SoundPressureLevel']
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback: create synthetic airfoil data
            print("Creating synthetic airfoil self-noise data...")
            np.random.seed(42)
            n_samples = 1503
            
            frequency = np.random.uniform(200, 20000, n_samples)
            angle = np.random.uniform(0, 22.2, n_samples)
            chord = np.random.uniform(0.025, 0.305, n_samples)
            velocity = np.random.uniform(31.7, 71.3, n_samples)
            thickness = np.random.uniform(0.002, 0.058, n_samples)
            
            # Sound pressure level with linear relationships
            sound = (100 + 0.001 * frequency - 0.5 * angle + 50 * chord + 
                    0.3 * velocity + 200 * thickness + np.random.normal(0, 3, n_samples))
            
            df = pd.DataFrame({
                'Frequency': frequency,
                'AngleOfAttack': angle,
                'ChordLength': chord,
                'FreeStreamVelocity': velocity,
                'SuctionSideThickness': thickness,
                'SoundPressureLevel': sound
            })
            print("✓ Synthetic dataset created")
        
        # All features are numerical
        feature_cols = ['Frequency', 'AngleOfAttack', 'ChordLength', 
                       'FreeStreamVelocity', 'SuctionSideThickness']
        
        self.X_full = df[feature_cols]
        self.y_full = df['SoundPressureLevel']
        self.feature_names = feature_cols
        self.target_name = 'SoundPressureLevel'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class WineQualityRedExample(LinearRegressionExample):
    """
    Wine Quality (Red) Dataset - Quality score prediction
    
    Predicts wine quality score based on physicochemical properties.
    All features are continuous numerical measurements from laboratory tests.
    
    Features: 11 numerical features (acidity, pH, sulfur dioxide, alcohol, etc.)
    Target: Quality score (0-10)
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Wine Quality (Red) dataset...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        
        try:
            df = pd.read_csv(url, sep=';')
            print("✓ Dataset loaded from UCI repository")
        except:
            # Fallback: create synthetic wine data
            print("Creating synthetic wine quality data...")
            np.random.seed(42)
            n_samples = 1599
            
            fixed_acidity = np.random.uniform(4.6, 15.9, n_samples)
            volatile_acidity = np.random.uniform(0.12, 1.58, n_samples)
            citric_acid = np.random.uniform(0, 1, n_samples)
            residual_sugar = np.random.uniform(0.9, 15.5, n_samples)
            chlorides = np.random.uniform(0.012, 0.611, n_samples)
            free_sulfur = np.random.uniform(1, 72, n_samples)
            total_sulfur = np.random.uniform(6, 289, n_samples)
            density = np.random.uniform(0.99, 1.004, n_samples)
            pH = np.random.uniform(2.74, 4.01, n_samples)
            sulphates = np.random.uniform(0.33, 2.0, n_samples)
            alcohol = np.random.uniform(8.4, 14.9, n_samples)
            
            # Quality with relationships to key features
            quality = (3 + 0.3 * alcohol - 5 * volatile_acidity + 0.5 * citric_acid + 
                      0.3 * sulphates + np.random.normal(0, 0.8, n_samples))
            quality = np.clip(quality, 0, 10)
            
            df = pd.DataFrame({
                'fixed acidity': fixed_acidity,
                'volatile acidity': volatile_acidity,
                'citric acid': citric_acid,
                'residual sugar': residual_sugar,
                'chlorides': chlorides,
                'free sulfur dioxide': free_sulfur,
                'total sulfur dioxide': total_sulfur,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol,
                'quality': quality
            })
            print("✓ Synthetic dataset created")
        
        # All features are numerical
        feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 
                       'residual sugar', 'chlorides', 'free sulfur dioxide',
                       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        
        self.X_full = df[feature_cols]
        self.y_full = df['quality']
        self.feature_names = feature_cols
        self.target_name = 'quality'
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names[:5]}... (showing first 5 of {len(self.feature_names)})")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()


class ENBEnergyEfficiencyExample(LinearRegressionExample):
    """
    Energy Efficiency Dataset - Heating/Cooling load prediction
    
    Predicts heating load of buildings based on building parameters.
    All features are numerical measurements of building characteristics.
    
    Features: 8 numerical features (relative compactness, surface area, wall area, etc.)
    Target: Heating load (Y1)
    
    Source: UCI Machine Learning Repository
    """
    
    def load_data(self):
        print("Loading Energy Efficiency dataset...")
        
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(name='energy-efficiency', version=1, as_frame=True, parser='auto')
            df = data.frame
            print("✓ Dataset loaded from OpenML")
        except:
            # Fallback: create synthetic energy efficiency data
            print("Creating synthetic energy efficiency data...")
            np.random.seed(42)
            n_samples = 768
            
            relative_compactness = np.random.uniform(0.62, 0.98, n_samples)
            surface_area = np.random.uniform(514.5, 808.5, n_samples)
            wall_area = np.random.uniform(245, 416.5, n_samples)
            roof_area = np.random.uniform(110.25, 220.5, n_samples)
            overall_height = np.random.uniform(3.5, 7, n_samples)
            orientation = np.random.randint(2, 6, n_samples)
            glazing_area = np.random.uniform(0, 0.4, n_samples)
            glazing_dist = np.random.randint(0, 6, n_samples)
            
            # Heating load with linear relationships
            heating_load = (50 - 30 * relative_compactness + 0.05 * surface_area - 
                           0.03 * wall_area + 5 * overall_height - 20 * glazing_area +
                           np.random.normal(0, 2, n_samples))
            heating_load = np.maximum(heating_load, 6)
            
            df = pd.DataFrame({
                'X1': relative_compactness,
                'X2': surface_area,
                'X3': wall_area,
                'X4': roof_area,
                'X5': overall_height,
                'X6': orientation,
                'X7': glazing_area,
                'X8': glazing_dist,
                'Y1': heating_load
            })
            print("✓ Synthetic dataset created")
        
        # All features are numerical
        feature_cols = [col for col in df.columns if col.startswith('X')]
        if not feature_cols:
            feature_cols = df.columns[:-1].tolist()
        
        target_col = 'Y1' if 'Y1' in df.columns else df.columns[-1]
        
        self.X_full = df[feature_cols]
        self.y_full = df[target_col]
        self.feature_names = feature_cols
        self.target_name = target_col
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2, random_state=42
        )
        
        print(f"Dataset shape: {self.X_full.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set:  {len(self.X_test)} samples")
        
        self.show_dataset_summary()
