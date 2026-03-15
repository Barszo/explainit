"""
Data Downloader and Preprocessor

Handles downloading, caching, and preprocessing of datasets for CF experiments.
Checks if data exists locally before downloading, and saves all intermediate steps.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
import ssl
import pickle


# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def download_with_ssl_context(url, filepath):
    """Download file with SSL context handling."""
    try:
        # Try with default SSL context
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"Standard download failed: {e}")
        print("Trying with unverified SSL context...")
        try:
            # Create unverified SSL context
            ssl_context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            return True
        except Exception as e2:
            print(f"SSL download also failed: {e2}")
            return False


def load_communities_and_crime(test_size=0.2, random_state=42, force_download=False):
    """
    Load and preprocess Communities and Crime dataset.
    
    Checks for cached data first:
    1. Preprocessed train/test splits (fastest)
    2. Raw downloaded data
    3. Downloads if needed
    
    Args:
        test_size: Test set size
        random_state: Random seed
        force_download: Force re-download even if cached data exists
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*80)
    print("LOADING COMMUNITIES AND CRIME DATASET")
    print("="*80)
    
    # File paths
    raw_file = os.path.join(DATA_DIR, 'communities_crime_raw.csv')
    processed_file = os.path.join(DATA_DIR, 'communities_crime_processed.pkl')
    
    # Check for preprocessed data
    if not force_download and os.path.exists(processed_file):
        print(f"Loading preprocessed data from: {processed_file}")
        with open(processed_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"✓ Loaded preprocessed data")
        print(f"  Training samples: {len(data_dict['X_train'])}")
        print(f"  Test samples: {len(data_dict['X_test'])}")
        print(f"  Features: {len(data_dict['feature_names'])}")
        return (data_dict['X_train'], data_dict['X_test'], 
                data_dict['y_train'], data_dict['y_test'],
                data_dict['feature_names'], data_dict['scaler'])
    
    # Check for raw data
    if not force_download and os.path.exists(raw_file):
        print(f"Loading raw data from: {raw_file}")
        data = pd.read_csv(raw_file)
    else:
        # Download data
        print("Downloading dataset from UCI repository...")
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
        
        temp_file = raw_file + '.tmp'
        success = download_with_ssl_context(url, temp_file)
        
        if not success:
            raise RuntimeError("ERROR: Could not download Communities and Crime dataset from UCI repository")
        
        # Read downloaded data
        try:
            data = pd.read_csv(temp_file, header=None, na_values='?')
            
            # Generate feature names (128 attributes total)
            feature_names_all = [f'feature_{i}' for i in range(128)]
            data.columns = feature_names_all
            
            # Save raw data
            data.to_csv(raw_file, index=False)
            print(f"✓ Downloaded and saved raw data to: {raw_file}")
            
            # Clean up temp file
            os.remove(temp_file)
        except Exception as e:
            print(f"ERROR reading downloaded file: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise RuntimeError(f"Failed to process Communities and Crime dataset: {e}")
    
    # Process data
    print("Processing data...")
    
    # Remove non-predictive features (first 5 columns)
    data = data.iloc[:, 5:]
    
    # Remove rows with missing values
    data = data.dropna()
    print(f"  Samples after removing missing values: {len(data)}")
    
    # Extract target (last column) and convert to binary
    target_col = data.columns[-1]
    crime_rate = data[target_col]
    threshold = crime_rate.median()
    
    # Binary: 1 = low crime (positive), 0 = high crime (negative)
    y = (crime_rate <= threshold).astype(int)
    
    # Features (all continuous, excluding target)
    X = data.iloc[:, :-1]
    
    # Select features with sufficient variance
    feature_variance = X.var()
    valid_features = feature_variance[feature_variance > 1e-6].index.tolist()
    
    # Use 99 features as per paper
    if len(valid_features) > 99:
        valid_features = feature_variance.nlargest(99).index.tolist()
    
    X = X[valid_features]
    feature_names = list(X.columns)
    
    print(f"  Using {len(feature_names)} continuous features")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data (80/20 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply 0 mean, unit variance scaling (as per paper)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Convert to numpy arrays
    y_train = y_train.values
    y_test = y_test.values
    
    # Save preprocessed data
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }
    with open(processed_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✓ Saved preprocessed data to: {processed_file}")
    
    print(f"\nDataset ready:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def load_german_credit(test_size=0.2, random_state=42, force_download=False):
    """
    Load and preprocess German Credit dataset.
    
    Checks for cached data first:
    1. Preprocessed train/test splits (fastest)
    2. Raw downloaded data
    3. Downloads if needed
    
    Args:
        test_size: Test set size
        random_state: Random seed
        force_download: Force re-download even if cached data exists
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*80)
    print("LOADING GERMAN CREDIT DATASET")
    print("="*80)
    
    # File paths
    raw_file = os.path.join(DATA_DIR, 'german_credit_raw.csv')
    processed_file = os.path.join(DATA_DIR, 'german_credit_processed.pkl')
    
    # Check for preprocessed data
    if not force_download and os.path.exists(processed_file):
        print(f"Loading preprocessed data from: {processed_file}")
        with open(processed_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"✓ Loaded preprocessed data")
        print(f"  Training samples: {len(data_dict['X_train'])}")
        print(f"  Test samples: {len(data_dict['X_test'])}")
        print(f"  Features: {len(data_dict['feature_names'])}")
        return (data_dict['X_train'], data_dict['X_test'],
                data_dict['y_train'], data_dict['y_test'],
                data_dict['feature_names'], data_dict['scaler'])
    
    # Check for raw data
    if not force_download and os.path.exists(raw_file):
        print(f"Loading raw data from: {raw_file}")
        data = pd.read_csv(raw_file)
    else:
        # Download data
        print("Downloading dataset from UCI repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        
        temp_file = raw_file + '.tmp'
        success = download_with_ssl_context(url, temp_file)
        
        if not success:
            raise RuntimeError("ERROR: Could not download German Credit dataset from UCI repository")
        
        # Read downloaded data
        try:
            column_names = [
                'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
                'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
                'residence', 'property', 'age', 'other_installment', 'housing',
                'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker',
                'credit_risk'
            ]
            
            data = pd.read_csv(temp_file, sep='\\s+', names=column_names, engine='python')
            
            # Convert target: 1=good, 2=bad -> 1=good (low risk), 0=bad (high risk)
            data['credit_risk'] = (data['credit_risk'] == 1).astype(int)
            
            # Save raw data
            data.to_csv(raw_file, index=False)
            print(f"✓ Downloaded and saved raw data to: {raw_file}")
            
            # Clean up temp file
            os.remove(temp_file)
        except Exception as e:
            print(f"ERROR reading downloaded file: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise RuntimeError(f"Failed to process German Credit dataset: {e}")
    
    # Process data
    print("Processing data...")
    
    # According to paper, use 7 numerical features (subset of 27 total)
    numerical_features = [
        'duration',           # Duration in months
        'credit_amount',      # Credit amount
        'age',               # Age in years
        'installment_rate',  # Installment rate in percentage of disposable income
        'residence',         # Present residence since
        'existing_credits',  # Number of existing credits at this bank
        'num_dependents'     # Number of people being liable to provide maintenance for
    ]
    
    # Extract features and target
    X = data[numerical_features]
    y = data['credit_risk']
    
    feature_names = numerical_features
    
    print(f"  Using {len(feature_names)} numerical features")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data (80/20 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply 0 mean, unit variance scaling (as per paper)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Convert to numpy arrays
    y_train = y_train.values
    y_test = y_test.values
    
    # Save preprocessed data
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }
    with open(processed_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✓ Saved preprocessed data to: {processed_file}")
    
    print(f"\nDataset ready:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    """Test data loading."""
    print("Testing data loading...")
    
    # Test Communities and Crime
    X_train, X_test, y_train, y_test, features, scaler = load_communities_and_crime()
    print(f"\n✓ Communities and Crime loaded successfully")
    print(f"  Shape: {X_train.shape}")
    
    # Test German Credit
    X_train, X_test, y_train, y_test, features, scaler = load_german_credit()
    print(f"\n✓ German Credit loaded successfully")
    print(f"  Shape: {X_train.shape}")
    
    print("\n" + "="*80)
    print("DATA LOADING TEST COMPLETE")
    print("="*80)
