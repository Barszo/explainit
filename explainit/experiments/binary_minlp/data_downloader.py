# --- Lending Club Dataset Loader ---
import zipfile
import requests
from model_builder import preprocess_lending_club_data
def load_lending_club_selected_features(csv_path, test_size=0.2, random_state=42):
    """
    Load Lending Club dataset using only 8 selected features and new preprocessing.
    Args:
        csv_path: Path to Lending Club CSV file
        test_size: Test set size
        random_state: Random seed
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    import os
    import zipfile
    import requests
    DATA_DIR = os.path.dirname(csv_path)
    if not os.path.exists(csv_path):
        # Download Lending Club dataset from Kaggle
        raw_zip = os.path.join(DATA_DIR, 'lending_club_raw.zip')
        kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club"
        headers = {"User-Agent": "Mozilla/5.0"}
        print(f"Downloading Lending Club dataset from Kaggle API...")
        response = requests.get(kaggle_url, headers=headers)
        if response.status_code == 200:
            with open(raw_zip, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded zip file to: {raw_zip}")
            # Extract CSV
            with zipfile.ZipFile(raw_zip, 'r') as zip_ref:
                found_csv = False
                for file in zip_ref.namelist():
                    if file.endswith('.csv'):
                        zip_ref.extract(file, DATA_DIR)
                        os.rename(os.path.join(DATA_DIR, file), csv_path)
                        print(f"✓ Extracted CSV to: {csv_path}")
                        found_csv = True
                        break
                if not found_csv:
                    raise RuntimeError("ERROR: No CSV file found in Lending Club zip archive.")
        else:
            raise RuntimeError("ERROR: Could not download Lending Club dataset from Kaggle API")
    else:
        print(f"✓ Using local Lending Club CSV: {csv_path}")

    # Validate CSV content before preprocessing
    # LoanStats3a.csv has headers on row 0 (no leading notes row)
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    required_cols = ["emp_length", "annual_inc", "open_acc", "earliest_cr_line", "grade", "home_ownership", "purpose", "addr_state", "loan_status"]
    if df.empty or not all(col in df.columns for col in required_cols):
        raise RuntimeError(f"ERROR: CSV file {csv_path} is empty or missing required columns.\nColumns found: {list(df.columns)}\nPlease ensure you have downloaded the correct Lending Club file (e.g., LoanStats3a.csv from Kaggle Lending Club).")

    X, y = preprocess_lending_club_data(csv_path)
    feature_names = list(X.columns)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Standardize: fit on train, transform both splits.
    # Required because raw features (annual_inc etc.) span very different scales.
    # Without this, tanh activations saturate and the model outputs the prior.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    return X_train, X_test, y_train, y_test, feature_names, scaler

def load_lending_club(test_size=0.2, random_state=42, force_download=False):
    """
    Load and preprocess Lending Club dataset from Kaggle.
    Only continuous features are used. Target is loan_status (binary).
    Args:
        test_size: Test set size
        random_state: Random seed
        force_download: Force re-download even if cached data exists
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*80)
    print("LOADING LENDING CLUB DATASET")
    print("="*80)

    # File paths
    raw_zip = os.path.join(DATA_DIR, 'lending_club_raw.zip')
    raw_csv = os.path.join(DATA_DIR, 'lending_club_raw.csv')
    processed_file = os.path.join(DATA_DIR, 'lending_club_processed.pkl')

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

    # Download from Kaggle only if needed
    if force_download or not os.path.exists(raw_csv):
        print("Downloading Lending Club dataset from Kaggle API...")
        kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/wordsforthewise/lending-club"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(kaggle_url, headers=headers)
        if response.status_code == 200:
            with open(raw_zip, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded zip file to: {raw_zip}")
            # Extract CSV
            with zipfile.ZipFile(raw_zip, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith('.csv'):
                        zip_ref.extract(file, DATA_DIR)
                        os.rename(os.path.join(DATA_DIR, file), raw_csv)
                        print(f"✓ Extracted CSV to: {raw_csv}")
                        break
        else:
            raise RuntimeError("ERROR: Could not download Lending Club dataset from Kaggle API")
    else:
        print(f"✓ Using local Lending Club CSV: {raw_csv}")

    # Load CSV
    print(f"Loading raw data from: {raw_csv}")
    df = pd.read_csv(raw_csv, low_memory=False)

    # Preprocessing steps:
    # 1. Select only continuous features (float or int, not object/categorical)
    continuous_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    id_cols = ['id', 'member_id']
    continuous_cols = [col for col in continuous_cols if col not in id_cols]
    # 2. Drop continuous columns with >20% missing values
    missing_props = df[continuous_cols].isnull().mean()
    filtered_cols = [col for col in continuous_cols if missing_props[col] <= 0.2]
    print(f"  Filtered continuous columns (<=20% missing): {filtered_cols}", flush=True)
    # 3. Drop rows with missing values in filtered features
    df = df.dropna(subset=filtered_cols, axis=0)
    print(f"  Shape after dropping rows with missing values in filtered features: {df.shape}", flush=True)
    continuous_cols = filtered_cols

    # 3. Define target: loan_status (binary: 1=good, 0=bad)
    # Common mapping: Charged Off, Default, Late, etc. = 0; Fully Paid = 1
    if 'loan_status' not in df.columns:
        raise RuntimeError("ERROR: 'loan_status' column not found in Lending Club dataset")
    status_map = {
        'Fully Paid': 1,
        'Charged Off': 0,
        'Default': 0,
        'Late (16-30 days)': 0,
        'Late (31-120 days)': 0,
        'In Grace Period': 0,
        'Does not meet the credit policy. Status: Charged Off': 0,
        'Does not meet the credit policy. Status: Fully Paid': 1,
        'Current': 1,
        'Issued': 1,
        'Expired': 0,
        'Removed': 0
    }
    df['loan_status_bin'] = df['loan_status'].map(status_map)
    df = df[df['loan_status_bin'].notnull()]
    y = df['loan_status_bin'].astype(int)

    # 4. Features
    X = df[continuous_cols]
    feature_names = list(X.columns)

    print(f"  Using {len(feature_names)} continuous features")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_names)

    # 6. Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 7. Save preprocessed data
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
        'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
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


def _download_credit_card_default(raw_file):
    """
    Download the UCI Default of Credit Card Clients dataset.
    Tries three strategies in order:
      1. sklearn fetch_openml by dataset name
      2. sklearn fetch_openml by data_id 42477
      3. Direct ZIP download from UCI archive (requires openpyxl)
    """
    print("Downloading Default of Credit Card Clients dataset...")

    # Strategy 1: fetch_openml by name
    try:
        from sklearn.datasets import fetch_openml
        print("  Trying sklearn fetch_openml (by name)...")
        raw = fetch_openml(
            name='default-of-credit-card-clients',
            as_frame=True, parser='auto'
        )
        df = raw.frame
        df.to_csv(raw_file, index=False)
        print(f"✓ Downloaded via OpenML (name), saved to: {raw_file}")
        return df
    except Exception as e:
        print(f"  fetch_openml (name) failed: {e}")

    # Strategy 2: fetch_openml by known data_id
    try:
        from sklearn.datasets import fetch_openml
        print("  Trying sklearn fetch_openml (data_id=42477)...")
        raw = fetch_openml(data_id=42477, as_frame=True, parser='auto')
        df = raw.frame
        df.to_csv(raw_file, index=False)
        print(f"✓ Downloaded via OpenML (id=42477), saved to: {raw_file}")
        return df
    except Exception as e:
        print(f"  fetch_openml (id) failed: {e}")

    # Strategy 3: direct ZIP + XLS download from UCI (with SSL fallback)
    import zipfile
    import io
    import requests as _req
    from urllib3.exceptions import InsecureRequestWarning

    zip_url = (
        "https://archive.ics.uci.edu/static/public/350/"
        "default+of+credit+card+clients.zip"
    )

    def _try_uci_download(verify_ssl):
        print(f"  Trying direct download from UCI archive (verify={verify_ssl})...")
        if not verify_ssl:
            _req.packages.urllib3.disable_warnings(InsecureRequestWarning)
        response = _req.get(zip_url, timeout=120, verify=verify_ssl)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            xls_name = next(
                f for f in z.namelist()
                if f.lower().endswith('.xls') or f.lower().endswith('.xlsx')
            )
            with z.open(xls_name) as xls_bytes:
                return pd.read_excel(io.BytesIO(xls_bytes.read()), header=1)

    last_err = None
    for verify in (True, False):
        try:
            df = _try_uci_download(verify_ssl=verify)
            df.to_csv(raw_file, index=False)
            note = "" if verify else " (SSL verification disabled fallback)"
            print(f"✓ Downloaded from UCI (XLS){note}, saved to: {raw_file}")
            return df
        except ImportError:
            print("  openpyxl not installed (needed for XLS). Run: pip install openpyxl")
            break
        except Exception as e:
            last_err = e
            print(f"  UCI direct download failed (verify={verify}): {e}")

    raise RuntimeError(
        "Could not download Default of Credit Card Clients dataset via any method.\n"
        f"Last error: {last_err}\n"
        "Please download manually from https://archive.ics.uci.edu/dataset/350\n"
        f"and place the CSV at: {raw_file}"
    )


def _process_credit_card_default(df, processed_file, test_size=0.2, random_state=42):
    """Process the raw Credit Card Default DataFrame into model-ready splits."""
    print("Processing data...")
    print(f"  Raw shape: {df.shape}")

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # Locate target column
    target_candidates = [
        c for c in df.columns
        if 'default' in c.lower() and 'payment' in c.lower()
    ]
    if not target_candidates:
        target_candidates = [
            c for c in df.columns if c.lower() in ('y', 'target', 'class')
        ]
    if not target_candidates:
        target_candidates = [df.columns[-1]]
    target_col = target_candidates[0]
    print(f"  Target column: '{target_col}'")

    # The 14 purely continuous features (dollar amounts + age/limit)
    wanted = [
        'LIMIT_BAL', 'AGE',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',  'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6',
    ]
    col_upper_map = {c.upper(): c for c in df.columns}
    feature_cols = [col_upper_map[w] for w in wanted if w in col_upper_map]

    if len(feature_cols) < 2:
        # Fallback: all numeric columns except target and known categoricals
        exclude = {target_col}
        for c in df.columns:
            cu = c.upper()
            if cu in ('ID', 'SEX', 'EDUCATION', 'MARRIAGE'):
                exclude.add(c)
            if cu.startswith('PAY_') and not any(
                cu == f'PAY_AMT{i}' for i in range(1, 7)
            ):
                exclude.add(c)
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        print(f"  Warning: expected columns not found, using {len(feature_cols)} numeric columns")

    X = df[feature_cols].copy().astype(float)
    # Flip target so encoding is consistent with other datasets:
    # 0 = defaulted (undesired / "bad"), 1 = paid on time (desired / "good")
    # This matches German Credit (0=bad, 1=good) and C&C (0=high crime, 1=low crime),
    # so source_class=0 always means the "bad" group seeking a favorable CF.
    y = (1 - df[target_col].astype(float)).astype(int)

    valid_mask = X.notna().all(axis=1) & y.notna()
    X, y = X[valid_mask], y[valid_mask]

    feature_names = list(feature_cols)
    print(f"  Using {len(feature_names)} continuous features: {feature_names}")
    print(f"  Samples after cleaning: {len(X)}")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test  = pd.DataFrame(X_test_scaled,  columns=feature_names)
    y_train = y_train.values
    y_test  = y_test.values

    data_dict = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': feature_names, 'scaler': scaler,
    }
    with open(processed_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✓ Saved preprocessed data to: {processed_file}")

    print(f"\nDataset ready:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Features:         {len(feature_names)}")

    return X_train, X_test, y_train, y_test, feature_names, scaler


def load_credit_card_default(test_size=0.2, random_state=42, force_download=False):
    """
    Load and preprocess the UCI Default of Credit Card Clients dataset (Taiwan, 2005).

    14 purely continuous features:
        LIMIT_BAL, AGE, BILL_AMT1-6, PAY_AMT1-6
    Binary target:
        1 = defaulted on payment next month
        0 = did not default

    Source: https://archive.ics.uci.edu/dataset/350

    Args:
        test_size: Fraction reserved for test set (default 0.2)
        random_state: Random seed
        force_download: Re-download and re-process even if cache exists

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n" + "="*80)
    print("LOADING DEFAULT OF CREDIT CARD CLIENTS DATASET")
    print("="*80)

    processed_file = os.path.join(DATA_DIR, 'credit_card_default_processed.pkl')
    raw_file       = os.path.join(DATA_DIR, 'credit_card_default_raw.csv')

    # Return preprocessed cache if available
    if not force_download and os.path.exists(processed_file):
        print(f"Loading preprocessed data from: {processed_file}")
        with open(processed_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f"✓ Loaded preprocessed data")
        print(f"  Training samples: {len(data_dict['X_train'])}")
        print(f"  Test samples:     {len(data_dict['X_test'])}")
        print(f"  Features:         {len(data_dict['feature_names'])}")
        return (
            data_dict['X_train'], data_dict['X_test'],
            data_dict['y_train'], data_dict['y_test'],
            data_dict['feature_names'], data_dict['scaler'],
        )

    # Load raw CSV cache if available
    if not force_download and os.path.exists(raw_file):
        print(f"Loading raw CSV from: {raw_file}")
        df = pd.read_csv(raw_file, low_memory=False)
    else:
        df = _download_credit_card_default(raw_file)

    return _process_credit_card_default(df, processed_file, test_size, random_state)


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

    # Test Credit Card Default
    X_train, X_test, y_train, y_test, features, scaler = load_credit_card_default()
    print(f"\n✓ Credit Card Default loaded successfully")
    print(f"  Shape: {X_train.shape}")
    
    print("\n" + "="*80)
    print("DATA LOADING TEST COMPLETE")
    print("="*80)
