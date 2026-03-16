import os
import sys
import pandas as pd
from data_downloader import load_lending_club_selected_features

print("\n--- Lending Club Loader Debug ---\n", flush=True)
raw_csv = os.path.join(os.path.dirname(__file__), 'data', 'lending_club_raw.csv')
if not os.path.exists(raw_csv):
    print("ERROR: lending_club_raw.csv not found. Please download it first.", flush=True)
    sys.exit(1)

try:
    X_train, X_test, y_train, y_test, feature_names, scaler = load_lending_club_selected_features()
    print(f"X_train shape: {X_train.shape}", flush=True)
    print(f"X_test shape: {X_test.shape}", flush=True)
    print(f"y_train shape: {y_train.shape}", flush=True)
    print(f"y_test shape: {y_test.shape}", flush=True)
    print(f"feature_names: {feature_names}", flush=True)
    print(f"Scaler: {scaler}", flush=True)
    print("\nFirst rows of X_train:", flush=True)
    print(X_train.head(), flush=True)
    print("\nFirst rows of y_train:", flush=True)
    print(y_train[:10], flush=True)
    # Check for missing values in features
    print("\nMissing values in X_train:", flush=True)
    print(X_train.isnull().sum(), flush=True)
    # Check target balance
    print("\nTarget distribution in y_train:", flush=True)
    print(pd.Series(y_train).value_counts(), flush=True)
    # Check if ready for model training
    if X_train.shape[0] == 0:
        print("ERROR: No training samples available.", flush=True)
    elif X_train.isnull().sum().sum() > 0:
        print("ERROR: Missing values remain in training features.", flush=True)
    elif pd.Series(y_train).nunique() < 2:
        print("ERROR: Target is not binary or not balanced.", flush=True)
    else:
        print("✓ Data is ready for model training.", flush=True)
except Exception as e:
    print(f"Exception: {e}", flush=True)

# Additionally, print some stats from the raw CSV
raw_csv = os.path.join(os.path.dirname(__file__), 'data', 'lending_club_raw.csv')
if os.path.exists(raw_csv):
    print("\n--- Raw CSV Quick Stats ---\n", flush=True)
    try:
        df = pd.read_csv(raw_csv, low_memory=False, nrows=1000)
        print(f"Raw CSV shape (first 1000 rows): {df.shape}", flush=True)
        print(f"Columns: {list(df.columns)}", flush=True)
        print(df.head(), flush=True)
        if 'loan_status' in df.columns:
            print(f"loan_status value counts:\n{df['loan_status'].value_counts()}", flush=True)
        # Print missing value proportions for continuous features
        continuous_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        print("\nMissing value proportions for continuous features:", flush=True)
        for col in continuous_cols:
            missing = df[col].isnull().sum()
            total = df.shape[0]
            print(f"  {col}: {missing}/{total} ({missing/total:.2%}) missing", flush=True)
    except Exception as e:
        print(f"Exception reading raw CSV: {e}", flush=True)
else:
    print("Raw CSV not found.", flush=True)
