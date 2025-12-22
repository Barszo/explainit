import numpy as np
import pandas as pd
import pickle
from categorical_shapley import CategoricalShapley


def example_diamond_price():
    """Example using real Diamond Price dataset with XGBoost model
    
    This demonstrates all 4 calculation methods:
    1. Exact without grouping - treats all 23 features independently
    2. Approximate without grouping - uses sampling for faster computation
    3. Exact with grouping - groups one-hot encoded categorical features
    4. Approximate with grouping - combines grouping with sampling
    
    The dataset has real one-hot encoded categorical features:
    - cut (4 one-hot features, indices 6-9)
    - color (6 one-hot features, indices 10-15)
    - clarity (7 one-hot features, indices 16-22)
    """
    # ========== CONFIGURATION ==========
    NUM_SAMPLES_TO_EXPLAIN = 100  # Modify this to change how many samples to process
    NUM_APPROXIMATION_SAMPLES = 1000  # Number of random coalitions for approximation
    # ===================================
    
    print("\n" + "="*80)
    print("Diamond Price Dataset - Shapley Value Analysis with XGBoost")
    print("="*80)
    
    # Load the trained model
    model_path = "../ML_models_for_tests/one_hot_regression/one_hot_regression_results/Diamond_Price/XGBoost_model.pkl"
    preprocessor_path = "../ML_models_for_tests/one_hot_regression/one_hot_regression_results/Diamond_Price/preprocessor.pkl"
    data_path = "../ML_models_for_tests/one_hot_regression/one_hot_regression_results/Diamond_Price/X_data_encoded.csv"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Load encoded dataset (already transformed)
        X_data = pd.read_csv(data_path, header=None)
        
        print("✓ Model and data loaded successfully")
        print(f"Dataset shape: {X_data.shape}")
        print(f"Number of features (encoded): {X_data.shape[1]}")
    except Exception as e:
        print(f"Error loading model/data: {e}")
        return None
    
    # Create prediction function (data is already encoded and scaled)
    def predict_diamond_price(X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return model.predict(X)
    
    # Select random samples from the dataset
    np.random.seed(42)
    sample_indices = np.random.choice(X_data.index, size=NUM_SAMPLES_TO_EXPLAIN, replace=False)
    
    samples_to_process = [X_data.iloc[idx].values for idx in sample_indices]
    
    # Base values: Use mean as reference for all samples
    base_value = X_data.mean().values
    base_values_list = [base_value for _ in range(NUM_SAMPLES_TO_EXPLAIN)]
    
    # Feature names (23 features total)
    feature_names = (['carat', 'depth', 'table', 'x', 'y', 'z'] +  # 0-5: numerical
                     [f'cut_{i}' for i in range(4)] +                # 6-9: cut one-hot
                     [f'color_{i}' for i in range(6)] +              # 10-15: color one-hot
                     [f'clarity_{i}' for i in range(7)])             # 16-22: clarity one-hot
    
    print(f"\n--- Selected Samples ---")
    print(f"Number of samples: {NUM_SAMPLES_TO_EXPLAIN}")
    print(f"\nFirst 3 samples (preview):")
    for i in range(min(3, NUM_SAMPLES_TO_EXPLAIN)):
        sample = samples_to_process[i]
        print(f"\nSample {i+1}:")
        print(f"  Predicted price: ${predict_diamond_price(sample)[0]:,.2f}")
        print(f"  First 6 numerical features: {sample[:6]}")
    
    print(f"\nBase value (Mean of dataset):")
    print(f"  First 6 numerical features: {base_value[:6]}")
    
    # Define categorical groups (one-hot encoded features)
    # cut: indices 6-9 (4 features)
    # color: indices 10-15 (6 features)
    # clarity: indices 16-22 (7 features)
    categorical_groups = [
        list(range(6, 10)),    # cut one-hot group
        list(range(10, 16)),   # color one-hot group
        list(range(16, 23))    # clarity one-hot group
    ]
    
    print(f"\n--- Grouping Configuration ---")
    print(f"Categorical groups defined:")
    print(f"  - cut (features 6-9): {len(categorical_groups[0])} one-hot features")
    print(f"  - color (features 10-15): {len(categorical_groups[1])} one-hot features")
    print(f"  - clarity (features 16-22): {len(categorical_groups[2])} one-hot features")
    print(f"Without grouping: 23 features")
    print(f"With grouping: 6 numerical + 3 categorical = 9 units")
    
    # Create explainer WITH categorical groups
    explainer = CategoricalShapley(
        model_pred=predict_diamond_price,
        categorical_groups=categorical_groups
    )
    
    # Calculate using approximation methods only (faster!)
    print("\n" + "="*80)
    print("Calculating Shapley values using APPROXIMATION methods only...")
    print("="*80)
    print(f"Processing {NUM_SAMPLES_TO_EXPLAIN} samples with {NUM_APPROXIMATION_SAMPLES} approximation samples each")
    print("(To enable exact methods, set use_exact=True, but beware: very slow with 23 features!)")
    results = explainer.calculate_all(samples_to_process, base_values_list, 
                                     num_samples=NUM_APPROXIMATION_SAMPLES, use_exact=False)
    
    # Get DataFrame
    df = explainer.get_dataframe()
    
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Rows: {NUM_SAMPLES_TO_EXPLAIN} samples × 2 methods = {df.shape[0]} rows")
    print(f"\nColumns: {list(df.columns)}")
    
    print(f"\n--- Calculation Times ---")
    print(df[['sample_id', 'type', 'time_seconds']].to_string(index=False))
    
    print(f"\n--- Top 5 Features by Importance (First 3 Samples) ---")
    for sample_id in range(min(3, NUM_SAMPLES_TO_EXPLAIN)):
        print(f"\nSample {sample_id + 1}:")
        for method in ['approx_wo_grouping', 'approx_w_grouping']:
            matching_rows = df[(df['sample_id'] == sample_id) & (df['type'] == method)]
            if matching_rows.empty:
                continue
            row = matching_rows.iloc[0]
            shap_vals = row['shap_values']
            # Get top 5 by absolute value
            top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
            print(f"  {method}:")
            for idx in top_indices:
                print(f"    {feature_names[idx]:15s}: {shap_vals[idx]:8.4f}")
    
    # Save results
    csv_path = "diamond_price_shapley.csv"
    explainer.save_dataframe(csv_path)
    
    # Generate full report
    print("\n" + "="*80)
    print("Detailed Report")
    print("="*80)
    report = explainer.generate_report()
    print(report)
    
    return df


if __name__ == "__main__":
    example_diamond_price()
