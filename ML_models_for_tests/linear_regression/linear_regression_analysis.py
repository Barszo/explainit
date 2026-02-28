"""
Comprehensive Linear Regression Analysis

This script performs thorough analysis of multiple real-world datasets using Linear Regression.
It focuses on datasets known to have strong linear relationships and saves detailed results.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import dataset examples
from linear_regression_examples import (
    AdvertisingExample,
    BostonHousingExample,
    StudentPerformanceExample,
    InsuranceCostExample,
    RealEstateValuationExample,
    FishMarketExample,
    YachtHydrodynamicsExample,
    AirfoilSelfNoiseExample,
    WineQualityRedExample,
    ENBEnergyEfficiencyExample
)


def grade_model(r2_score):
    """Grade the model based on R² score"""
    if r2_score >= 0.90:
        return "PERFECT (can be used)"
    elif r2_score >= 0.80:
        return "VERY GOOD (can be used)"
    elif r2_score >= 0.65:
        return "GOOD (could be better but can be used)"
    elif r2_score >= 0.40:
        return "NOT ACCEPTABLE (better than bad but still can't be used)"
    elif r2_score >= 0.20:
        return "BAD (can't be used)"
    else:
        return "VERY BAD (can't be used)"


def train_linear_regression(X_train, X_test, y_train, y_test, dataset_name):
    """Train Linear Regression model and return metrics"""
    
    print(f"\n  Training Linear Regression model...")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    overfitting = abs(train_r2 - test_r2)
    
    print(f"    ✓ Model trained")
    print(f"    Train R² Score: {train_r2:.4f}")
    print(f"    Test R² Score: {test_r2:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    result = {
        'Model': 'Linear Regression',
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test RMSE': rmse,
        'Test MAE': mae,
        'Overfitting': overfitting,
        'Grade': grade_model(test_r2),
        'Coefficients': model.coef_,
        'Intercept': model.intercept_
    }
    
    return result, model


def analyze_and_save_dataset(example_class, dataset_name, base_dir="linear_regression_results"):
    """Complete analysis of dataset with Linear Regression and save everything"""
    
    print("\n" + "="*100)
    print(f"DATASET: {dataset_name}")
    print("="*100)
    
    # Create directory for this dataset
    dataset_dir = os.path.join(base_dir, dataset_name.replace(" ", "_"))
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Load data
    example = example_class()
    example.load_data()
    
    X_full = example.X_full
    y_full = example.y_full
    X_train = example.X_train
    X_test = example.X_test
    y_train = example.y_train
    y_test = example.y_test
    
    # Save raw dataset
    print(f"\n  Saving dataset to {dataset_dir}...")
    X_full.to_csv(os.path.join(dataset_dir, "X_data.csv"), index=False)
    y_full.to_csv(os.path.join(dataset_dir, "y_data.csv"), index=False)
    print(f"    ✓ Saved: X_data.csv, y_data.csv")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(os.path.join(dataset_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"    ✓ Saved: scaler.pkl")
    
    # Prepare summary content
    summary_lines = []
    
    summary_lines.append("="*100)
    summary_lines.append(f"LINEAR REGRESSION ANALYSIS: {dataset_name}")
    summary_lines.append("="*100)
    summary_lines.append("")
    
    # 1. Dataset Name
    summary_lines.append("-"*100)
    summary_lines.append("1. DATASET NAME")
    summary_lines.append("-"*100)
    summary_lines.append(f"{dataset_name}")
    summary_lines.append("")
    
    # 2. Dataset Summary
    summary_lines.append("-"*100)
    summary_lines.append("2. DATASET SUMMARY")
    summary_lines.append("-"*100)
    summary_lines.append("")
    summary_lines.append(f"Number of samples in dataset: {len(X_full)}")
    summary_lines.append(f"Number of features: {X_full.shape[1]}")
    summary_lines.append("")
    summary_lines.append("Number of unique values per feature:")
    for col in X_full.columns:
        n_unique = X_full[col].nunique()
        summary_lines.append(f"  • {col}: {n_unique} unique values")
    
    summary_lines.append("")
    summary_lines.append("Dataset description (target variable):")
    summary_lines.append(f"  • Mean: {y_full.mean():.4f}")
    summary_lines.append(f"  • Median: {y_full.median():.4f}")
    summary_lines.append(f"  • Min: {y_full.min():.4f}")
    summary_lines.append(f"  • Max: {y_full.max():.4f}")
    summary_lines.append(f"  • Std: {y_full.std():.4f}")
    summary_lines.append("")
    
    summary_lines.append("Feature statistics:")
    summary_lines.append(X_full.describe().to_string())
    summary_lines.append("")
    
    # 3. Preprocessing Steps
    summary_lines.append("-"*100)
    summary_lines.append("3. PREPROCESSING STEPS")
    summary_lines.append("-"*100)
    summary_lines.append("Applied preprocessing:")
    summary_lines.append("  1. Train-test split (80/20)")
    summary_lines.append("  2. StandardScaler normalization (zero mean, unit variance)")
    summary_lines.append("  3. Linear Regression assumes linear relationship between features and target")
    summary_lines.append("")
    
    # Train model
    summary_lines.append("-"*100)
    summary_lines.append("4. LINEAR REGRESSION MODEL RESULTS")
    summary_lines.append("-"*100)
    summary_lines.append("")
    
    result, model = train_linear_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, dataset_name
    )
    
    # Display results
    print("\n  Model Performance:")
    header = f"{'Metric':<25} {'Value':<20}"
    print("  " + header)
    print("  " + "-"*50)
    
    summary_lines.append("Model Performance Metrics:")
    summary_lines.append(f"{'Metric':<25} {'Value':<20}")
    summary_lines.append("-"*50)
    
    metrics = [
        ('Train R² Score', f"{result['Train R²']:.4f}"),
        ('Test R² Score', f"{result['Test R²']:.4f}"),
        ('RMSE', f"{result['Test RMSE']:.4f}"),
        ('MAE', f"{result['Test MAE']:.4f}"),
        ('Overfitting', f"{result['Overfitting']:.4f}"),
        ('Grade', result['Grade'])
    ]
    
    for metric_name, metric_value in metrics:
        line = f"{metric_name:<25} {metric_value:<20}"
        print("  " + line)
        summary_lines.append(line)
    
    summary_lines.append("")
    summary_lines.append("Model Coefficients (weights for each feature):")
    summary_lines.append(f"  Intercept: {result['Intercept']:.4f}")
    for i, (feature, coef) in enumerate(zip(X_full.columns, result['Coefficients'])):
        summary_lines.append(f"  {feature}: {coef:.4f}")
    summary_lines.append("")
    
    # Interpretation
    summary_lines.append("Coefficient Interpretation:")
    summary_lines.append("  Positive coefficients: Feature increases → Target increases")
    summary_lines.append("  Negative coefficients: Feature increases → Target decreases")
    summary_lines.append("  Larger absolute value: Stronger influence on target")
    summary_lines.append("")
    
    # Most important features
    coef_abs = np.abs(result['Coefficients'])
    sorted_idx = np.argsort(coef_abs)[::-1]
    summary_lines.append("Features ranked by importance (absolute coefficient value):")
    for rank, idx in enumerate(sorted_idx, 1):
        feature = X_full.columns[idx]
        coef = result['Coefficients'][idx]
        summary_lines.append(f"  {rank}. {feature}: {coef:.4f}")
    summary_lines.append("")
    
    # Save model
    print(f"\n  Saving model to {dataset_dir}...")
    model_filename = "LinearRegression_model.pkl"
    with open(os.path.join(dataset_dir, model_filename), 'wb') as f:
        pickle.dump(model, f)
    print(f"    ✓ Saved: {model_filename}")
    
    # Add saved files section to summary
    summary_lines.append("-"*100)
    summary_lines.append("5. SAVED FILES AND LOCATIONS")
    summary_lines.append("-"*100)
    summary_lines.append("")
    summary_lines.append(f"All files saved to directory: {os.path.abspath(dataset_dir)}")
    summary_lines.append("")
    summary_lines.append("Dataset files:")
    summary_lines.append("  • X_data.csv - Feature data (all samples)")
    summary_lines.append("  • y_data.csv - Target variable (all samples)")
    summary_lines.append("  • scaler.pkl - StandardScaler for feature normalization")
    summary_lines.append("")
    summary_lines.append("Trained model:")
    summary_lines.append(f"  • {model_filename} - Linear Regression model")
    summary_lines.append("")
    summary_lines.append("Summary file:")
    summary_lines.append("  • ANALYSIS_SUMMARY.txt - This file")
    summary_lines.append("")
    
    # Final assessment
    summary_lines.append("="*100)
    summary_lines.append(f"FINAL ASSESSMENT")
    summary_lines.append(f"Test R²: {result['Test R²']:.4f}")
    summary_lines.append(f"Grade: *** {result['Grade']} ***")
    summary_lines.append("="*100)
    
    print("\n  " + "="*100)
    print(f"  FINAL ASSESSMENT")
    print(f"  Test R²: {result['Test R²']:.4f}")
    print(f"  Grade: *** {result['Grade']} ***")
    print("  " + "="*100)
    
    # Save summary
    summary_path = os.path.join(dataset_dir, "ANALYSIS_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\n  ✓ Saved summary: ANALYSIS_SUMMARY.txt")
    
    return result


def main():
    """Run linear regression analysis on all datasets"""
    
    print("\n" + "#"*100)
    print("# COMPREHENSIVE LINEAR REGRESSION ANALYSIS")
    print("# Real-world datasets with strong linear relationships")
    print("# Analyzing practical examples ideal for linear regression")
    print("#"*100)
    
    # Create base directory
    base_dir = "linear_regression_results"
    os.makedirs(base_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {os.path.abspath(base_dir)}")
    
    datasets = [
        (AdvertisingExample, "Advertising Sales"),
        (BostonHousingExample, "Boston Housing"),
        (StudentPerformanceExample, "Student Performance"),
        (InsuranceCostExample, "Medical Insurance Cost"),
        (RealEstateValuationExample, "Real Estate Valuation"),
        (FishMarketExample, "Fish Market Weight"),
        (YachtHydrodynamicsExample, "Yacht Hydrodynamics"),
        (AirfoilSelfNoiseExample, "Airfoil Self-Noise"),
        (WineQualityRedExample, "Wine Quality Red"),
        (ENBEnergyEfficiencyExample, "Energy Efficiency")
    ]
    
    all_results = {}
    
    for example_class, dataset_name in datasets:
        try:
            result = analyze_and_save_dataset(example_class, dataset_name, base_dir)
            all_results[dataset_name] = result
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    # Create final summary
    print("\n" + "#"*100)
    print("# FINAL SUMMARY - ALL DATASETS")
    print("#"*100)
    print()
    
    summary_lines = []
    summary_lines.append("="*100)
    summary_lines.append("COMPREHENSIVE LINEAR REGRESSION ANALYSIS - FINAL SUMMARY")
    summary_lines.append("="*100)
    summary_lines.append("")
    summary_lines.append(f"{'Dataset':<35} {'Test R²':<12} {'RMSE':<12} {'Grade':<50}")
    summary_lines.append("-"*100)
    
    print(f"{'Dataset':<35} {'Test R²':<12} {'RMSE':<12} {'Grade':<50}")
    print("-"*100)
    
    for dataset_name, result in all_results.items():
        line = f"{dataset_name:<35} {result['Test R²']:<12.4f} {result['Test RMSE']:<12.4f} {result['Grade']:<50}"
        print(line)
        summary_lines.append(line)
    
    summary_lines.append("")
    summary_lines.append("="*100)
    summary_lines.append("Analysis Insights:")
    summary_lines.append("="*100)
    summary_lines.append("")
    summary_lines.append("Linear Regression Performance:")
    summary_lines.append("  • These datasets were selected for their known linear relationships")
    summary_lines.append("  • R² score indicates how well the linear model explains variance")
    summary_lines.append("  • High R² (>0.80) suggests strong linear relationships")
    summary_lines.append("  • Lower R² may indicate non-linear patterns or high variance")
    summary_lines.append("")
    summary_lines.append("Interpretation:")
    summary_lines.append("  • Linear Regression is interpretable - coefficients show feature importance")
    summary_lines.append("  • Positive coefficient: increase in feature → increase in target")
    summary_lines.append("  • Negative coefficient: increase in feature → decrease in target")
    summary_lines.append("  • Works best when relationship between features and target is approximately linear")
    summary_lines.append("")
    
    # Save final summary
    final_summary_path = os.path.join(base_dir, "FINAL_SUMMARY.txt")
    with open(final_summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print()
    print("="*100)
    print("✅ ANALYSIS COMPLETE")
    print(f"📁 All results saved to: {os.path.abspath(base_dir)}")
    print(f"📄 Final summary: {final_summary_path}")
    print("="*100)
    
    # Print directory structure
    print("\n📂 Directory Structure:")
    for dataset_name in all_results.keys():
        dir_name = dataset_name.replace(" ", "_")
        print(f"  {base_dir}/{dir_name}/")
        print(f"    ├── X_data.csv")
        print(f"    ├── y_data.csv")
        print(f"    ├── scaler.pkl")
        print(f"    ├── LinearRegression_model.pkl")
        print(f"    └── ANALYSIS_SUMMARY.txt")
        print()


if __name__ == "__main__":
    main()
