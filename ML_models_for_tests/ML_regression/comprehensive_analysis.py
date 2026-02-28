"""
Comprehensive ML Regression Analysis - All Models
Organizes datasets, trains all models (traditional + neural networks), and saves everything
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
    TENSORFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Import dataset loaders
from regression_examples import (
    CaliforniaHousingExample,
    DiabetesExample,
    WineQualityExample,
    AmesHousingExample,
    AutoMPGExample,
    ConcreteStrengthExample,
    EnergyEfficiencyExample,
    BikeSharingExample
)


def create_shallow_nn(input_dim):
    """
    Shallow Neural Network - 2 hidden layers
    Architecture: 64 → 32 → 1
    Activation: ReLU
    Regularization: Dropout (0.2)
    Optimizer: Adam
    Best for: Small datasets, fast training
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_medium_nn(input_dim):
    """
    Medium Neural Network - 4 hidden layers
    Architecture: 128 → 64 → 32 → 16 → 1
    Activation: ReLU
    Regularization: Dropout (0.3, 0.3, 0.2)
    Optimizer: Adam
    Best for: Medium datasets, balanced complexity
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_deep_nn(input_dim):
    """
    Deep Neural Network - 6 hidden layers
    Architecture: 256 → 128 → 64 → 32 → 16 → 8 → 1
    Activation: ReLU
    Regularization: BatchNormalization + Dropout (0.3, 0.3, 0.2, 0.2)
    Optimizer: Adam
    Best for: Large datasets, complex patterns
    """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_wide_nn(input_dim):
    """
    Wide Neural Network - 3 hidden layers with many neurons
    Architecture: 512 → 256 → 128 → 1
    Activation: ReLU
    Regularization: Dropout (0.4, 0.3, 0.2)
    Optimizer: Adam
    Best for: Learning complex feature interactions
    """
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_dim=input_dim),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_regularized_nn(input_dim):
    """
    Regularized Neural Network - L2 regularization
    Architecture: 128 → 64 → 32 → 1
    Activation: ReLU
    Regularization: L2 (0.001) + Dropout (0.3, 0.2)
    Optimizer: Adam
    Best for: Preventing overfitting
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_residual_nn(input_dim):
    """
    Residual Neural Network - Skip connections
    Architecture: 128 → [128 + residual] → 64 → 32 → 1
    Activation: ReLU
    Regularization: BatchNormalization + Dropout (0.3)
    Optimizer: Adam
    Best for: Deep learning with gradient flow
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # First block
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Second block with residual
    x2 = layers.Dense(128, activation='relu')(x)
    x2 = layers.BatchNormalization()(x2)
    x = layers.Add()([x, x2])  # Skip connection
    x = layers.Dropout(0.3)(x)
    
    # Output layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def grade_model(r2_score):
    """Grade model performance"""
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


def train_traditional_models(X_train, X_test, y_train, y_test, dataset_name):
    """Train all traditional ML models"""
    results = []
    models_dict = {}
    
    print(f"\n  Training traditional ML models...")
    
    # Ridge Regression
    print(f"    • Ridge Regression...", end=" ")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        'Model': 'Ridge Regression',
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
        'Grade': grade_model(r2_score(y_test, y_pred_test))
    })
    models_dict['Ridge'] = model
    print("✓")
    
    # Lasso Regression
    print(f"    • Lasso Regression...", end=" ")
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        'Model': 'Lasso Regression',
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
        'Grade': grade_model(r2_score(y_test, y_pred_test))
    })
    models_dict['Lasso'] = model
    print("✓")
    
    # ElasticNet
    print(f"    • ElasticNet...", end=" ")
    model = ElasticNet(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        'Model': 'ElasticNet',
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
        'Grade': grade_model(r2_score(y_test, y_pred_test))
    })
    models_dict['ElasticNet'] = model
    print("✓")
    
    # Random Forest
    print(f"    • Random Forest...", end=" ")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        'Model': 'Random Forest',
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
        'Grade': grade_model(r2_score(y_test, y_pred_test))
    })
    models_dict['RandomForest'] = model
    print("✓")
    
    # Gradient Boosting
    print(f"    • Gradient Boosting...", end=" ")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    results.append({
        'Model': 'Gradient Boosting',
        'Train R²': r2_score(y_train, y_pred_train),
        'Test R²': r2_score(y_test, y_pred_test),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
        'Grade': grade_model(r2_score(y_test, y_pred_test))
    })
    models_dict['GradientBoosting'] = model
    print("✓")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print(f"    • XGBoost...", end=" ")
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        results.append({
            'Model': 'XGBoost',
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Overfitting': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
            'Grade': grade_model(r2_score(y_test, y_pred_test))
        })
        models_dict['XGBoost'] = model
        print("✓")
    
    return results, models_dict


def train_neural_network(X_train, X_test, y_train, y_test, dataset_size):
    """Train multiple neural network architectures"""
    if not TENSORFLOW_AVAILABLE:
        return [], {}
    
    print(f"\n  Neural Networks (6 architectures):")
    
    input_dim = X_train.shape[1]
    
    # Define all architectures with descriptions
    nn_architectures = [
        ('NN_Shallow', create_shallow_nn, '2 layers (64→32), Dropout 0.2'),
        ('NN_Medium', create_medium_nn, '4 layers (128→64→32→16), Dropout 0.2-0.3'),
        ('NN_Deep', create_deep_nn, '6 layers (256→128→64→32→16→8), BatchNorm + Dropout'),
        ('NN_Wide', create_wide_nn, '3 wide layers (512→256→128), Dropout 0.2-0.4'),
        ('NN_Regularized', create_regularized_nn, 'L2 reg (128→64→32), L2=0.001 + Dropout'),
        ('NN_Residual', create_residual_nn, 'Skip connections (128+res→64→32), BatchNorm'),
    ]
    
    results = []
    models_dict = {}
    
    for model_name, create_model_func, description in nn_architectures:
        print(f"    • {model_name} ({description})...", end=" ")
        
        # Create model
        model = create_model_func(input_dim)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Predictions
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        epochs_trained = len(history.history['loss'])
        
        result = {
            'Model': f'{model_name} ({description})',
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Overfitting': train_r2 - test_r2,
            'Grade': grade_model(test_r2)
        }
        
        results.append(result)
        models_dict[model_name] = model
        
        print(f"✓ (epoch {epochs_trained}, R²={test_r2:.4f})")
    
    return results, models_dict


def analyze_and_save_dataset(example_class, dataset_name, base_dir="ML_regression_results"):
    """Complete analysis of dataset with all models and save everything"""
    
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
    summary_lines.append(f"COMPREHENSIVE ML ANALYSIS: {dataset_name}")
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
    summary_lines.append("  3. Same preprocessing applied to all models")
    summary_lines.append("")
    
    # Train all models
    summary_lines.append("-"*100)
    summary_lines.append("4. ML ALGORITHMS AND RESULTS")
    summary_lines.append("-"*100)
    summary_lines.append("")
    
    # Traditional models
    traditional_results, traditional_models = train_traditional_models(
        X_train_scaled, X_test_scaled, y_train, y_test, dataset_name
    )
    
    # Neural networks
    if TENSORFLOW_AVAILABLE:
        nn_results, nn_models = train_neural_network(
            X_train_scaled, X_test_scaled, y_train, y_test, len(X_train)
        )
        if nn_results:
            traditional_results.extend(nn_results)
            traditional_models.update(nn_models)
    
    # Create results DataFrame
    df_results = pd.DataFrame(traditional_results)
    
    # Display results
    print("\n  Model Performance:")
    header = f"{'Model':<35} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'MAE':<12} {'Overfit':<12} {'GRADE':<45}"
    print("  " + header)
    print("  " + "-"*100)
    
    summary_lines.append(header)
    summary_lines.append("-"*100)
    
    for idx, row in df_results.iterrows():
        line = f"{row['Model']:<35} {row['Train R²']:<12.4f} {row['Test R²']:<12.4f} " \
               f"{row['Test RMSE']:<12.4f} {row['Test MAE']:<12.4f} " \
               f"{row['Overfitting']:<12.4f} {row['Grade']:<45}"
        print("  " + line)
        summary_lines.append(line)
    
    summary_lines.append("")
    
    # Save models
    print(f"\n  Saving models to {dataset_dir}...")
    saved_files = []
    
    for model_name, model in traditional_models.items():
        if model_name.startswith('NN_') and TENSORFLOW_AVAILABLE:
            filename = f"{model_name}_model.h5"
            model.save(os.path.join(dataset_dir, filename))
            print(f"    ✓ Saved: {filename}")
            saved_files.append(filename)
        else:
            filename = f"{model_name}_model.pkl"
            with open(os.path.join(dataset_dir, filename), 'wb') as f:
                pickle.dump(model, f)
            print(f"    ✓ Saved: {filename}")
            saved_files.append(filename)
    
    # Best model
    best_idx = df_results['Test R²'].idxmax()
    best_model = df_results.loc[best_idx]
    
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
    summary_lines.append("Trained models:")
    for filename in saved_files:
        summary_lines.append(f"  • {filename}")
    summary_lines.append("")
    summary_lines.append("Summary file:")
    summary_lines.append("  • ANALYSIS_SUMMARY.txt - This file")
    summary_lines.append("")
    
    summary_lines.append("="*100)
    summary_lines.append(f"🏆 BEST MODEL: {best_model['Model']}")
    summary_lines.append(f"   Test R²: {best_model['Test R²']:.4f}")
    summary_lines.append(f"   Grade: *** {best_model['Grade']} ***")
    summary_lines.append("="*100)
    
    print("\n  " + "="*100)
    print(f"  🏆 BEST MODEL: {best_model['Model']}")
    print(f"     Test R²: {best_model['Test R²']:.4f}")
    print(f"     Grade: *** {best_model['Grade']} ***")
    print("  " + "="*100)
    
    # Save summary
    summary_path = os.path.join(dataset_dir, "ANALYSIS_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\n  ✓ Saved summary: ANALYSIS_SUMMARY.txt")
    
    return df_results, best_model


def main():
    """Run comprehensive analysis on all datasets"""
    
    print("\n" + "#"*100)
    print("# COMPREHENSIVE ML REGRESSION ANALYSIS - ALL MODELS")
    print("# Traditional ML + Neural Networks")
    print("# Organizing by dataset, saving all models and summaries")
    print("#"*100)
    
    if not TENSORFLOW_AVAILABLE:
        print("\n⚠ TensorFlow not available. Neural networks will be skipped.")
    if not XGBOOST_AVAILABLE:
        print("\n⚠ XGBoost not available. XGBoost models will be skipped.")
    
    # Create base directory
    base_dir = "ML_regression_results"
    os.makedirs(base_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {os.path.abspath(base_dir)}")
    
    datasets = [
        (CaliforniaHousingExample, "California Housing"),
        (DiabetesExample, "Diabetes"),
        (WineQualityExample, "Wine Quality"),
        (AmesHousingExample, "Ames Housing"),
        (AutoMPGExample, "Auto MPG"),
        (ConcreteStrengthExample, "Concrete Strength"),
        (EnergyEfficiencyExample, "Energy Efficiency"),
        (BikeSharingExample, "Bike Sharing")
    ]
    
    all_results = {}
    all_best_models = {}
    
    for example_class, dataset_name in datasets:
        try:
            results, best_model = analyze_and_save_dataset(example_class, dataset_name, base_dir)
            all_results[dataset_name] = results
            all_best_models[dataset_name] = best_model
        except Exception as e:
            print(f"\n⚠ Error analyzing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final comprehensive summary
    print("\n" + "#"*100)
    print("# FINAL SUMMARY - BEST MODEL PER DATASET")
    print("#"*100)
    
    final_summary = []
    final_summary.append("#"*100)
    final_summary.append("FINAL COMPREHENSIVE SUMMARY - ALL 8 DATASETS")
    final_summary.append("#"*100)
    final_summary.append("")
    
    header = f"{'Dataset':<30} {'Best Model':<35} {'Test R²':<12} {'Grade':<45}"
    print(f"\n{header}")
    print("-"*100)
    final_summary.append(header)
    final_summary.append("-"*100)
    
    for dataset_name, best_model in all_best_models.items():
        line = f"{dataset_name:<30} {best_model['Model']:<35} {best_model['Test R²']:<12.4f} {best_model['Grade']:<45}"
        print(line)
        final_summary.append(line)
    
    final_summary.append("")
    final_summary.append("="*100)
    final_summary.append("ANALYSIS COMPLETE")
    final_summary.append(f"Results saved to: {os.path.abspath(base_dir)}")
    final_summary.append("="*100)
    
    # Save final summary
    final_summary_path = os.path.join(base_dir, "FINAL_SUMMARY.txt")
    with open(final_summary_path, 'w') as f:
        f.write('\n'.join(final_summary))
    
    print("\n" + "="*100)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"📁 All results saved to: {os.path.abspath(base_dir)}")
    print(f"📄 Final summary: {final_summary_path}")
    print("="*100)
    
    # Print directory structure
    print("\n📂 Directory Structure:")
    for dataset_name in all_results.keys():
        dataset_dir = dataset_name.replace(" ", "_")
        print(f"  {base_dir}/{dataset_dir}/")
        print(f"    ├── X_data.csv")
        print(f"    ├── y_data.csv")
        print(f"    ├── scaler.pkl")
        print(f"    ├── Ridge_model.pkl")
        print(f"    ├── Lasso_model.pkl")
        print(f"    ├── ElasticNet_model.pkl")
        print(f"    ├── RandomForest_model.pkl")
        print(f"    ├── GradientBoosting_model.pkl")
        if XGBOOST_AVAILABLE:
            print(f"    ├── XGBoost_model.pkl")
        if TENSORFLOW_AVAILABLE:
            print(f"    ├── NeuralNetwork_model.h5")
        print(f"    └── ANALYSIS_SUMMARY.txt")
        print()


if __name__ == "__main__":
    main()
