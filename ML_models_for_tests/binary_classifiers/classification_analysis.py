"""
Comprehensive Binary Classification Analysis Script

This script performs systematic analysis of multiple binary classification datasets
using various models including traditional ML and neural networks.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
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
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Import dataset classes
from classification_examples import (
    BreastCancerExample,
    HeartDiseaseExample,
    DiabetesClassificationExample,
    BankMarketingExample,
    CreditCardFraudExample,
    SpamDetectionExample,
    IonosphereExample,
    AdultIncomeExample
)

# Check for XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available")

# Check for Neural Networks
try:
    from sklearn.neural_network import MLPClassifier
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    print("⚠ Neural Networks not available")

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠ TensorFlow not available")


def grade_model(accuracy, f1_score):
    """
    Grade the model based on accuracy and F1 score.
    
    Args:
        accuracy: Test accuracy
        f1_score: Test F1 score
    
    Returns:
        String grade
    """
    # Average of accuracy and F1 for overall performance
    avg_score = (accuracy + f1_score) / 2
    
    if avg_score >= 0.95:
        return "EXCELLENT"
    elif avg_score >= 0.90:
        return "VERY GOOD"
    elif avg_score >= 0.85:
        return "GOOD"
    elif avg_score >= 0.75:
        return "ACCEPTABLE"
    elif avg_score >= 0.65:
        return "POOR"
    else:
        return "VERY POOR"


def train_single_model(X_train, X_test, y_train, y_test, model_type, model_name, random_state=42):
    """
    Train a single model and return results.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        model_type: Type of model to train
        model_name: Display name for the model
        random_state: Random seed
    
    Returns:
        Dictionary with model results
    """
    print(f"\n  Training {model_name}...")
    
    # Create model based on type
    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        needs_scaling = True
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        needs_scaling = False
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        needs_scaling = False
    elif model_type == 'svm':
        model = SVC(random_state=random_state, probability=True)
        needs_scaling = True
    elif model_type == 'naive_bayes':
        model = GaussianNB()
        needs_scaling = False
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state)
        needs_scaling = False
    elif model_type == 'xgb' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        needs_scaling = False
    elif model_type == 'nn_small' and NEURAL_NET_AVAILABLE:
        model = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        needs_scaling = True
    elif model_type == 'nn_medium' and NEURAL_NET_AVAILABLE:
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        needs_scaling = True
    elif model_type == 'nn_large' and NEURAL_NET_AVAILABLE:
        model = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        needs_scaling = True
    elif model_type == 'tensorflow_nn' and TENSORFLOW_AVAILABLE:
        # TensorFlow neural network for DiCE gradient compatibility
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
        
        # Build TensorFlow model
        input_dim = X_train.shape[1]
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Train model with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print(f"    Training TensorFlow model...")
        model.fit(
            X_train_processed, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Make predictions
        y_pred_train = (model.predict(X_train_processed, verbose=0) > 0.5).astype(int).flatten()
        y_pred_test = (model.predict(X_test_processed, verbose=0) > 0.5).astype(int).flatten()
        
        # Get probability predictions
        y_prob_train = model.predict(X_train_processed, verbose=0).flatten()
        y_prob_test = model.predict(X_test_processed, verbose=0).flatten()
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'model': model,
            'scaler': scaler,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_prob_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_prob_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }
        
        # Calculate overfitting measure
        results['overfit_accuracy'] = results['train_accuracy'] - results['test_accuracy']
        results['overfit_f1'] = results['train_f1'] - results['test_f1']
        
        # Grade the model
        results['grade'] = grade_model(results['test_accuracy'], results['test_f1'])
        
        print(f"    Test Accuracy: {results['test_accuracy']:.4f} | F1: {results['test_f1']:.4f} | Grade: {results['grade']}")
        
        return results
    else:
        return None
    
    # Scale data if needed
    if needs_scaling:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        scaler = None
        X_train_processed = X_train
        X_test_processed = X_test
    
    # Train model
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_processed)
    y_pred_test = model.predict(X_test_processed)
    
    # Get probability predictions for ROC AUC
    try:
        y_prob_train = model.predict_proba(X_train_processed)[:, 1]
        y_prob_test = model.predict_proba(X_test_processed)[:, 1]
    except:
        y_prob_train = y_pred_train
        y_prob_test = y_pred_test
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'model': model,
        'scaler': scaler,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
        'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
        'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
        'train_roc_auc': roc_auc_score(y_train, y_prob_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test, y_prob_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }
    
    # Calculate overfitting measure
    results['overfit_accuracy'] = results['train_accuracy'] - results['test_accuracy']
    results['overfit_f1'] = results['train_f1'] - results['test_f1']
    
    # Grade the model
    results['grade'] = grade_model(results['test_accuracy'], results['test_f1'])
    
    print(f"    Test Accuracy: {results['test_accuracy']:.4f} | F1: {results['test_f1']:.4f} | Grade: {results['grade']}")
    
    return results


def analyze_and_save_dataset(example_class, dataset_name, base_dir="binary_classification_results"):
    """
    Perform comprehensive analysis on a dataset with multiple models.
    
    Args:
        example_class: Dataset class to instantiate
        dataset_name: Name of the dataset
        base_dir: Base directory for saving results
    
    Returns:
        Tuple of (results_dict, best_model_name)
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*80}")
    
    # Load data
    example = example_class()
    example.load_data()
    
    X_full = example.X_full
    y_full = example.y_full
    X_train = example.X_train
    X_test = example.X_test
    y_train = example.y_train
    y_test = example.y_test
    
    # Create directory for this dataset
    dataset_dir = os.path.join(base_dir, dataset_name.replace(' ', '_'))
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save raw data
    print(f"\n💾 Saving data files...")
    X_full.to_csv(os.path.join(dataset_dir, 'X_data.csv'), index=False)
    pd.Series(y_full, name='target').to_csv(os.path.join(dataset_dir, 'y_data.csv'), index=False)
    
    # Define models to train
    models_to_train = [
        ('logistic', 'Logistic Regression'),
        ('rf', 'Random Forest'),
        ('gbm', 'Gradient Boosting'),
        ('svm', 'Support Vector Machine'),
        ('naive_bayes', 'Naive Bayes'),
        ('decision_tree', 'Decision Tree'),
    ]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_to_train.append(('xgb', 'XGBoost'))
    
    # Add Neural Networks if available
    if NEURAL_NET_AVAILABLE:
        models_to_train.extend([
            ('nn_small', 'Neural Network (Small)'),
            ('nn_medium', 'Neural Network (Medium)'),
            ('nn_large', 'Neural Network (Large)')
        ])
    
    # Add TensorFlow if available
    if TENSORFLOW_AVAILABLE:
        models_to_train.append(('tensorflow_nn', 'TensorFlow Neural Network'))
    
    # Train all models
    print(f"\n🤖 Training {len(models_to_train)} models...")
    results = {}
    
    for model_type, model_name in models_to_train:
        result = train_single_model(X_train, X_test, y_train, y_test, model_type, model_name)
        if result is not None:
            results[model_name] = result
    
    # Find best model based on F1 score (more balanced than accuracy)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   Test F1 Score: {best_result['test_f1']:.4f}")
    print(f"   Test ROC AUC: {best_result['test_roc_auc']:.4f}")
    
    # Save all models
    print(f"\n💾 Saving models...")
    for model_name, result in results.items():
        # Check if TensorFlow model
        if result['model_type'] == 'tensorflow_nn':
            model_filename = f"{model_name.replace(' ', '_')}_model.keras"
            model_path = os.path.join(dataset_dir, model_filename)
            result['model'].save(model_path, save_format='keras')
        else:
            model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
            model_path = os.path.join(dataset_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save scaler if exists
        if result['scaler'] is not None:
            scaler_filename = f"{model_name.replace(' ', '_')}_scaler.pkl"
            scaler_path = os.path.join(dataset_dir, scaler_filename)
            with open(scaler_path, 'wb') as f:
                pickle.dump(result['scaler'], f)
    
    # Create comprehensive summary
    print(f"\n📊 Creating analysis summary...")
    
    summary_path = os.path.join(dataset_dir, 'ANALYSIS_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BINARY CLASSIFICATION ANALYSIS: {dataset_name}\n")
        f.write("="*80 + "\n")
        f.write(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset Information
        f.write(f"\n{'='*80}\n")
        f.write("DATASET INFORMATION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total Samples: {len(X_full)}\n")
        f.write(f"Number of Features: {X_full.shape[1]}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        
        # Class distribution
        class_counts = pd.Series(y_full).value_counts().sort_index()
        f.write(f"\nClass Distribution:\n")
        for class_label, count in class_counts.items():
            percentage = (count / len(y_full)) * 100
            f.write(f"  Class {class_label}: {count} samples ({percentage:.2f}%)\n")
        
        balance_ratio = class_counts.min() / class_counts.max()
        f.write(f"  Balance Ratio: {balance_ratio:.3f}\n")
        
        # Feature statistics
        f.write(f"\nFeature Statistics:\n")
        f.write(f"{'Feature':<25} {'Unique Values':<15} {'Mean':<12} {'Std':<12}\n")
        f.write("-" * 80 + "\n")
        for col in X_full.columns:
            f.write(f"{col:<25} {X_full[col].nunique():<15} {X_full[col].mean():<12.4f} {X_full[col].std():<12.4f}\n")
        
        # Model Comparison
        f.write(f"\n{'='*80}\n")
        f.write("MODEL COMPARISON\n")
        f.write(f"{'='*80}\n")
        f.write(f"\n{'Model':<25} {'Grade':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10}\n")
        f.write("-" * 95 + "\n")
        
        # Sort by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
        
        for model_name, result in sorted_results:
            f.write(f"{model_name:<25} {result['grade']:<12} {result['test_accuracy']:<10.4f} "
                   f"{result['test_precision']:<10.4f} {result['test_recall']:<10.4f} "
                   f"{result['test_f1']:<10.4f} {result['test_roc_auc']:<10.4f}\n")
        
        # Best Model Details
        f.write(f"\n{'='*80}\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nTraining Performance:\n")
        f.write(f"  Accuracy:  {best_result['train_accuracy']:.4f}\n")
        f.write(f"  Precision: {best_result['train_precision']:.4f}\n")
        f.write(f"  Recall:    {best_result['train_recall']:.4f}\n")
        f.write(f"  F1 Score:  {best_result['train_f1']:.4f}\n")
        f.write(f"  ROC AUC:   {best_result['train_roc_auc']:.4f}\n")
        
        f.write(f"\nTest Performance:\n")
        f.write(f"  Accuracy:  {best_result['test_accuracy']:.4f}\n")
        f.write(f"  Precision: {best_result['test_precision']:.4f}\n")
        f.write(f"  Recall:    {best_result['test_recall']:.4f}\n")
        f.write(f"  F1 Score:  {best_result['test_f1']:.4f}\n")
        f.write(f"  ROC AUC:   {best_result['test_roc_auc']:.4f}\n")
        f.write(f"  Grade:     {best_result['grade']}\n")
        
        f.write(f"\nOverfitting Analysis:\n")
        f.write(f"  Accuracy Gap: {best_result['overfit_accuracy']:.4f}\n")
        f.write(f"  F1 Score Gap: {best_result['overfit_f1']:.4f}\n")
        if best_result['overfit_f1'] < 0.05:
            f.write(f"  Assessment: Excellent generalization ✓\n")
        elif best_result['overfit_f1'] < 0.10:
            f.write(f"  Assessment: Good generalization ✓\n")
        elif best_result['overfit_f1'] < 0.15:
            f.write(f"  Assessment: Moderate overfitting ⚠\n")
        else:
            f.write(f"  Assessment: Significant overfitting ✗\n")
        
        f.write(f"\nConfusion Matrix (Test Set):\n")
        cm = best_result['confusion_matrix']
        f.write(f"  [[TN={cm[0,0]:<6} FP={cm[0,1]:<6}]\n")
        f.write(f"   [FN={cm[1,0]:<6} TP={cm[1,1]:<6}]]\n")
        
        # Performance Interpretation
        f.write(f"\n{'='*80}\n")
        f.write("PERFORMANCE INTERPRETATION\n")
        f.write(f"{'='*80}\n")
        
        f.write(f"\nAccuracy: {best_result['test_accuracy']:.4f}\n")
        f.write(f"  - Percentage of correct predictions overall\n")
        f.write(f"  - Note: Can be misleading with imbalanced datasets\n")
        
        f.write(f"\nPrecision: {best_result['test_precision']:.4f}\n")
        f.write(f"  - Of predicted positives, how many are actually positive\n")
        f.write(f"  - Important when false positives are costly\n")
        f.write(f"  - Formula: TP / (TP + FP)\n")
        
        f.write(f"\nRecall (Sensitivity): {best_result['test_recall']:.4f}\n")
        f.write(f"  - Of actual positives, how many were correctly identified\n")
        f.write(f"  - Important when false negatives are costly\n")
        f.write(f"  - Formula: TP / (TP + FN)\n")
        
        f.write(f"\nF1 Score: {best_result['test_f1']:.4f}\n")
        f.write(f"  - Harmonic mean of precision and recall\n")
        f.write(f"  - Balanced measure, good for comparing models\n")
        f.write(f"  - Formula: 2 * (Precision * Recall) / (Precision + Recall)\n")
        
        f.write(f"\nROC AUC: {best_result['test_roc_auc']:.4f}\n")
        f.write(f"  - Area Under the Receiver Operating Characteristic Curve\n")
        f.write(f"  - Measures model's ability to distinguish between classes\n")
        f.write(f"  - Range: 0.5 (random) to 1.0 (perfect)\n")
        
        # Saved Files
        f.write(f"\n{'='*80}\n")
        f.write("SAVED FILES AND LOCATIONS\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nAll files saved to: {os.path.abspath(dataset_dir)}\n")
        f.write(f"\nData Files:\n")
        f.write(f"  - X_data.csv: Feature matrix ({X_full.shape[0]} samples × {X_full.shape[1]} features)\n")
        f.write(f"  - y_data.csv: Target labels ({len(y_full)} samples)\n")
        
        f.write(f"\nModel Files:\n")
        for model_name in results.keys():
            # Check model type for correct extension
            if results[model_name]['model_type'] == 'tensorflow_nn':
                model_filename = f"{model_name.replace(' ', '_')}_model.keras"
            else:
                model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
            f.write(f"  - {model_filename}: Trained {model_name} model\n")
            if results[model_name]['scaler'] is not None:
                scaler_filename = f"{model_name.replace(' ', '_')}_scaler.pkl"
                f.write(f"  - {scaler_filename}: StandardScaler for {model_name}\n")
        
        f.write(f"\nSummary Files:\n")
        f.write(f"  - ANALYSIS_SUMMARY.txt: This comprehensive analysis report\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("END OF ANALYSIS\n")
        f.write(f"{'='*80}\n")
    
    print(f"✅ Analysis complete for {dataset_name}")
    print(f"📁 Results saved to: {os.path.abspath(dataset_dir)}")
    
    return results, best_model_name


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("COMPREHENSIVE BINARY CLASSIFICATION ANALYSIS")
    print("="*80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create base directory
    base_dir = "binary_classification_results"
    os.makedirs(base_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {os.path.abspath(base_dir)}")
    
    datasets = [
        (BreastCancerExample, "Breast Cancer"),
        (HeartDiseaseExample, "Heart Disease"),
        (DiabetesClassificationExample, "Diabetes"),
        (BankMarketingExample, "Bank Marketing"),
        (CreditCardFraudExample, "Credit Card Fraud"),
        (SpamDetectionExample, "Spam Detection"),
        (IonosphereExample, "Ionosphere"),
        (AdultIncomeExample, "Adult Income")
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
            continue
    
    # Create final summary
    print(f"\n{'='*80}")
    print("Creating Final Summary...")
    print(f"{'='*80}")
    
    final_summary_path = os.path.join(base_dir, 'FINAL_SUMMARY.txt')
    with open(final_summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL SUMMARY - BINARY CLASSIFICATION ANALYSIS\n")
        f.write("="*80 + "\n")
        f.write(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Datasets Analyzed: {len(all_results)}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("BEST MODEL PER DATASET\n")
        f.write(f"{'='*80}\n")
        f.write(f"\n{'Dataset':<25} {'Best Model':<25} {'Test Accuracy':<15} {'Test F1':<15} {'Grade':<12}\n")
        f.write("-" * 95 + "\n")
        
        for dataset_name in all_results.keys():
            best_model = all_best_models[dataset_name]
            result = all_results[dataset_name][best_model]
            f.write(f"{dataset_name:<25} {best_model:<25} {result['test_accuracy']:<15.4f} "
                   f"{result['test_f1']:<15.4f} {result['grade']:<12}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("MODEL TYPE SUMMARY\n")
        f.write(f"{'='*80}\n")
        
        # Count how many times each model type was best
        model_type_counts = {}
        for best_model in all_best_models.values():
            model_type_counts[best_model] = model_type_counts.get(best_model, 0) + 1
        
        f.write(f"\nBest Model Frequency:\n")
        for model, count in sorted(model_type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model}: {count} dataset(s)\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("AVERAGE PERFORMANCE BY MODEL TYPE\n")
        f.write(f"{'='*80}\n")
        
        # Calculate average performance for each model type
        model_averages = {}
        for dataset_name, results in all_results.items():
            for model_name, result in results.items():
                if model_name not in model_averages:
                    model_averages[model_name] = {'accuracy': [], 'f1': [], 'roc_auc': []}
                model_averages[model_name]['accuracy'].append(result['test_accuracy'])
                model_averages[model_name]['f1'].append(result['test_f1'])
                model_averages[model_name]['roc_auc'].append(result['test_roc_auc'])
        
        f.write(f"\n{'Model':<25} {'Avg Accuracy':<15} {'Avg F1':<15} {'Avg ROC AUC':<15}\n")
        f.write("-" * 70 + "\n")
        
        for model_name in sorted(model_averages.keys()):
            avg_acc = np.mean(model_averages[model_name]['accuracy'])
            avg_f1 = np.mean(model_averages[model_name]['f1'])
            avg_auc = np.mean(model_averages[model_name]['roc_auc'])
            f.write(f"{model_name:<25} {avg_acc:<15.4f} {avg_f1:<15.4f} {avg_auc:<15.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("DATASET CHARACTERISTICS\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nKey Insights:\n")
        f.write(f"  - Total datasets analyzed: {len(all_results)}\n")
        f.write(f"  - Multiple models trained per dataset (6-12 depending on availability)\n")
        f.write(f"  - All models include proper train/test split with stratification\n")
        f.write(f"  - Performance metrics: Accuracy, Precision, Recall, F1, ROC AUC\n")
        f.write(f"  - Overfitting analysis included for all models\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("RECOMMENDATIONS\n")
        f.write(f"{'='*80}\n")
        f.write(f"\nGeneral Guidelines:\n")
        f.write(f"  1. For balanced datasets: Use accuracy as primary metric\n")
        f.write(f"  2. For imbalanced datasets: Use F1 score or ROC AUC\n")
        f.write(f"  3. When false positives are costly: Optimize precision\n")
        f.write(f"  4. When false negatives are costly: Optimize recall\n")
        f.write(f"  5. For model comparison: Use F1 score for balanced view\n")
        f.write(f"  6. Always check confusion matrix for detailed performance\n")
        f.write(f"  7. Monitor overfitting gap between train and test performance\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("END OF FINAL SUMMARY\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n✅ All analyses complete!")
    print(f"📊 Final summary saved to: {os.path.abspath(final_summary_path)}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
