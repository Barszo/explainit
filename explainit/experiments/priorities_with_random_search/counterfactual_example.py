"""
Counterfactual Explanation Example using German Credit Dataset

This script demonstrates how to:
1. Load and preprocess the German Credit dataset
2. Train a Logistic Regression model
3. Define a sample for analysis
4. Define preferences (actionability and desirability) for features
5. Generate counterfactual explanations using random search
"""

import logging
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings

from explainit.priorities.nonlinear import exponential
from explainit.explainers.random_search import RandomSearchExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def load_and_preprocess_data(data_path='data/german_credit_data.csv'):
    """
    Load and preprocess the German Credit dataset.
    
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
        scaler: Fitted MinMaxScaler
        encoder: Fitted OneHotEncoder
        feature_names: List of all feature names
    """
    logger.info("Loading German Credit dataset...")
    df = pd.read_csv(data_path)
    
    # Drop index column
    df = df.drop(columns=['Unnamed: 0'])
    
    # Define categorical and numerical columns
    cat_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    num_cols = ['Age', 'Credit amount', 'Duration']
    
    # Handle missing values in categorical columns
    df[cat_cols] = df[cat_cols].fillna('missing')
    
    # Encode target: 'good' = 1, 'bad' = 0
    df['target'] = (df['Risk'] == 'good').astype(int)
    df = df.drop(columns=['Risk'])
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale numerical features
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]), 
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )
    
    # Combine numerical and categorical features
    X = pd.concat([X[num_cols], X_cat], axis=1)
    
    # Set categorical columns as category dtype
    for col in encoder.get_feature_names_out(encoder.feature_names_in_):
        X[col] = X[col].astype('category')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X_train, X_test, y_train, y_test, scaler, encoder, list(X.columns)


def train_model(X_train, y_train, X_test, y_test):
    """
    Train a Logistic Regression classifier.
    
    Returns:
        model: Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    return model


def find_exemplar(model, X_test, target=0.60):
    """
    Find the exemplar - the sample in the dataset closest to the target prediction.
    
    Args:
        model: Trained classifier
        X_test: Test dataset
        target: Target prediction value
    
    Returns:
        exemplar: Sample closest to target
        exemplar_prediction: Prediction for the exemplar
        exemplar_index: Index of exemplar in X_test
    """
    logger.info(f"Finding exemplar closest to target={target}...")
    
    # Get predictions for all test samples (probability of class 1)
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Find sample closest to target
    distances = np.abs(predictions - target)
    exemplar_index = np.argmin(distances)
    
    exemplar = X_test.iloc[exemplar_index].values.tolist()
    exemplar_prediction = predictions[exemplar_index]
    
    logger.info(f"Found exemplar at index {exemplar_index}")
    logger.info(f"Exemplar prediction: {exemplar_prediction:.6f} (target: {target})")
    logger.info(f"Distance from target: {abs(exemplar_prediction - target):.6f}")
    
    return exemplar, exemplar_prediction, exemplar_index


def define_sample():
    """
    Define a sample instance for counterfactual analysis.
    
    Feature order:
    - Age, Credit amount, Duration (numerical)
    - Sex (2 categories)
    - Job (4 categories)
    - Housing (3 categories)
    - Saving accounts (5 categories)
    - Checking account (4 categories)
    - Purpose (8 categories)
    
    Returns:
        sample: List of feature values
    """
    sample = [
        0.69642857,                     # Age
        0.86359635,                     # Credit amount
        0.73529412,                     # Duration
        0., 1.,                         # Sex (female=0, male=1)
        0., 0., 1., 0.,                 # Job (0, 1, 2, 3)
        0., 0., 1.,                     # Housing (free, own, rent)
        1., 0., 0., 0., 0.,             # Saving accounts (little, missing, moderate, quite rich, rich)
        0., 0., 1., 0.,                 # Checking account (little, missing, moderate, rich)
        1., 0., 0., 0., 0., 0., 0., 0.  # Purpose (business, car, domestic appliances, education, furniture/equipment, radio/TV, repairs, vacation/others)
    ]
    return sample


def create_numerical_preference_function(sample_value, exemplar_value, min_val=0.0, max_val=1.0, exemplar_weight=0.5):
    """
    Create a preference function for a numerical feature.
    
    The function ensures:
    - f(x) = 1 when x >= sample_value
    - f(exemplar_value) = exemplar_weight
    - f(x) -> 0 as x approaches x0
    
    Args:
        sample_value: Value in the sample (weight = 1)
        exemplar_value: Value in the exemplar (weight = exemplar_weight)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        exemplar_weight: Weight assigned to exemplar value (default: 0.5)
    
    Returns:
        Preference function and x0 value
    """
    # For exponential function with a=5:
    # We want f(exemplar_value) = exemplar_weight
    # exponential: f(x) = 0 at x0, f(x) = 1 at x1
    # t = (x - x0) / (x1 - x0)
    # curve = (exp(5*t) - 1) / (exp(5) - 1)
    # For f = exemplar_weight: (exp(5*t) - 1) / (exp(5) - 1) = exemplar_weight
    # exp(5*t) = 1 + exemplar_weight * (exp(5) - 1)
    # 5*t = ln(1 + exemplar_weight * (exp(5) - 1))
    # t = ln(1 + exemplar_weight * (exp(5) - 1)) / 5
    
    a = 5
    t_target = np.log(1 + exemplar_weight * (np.exp(a) - 1)) / a
    
    # Now: t_target = (exemplar_value - x0) / (sample_value - x0)
    # exemplar_value - x0 = t_target * (sample_value - x0)
    # exemplar_value = x0 + t_target * sample_value - t_target * x0
    # exemplar_value = x0 * (1 - t_target) + t_target * sample_value
    # x0 = (exemplar_value - t_target * sample_value) / (1 - t_target)
    
    x0 = (exemplar_value - t_target * sample_value) / (1 - t_target)
    x1 = sample_value
    
    def preference_func(x):
        return exponential(x, x0=x0, x1=x1, increasing=True, a=a)
    
    return preference_func, x0


def define_preferences(sample, exemplar, X_train, exemplar_weight=0.5):
    """
    Automatically define preferences based on sample and exemplar.
    
    Preferences dictionary structure:
    - 'numerical': Maps feature index to:
        * None (unactionable - fixed at sample value)
        * {'function': f(x), 'min': min_val, 'max': max_val} (actionable with preference function)
    - 'categorical': Maps feature group indices to:
        * Dictionary mapping category combinations to weights (0 = forbidden, >0 = preference weight)
    
    Feature indices:
    0: Age (unactionable), 1: Credit amount, 2: Duration
    3-4: Sex, 5-8: Job, 9-11: Housing
    12-16: Saving accounts, 17-20: Checking account, 21-28: Purpose
    
    Args:
        sample: Sample to explain
        exemplar: Exemplar (sample closest to target)
        X_train: Training dataset to extract min/max values
        exemplar_weight: Weight assigned to exemplar value (default: 0.5)
    
    Returns:
        preferences: Dictionary defining preferences
    """
    # List of unactionable features
    unactionable_features = [0]  # Age
    
    # Define actionable numerical features (just the indices)
    actionable_numerical = [1, 2]  # Credit amount, Duration
    
    # Build numerical preferences
    numerical_preferences = {}
    
    # Add unactionable features
    for idx in unactionable_features:
        numerical_preferences[idx] = None
        logger.info(f"Feature {idx}: Unactionable (fixed at sample value)")
    
    # Add actionable features with automatically generated preference functions
    for idx in actionable_numerical:
        sample_val = sample[idx]
        exemplar_val = exemplar[idx]
        
        # Get actual min/max from dataset
        dataset_min = X_train.iloc[:, idx].min()
        dataset_max = X_train.iloc[:, idx].max()
        
        preference_func, x0_calculated = create_numerical_preference_function(
            sample_value=sample_val,
            exemplar_value=exemplar_val,
            min_val=dataset_min,
            max_val=dataset_max,
            exemplar_weight=exemplar_weight
        )
        
        # Determine acceptable min/max based on where weight function = 0
        # x0_calculated is where weight = 0
        # If dataset_min < x0_calculated, use x0_calculated as min (values below have weight 0)
        # If dataset_min >= x0_calculated, use dataset_min
        acceptable_min = max(dataset_min, x0_calculated)
        acceptable_max = dataset_max
        
        logger.info(f"Feature {idx}: Creating preference function")
        logger.info(f"  Dataset min: {dataset_min:.6f}, Dataset max: {dataset_max:.6f}")
        logger.info(f"  Sample value: {sample_val:.6f} (weight = 1.0)")
        logger.info(f"  Exemplar value: {exemplar_val:.6f} (weight = {exemplar_weight})")
        logger.info(f"  Calculated x0: {x0_calculated:.6f} (weight = 0.0)")
        logger.info(f"  Acceptable min: {acceptable_min:.6f}, Acceptable max: {acceptable_max:.6f}")
        logger.info(f"  Function: exponential(x, x0={x0_calculated:.6f}, x1={sample_val:.6f}, increasing=True, a=5)")
        
        numerical_preferences[idx] = {
            'function': preference_func,
            'min': acceptable_min,
            'max': acceptable_max
        }
    
    # Define categorical preferences
    preferences = {
        'numerical': numerical_preferences,
        'categorical': {
            # Sex (indices 3-4): prefer male
            (3, 4): {
                (1.0, 0.0): 0,      # female: forbidden
                (0.0, 1.0): 1       # male: allowed
            },
            # Job (indices 5-8): prefer skilled jobs (2, 3)
            (5, 6, 7, 8): {
                (1.0, 0.0, 0.0, 0.0): 0,    # unskilled non-resident: forbidden
                (0.0, 1.0, 0.0, 0.0): 0,    # unskilled resident: forbidden
                (0.0, 0.0, 1.0, 0.0): 1,    # skilled: allowed
                (0.0, 0.0, 0.0, 1.0): 1     # highly skilled: allowed
            },
            # Housing (indices 9-11): all allowed
            (9, 10, 11): {
                (1.0, 0.0, 0.0): 1,  # free: allowed
                (0.0, 1.0, 0.0): 1,  # own: allowed
                (0.0, 0.0, 1.0): 1   # rent: allowed
            },
            # Saving accounts (indices 12-16): forbid 'quite rich' and 'rich'
            (12, 13, 14, 15, 16): {
                (1.0, 0.0, 0.0, 0.0, 0.0): 1,  # little: allowed
                (0.0, 1.0, 0.0, 0.0, 0.0): 1,  # missing: allowed
                (0.0, 0.0, 1.0, 0.0, 0.0): 1,  # moderate: allowed
                (0.0, 0.0, 0.0, 1.0, 0.0): 0,  # quite rich: forbidden
                (0.0, 0.0, 0.0, 0.0, 1.0): 0   # rich: forbidden
            },
            # Checking account (indices 17-20): forbid 'rich'
            (17, 18, 19, 20): {
                (1.0, 0.0, 0.0, 0.0): 1,  # little: allowed
                (0.0, 1.0, 0.0, 0.0): 1,  # missing: allowed
                (0.0, 0.0, 1.0, 0.0): 1,  # moderate: allowed
                (0.0, 0.0, 0.0, 1.0): 0   # rich: forbidden
            },
            # Purpose (indices 21-28): all allowed
            (21, 22, 23, 24, 25, 26, 27, 28): {
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 1,  # business
                (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0): 1,  # car
                (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0): 1,  # domestic appliances
                (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0): 1,  # education
                (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0): 1,  # furniture/equipment
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0): 1,  # radio/TV
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0): 1,  # repairs
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0): 1   # vacation/others
            }
        }
    }
    
    return preferences


def display_preference_breakdown(explainer, cf_sample, cf_index):
    """
    Display detailed breakdown of preference contributions for a counterfactual.
    
    Args:
        explainer: RandomSearchExplainer instance
        cf_sample: Counterfactual sample
        cf_index: Index of the counterfactual (for display)
    """
    breakdown = explainer.get_preference_breakdown(cf_sample)
    
    logger.info(f"\n  Counterfactual #{cf_index} - Preference Breakdown:")
    logger.info(f"  {'='*70}")
    
    # Display numerical features
    if breakdown['numerical']:
        logger.info(f"  Numerical Features:")
        for idx, info in breakdown['numerical'].items():
            if info.get('actionable', True):
                logger.info(f"    Feature {idx}: value={info['value']:.6f}, weight={info['weight']:.6f}")
            else:
                logger.info(f"    Feature {idx}: value={info['value']:.6f}, weight=N/A (unactionable - fixed)")
    
    # Display categorical features
    if breakdown['categorical']:
        logger.info(f"  Categorical Features:")
        for group_indices, info in breakdown['categorical'].items():
            combo_str = ', '.join([f"{v:.1f}" for v in info['combination']])
            logger.info(f"    Group {group_indices}: combination=({combo_str}), weight={info['weight']:.6f}")
    
    # Calculate and display the product
    actionable_weights = []
    for idx, info in breakdown['numerical'].items():
        if info.get('actionable', True):
            actionable_weights.append(info['weight'])
    for group_indices, info in breakdown['categorical'].items():
        actionable_weights.append(info['weight'])
    
    logger.info(f"  \n  Preference Calculation:")
    weight_str = ' + '.join([f"{w:.6f}" for w in actionable_weights])
    logger.info(f"    {weight_str} = {breakdown['overall']:.6f}")
    logger.info(f"  Overall Preference Weight (Sum): {breakdown['overall']:.6f}")
    logger.info(f"  {'='*70}")


def generate_counterfactuals(model, sample, preferences, target=0.60, 
                            n_samples=10000, epsilon=0.05, use_monte_carlo=True, return_top_n=None, exemplar=None):
    """
    Generate counterfactual explanations using random search.
    
    Args:
        model: Trained classifier
        sample: Instance to explain
        preferences: Preference dictionary
        target: Target prediction value (default: 0.60 for 60% probability of class 1)
        n_samples: Number of random samples to generate
        epsilon: Acceptable deviation from target
        use_monte_carlo: Whether to use Monte Carlo sampling
        return_top_n: If specified, return only top N most preferable counterfactuals
        exemplar: Optional exemplar sample to display on preference plots
    
    Returns:
        samples: List of counterfactual samples
        predictions: List of corresponding predictions
        preference_scores: List of preference scores for each sample
        explainer: RandomSearchExplainer instance
    """
    logger.info("Initializing counterfactual explainer...")
    
    # Define model prediction function (probability of class 1)
    model_pred = lambda x: model.predict_proba(x)[:, 1]
    
    # Initialize explainer
    explainer = RandomSearchExplainer(
        model_pred=model_pred,
        priorities=preferences,
        sample=sample,
        target=target
    )
    
    # Display preferences
    logger.info("Displaying preference functions...")
    logger.info("Saving preference plots to 'images/' directory...")
    explainer.display_priorities(exemplar=exemplar)
    logger.info("Preference plots saved.")
    
    # Investigate probability distribution
    logger.info("Investigating probability distribution...")
    logger.info("Saving probability distribution plots to 'images/' directory...")
    explainer.investigate_probability_distribution()
    logger.info("Probability distribution plots saved.")
    
    # Generate counterfactual samples
    logger.info(f"Generating {n_samples} counterfactual samples...")
    samples, predictions, preference_scores = explainer.generate_random_samples(
        n_samples=n_samples,
        epsilon=epsilon,
        use_monte_carlo=use_monte_carlo,
        random_seed=42,
        max_tries=100,
        return_top_n=return_top_n
    )
    
    logger.info(f"Found {len(samples)} counterfactuals within epsilon={epsilon}")
    
    return samples, predictions, preference_scores, explainer


def save_parameters_csv(X_train, preferences, filename='experiment_parameters.csv'):
    """
    Save experiment parameters and feature statistics to CSV.
    Each categorical feature gets its own row.
    
    Args:
        X_train: Training dataset
        preferences: Preference dictionary
        filename: Output CSV filename
    """
    logger.info(f"Saving parameters to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'feature_id', 'feature_type', 'grouped_with',
            'dataset_min', 'dataset_max', 'dataset_mean', 'dataset_median', 'dataset_std',
            'count_value_1', 'count_value_0',
            'priority_type', 'priority_min', 'priority_max', 
            'priority_function', 'priority_function_params',
            'allowed_value', 'category_weight'
        ])
        
        # Numerical features
        for idx, constraint in preferences['numerical'].items():
            row = [idx, 'numerical', '']
            
            # Dataset statistics
            feature_values = X_train.iloc[:, idx]
            row.extend([
                feature_values.min(),
                feature_values.max(),
                feature_values.mean(),
                feature_values.median(),
                feature_values.std()
            ])
            
            # Categories (N/A for numerical)
            row.extend(['', ''])
            
            # Priority information
            if constraint is None:
                row.extend(['unactionable', '', '', '', '', '', ''])
            elif isinstance(constraint, dict) and 'function' in constraint:
                row.extend([
                    'actionable',
                    constraint['min'],
                    constraint['max'],
                    'exponential',
                    f"a=5, increasing=True",
                    '',
                    ''
                ])
            else:
                row.extend(['fixed', constraint, '', '', '', '', ''])
            
            writer.writerow(row)
        
        # Categorical features - split into individual rows
        for group_indices, possible_values in preferences['categorical'].items():
            group_indices = list(group_indices)
            
            # Process each feature in the group separately
            for feature_idx in group_indices:
                # Other features in the group
                other_features = [idx for idx in group_indices if idx != feature_idx]
                grouped_with = ','.join(map(str, other_features)) if other_features else ''
                
                row = [feature_idx, 'categorical', grouped_with]
                
                # Dataset statistics (N/A for categorical)
                row.extend(['', '', '', '', ''])
                
                # Count how many samples have 1 vs 0 for this feature
                count_1 = (X_train.iloc[:, feature_idx] == 1).sum()
                count_0 = (X_train.iloc[:, feature_idx] == 0).sum()
                row.extend([count_1, count_0])
                
                # Priority information
                # Check if this feature can be 1 in any allowed combination
                can_be_1 = False
                weight_when_1 = None
                
                for combo, weight in possible_values.items():
                    if weight > 0:  # Allowed combination
                        # Find position of feature_idx in group_indices
                        pos = group_indices.index(feature_idx)
                        if combo[pos] == 1:
                            can_be_1 = True
                            if weight_when_1 is None:
                                weight_when_1 = weight
                            # Use the first weight found (they should be the same for given feature value)
                
                row.extend([
                    'categorical',
                    '',
                    '',
                    '',
                    '',
                    '1' if can_be_1 else '0',
                    str(weight_when_1) if weight_when_1 is not None else '0'
                ])
                
                writer.writerow(row)
    
    logger.info(f"Parameters saved to {filename}")


def save_results_csv(results, config, filename='experiment_results.csv'):
    """
    Save experiment results to CSV.
    Each counterfactual gets its own row.
    
    Args:
        results: List of experiment results
        config: Experiment configuration
        filename: Output CSV filename
    """
    logger.info(f"Saving results to {filename}...")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header with config columns
        config_keys = list(config.keys())
        writer.writerow([
            'sample_idx', 'sample_prediction', 'sample_values',
            'target_idx', 'target_prediction', 
            'exemplar_prediction', 'exemplar_values',
            'cf_rank', 'cf_prediction', 'cf_distance_from_target',
            'cf_preference_score', 'cf_values'
        ] + config_keys)
        
        # Data rows - one per counterfactual
        for result in results:
            # Base information for this experiment
            base_row = [
                result['sample_idx'],
                result['sample_prediction'],
                str(result['sample_values']),
                result['target_idx'],
                result['target_prediction'],
                result['target_prediction'],  # exemplar = target
                str(result['exemplar_values'])
            ]
            
            # Config values
            config_values = [config[key] for key in config_keys]
            
            if result['n_counterfactuals'] == 0:
                # No counterfactuals found - write one row with empty CF fields
                writer.writerow(base_row + ['', '', '', '', ''] + config_values)
            else:
                # Write one row per counterfactual
                for cf in result['counterfactuals']:
                    cf_row = base_row + [
                        cf['rank'],
                        cf['prediction'],
                        abs(cf['prediction'] - result['target_prediction']),
                        cf['preference_score'],
                        str(cf['sample'])
                    ] + config_values
                    writer.writerow(cf_row)
    
    logger.info(f"Results saved to {filename}")


def run_counterfactual_experiment(model, X_train, X_test, config):
    """
    Run a comprehensive counterfactual experiment with multiple samples and targets.
    
    Args:
        model: Trained classifier
        X_train: Training dataset
        X_test: Test dataset
        config: Dictionary with experiment parameters:
            - n_quantiles: Number of equally distributed prediction points
            - return_top_n: Number of top counterfactuals to return
            - epsilon: Target prediction tolerance
            - exemplar_weight: Weight assigned to exemplar value
            - n_samples: Number of samples to generate
            - use_monte_carlo: Whether to use Monte Carlo sampling
    
    Returns:
        results: List of experiment results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING COUNTERFACTUAL EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    
    # Get predictions for all test samples
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Select equally distributed prediction quantiles
    quantiles = np.linspace(0, 1, config['n_quantiles'])
    quantile_values = np.quantile(predictions, quantiles)
    
    # Find samples closest to each quantile
    sample_points = []
    for q_val in quantile_values:
        distances = np.abs(predictions - q_val)
        idx = np.argmin(distances)
        sample_points.append({
            'index': idx,
            'sample': X_test.iloc[idx].values.tolist(),
            'prediction': predictions[idx]
        })
    
    logger.info(f"Selected {len(sample_points)} sample points with predictions:")
    for i, sp in enumerate(sample_points):
        logger.info(f"  Sample {i+1}: prediction = {sp['prediction']:.6f}")
    logger.info("\n")
    
    # Run experiment for each sample-target pair
    results = []
    total_combinations = len(sample_points) * (len(sample_points) - 1)
    current_combination = 0
    
    for sample_idx, sample_point in enumerate(sample_points):
        sample = sample_point['sample']
        sample_pred = sample_point['prediction']
        
        for target_idx, target_point in enumerate(sample_points):
            if sample_idx == target_idx:
                continue  # Skip same point
            
            current_combination += 1
            target_pred = target_point['prediction']
            exemplar = target_point['sample']
            
            logger.info(f"[{current_combination}/{total_combinations}] Sample {sample_idx+1} → Target {target_idx+1}")
            logger.info(f"  Sample prediction: {sample_pred:.6f}, Target prediction: {target_pred:.6f}")
            
            # Define preferences
            preferences = define_preferences(sample, exemplar, X_train, config['exemplar_weight'])
            
            # Generate counterfactuals (without displaying plots)
            model_pred = lambda x: model.predict_proba(x)[:, 1]
            explainer = RandomSearchExplainer(
                model_pred=model_pred,
                priorities=preferences,
                sample=sample,
                target=target_pred
            )
            
            # Generate ALL counterfactuals (don't filter to top N yet)
            cf_samples, cf_predictions, cf_scores = explainer.generate_random_samples(
                n_samples=config['n_samples'],
                epsilon=config['epsilon'],
                use_monte_carlo=config['use_monte_carlo'],
                random_seed=42,
                max_tries=100,
                return_top_n=None  # Get all counterfactuals
            )
            
            # Sort by preference score descending for ranking
            if len(cf_samples) > 0:
                sorted_indices = np.argsort(cf_scores)[::-1]
                cf_samples = [cf_samples[i] for i in sorted_indices]
                cf_predictions = [cf_predictions[i] for i in sorted_indices]
                cf_scores = [cf_scores[i] for i in sorted_indices]
            
            # Store results
            result = {
                'sample_idx': sample_idx + 1,
                'target_idx': target_idx + 1,
                'sample_prediction': sample_pred,
                'target_prediction': target_pred,
                'sample_values': sample,
                'exemplar_values': exemplar,
                'n_counterfactuals': len(cf_samples),
                'counterfactuals': []
            }
            
            for i, (cf_sample, cf_pred, cf_score) in enumerate(zip(cf_samples, cf_predictions, cf_scores)):
                result['counterfactuals'].append({
                    'rank': i + 1,
                    'prediction': cf_pred,
                    'preference_score': cf_score,
                    'sample': cf_sample
                })
            
            results.append(result)
            
            if len(cf_samples) > 0:
                logger.info(f"  Found {len(cf_samples)} counterfactuals")
                logger.info(f"  Prediction range: [{min(cf_predictions):.4f}, {max(cf_predictions):.4f}]")
                logger.info(f"  Preference score range: [{min(cf_scores):.4f}, {max(cf_scores):.4f}]")
            else:
                logger.info(f"  No counterfactuals found")
            logger.info("")
    
    return results


def print_experiment_results(results, config):
    """
    Print comprehensive results from the counterfactual experiment.
    
    Args:
        results: List of experiment results
        config: Experiment configuration
    """
    import sys
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 80)
    sys.stderr.flush()
    
    # Overall statistics
    total_experiments = len(results)
    experiments_with_cf = sum(1 for r in results if r['n_counterfactuals'] > 0)
    total_cf = sum(r['n_counterfactuals'] for r in results)
    
    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total sample-target pairs: {total_experiments}")
    logger.info(f"  Pairs with counterfactuals: {experiments_with_cf} ({100*experiments_with_cf/total_experiments:.1f}%)")
    logger.info(f"  Total counterfactuals generated: {total_cf}")
    logger.info(f"  Average counterfactuals per pair: {total_cf/total_experiments:.2f}")
    sys.stderr.flush()
    
    # Detailed results for each experiment - only show summary, not all details
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY BY SAMPLE-TARGET PAIR")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"\nSample {result['sample_idx']} → Target {result['target_idx']}: "
                    f"{result['n_counterfactuals']} counterfactuals found "
                    f"(distance: {abs(result['sample_prediction'] - result['target_prediction']):.4f})")
        
        if result['n_counterfactuals'] > 0 and len(result['counterfactuals']) > 0:
            best_cf = result['counterfactuals'][0]
            best_score = best_cf['preference_score']
            
            # Count how many CFs have the same score as the best one
            count_with_best_score = sum(1 for cf in result['counterfactuals'] 
                                       if cf['preference_score'] == best_score)
            
            logger.info(f"  Best CF: prediction={best_cf['prediction']:.4f}, "
                       f"preference={best_score:.2f}, "
                       f"distance_from_target={abs(best_cf['prediction'] - result['target_prediction']):.4f}")
            logger.info(f"  CFs with best score ({best_score:.2f}): {count_with_best_score} out of {result['n_counterfactuals']}")
    
    sys.stderr.flush()
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT RESULTS SUMMARY COMPLETE")
    logger.info("=" * 80)
    sys.stderr.flush()


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Counterfactual Explanation Example - German Credit Dataset")
    logger.info("=" * 80)
    
    # ============================================================================
    # EXPERIMENT CONFIGURATION
    # ============================================================================
    config = {
        'n_quantiles': 3,           # Number of equally distributed prediction points (3 points = 6 pairs)
        'return_top_n': 5,          # Number of top counterfactuals to return per experiment
        'epsilon': 0.05,            # Target prediction tolerance (±5%)
        'exemplar_weight': 0.01,    # Weight assigned to exemplar value (0.01 = almost undesirable)
        'n_samples': 10000,         # Number of samples to generate per experiment
        'use_monte_carlo': True,    # Use Monte Carlo sampling
        'run_experiment': True,     # Set to False to run single example instead
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    # ============================================================================
    
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = \
        load_and_preprocess_data()
    
    # 2. Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    if config['run_experiment']:
        # ========================================================================
        # RUN FULL EXPERIMENT
        # ========================================================================
        results = run_counterfactual_experiment(model, X_train, X_test, config)
        
        # Save results to CSV files
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS TO CSV")
        logger.info("=" * 80)
        
        # Get preferences for CSV (using first sample-target pair)
        predictions = model.predict_proba(X_test)[:, 1]
        sample_point = X_test.iloc[0].values.tolist()
        exemplar_point = X_test.iloc[1].values.tolist()
        preferences = define_preferences(sample_point, exemplar_point, X_train, config['exemplar_weight'])
        
        save_parameters_csv(X_train, preferences)
        save_results_csv(results, config)
        
        logger.info("CSV files saved successfully.")
        
        # Print comprehensive summary at the end
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        try:
            print_experiment_results(results, config)
        except Exception as e:
            logger.error(f"Error printing results: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        # ========================================================================
        # RUN SINGLE EXAMPLE (Original workflow with visualizations)
        # ========================================================================
        # 3. Find exemplar - sample closest to target
        exemplar, exemplar_prediction, exemplar_index = find_exemplar(
            model, X_test, target=0.60
        )
        logger.info("\n" + "=" * 80)
        logger.info("EXEMPLAR SAMPLE")
        logger.info("=" * 80)
        logger.info(f"Exemplar (sample closest to target):")
        logger.info(f"  Prediction: {exemplar_prediction:.6f}")
        logger.info(f"  Features: {exemplar}")
        logger.info("=" * 80 + "\n")
        
        # 4. Define sample for analysis
        sample = define_sample()
        logger.info(f"Sample to explain: {len(sample)} features")
        
        # 5. Define preferences (automatically generated from sample and exemplar)
        exemplar_weight = config['exemplar_weight']
        preferences = define_preferences(sample, exemplar, X_train, exemplar_weight)
        logger.info(f"Preferences automatically generated from sample and exemplar (exemplar_weight={exemplar_weight})")
        logger.info("\n" + "=" * 80)
        logger.info("PREFERENCES")
        logger.info("=" * 80)
        logger.info(f"Numerical preferences:")
        for idx, pref in preferences['numerical'].items():
            if pref is None:
                logger.info(f"  Feature {idx}: None (unactionable)")
            elif isinstance(pref, dict):
                logger.info(f"  Feature {idx}: min={pref['min']}, max={pref['max']}, function={pref['function'].__name__ if hasattr(pref['function'], '__name__') else 'custom'}")
            else:
                logger.info(f"  Feature {idx}: {pref}")
        logger.info(f"Categorical preferences:")
        for group, mappings in preferences['categorical'].items():
            logger.info(f"  Group {group}: {len(mappings)} combinations defined")
        logger.info("=" * 80 + "\n")
        
        samples, predictions, preference_scores, explainer = generate_counterfactuals(
            model=model,
            sample=sample,
            preferences=preferences,
            target=0.60,
            n_samples=config['n_samples'],
            epsilon=config['epsilon'],
            use_monte_carlo=config['use_monte_carlo'],
            return_top_n=config['return_top_n'],
            exemplar=exemplar
        )
        
        # 7. Display results
        if len(samples) > 0:
            logger.info("\n" + "=" * 80)
            logger.info("COUNTERFACTUAL RESULTS")
            logger.info("=" * 80)
            logger.info(f"Total counterfactuals found: {len(samples)}")
            logger.info(f"Target prediction: 0.60 ± {config['epsilon']}")
            logger.info(f"Prediction range: [{min(predictions):.4f}, {max(predictions):.4f}]")
            logger.info(f"Mean prediction: {np.mean(predictions):.4f}")
            logger.info(f"Std prediction: {np.std(predictions):.4f}")
            
            # Display top counterfactuals with predictions and preference weights
            logger.info(f"\nTop {len(samples)} counterfactuals (sorted by preference):")
            for i, (cf_sample, pred, pref_score) in enumerate(zip(samples, predictions, preference_scores)):
                logger.info(f"\n  {i+1}. Prediction: {pred:.4f}, Preference Weight: {pref_score:.6f}")
                # Display detailed breakdown for each counterfactual
                display_preference_breakdown(explainer, cf_sample, i+1)
        else:
            logger.warning("No counterfactuals found within the specified epsilon!")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
