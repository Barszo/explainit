"""
Example demonstrating MINLSearchExplainer for binary classification.

This example shows how to use the MINLP (Mixed Integer Non-Linear Programming) approach
to find counterfactual explanations for binary classification models. Unlike random search,
MINLP uses optimization with Shapley values to find more targeted counterfactuals.

Key differences from random search:
- MINLP requires a dataset (used to find target exemplar with desired prediction)
- Uses Shapley values to understand feature importance
- Performs mathematical optimization to find counterfactuals
- More computationally intensive but potentially better quality solutions
- Returns multiple ranked candidates
"""

import os
import sys
import ssl
import types
import importlib.util
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a minimal explainit package stub so we can import the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
explainit_module = types.ModuleType('explainit')
explainit_module.__path__ = [os.path.join(project_root, 'explainit')]
sys.modules['explainit'] = explainit_module

logging_config_path = os.path.abspath(os.path.join(project_root, 'explainit', 'logging_config.py'))
spec = importlib.util.spec_from_file_location('explainit.logging_config', logging_config_path)
logging_config = importlib.util.module_from_spec(spec)
sys.modules['explainit.logging_config'] = logging_config
spec.loader.exec_module(logging_config)

# Import MINLP search explainer
minlp_search_path = os.path.abspath(
    os.path.join(project_root, 'explainit', 'explainers', 'minlp_search.py')
)
spec = importlib.util.spec_from_file_location('minlp_search', minlp_search_path)
minlp_search = importlib.util.module_from_spec(spec)
sys.modules['minlp_search'] = minlp_search
spec.loader.exec_module(minlp_search)
MINLSearchExplainer = minlp_search.MINLSearchExplainer

# Import priority helper modules (linear and nonlinear)
linear_path = os.path.abspath(os.path.join(project_root, 'explainit', 'priorities', 'linear.py'))
spec = importlib.util.spec_from_file_location('explainit.priorities.linear', linear_path)
linear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linear)

nonlinear_path = os.path.abspath(os.path.join(project_root, 'explainit', 'priorities', 'nonlinear.py'))
spec = importlib.util.spec_from_file_location('explainit.priorities.nonlinear', nonlinear_path)
nonlinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonlinear)


def load_heart_disease_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Download and prepare the UCI Heart Disease dataset for binary classification."""
    logger.info("Downloading Heart Disease dataset from UCI...")

    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    try:
        data = pd.read_csv(url, names=column_names, na_values='?')
        logger.info("Dataset downloaded successfully from UCI")
    except Exception as e:
        logger.warning("Primary download failed, trying fallback raw GitHub dataset: %s", e)
        fallback_url = "https://raw.githubusercontent.com/rashida048/Datasets/master/heart.csv"
        data = pd.read_csv(fallback_url)
        if 'target' not in data.columns:
            raise ValueError("Dataset does not contain 'target' column")
        logger.info("Dataset downloaded successfully from fallback source")

    data = data.dropna()
    data['target'] = (data['target'] > 0).astype(int)

    logger.info("Dataset ready: %s samples, %s features", data.shape[0], data.shape[1] - 1)

    X = data.drop(columns=['target'])
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Train/test split created: %s train, %s test", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_binary_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a simple binary classifier for heart disease probability prediction."""
    logger.info("Training logistic regression model...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def make_preference_from_exponential(center: float, span: float, a: float = 5.0):
    """Create a preference function using exponential weight curve."""
    half = span / 2.0
    x0 = center - half
    x1 = center + half

    def pref(x):
        return float(np.asarray(nonlinear.exponential(x, x0, x1, increasing=True, a=a)).squeeze())

    return pref


def build_priorities(feature_names: list, sample: np.ndarray, X_train: pd.DataFrame) -> Dict[str, Any]:
    """Build numerical and categorical priorities for MINLSearchExplainer."""
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    numerical_priorities = {}

    def add_num(name, center, span, a=5.0):
        idx = feature_indices[name]
        col = X_train[name]
        minv = float(col.min())
        maxv = float(col.max())
        numerical_priorities[idx] = {
            'min': minv,
            'max': maxv,
            'function': make_preference_from_exponential(center=center, span=span, a=a)
        }

    # Add numerical features with preferences
    add_num('age', center=50.0, span=20.0, a=6.0)
    add_num('trestbps', center=130.0, span=40.0, a=5.0)
    add_num('chol', center=240.0, span=120.0, a=4.0)
    add_num('thalach', center=150.0, span=35.0, a=5.0)

    # Mark fbs as unactionable (non-modifiable)
    numerical_priorities[feature_indices['fbs']] = 0

    # Use linear helper for one feature to show diversity
    chol_idx = feature_indices['chol']
    chol_min = numerical_priorities[chol_idx]['min']
    chol_max = numerical_priorities[chol_idx]['max']
    chol_center = 240.0
    chol_span = 120.0
    chol_x0 = chol_center - chol_span / 2.0
    chol_x1 = chol_center + chol_span / 2.0
    numerical_priorities[chol_idx]['function'] = lambda x, x0=chol_x0, x1=chol_x1: float(
        np.asarray(linear.basic_linear(x, x0, x1, increasing=True)).squeeze()
    )

    # Categorical priorities (simple case: prefer certain chest pain types)
    cp_values = sorted(X_train['cp'].unique())
    categorical_priorities = {}

    # Example: prefer specific chest pain types
    categorical_priorities[(feature_indices['cp'],)] = {
        (int(val),): (1.0 if int(val) == 3 else 0.8 if int(val) == 4 else 0.6)
        for val in cp_values
    }

    return {
        'numerical': numerical_priorities,
        'categorical': categorical_priorities,
    }


def main() -> None:
    """Main example demonstrating MINLP-based binary classification counterfactuals."""
    
    logger.info("=" * 80)
    logger.info("MINLP Search Example for Binary Classification")
    logger.info("=" * 80)

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_heart_disease_data()
    model = train_binary_classifier(X_train, y_train)

    feature_names = list(X_train.columns)
    
    # Select a sample to explain
    sample_idx = 0
    sample = X_test.iloc[sample_idx].to_numpy(dtype=float)
    original_prob = model.predict_proba(sample.reshape(1, -1))[:, 1][0]
    original_class = int(original_prob > 0.5)

    logger.info("")
    logger.info("Selected sample index %s from the test set", sample_idx)
    logger.info("Original prediction probability: %.4f", original_prob)
    logger.info("Original predicted class (threshold=0.5): %s", original_class)
    logger.info("")

    # Build priorities
    priorities = build_priorities(feature_names, sample, X_train)
    logger.info("Priorities defined for %d numerical and %d categorical features",
               len(priorities['numerical']), len(priorities['categorical']))

    # Define model prediction function
    def model_pred(X_batch: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        return model.predict_proba(X_batch)[:, 1]

    # Create MINLP explainer
    logger.info("")
    logger.info("Creating MINLSearchExplainer...")
    try:
        explainer = MINLSearchExplainer(
            model_pred=model_pred,
            priorities=priorities,
            sample=sample,
            target=0.7,  # This will be overridden by find_counterfactuals_for_binary
            dataset=X_train.to_numpy(),
            target_exemplar_epsilon=0.05,
            epsilon=0.05
        )
        logger.info("Explainer created successfully")
    except Exception as e:
        logger.error("Failed to create explainer: %s", e)
        return

    # Find counterfactuals for binary classification
    logger.info("")
    logger.info("Finding counterfactuals to change prediction to class 1 (disease present)...")
    logger.info("")
    
    try:
        cfs, predictions, scores, iterations = explainer.find_counterfactuals_for_binary(
            target_class=1,
            threshold=0.5,
            expected_counterfactuals=3,
            max_iterations=200,
            shap_approx=False,
            return_top_n=3
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("COUNTERFACTUAL RESULTS")
        logger.info("=" * 80)
        logger.info("")

        if len(cfs) == 0:
            logger.warning("No valid counterfactuals found!")
            logger.warning("This may indicate:")
            logger.warning("  1. The target class is already achieved")
            logger.warning("  2. Priorities are too restrictive")
            logger.warning("  3. The dataset/model combination makes it infeasible")
            return

        for idx, (cf, pred, score, iteration) in enumerate(zip(cfs, predictions, scores, iterations), start=1):
            logger.info("--- Counterfactual %d (Iteration: %d) ---", idx, iteration)
            logger.info("Prediction probability: %.4f", pred)
            logger.info("Priority score (weight): %.4f", score)
            logger.info("")
            
            # Show feature changes
            logger.info("Feature changes from original:")
            for feature_idx, feature_name in enumerate(feature_names):
                original_val = sample[feature_idx]
                cf_val = cf[feature_idx]
                if abs(original_val - cf_val) > 1e-6:
                    logger.info("  %s: %.4f → %.4f (change: %.4f)", 
                              feature_name, original_val, cf_val, cf_val - original_val)
            logger.info("")

        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info("Original prediction: %.4f (class %d)", original_prob, original_class)
        logger.info("Found %d valid counterfactuals to flip class", len(cfs))
        logger.info("Best (lowest cost) counterfactual prediction: %.4f", predictions[0])
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Error during counterfactual search: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        return

    # Optional: Find counterfactuals for opposite class
    logger.info("")
    logger.info("(Optional) Finding counterfactuals to change prediction to class 0 (disease absent)...")
    logger.info("")
    
    try:
        cfs_class0, predictions_class0, scores_class0, _ = explainer.find_counterfactuals_for_binary(
            target_class=0,
            threshold=0.5,
            expected_counterfactuals=2,
            max_iterations=200,
            shap_approx=False,
            return_top_n=2
        )
        
        if len(cfs_class0) > 0:
            logger.info("Found %d counterfactuals to class 0", len(cfs_class0))
            logger.info("Best counterfactual prediction: %.4f", predictions_class0[0])
        else:
            logger.info("No counterfactuals found for class 0")
    
    except Exception as e:
        logger.warning("Could not find counterfactuals for opposite class: %s", e)

    logger.info("")
    logger.info("Example completed!")
    logger.info("")
    logger.info("Notes on MINLP vs Random Search:")
    logger.info("- MINLP requires a dataset (for finding target exemplar with Shapley values)")
    logger.info("- More computationally intensive but uses mathematical optimization")
    logger.info("- Returns counterfactuals ranked by preference (priority) weight")
    logger.info("- Better for problems where you want specific, high-quality solutions")
    logger.info("")


if __name__ == '__main__':
    main()
