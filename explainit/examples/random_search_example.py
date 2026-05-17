import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import importlib.util
import logging
import os
import ssl
import sys
import types
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a minimal explainit package stub so random_search.py can import
# explainit.logging_config without triggering explainit/__init__.py.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
explainit_module = types.ModuleType('explainit')
explainit_module.__path__ = [os.path.join(project_root, 'explainit')]
sys.modules['explainit'] = explainit_module

logging_config_path = os.path.abspath(os.path.join(project_root, 'explainit', 'logging_config.py'))
spec = importlib.util.spec_from_file_location('explainit.logging_config', logging_config_path)
logging_config = importlib.util.module_from_spec(spec)
sys.modules['explainit.logging_config'] = logging_config
spec.loader.exec_module(logging_config)

random_search_path = os.path.abspath(
    os.path.join(project_root, 'explainit', 'explainers', 'random_search.py')
)
spec = importlib.util.spec_from_file_location('random_search', random_search_path)
random_search = importlib.util.module_from_spec(spec)
sys.modules['random_search'] = random_search
spec.loader.exec_module(random_search)
RandomSearchExplainer = random_search.RandomSearchExplainer

# Import priority helper modules (linear and nonlinear) directly from the repository
linear_path = os.path.abspath(os.path.join(project_root, 'explainit', 'priorities', 'linear.py'))
spec = importlib.util.spec_from_file_location('explainit.priorities.linear', linear_path)
linear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linear)

nonlinear_path = os.path.abspath(os.path.join(project_root, 'explainit', 'priorities', 'nonlinear.py'))
spec = importlib.util.spec_from_file_location('explainit.priorities.nonlinear', nonlinear_path)
nonlinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonlinear)

# Configure informative logging for the example.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_heart_disease_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Download and prepare the UCI Heart Disease dataset for a binary classification example."""
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
            target_col = [c for c in data.columns if 'target' in c.lower() or 'disease' in c.lower()][0]
            data = data.rename(columns={target_col: 'target'})
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
    logger.info("Training simple logistic regression model...")
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear', random_state=42))
    ])
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def make_preference_from_exponential(center: float, span: float, a: float = 5.0):
    """Create a preference function using `nonlinear.exponential`.

    This wraps the library exponential curve into the callable format expected
    by RandomSearchExplainer: f(x) -> [0,1]. We convert the (center, span)
    description into (x0, x1) bounds used by the exponential helper.
    """
    half = span / 2.0
    x0 = center - half
    x1 = center + half

    def pref(x):
        # nonlinear.exponential accepts array-like inputs and returns numpy arrays
        return float(np.asarray(nonlinear.exponential(x, x0, x1, increasing=True, a=a)).squeeze())

    return pref


def build_priorities(feature_names: list, sample: np.ndarray, X_train: pd.DataFrame) -> Dict[str, Any]:
    """Build numerical and categorical priorities for the RandomSearchExplainer.

    Explanation (tutorial):
    - Priorities is a dictionary with two keys: `'numerical'` and `'categorical'`.
      * `'numerical'` maps a feature index -> either:
          - a dict {'function': f, 'min': ..., 'max': ...} where `f(x)` -> [0,1]
            is a preference weight (1 = highly preferred, 0 = forbidden),
          - OR a numeric/0 to indicate the feature is unactionable (fixed to sample value).
      * `'categorical'` maps a tuple of feature indices (group) -> a mapping of
         possible category tuples -> weight in [0,1]. Categories with weight 0
         are forbidden and will not be sampled.

    Numerical priorities are continuous preference functions (we recommend
    using `linear` or `nonlinear` helpers). Categorical priorities are discrete
    maps that allow/forbid specific combinations (useful for one-hot groups or
    logically tied categorical features).
    """
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    # Numerical preferences: use continuous clinical measurements where we want
    # the explainer to prefer some regions over others. We build these
    # using the `nonlinear.exponential` helper to demonstrate usage of the
    # repository-provided priority functions.
    numerical_priorities = {}

    # Use the trained data range to set sensible min/max values, and use the
    # exponential preference function (steepness controlled by `a`). This
    # demonstrates using the provided `nonlinear.py` helper.
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

    add_num('age', center=55.0, span=25.0, a=6.0)
    add_num('trestbps', center=130.0, span=40.0, a=5.0)
    add_num('chol', center=240.0, span=120.0, a=4.0)
    add_num('thalach', center=150.0, span=35.0, a=5.0)

    # Example of an unactionable numerical feature: keep fasting blood sugar fixed.
    numerical_priorities[feature_indices['fbs']] = 0

    # Demonstrate using `linear` helper instead of exponential for one feature
    # (linear transition between x0 and x1). We'll replace `chol`'s function
    # with a linear step to show both options.
    chol_idx = feature_indices['chol']
    chol_min = numerical_priorities[chol_idx]['min']
    chol_max = numerical_priorities[chol_idx]['max']
    chol_center = 240.0
    chol_span = 120.0
    chol_x0 = chol_center - chol_span / 2.0
    chol_x1 = chol_center + chol_span / 2.0
    numerical_priorities[chol_idx]['function'] = lambda x, x0=chol_x0, x1=chol_x1: float(np.asarray(linear.basic_linear(x, x0, x1, increasing=True)).squeeze())

    # Categorical priorities: define preferred categories or category combinations.
    # Keep the keys as tuples of indices for compatibility with RandomSearchExplainer.
    cp_values = sorted(X_train['cp'].unique())
    thal_values = sorted(X_train['thal'].unique())
    exang_slope_values = sorted(set(zip(X_train['exang'], X_train['slope'])))

    # Categorical priorities: define preferred categories or category combinations.
    # Use tuples of indices for groups (this is how RandomSearchExplainer expects them).
    categorical_priorities = {}

    # For chest pain type (`cp`) prefer type 3 strongly, others less so.
    categorical_priorities[(feature_indices['cp'],)] = {
        (int(val),): (1.0 if int(val) == 3 else 0.8 if int(val) == 4 else 0.6 if int(val) == 2 else 0.4)
        for val in cp_values
    }

    # For `thal` prefer value 3 (example) and give moderate weight to others.
    categorical_priorities[(feature_indices['thal'],)] = {
        (int(val),): (1.0 if int(val) == 3 else 0.7)
        for val in thal_values
    }

    # Example of a grouped categorical preference: prefer no exercise-induced angina
    # combined with a favorable slope value. Group keys are (exang_index, slope_index).
    categorical_priorities[(feature_indices['exang'], feature_indices['slope'])] = {
        (int(exang), int(slope)): (1.0 if exang == 0 and slope == 2 else 0.8 if exang == 0 and slope == 3 else 0.4)
        for exang, slope in exang_slope_values
    }

    return {
        'numerical': numerical_priorities,
        'categorical': categorical_priorities,
    }


def main() -> None:
    logger.info("Starting Random Search example for Heart Disease binary classification")

    X_train, X_test, y_train, y_test = load_heart_disease_data()
    model = train_binary_classifier(X_train, y_train)

    feature_names = list(X_train.columns)
    sample_idx = 0
    sample = X_test.iloc[sample_idx].to_numpy(dtype=float)
    original_prob = model.predict_proba(sample.reshape(1, -1))[:, 1][0]
    original_class = int(original_prob > 0.5)

    logger.info("Selected sample index %s from the test set", sample_idx)
    logger.info("Original prediction probability: %.4f", original_prob)
    logger.info("Original predicted class (threshold=0.5): %s", original_class)

    priorities = build_priorities(feature_names, sample, X_train)
    logger.info("Numerical priorities defined for indices: %s", sorted(priorities['numerical'].keys()))
    logger.info("Categorical priorities defined for groups: %s", list(priorities['categorical'].keys()))

    def model_pred(X_batch: np.ndarray) -> np.ndarray:
        return model.predict_proba(X_batch)[:, 1]

    explainer = RandomSearchExplainer(
        model_pred=model_pred,
        priorities=priorities,
        sample=sample,
        target=1,
    )

    logger.info("Filtered priorities prepared by RandomSearchExplainer")
    logger.info("Filtered numerical priorities: %s", explainer.filtered_priorities['numerical'].keys())
    logger.info("Filtered categorical priorities: %s", explainer.filtered_priorities['categorical'].keys())

    # --- Tutorial step: display priorities ---
    # The explainer can visualise the numerical priority functions and the
    # categorical weight tables. This helps to verify that your priorities
    # express the intended constraints before running the expensive search.
    logger.info("Displaying priority functions and categorical weights (saved to images/)")
    try:
        explainer.display_priorities(exemplar=sample)
    except Exception as e:
        logger.warning("Could not display priorities (matplotlib may not be available): %s", e)

    # Generate counterfactuals for the binary classification target class.
    # target_class=1 means we want samples that the model predicts as positive.
    cfs, predictions, scores, iterations = explainer.generate_for_binary(
        expected_counterfactuals=2,
        max_iterations=5000,
        target_class=1,
        threshold=0.5,
        random_seed=42,
        use_monte_carlo=True,
        max_tries=200,
        n_candidates_per_cf=3,
    )

    logger.info("Counterfactual search complete")

    for idx, (cf, pred, score, iteration) in enumerate(zip(cfs, predictions, scores, iterations), start=1):
        logger.info("--- Counterfactual %d ---", idx)
        logger.info("Prediction probability: %.4f", pred)
        logger.info("Preference score: %.4f", score)
        logger.info("Iterations required: %d", iteration)
        logger.info("Counterfactual sample: %s", np.round(cf, 3).tolist())
        breakdown = explainer.get_preference_breakdown(cf)
        logger.info("Preference breakdown: %s", breakdown)

    if len(cfs) == 0:
        logger.warning("No counterfactuals were found with the current priorities and search settings.")
        logger.warning("Try increasing max_iterations, relaxing the target threshold, or broadening the priority functions.")

    logger.info("Example completed. You can now adapt the priorities and target settings for your own binary case.")


if __name__ == '__main__':
    main()
