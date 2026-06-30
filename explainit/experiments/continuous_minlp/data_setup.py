"""Stage 1: prepare datasets for the continuous-target MINLP experiment.

For each registered dataset key this script:

* loads the raw features and target
* train/test-splits and standard-scales the features
* MinMax-scales the target into ``[0, 1]``
* pickles everything (plus the scalers) under ``data/<dataset_key>/data.pkl``

Adding a new dataset is a single line in :data:`DATASETS` plus the matching
loader function returning ``(X_df, y_array, feature_names)``.

CLI usage::

    python -m explainit.experiments.continuos_minlp.data_setup
    python -m explainit.experiments.continuos_minlp.data_setup --datasets diabetes
    python -m explainit.experiments.continuos_minlp.data_setup --force
"""

from __future__ import annotations

import argparse
import logging
import pickle
import ssl
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_ANALYSIS_DIR = EXPERIMENT_DIR / "data_analysis"
DATA_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("explainit.experiments.continuos_minlp.data_setup")


# ---------------------------------------------------------------------------
# Raw loaders. Each returns (X dataframe, y array, feature_names list).
# ---------------------------------------------------------------------------


def _load_diabetes_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import load_diabetes

    logger.info(
        "[%s] Raw load: using sklearn.load_diabetes (bundled dataset, no download step).",
        "diabetes",
    )
    bunch = load_diabetes(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    return X, y, list(bunch.feature_names)


def _load_california_housing_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from sklearn.datasets import fetch_california_housing

    try:
        logger.info(
            "[%s] Raw load: calling sklearn.fetch_california_housing "
            "(uses sklearn cache; first run may download).",
            "california_housing",
        )
        bunch = fetch_california_housing(as_frame=True)
    except Exception as exc:  # pragma: no cover - macOS SSL fallback
        logger.warning(
            "fetch_california_housing failed (%s); retrying with SSL disabled.", exc,
        )
        ssl._create_default_https_context = ssl._create_unverified_context
        bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.astype(float).to_numpy()
    return X, y, list(bunch.feature_names)


def _load_synthetic_raw() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    rng = np.random.default_rng(42)
    n_samples = 2000
    n_features = 6
    X_np = rng.normal(size=(n_samples, n_features))
    weights = np.linspace(1.0, 0.2, n_features)
    bias = 0.3
    y = X_np @ weights + bias + rng.normal(scale=0.1, size=n_samples)
    feature_names = [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(X_np, columns=feature_names)
    return X, y.astype(float), feature_names


# Registry. Add new datasets here.
DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, np.ndarray, List[str]]]] = {
    "diabetes": _load_diabetes_raw,
    # "california_housing": _load_california_housing_raw,
    # "synthetic": _load_synthetic_raw,
}


TARGET_NAMES: Dict[str, str] = {
    "diabetes": "disease_progression",
    "california_housing": "median_house_value",
    "synthetic": "synthetic_target",
}


# Manual categorical overrides per dataset (by raw column name). These names
# are always treated as categorical and override automatic detection.
CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "diabetes": ["sex"],
}


AUTO_CATEGORICAL_MAX_UNIQUE_VALUES = 20
AUTO_CATEGORICAL_MAX_UNIQUE_VALUES_NON_INTEGER = 5
AUTO_CATEGORICAL_MAX_UNIQUE_RATIO = 0.05
MISSING_CATEGORY_TOKEN = "__MISSING__"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _one_hot_frame(
    series: pd.Series, code_map: Dict[object, int], n_categories: int, col_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode ``series`` (mapped through ``code_map``) into 0/1 columns."""
    normalized = series.astype(object).where(series.notna(), MISSING_CATEGORY_TOKEN)
    codes = normalized.map(code_map).to_numpy()
    missing_mask = pd.isna(codes)
    if np.any(missing_mask):
        unknown = sorted({str(v) for v in normalized[missing_mask].tolist()})
        raise ValueError(
            f"Categorical column '{col_name}' has unknown value(s) not seen in raw fit set: {unknown}."
        )
    codes = codes.astype(int, copy=False)
    data: Dict[str, np.ndarray] = {}
    names: List[str] = []
    for i in range(n_categories):
        name = f"{col_name}={i}"
        data[name] = (codes == i).astype(float)
        names.append(name)
    return pd.DataFrame(data, index=series.index), names


def _sorted_unique_categorical_values(series: pd.Series) -> List[Any]:
    normalized = series.astype(object).where(series.notna(), MISSING_CATEGORY_TOKEN)
    values = list(pd.unique(normalized))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda v: str(v))


def _is_integer_like(series: pd.Series) -> bool:
    non_na = series.dropna()
    if non_na.empty:
        return False
    arr = non_na.to_numpy(dtype=float)
    return bool(np.all(np.isclose(arr, np.round(arr), rtol=0.0, atol=1e-9)))


def _detect_categorical_features(
    X_raw: pd.DataFrame, feature_names: List[str], manual_overrides: Sequence[str],
) -> Tuple[List[str], Dict[str, str], List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    manual_set = set(manual_overrides)
    categorical: List[str] = []
    reasoning: Dict[str, str] = {}
    auto_categorical: List[str] = []
    numerical_summary_rows: List[Dict[str, object]] = []
    categorical_summary_rows: List[Dict[str, object]] = []

    for col in feature_names:
        series = X_raw[col]
        n_all = int(len(series))
        non_na = series.dropna()
        n_non_na = int(non_na.size)
        n_unique = int(non_na.nunique(dropna=True))
        pct_non_na = (100.0 * n_non_na / n_all) if n_all else 0.0
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        is_bool = bool(pd.api.types.is_bool_dtype(series))
        integer_like = is_numeric and _is_integer_like(series)
        unique_ratio = (float(n_unique) / float(n_non_na)) if n_non_na else 0.0

        auto_reason = "manual review required"
        auto_is_categorical = False
        if not is_numeric:
            auto_is_categorical = True
            auto_reason = f"non-numeric dtype ({series.dtype})"
        elif is_bool:
            auto_is_categorical = True
            auto_reason = "boolean feature"
        elif (
            n_unique <= AUTO_CATEGORICAL_MAX_UNIQUE_VALUES
            and unique_ratio <= AUTO_CATEGORICAL_MAX_UNIQUE_RATIO
            and (
                integer_like
                or n_unique <= AUTO_CATEGORICAL_MAX_UNIQUE_VALUES_NON_INTEGER
            )
        ):
            auto_is_categorical = True
            auto_reason = (
                "low cardinality numeric "
                f"(unique={n_unique}, unique/non-nan={unique_ratio:.2%}, "
                f"integer_like={integer_like}, thresholds<={AUTO_CATEGORICAL_MAX_UNIQUE_VALUES} "
                f"(or <= {AUTO_CATEGORICAL_MAX_UNIQUE_VALUES_NON_INTEGER} for non-integer-like) "
                f"and <={AUTO_CATEGORICAL_MAX_UNIQUE_RATIO:.2%})"
            )
        else:
            auto_reason = (
                "kept numerical "
                f"(dtype={series.dtype}, integer_like={integer_like}, unique={n_unique}, unique/non-nan={unique_ratio:.2%})"
            )

        final_is_categorical = auto_is_categorical or (col in manual_set)
        if col in manual_set:
            reason = f"manual override in CATEGORICAL_FEATURES (auto: {auto_reason})"
        else:
            reason = f"automatic detection: {auto_reason}"

        if final_is_categorical:
            categorical.append(col)
        if auto_is_categorical:
            auto_categorical.append(col)
        reasoning[col] = reason

        if is_numeric:
            num_arr = non_na.to_numpy(dtype=float) if n_non_na else np.array([], dtype=float)
            numerical_summary_rows.append(
                {
                    "feature_name": col,
                    "auto_considered_categorical": "yes" if auto_is_categorical else "no",
                    "number_of_unique_values": n_unique,
                    "number_of_all_values": n_all,
                    "number_of_not_nan_values": n_non_na,
                    "percentage_of_not_nan_values": pct_non_na,
                    "min": float(np.min(num_arr)) if n_non_na else np.nan,
                    "max": float(np.max(num_arr)) if n_non_na else np.nan,
                    "average": float(np.mean(num_arr)) if n_non_na else np.nan,
                    "median": float(np.median(num_arr)) if n_non_na else np.nan,
                    "standard_deviation": float(np.std(num_arr)) if n_non_na else np.nan,
                }
            )

        if final_is_categorical:
            counts = non_na.value_counts(dropna=True)
            row: Dict[str, object] = {
                "feature_name": col,
                "number_of_unique_values": n_unique,
                "number_of_all_values": n_all,
                "number_of_all_not_nan_values": n_non_na,
                "percentage_of_all_not_nan_values": pct_non_na,
            }
            for idx in range(20):
                rank = idx + 1
                if idx < len(counts):
                    value = counts.index[idx]
                    count = int(counts.iloc[idx])
                    pct = (100.0 * count / n_non_na) if n_non_na else 0.0
                    row[f"most_popular_value_{rank}"] = str(value)
                    row[f"percentage_of_most_popular_value_{rank}"] = pct
                else:
                    row[f"most_popular_value_{rank}"] = ""
                    row[f"percentage_of_most_popular_value_{rank}"] = ""
            categorical_summary_rows.append(row)

    return categorical, reasoning, auto_categorical, numerical_summary_rows, categorical_summary_rows


def _save_feature_analysis_csvs(
    key: str,
    numerical_summary_rows: List[Dict[str, object]],
    categorical_summary_rows: List[Dict[str, object]],
) -> Tuple[Path, Path]:
    dataset_analysis_dir = DATA_ANALYSIS_DIR / key
    dataset_analysis_dir.mkdir(parents=True, exist_ok=True)
    numerical_path = dataset_analysis_dir / "numerical_features.csv"
    categorical_path = dataset_analysis_dir / "categorical_features.csv"

    pd.DataFrame(numerical_summary_rows).to_csv(numerical_path, index=False)
    pd.DataFrame(categorical_summary_rows).to_csv(categorical_path, index=False)
    return numerical_path, categorical_path


def _prepare_dataset(
    key: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{key}'. Registered: {sorted(DATASETS)}"
        )

    logger.info("[%s] Starting preprocessing pipeline.", key)
    logger.info("[%s] Step 1/6: load raw dataset.", key)
    X_raw, y_raw, feature_names = DATASETS[key]()
    logger.info(
        "[%s] Step 1/6 done: loaded raw shape X=%s, y=%s, features=%d.",
        key, X_raw.shape, y_raw.shape, len(feature_names),
    )

    manual_cat_cols = list(CATEGORICAL_FEATURES.get(key, []))
    for col in manual_cat_cols:
        if col not in feature_names:
            raise KeyError(
                f"Categorical feature '{col}' for dataset '{key}' is not a known "
                f"column. Known columns: {feature_names}"
            )

    cat_cols, feature_reasoning, auto_cat_cols, numerical_summary_rows, categorical_summary_rows = _detect_categorical_features(
        X_raw, feature_names, manual_cat_cols
    )
    num_cols = [c for c in feature_names if c not in set(cat_cols)]
    logger.info(
        "[%s] Step 2/6: automatic feature typing -> numerical=%s | categorical=%s.",
        key, num_cols, cat_cols,
    )
    logger.info("[%s] Feature typing details:", key)
    for col in feature_names:
        logger.info(
            "[%s]   - %s: %s",
            key, col, feature_reasoning[col],
        )
    numerical_csv_path, categorical_csv_path = _save_feature_analysis_csvs(
        key, numerical_summary_rows, categorical_summary_rows
    )
    logger.info(
        "[%s] Saved feature analysis CSVs: numerical=%s | categorical=%s.",
        key, numerical_csv_path, categorical_csv_path,
    )

    logger.info(
        "[%s] Step 3/6: train/test split (test_size=%.2f, random_state=%d).",
        key, float(test_size), int(random_state),
    )
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state,
    )
    logger.info(
        "[%s] Step 3/6 done: X_train=%s X_test=%s.",
        key, X_train_raw.shape, X_test_raw.shape,
    )

    # Standard-scale numerical columns only.
    logger.info("[%s] Step 4/6: scale numerical features + one-hot encode categoricals.", key)
    x_scaler = StandardScaler()
    if num_cols:
        train_num = pd.DataFrame(
            x_scaler.fit_transform(X_train_raw[num_cols]),
            columns=num_cols, index=X_train_raw.index,
        )
        test_num = pd.DataFrame(
            x_scaler.transform(X_test_raw[num_cols]),
            columns=num_cols, index=X_test_raw.index,
        )
    else:
        train_num = pd.DataFrame(index=X_train_raw.index)
        test_num = pd.DataFrame(index=X_test_raw.index)

    # One-hot encode categorical columns (codes derived from the full raw data).
    cat_meta: Dict[str, Dict[str, object]] = {}
    train_cat: Dict[str, pd.DataFrame] = {}
    test_cat: Dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        source_values = _sorted_unique_categorical_values(X_raw[col])
        code_map = {v: i for i, v in enumerate(source_values)}
        tr_df, names = _one_hot_frame(X_train_raw[col], code_map, len(source_values), col)
        te_df, _ = _one_hot_frame(X_test_raw[col], code_map, len(source_values), col)
        train_cat[col] = tr_df
        test_cat[col] = te_df
        cat_meta[col] = {
            "categories": list(range(len(source_values))),
            "source_values": [str(v) for v in source_values],
            "columns": names,
        }
        logger.info(
            "[%s] Encoded categorical '%s': source_values=%s -> one-hot columns=%s.",
            key, col, cat_meta[col]["source_values"], names,
        )
    if not cat_cols:
        logger.info("[%s] No categorical columns declared; one-hot encoding skipped.", key)

    # Reassemble the feature matrix in the original column order, replacing each
    # categorical column with its one-hot columns.
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    final_names: List[str] = []
    for name in feature_names:
        if name in cat_cols:
            train_parts.append(train_cat[name])
            test_parts.append(test_cat[name])
            final_names.extend(cat_meta[name]["columns"])  # type: ignore[arg-type]
        else:
            train_parts.append(train_num[[name]])
            test_parts.append(test_num[[name]])
            final_names.append(name)

    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1)

    name_to_idx = {n: i for i, n in enumerate(final_names)}
    categorical_groups: Dict[str, Dict[str, object]] = {}
    for col in cat_cols:
        cols = cat_meta[col]["columns"]  # type: ignore[assignment]
        categorical_groups[col] = {
            "indices": [name_to_idx[c] for c in cols],
            "categories": cat_meta[col]["categories"],
            "source_values": cat_meta[col]["source_values"],
            "columns": list(cols),
        }

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    logger.info(
        "[%s] Step 4/6 done: final features=%d (%s).",
        key, len(final_names), final_names,
    )
    logger.info(
        "[%s] Step 5/6: target scaling done with MinMaxScaler to [0, 1].",
        key,
    )

    return {
        "dataset_key": key,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": final_names,
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "auto_categorical_features": list(auto_cat_cols),
        "manual_categorical_overrides": list(manual_cat_cols),
        "categorical_groups": categorical_groups,
        "feature_analysis": {
            "numerical_csv": str(numerical_csv_path),
            "categorical_csv": str(categorical_csv_path),
            "classification_reasons": feature_reasoning,
        },
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "raw_target_min": float(np.min(y_raw)),
        "raw_target_max": float(np.max(y_raw)),
        "target_name": TARGET_NAMES.get(key, "target"),
    }


def _dataset_path(key: str) -> Path:
    return DATA_DIR / key / "data.pkl"


def setup_dataset(key: str, *, force: bool = False) -> Path:
    """Prepare and pickle a dataset, returning the path of the pickle file."""

    path = _dataset_path(key)
    if path.exists() and not force:
        logger.info(
            "[%s] SKIPPED: cached dataset already exists at %s (pass --force to recreate).",
            key, path,
        )
        logger.info("[%s] No raw-load/download/preprocessing was performed.", key)
        return path
    if path.exists() and force:
        logger.info("[%s] Existing cache found at %s and --force is set; recreating.", key, path)
    else:
        logger.info("[%s] No cache found; preparing dataset from raw source.", key)

    data = _prepare_dataset(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(data, handle)
    logger.info(
        "[%s] DONE: saved preprocessed dataset to %s | X_train=%s X_test=%s y_train=%d y_test=%d.",
        key,
        path,
        tuple(data["X_train"].shape),  # type: ignore[index]
        tuple(data["X_test"].shape),  # type: ignore[index]
        len(data["y_train"]),  # type: ignore[arg-type,index]
        len(data["y_test"]),  # type: ignore[arg-type,index]
    )
    auto_cat = list(data.get("auto_categorical_features", []))
    manual_overrides = list(data.get("manual_categorical_overrides", []))
    feature_analysis = data.get("feature_analysis", {})
    if isinstance(feature_analysis, dict):
        numerical_csv = feature_analysis.get("numerical_csv", "numerical_features.csv")
        categorical_csv = feature_analysis.get("categorical_csv", "categorical_features.csv")
    else:
        numerical_csv = "numerical_features.csv"
        categorical_csv = "categorical_features.csv"
    logger.info("[%s] Step 6/6: review automatic categorical decisions.", key)
    logger.warning(
        "[%s] Automatic categorical candidates: %s",
        key,
        auto_cat,
    )
    logger.warning(
        "[%s] Manual categorical overrides currently configured: %s",
        key,
        manual_overrides,
    )
    logger.warning(
        "[%s] Please review %s and %s, then update CATEGORICAL_FEATURES in data_setup.py "
        "to override any automatic decisions you do not want.",
        key,
        numerical_csv,
        categorical_csv,
    )
    return path


def load_dataset(key: str) -> Dict[str, object]:
    """Load a previously prepared dataset pickle."""

    path = _dataset_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached dataset for '{key}' at {path}. "
            f"Run: python -m explainit.experiments.continuos_minlp.data_setup "
            f"--datasets {key}"
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--datasets", nargs="+", default=sorted(DATASETS),
        help="Dataset keys to prepare. Default: all registered.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if a pickle already exists.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)
    for key in args.datasets:
        try:
            setup_dataset(key, force=args.force)
        except Exception as exc:
            logger.error("Failed to prepare '%s': %s", key, exc)


if __name__ == "__main__":
    main()
