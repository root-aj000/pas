"""Fix the Class over-trust. Four targeted attempts, plus the ceiling.

Run with: python research/fix_class_overtrust.py

Step 9 (research/read_errors.py) found the pattern: the model is wrong exactly where
the Class signal says it should be right. Satisfied Eco passengers get steamrolled
because the model learned "Eco = dissatisfied", and dissatisfied Business passengers
because it learned "Business = satisfied". 3,757 hard rows, 3.6% of validation.

The model is not missing a feature. It is over-trusting one. So this script does not
add features. It attacks the trust directly, four ways:

1. THE CEILING - if every hard row were ranked perfectly, how much AUC is even
   available? If the answer is +0.002, stop reading and go do something else.
2. SEGMENT MODELS - train one model per Class, with Class removed. A model that
   never sees Class cannot use it as a crutch.
3. CLASS-RELATIVE FEATURES - each rating minus its class mean. Forces the model to
   see deviation from class expectation instead of the raw value.
4. INTERACTION CONSTRAINTS - forbid Class from splitting alone. XGBoost's
   `interaction_constraints` makes Class appear only together with ratings.
5. DEEPER TREES - depth 10-12, so the model has capacity left to learn the
   within-class patterns after the Class splits. Already partially explored.

Validation only. The test split stays closed.
"""

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.utils.common import apply_categorical_encoding, load_json, save_dataframe

SEED = 42
TARGET = "satisfaction"
HARD_POSITIVE_CUTOFF = 0.10
HARD_NEGATIVE_CUTOFF = 0.90

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
FEATURES: list[str] = list(CONFIG["features"])

BASE_PARAMS = {
    "max_depth": 8,
    "n_estimators": 1400,
    "learning_rate": 0.10,
    "subsample": 1.0,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.01,
    "reg_lambda": 2.0,
    "gamma": 0.1,
    "max_bin": 1024,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_shipped() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load validation rows, the shipped model, and its probabilities.

    Returns:
        The validation frame, the feature list, and the shipped probabilities.
    """
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    validation = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
        "fix",
    )
    saved = load_json(Path("models/model_4/features.json"))
    model = joblib.load(Path("models/model_4/model.pkl"))
    probabilities = model.predict_proba(validation[list(saved["features"])])[:, 1]
    return validation, list(saved["features"]), probabilities


def measure_ceiling(labels: np.ndarray, probabilities: np.ndarray) -> None:
    """Print how much AUC is available if every hard row were ranked perfectly.

    Args:
        labels: The true answers.
        probabilities: The shipped model's scores.
    """
    hard_positive = (labels == 1) & (probabilities < HARD_POSITIVE_CUTOFF)
    hard_negative = (labels == 0) & (probabilities > HARD_NEGATIVE_CUTOFF)
    print(
        f"hard positives: {int(hard_positive.sum()):,}   hard negatives: {int(hard_negative.sum()):,}"
    )

    # Move every hard positive above every negative, and every hard negative below
    # every positive, without touching anything else. That is the best any fix that
    # only touches these rows can do.
    fixed = probabilities.copy()
    fixed[hard_positive] = 1.0
    fixed[hard_negative] = 0.0
    current = float(roc_auc_score(labels, probabilities))
    ceiling = float(roc_auc_score(labels, fixed))
    print(f"current AUC:  {current:.6f}")
    print(f"ceiling AUC:  {ceiling:.6f}")
    print(f"available:    {ceiling - current:+.6f}")
    print()


def fit_xgboost(train_features, train_labels, params) -> Any:
    """Fit XGBoost and return the model.

    Args:
        train_features: Rows to fit on.
        train_labels: Labels.
        params: Hyperparameters.

    Returns:
        The fitted model.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(train_features, train_labels)
    return model


def main() -> None:
    """Measure the ceiling, then test the four fixes."""
    validation, features, shipped_probabilities = load_shipped()
    labels = validation[TARGET].to_numpy()
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    train = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
        "fix",
    )

    print("=" * 92)
    print("1. THE CEILING - how much AUC is even available in the hard rows?")
    print("=" * 92)
    measure_ceiling(labels, shipped_probabilities)

    results = [
        {
            "variant": "shipped model_4",
            "roc_auc": round(float(roc_auc_score(labels, shipped_probabilities)), 6),
        }
    ]

    print("=" * 92)
    print("2. SEGMENT MODELS - one model per Class, Class removed")
    print("=" * 92)
    segment_probabilities = np.zeros(len(validation))
    for segment in ("Business", "Eco", "Eco Plus"):
        segment_train = train[train["Class"] == segment]
        segment_validation = validation[validation["Class"] == segment]
        if len(segment_validation) == 0:
            continue
        segment_features = [f for f in features if f != "Class"]
        started = time.monotonic()
        model = fit_xgboost(
            segment_train[segment_features], segment_train[TARGET], BASE_PARAMS
        )
        seconds = time.monotonic() - started
        segment_probabilities[segment_validation.index] = model.predict_proba(
            segment_validation[segment_features]
        )[:, 1]
        segment_auc = float(
            roc_auc_score(
                segment_validation[TARGET],
                segment_probabilities[segment_validation.index],
            )
        )
        print(
            f"  Class={segment:<10} train={len(segment_train):>7,}  val={len(segment_validation):>6,}  auc={segment_auc:.6f}  {seconds:.0f}s"
        )
    segment_auc = float(roc_auc_score(labels, segment_probabilities))
    print(f"  pooled segments: {segment_auc:.6f}")
    results.append(
        {"variant": "segment models (no Class)", "roc_auc": round(segment_auc, 6)}
    )

    print()
    print("=" * 92)
    print("3. CLASS-RELATIVE FEATURES - each rating minus its class mean")
    print("=" * 92)
    rating_columns = [
        c
        for c in features
        if c
        not in (
            "Class",
            "Gender",
            "Customer Type",
            "Type of Travel",
            "Age",
            "Flight Distance",
        )
    ]
    class_means = train.groupby("Class", observed=True)[rating_columns].mean()
    relative_train = train.copy()
    relative_validation = validation.copy()
    for column in rating_columns:
        mapping = class_means[column].to_dict()
        relative_train[f"{column}_vs_class"] = relative_train[column].astype(
            float
        ) - relative_train["Class"].map(mapping).astype(float)
        relative_validation[f"{column}_vs_class"] = relative_validation[column].astype(
            float
        ) - relative_validation["Class"].map(mapping).astype(float)
    relative_features = features + [f"{c}_vs_class" for c in rating_columns]
    started = time.monotonic()
    model = fit_xgboost(
        relative_train[relative_features], relative_train[TARGET], BASE_PARAMS
    )
    seconds = time.monotonic() - started
    relative_probabilities = model.predict_proba(
        relative_validation[relative_features]
    )[:, 1]
    relative_auc = float(roc_auc_score(labels, relative_probabilities))
    print(
        f"  +{len(rating_columns)} relative features: {relative_auc:.6f}  {seconds:.0f}s"
    )
    results.append(
        {"variant": "class-relative features", "roc_auc": round(relative_auc, 6)}
    )

    print()
    print("=" * 92)
    print("4. INTERACTION CONSTRAINTS - Class may never split alone")
    print("=" * 92)
    # Feature NAMES, not indices. With a pandas frame the DMatrix always carries
    # names, and integer indices then fail with "Constrained features are not a
    # subset of training data feature names".
    others = [f for f in features if f != "Class"]
    constrained_params = dict(BASE_PARAMS)
    constrained_params["interaction_constraints"] = [["Class"] + others]
    started = time.monotonic()
    model = fit_xgboost(train[features], train[TARGET], constrained_params)
    seconds = time.monotonic() - started
    constrained_probabilities = model.predict_proba(validation[features])[:, 1]
    constrained_auc = float(roc_auc_score(labels, constrained_probabilities))
    print(f"  Class forced to interact: {constrained_auc:.6f}  {seconds:.0f}s")
    results.append(
        {"variant": "interaction constraints", "roc_auc": round(constrained_auc, 6)}
    )

    print()
    print("=" * 92)
    print("5. DEEPER TREES - depth 12, room to learn within-class patterns")
    print("=" * 92)
    deep_params = dict(BASE_PARAMS)
    deep_params["max_depth"] = 12
    deep_params["min_child_weight"] = 5
    started = time.monotonic()
    model = fit_xgboost(train[features], train[TARGET], deep_params)
    seconds = time.monotonic() - started
    deep_probabilities = model.predict_proba(validation[features])[:, 1]
    deep_auc = float(roc_auc_score(labels, deep_probabilities))
    print(f"  depth 12: {deep_auc:.6f}  {seconds:.0f}s")
    results.append(
        {"variant": "deeper trees (depth 12)", "roc_auc": round(deep_auc, 6)}
    )

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    output = Path(CONFIG["report_path"]) / "class_overtrust_fixes.csv"
    save_dataframe(table, output)
    print()
    print("=" * 92)
    print("RANKED")
    print("=" * 92)
    print(table.to_string(index=False))
    print()
    print(f"wrote {output}")
    print("Validation only. The test split stays closed.")


if __name__ == "__main__":
    main()
