"""Round 2: the model is underfit. Push capacity, combine winners, try an ensemble.

Run with: python research/auc_experiments_round2.py

Round 1 (`research/auc_experiments.py`) found two things:

1. **Every feature idea failed.** Seven candidates landed between -0.000067 and
   +0.000044. Twelve zero-indicator columns moved the score by exactly 0.000000.
   The raw ratings already carry everything they carry.
2. **Every hyperparameter gain pointed the same way** - smaller leaves, more
   trees. That is the signature of an underfit model, not of a badly chosen
   setting. `min_samples_leaf=40` on 490,000 rows is very coarse.

So this round does three things: combine the winners, push further in the same
direction to find where it stops paying, and test whether two different model
families disagree enough to be worth averaging.

Validation split only. The test split stays closed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.utils.common import save_dataframe

SEED = 42
TARGET = "satisfaction"
CONFIG = yaml.safe_load(Path("config.yaml").read_text())
BASE_FEATURES: list[str] = list(CONFIG["features"])


def fit_and_score_hist(
    features, train_data, validation_data, params
) -> tuple[float, float, object]:
    """Fit HistGradientBoosting and return its AUC, fit time and model.

    Args:
        features: The feature list.
        train_data: Training rows.
        validation_data: Held-out rows.
        params: Hyperparameters.

    Returns:
        The validation ROC-AUC, the fit seconds, and the fitted model.
    """
    model = HistGradientBoostingClassifier(random_state=SEED, **params)
    started = time.monotonic()
    model.fit(train_data[features], train_data[TARGET])
    seconds = time.monotonic() - started
    auc = float(
        roc_auc_score(
            validation_data[TARGET],
            model.predict_proba(validation_data[features])[:, 1],
        )
    )
    return auc, seconds, model


def fit_and_score_xgboost(
    features, train_data, validation_data, params
) -> tuple[float, float, object]:
    """Fit XGBClassifier and return its AUC, fit time and model.

    Args:
        features: The feature list.
        train_data: Training rows.
        validation_data: Held-out rows.
        params: XGBoost hyperparameters.

    Returns:
        The validation ROC-AUC, the fit seconds, and the fitted model.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(random_state=SEED, **params)
    started = time.monotonic()
    model.fit(train_data[features], train_data[TARGET])
    seconds = time.monotonic() - started
    auc = float(
        roc_auc_score(
            validation_data[TARGET],
            model.predict_proba(validation_data[features])[:, 1],
        )
    )
    return auc, seconds, model


def main() -> None:
    """Run round 2 and print one ranked table."""
    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    validation_data = pd.read_csv("artifacts/data_cleaning_encoding/validation.csv")
    features = BASE_FEATURES

    results: list[dict[str, object]] = []
    keep_probabilities: dict[str, object] = {}

    def record(label: str, family: str, auc: float, seconds: float) -> None:
        """Store one result row."""
        results.append(
            {
                "variant": label,
                "family": family,
                "roc_auc": round(auc, 6),
                "fit_seconds": round(seconds, 1),
            }
        )

    print("=" * 92)
    print("COMBINING THE ROUND 1 WINNERS - capacity up, learning rate down")
    print("=" * 92)

    hist_candidates: list[tuple[str, dict[str, object]]] = [
        (
            "baseline (shipped): leaf=40, leaves=31, iter=400, lr=0.06",
            {
                "learning_rate": 0.06,
                "max_iter": 400,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 40,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
        (
            "leaf=10, iter=500 (round 1 winner)",
            {
                "learning_rate": 0.06,
                "max_iter": 500,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 10,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
        (
            "leaf=10 + leaves=63",
            {
                "learning_rate": 0.06,
                "max_iter": 500,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 10,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
        (
            "leaf=10 + leaves=63 + lr=0.03",
            {
                "learning_rate": 0.03,
                "max_iter": 900,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 10,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
        (
            "leaf=10 + leaves=63 + lr=0.03 + l2=0",
            {
                "learning_rate": 0.03,
                "max_iter": 900,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 10,
                "l2_regularization": 0.0,
                "early_stopping": False,
            },
        ),
        (
            "leaf=20 + leaves=63 + lr=0.03",
            {
                "learning_rate": 0.03,
                "max_iter": 900,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 20,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
        (
            "leaf=10 + leaves=127, push harder",
            {
                "learning_rate": 0.05,
                "max_iter": 700,
                "max_leaf_nodes": 127,
                "min_samples_leaf": 10,
                "l2_regularization": 0.5,
                "early_stopping": False,
            },
        ),
    ]

    for label, params in hist_candidates:
        auc, seconds, model = fit_and_score_hist(
            features, train_data, validation_data, params
        )
        record(label, "HistGradientBoosting", auc, seconds)
        keep_probabilities[label] = model.predict_proba(validation_data[features])[:, 1]
        print(f"  {label:<52} auc={auc:.6f}  fit={seconds:.1f}s")

    print()
    print("=" * 92)
    print("XGBoost WITH MORE CAPACITY - it won round 1 at defaults by 0.0010")
    print("=" * 92)

    xgb_candidates: list[tuple[str, dict[str, object]]] = [
        ("xgboost defaults", {}),
        (
            "xgboost depth 6, 600 trees, lr 0.05",
            {
                "max_depth": 6,
                "n_estimators": 600,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
            },
        ),
        (
            "xgboost depth 8, 1000 trees, lr 0.03",
            {
                "max_depth": 8,
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
            },
        ),
        (
            "xgboost depth 10, 1500 trees, lr 0.02",
            {
                "max_depth": 10,
                "n_estimators": 1500,
                "learning_rate": 0.02,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
            },
        ),
    ]

    for label, params in xgb_candidates:
        auc, seconds, model = fit_and_score_xgboost(
            features, train_data, validation_data, params
        )
        record(label, "XGBoost", auc, seconds)
        keep_probabilities[label] = model.predict_proba(validation_data[features])[:, 1]
        print(f"  {label:<52} auc={auc:.6f}  fit={seconds:.1f}s")

    print()
    print("=" * 92)
    print("ENSEMBLE - do two different families disagree enough to be worth averaging?")
    print("=" * 92)

    best_hist_label = max(
        (r for r in results if r["family"] == "HistGradientBoosting"),
        key=lambda r: r["roc_auc"],
    )["variant"]
    best_xgb_label = max(
        (r for r in results if r["family"] == "XGBoost"), key=lambda r: r["roc_auc"]
    )["variant"]
    hist_probabilities = keep_probabilities[best_hist_label]
    xgb_probabilities = keep_probabilities[best_xgb_label]
    target = validation_data[TARGET]

    for weight in (0.3, 0.4, 0.5, 0.6, 0.7):
        blended = weight * xgb_probabilities + (1 - weight) * hist_probabilities
        auc = float(roc_auc_score(target, blended))
        label = f"blend: {weight:.1f}*xgb + {1 - weight:.1f}*hist"
        record(label, "Blend", auc, 0.0)
        print(f"  {label:<52} auc={auc:.6f}")

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    output = Path(CONFIG["report_path"]) / "auc_experiments_round2.csv"
    save_dataframe(table, output)

    print()
    print("=" * 92)
    print("RANKED")
    print("=" * 92)
    print(table.to_string(index=False))
    print()
    print(f"wrote {output}")
    print("Validation only. The test split has not been opened.")
    print()
    print(
        f"Best single model : {table.iloc[0]['variant']} at {table.iloc[0]['roc_auc']:.6f}"
    )
    best_blend = table[table["family"] == "Blend"].iloc[0]
    print(f"Best blend       : {best_blend['variant']} at {best_blend['roc_auc']:.6f}")
    if float(best_blend["roc_auc"]) > float(table.iloc[0]["roc_auc"]):
        print(
            "  -> the blend beats the best single model, so the families do disagree."
        )


if __name__ == "__main__":
    main()
