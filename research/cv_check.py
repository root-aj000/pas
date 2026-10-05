"""Is the 0.9589 validation AUC real, or is the single split flattering it?

Run with: python research/cv_check.py

The question this answers: our score comes from one validation split of 104,946
rows. If that split happened to be easy, the number is inflated and every decision
made on it is suspect. Five-fold cross-validation re-estimates the same score on
five different training sets, so a tight spread means the model is not sensitive to
which rows it saw.

Two configurations are measured, not one:

* **shipped** - exactly what config.yaml holds and what model_4 was trained with.
* **proposed** - the settings suggested for checking: n_estimators 2000,
  learning_rate 0.03, max_depth 6, min_child_weight 3, subsample 0.8,
  colsample_bytree 0.8, reg_alpha 0.1, reg_lambda 2.0.

Two things this script has to handle that a naive snippet gets wrong:

1. **The categorical dtype.** `artifacts/data_cleaning_encoding/train.csv` is CSV,
   and CSV does not carry pandas `category`. Read straight, the four categorical
   columns arrive as `str` and xgboost rejects them. The dtype is re-applied from
   the contract stage 2 wrote.
2. **Cross-validation trains on 80%, our split trains on 70%.** So these CV scores
   are NOT comparable with the 0.958844 validation number - they should come out
   slightly higher, because each fold sees more data. They are only comparable with
   each other.

Nothing here changes the shipped model. Tuning on a cross-validation score would
make that score part of training, and the frozen test split is the only untouched
measurement we have.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.common import (
    apply_categorical_encoding,
    load_json,
    save_dataframe,
)

SEED = 42
N_SPLITS = 5

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
TARGET = str(CONFIG["target_column"])

SHIPPED_PARAMS = dict(CONFIG["model_params"])
SHIPPED_PARAMS["enable_categorical"] = True
SHIPPED_PARAMS["tree_method"] = "hist"
SHIPPED_PARAMS["random_state"] = SEED

PROPOSED_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": SEED,
}


def cross_validate_one(
    features: pd.DataFrame, labels: pd.Series, params: dict[str, object], name: str
) -> dict[str, object]:
    """Score one configuration across five stratified folds.

    Args:
        features: The prepared training rows.
        labels: The training labels.
        params: XGBoost hyperparameters.
        name: What this configuration is called.

    Returns:
        Per-fold scores plus the mean and spread.

    Note:
    The folds run sequentially and each fit uses every core. Passing n_jobs=-1 to
    both cross_val_score and the estimator oversubscribes the machine and is
    usually slower than doing one at a time.
    """
    from xgboost import XGBClassifier

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores: list[float] = []
    fold_rows: list[pd.Series] = []
    fold_seconds: list[float] = []

    for fold, (fitting_index, scoring_index) in enumerate(
        splitter.split(features, labels), start=1
    ):
        model = XGBClassifier(n_jobs=-1, **params)
        started = time.monotonic()
        model.fit(features.iloc[fitting_index], labels.iloc[fitting_index])
        seconds = time.monotonic() - started
        probabilities = model.predict_proba(features.iloc[scoring_index])[:, 1]
        score = float(roc_auc_score(labels.iloc[scoring_index], probabilities))
        scores.append(score)
        fold_seconds.append(seconds)
        fold_rows.append(
            pd.Series(
                {f"fold_{fold}": round(score, 6), f"seconds_{fold}": round(seconds, 1)}
            )
        )
        print(f"    fold {fold}/{N_SPLITS}  roc_auc={score:.6f}  fit={seconds:.1f}s")

    fold_table = pd.DataFrame(fold_rows).iloc[0]
    array = np.array(scores)
    return {
        "config": name,
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "spread": round(float(array.max() - array.min()), 6),
        "per_fold": ", ".join(f"{value:.6f}" for value in scores),
        "fold_table": fold_table,
        "mean_seconds": round(float(np.mean(fold_seconds)), 1),
    }


def main() -> None:
    """Run five-fold cross-validation on both configurations and interpret it."""
    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    train_data = apply_categorical_encoding(
        train_data,
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
        "cv",
    )

    features = train_data[list(CONFIG["features"])]
    labels = train_data[TARGET]
    print("=" * 92)
    print("FIVE-FOLD CROSS-VALIDATION - is the score real, or is the split easy?")
    print("=" * 92)
    print(
        f"rows {len(train_data):,}   features {features.shape[1]}   folds {N_SPLITS}   seed {SEED}"
    )
    print(f"categorical encoding: {contract.get('categorical_encoding')}")
    print()
    print("Each fold trains on 80% (392k rows). Our validation split trains on 70%")
    print("(490k rows), so these numbers are NOT comparable with 0.958844. They are")
    print("comparable with each other.")
    print()

    results: list[dict[str, object]] = []
    for name, params in (
        ("shipped (config.yaml)", SHIPPED_PARAMS),
        ("proposed (2000 trees, depth 6)", PROPOSED_PARAMS),
    ):
        print(f"  {name}")
        results.append(cross_validate_one(features, labels, params, name))
        print()

    summary = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "fold_table"} for row in results]
    )
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(summary.to_string(index=False))
    print()

    output_dir = Path(CONFIG["report_path"])
    save_dataframe(summary, output_dir / "cv_check_summary.csv")
    save_dataframe(
        pd.DataFrame([row["fold_table"] for row in results]),
        output_dir / "cv_check_folds.csv",
    )

    print(f"wrote {output_dir}/cv_check_summary.csv and cv_check_folds.csv")
    print()
    print("The number to compare against, from runs nobody tuned on:")
    print("  frozen test split, model_4:  0.957894")
    print("  single validation split:     0.958844")
    print("  gap:                         -0.000950")
    print()
    print("Read the spread column. Tight means the model does not care which rows it")
    print("trained on, so the score is a property of the model and not of the split.")


if __name__ == "__main__":
    main()
