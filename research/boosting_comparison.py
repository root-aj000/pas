"""XGBoost vs LightGBM vs CatBoost, same folds, blended out-of-fold.

Raw data -> features -> three boosters -> OOF predictions -> blending -> AUC.

Run full comparison with:  python research/boosting_comparison.py
Run a fast smoke test with: python research/boosting_comparison.py --quick

"Don't assume XGBoost is the strongest model." Only one of the three gradient
boosting implementations had ever been fitted here. LightGBM and CatBoost both
handle categoricals natively too.

### The stacking is genuinely out-of-fold

The rule, from the owner: *"Don't choose blend weights using the validation
labels and then report the same validation score as if it were untouched."*

```
1. 5-fold CV over TRAIN only -> out-of-fold predictions per model.
   Every OOF prediction comes from a model that never saw that row's label.
2. Fit blend weights on those OOF predictions -> never see a validation label.
3. Fit each model on the FULL train split. Predict validation once.
4. Blend, and score. Validation is touched exactly once, for scoring.
```

Three blends reported: uniform, the suggested 0.45/0.35/0.20, and OOF-fitted.

Validation only. The test split stays closed.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.common import apply_categorical_encoding, load_json, save_dataframe

SEED = 42
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]
MEMBER_NAMES = ["xgboost", "lightgbm", "catboost"]

# The suggested weights, used as given rather than tuned.
SUGGESTED_WEIGHTS = np.array([0.45, 0.35, 0.20])

logger = logging.getLogger("compare")


def build_model(name: str, seed: int, quick: bool):
    """Return an unfitted booster with the comparison settings.

    Args:
        name: One of xgboost, lightgbm, catboost.
        seed: Random seed.
        quick: If True, use 50 trees so a smoke test finishes in seconds.

    Returns:
        An unfitted classifier.

    Raises:
        ValueError: If the name is not one of the three.
    """
    trees = 50 if quick else None
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            max_depth=8,
            n_estimators=trees or 1400,
            learning_rate=0.10,
            subsample=1.0,
            colsample_bytree=0.6,
            reg_alpha=0.01,
            reg_lambda=2.0,
            gamma=0.1,
            max_bin=1024,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=trees or 1400,
            learning_rate=0.10,
            num_leaves=31,
            min_child_samples=40,
            colsample_bytree=0.6,
            reg_alpha=0.01,
            reg_lambda=2.0,
            max_bin=255,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=trees or 600,
            learning_rate=0.15,
            depth=6,
            l2_leaf_reg=2.0,
            cat_features=list(CATEGORICAL_COLUMNS),
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise ValueError(f"Unknown model '{name}'. Expected one of {MEMBER_NAMES}.")


def as_model_input(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return the frame in the form the named model reads.

    Args:
        frame: Prepared rows with `category` dtype categoricals.
        name: Which model will read it.

    Returns:
        The frame, with categoricals as plain strings for CatBoost only.
        CatBoost reads raw strings, not pandas `category`.
    """
    if name != "catboost":
        return frame
    out = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = out[column].astype(str)
    return out


def predict_member(
    name: str,
    fitting: pd.DataFrame,
    scoring: pd.DataFrame,
    features: list[str],
    target: str,
    seed: int,
) -> np.ndarray:
    """Fit one member and return its scores on the scoring rows.

    Args:
        name: Which model to fit.
        fitting: Rows to fit on.
        scoring: Rows to score.
        features: The feature list, identical for every member.
        target: The label column.
        seed: Random seed.

    Returns:
        Probability of satisfaction for each scoring row.
    """
    model = build_model(name, seed, quick=False)
    model.fit(as_model_input(fitting, name)[features], fitting[target])
    return model.predict_proba(as_model_input(scoring, name)[features])[:, 1]


def load_splits(quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Load the prepared splits with the categorical dtype re-applied.

    Args:
        quick: If True, use a 20k-row subsample so a smoke test is fast.

    Returns:
        The train rows, validation rows, feature list, and target name.
    """
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    categorical = list(contract.get("categorical_columns", []))
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    train = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        categorical,
        encoding,
        "compare",
    )
    validation = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        categorical,
        encoding,
        "compare",
    )
    config = yaml.safe_load(Path("config.yaml").read_text())
    if quick:
        train = train.sample(n=20000, random_state=SEED)
        validation = validation.sample(n=5000, random_state=SEED)
    return train, validation, list(config["features"]), str(config["target_column"])


def fit_blend_weights(oof_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit non-negative blend weights on out-of-fold predictions.

    Args:
        oof_matrix: Out-of-fold predictions, shape (rows, members).
        labels: Labels for those rows.

    Returns:
        Weights summing to one.
    """
    count = oof_matrix.shape[1]
    weights = np.full(count, 1.0 / count)
    for _ in range(4):
        improved = False
        for index in range(count):
            current = weights[index]
            best_value, best_score = current, -1.0
            for candidate in np.linspace(0.0, 1.0, 11):
                trial = weights.copy()
                trial[index] = candidate
                if trial.sum() <= 0:
                    continue
                trial = trial / trial.sum()
                score = roc_auc_score(labels, oof_matrix @ trial)
                if score > best_score:
                    best_value, best_score = candidate, score
            if best_value != current:
                weights[index] = best_value
                improved = True
        if not improved:
            break
    return weights / weights.sum()


def main(quick: bool) -> int:
    """Run the comparison and report members and blends.

    Args:
        quick: If True, 2 folds on a subsample with 50 trees. Tests every code
            path in under a minute. If False, the full 5-fold comparison.

    Returns:
        0 if every member and blend was scored.
    """
    folds = 2 if quick else 5
    train, validation, features, target = load_splits(quick)
    labels = validation[target].to_numpy()
    print(
        f"mode={'QUICK smoke test' if quick else 'FULL comparison'}  folds={folds}  "
        f"train={len(train):,}  validation={len(validation):,}  features={len(features)}"
    )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), len(MEMBER_NAMES)))
    started = time.monotonic()
    for fold, (fitting_index, scoring_index) in enumerate(
        splitter.split(train, train[target]), start=1
    ):
        for column, name in enumerate(MEMBER_NAMES):
            model = build_model(name, SEED, quick=quick)
            fitting = train.iloc[fitting_index]
            scoring = train.iloc[scoring_index]
            model.fit(as_model_input(fitting, name)[features], fitting[target])
            oof[scoring_index, column] = model.predict_proba(
                as_model_input(scoring, name)[features]
            )[:, 1]
        print(
            f"  fold {fold}/{folds} done  {time.monotonic() - started:.0f}s", flush=True
        )
    train_labels = train[target].to_numpy()

    print("\nOUT-OF-FOLD ON TRAIN")
    for column, name in enumerate(MEMBER_NAMES):
        print(f"  {name:<12} {roc_auc_score(train_labels, oof[:, column]):.6f}")

    # Provisional blend of the OOF matrix, so the run tells us something even
    # before the full-train fits finish.
    fitted_weights = fit_blend_weights(oof, train_labels)
    print(
        "\nweights fitted on OOF:",
        {n: round(float(w), 4) for n, w in zip(MEMBER_NAMES, fitted_weights)},
    )

    predictions = np.column_stack(
        [
            predict_member(name, train, validation, features, target, SEED)
            for name in MEMBER_NAMES
        ]
    )
    print("\nVALIDATION (touched once, for scoring)")
    member_auc = {}
    for column, name in enumerate(MEMBER_NAMES):
        member_auc[name] = float(roc_auc_score(labels, predictions[:, column]))
        print(f"  {name:<12} {member_auc[name]:.6f}")
    best_single = max(member_auc.values())

    blends = {
        "uniform average": predictions.mean(axis=1),
        "suggested 0.45/0.35/0.20": predictions @ SUGGESTED_WEIGHTS,
        "OOF-fitted weights": predictions @ fitted_weights,
    }
    print("\nBLENDS")
    blend_auc = {}
    for label, blended in blends.items():
        blend_auc[label] = float(roc_auc_score(labels, blended))
        print(
            f"  {label:<24} {blend_auc[label]:.6f}   {blend_auc[label] - best_single:+.6f} vs best single"
        )

    if not quick:
        output = Path("reports/boosting_comparison.csv")
        save_dataframe(
            pd.DataFrame(
                [
                    {"model": name, "validation_auc": round(member_auc[name], 6)}
                    for name in MEMBER_NAMES
                ]
                + [
                    {"model": label, "validation_auc": round(value, 6)}
                    for label, value in blend_auc.items()
                ]
            ),
            output,
        )
        np.save("reports/boosting_oof.npy", oof)
        np.save("reports/boosting_validation.npy", predictions)
        print(f"\nwrote {output} and the prediction matrices")
        with open("logs/dev/boosting_comparison_summary.txt", "w") as log:
            log.write(f"mode=FULL folds={folds} seed={SEED}\n")
            for name in MEMBER_NAMES:
                log.write(f"member {name} {member_auc[name]:.6f}\n")
            for label, value in blend_auc.items():
                log.write(f"blend {label} {value:.6f}\n")
        print("wrote logs/dev/boosting_comparison_summary.txt")
    print("\nValidation only. The test split stays closed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare three boosters with out-of-fold blending."
    )
    parser.add_argument(
        "--quick", action="store_true", help="Smoke test: 2 folds, subsample, 50 trees."
    )
    sys.exit(main(quick=parser.parse_args().quick))
