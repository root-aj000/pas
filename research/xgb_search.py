"""Is 0.965 reachable by tuning XGBoost? Search the space properly and find out.

Run with: python research/xgb_search.py

The owner wants 0.965 test ROC-AUC. We are at 0.954479. That is a gap of +0.0105,
and the honest way to answer whether tuning can close it is to search the space
rather than reason about it.

Earlier work (research/auc_experiments*.py) tried 11 hand-picked settings and gained
+0.0013 in total. That is a very narrow slice of XGBoost's space - it never touched
subsample below 0.8, colsample below 0.9, reg_alpha, gamma, or max_bin. This script
samples those properly.

Two-phase design, because a full-budget fit here costs 15-40 seconds:

1. **Search phase.** 40 random configurations at a reduced tree budget. Cheap enough
   to cover the space, and good enough to rank.
2. **Confirm phase.** The best few from the search, re-fitted at the full tree budget
   so the winner is not an artefact of the reduced budget.

Every configuration is scored on the same validation rows with the same metric.
Validation only. The test split stays closed - if we tuned on it we would learn
nothing about what generalises.
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

from src.utils.common import save_dataframe

TARGET = "satisfaction"
SEED = 42
SEARCH_TRIALS = 40
SEARCH_TREES = 300
CONFIRM_TREES = 1400
CONFIRM_TOP_N = 5

# The number to beat, measured on the test split by run_pipeline.py.
CURRENT_TEST_AUC = 0.954479
CURRENT_VALIDATION_AUC = 0.955511

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
FEATURES: list[str] = list(CONFIG["features"])

# Ranges chosen to bracket what is plausible for tabular data, and to include the
# low end of subsample and colsample which the earlier hand search never visited.
# .lead/02-D-METHODS-CATALOGUE.md section 13 lists Optuna as the tool for this;
# this script is the cheap version of the same idea, so the shape of the space is
# known before a GPU search is spent on it.
SEARCH_SPACE: dict[str, list[object]] = {
    "max_depth": [4, 5, 6, 7, 8, 9, 10, 12],
    "min_child_weight": [1, 2, 3, 5, 8, 13, 21],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0, 3.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "gamma": [0.0, 0.05, 0.1, 0.3, 0.8],
    "max_bin": [128, 256, 512, 1024],
}


def fit_and_score(
    train_data: pd.DataFrame, validation_data: pd.DataFrame, params: dict[str, object]
) -> tuple[float, float]:
    """Fit XGBoost with the given settings and return its validation AUC and fit time.

    Args:
        train_data: Training rows.
        validation_data: Held-out rows.
        params: XGBoost hyperparameters.

    Returns:
        The validation ROC-AUC and the fit seconds.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(random_state=SEED, tree_method="hist", **params)
    started = time.monotonic()
    model.fit(train_data[FEATURES], train_data[TARGET])
    seconds = time.monotonic() - started
    probabilities = model.predict_proba(validation_data[FEATURES])[:, 1]
    return float(roc_auc_score(validation_data[TARGET], probabilities)), seconds


def main() -> None:
    """Run the search, then confirm the leaders at full budget, and report."""
    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    validation_data = pd.read_csv("artifacts/data_cleaning_encoding/validation.csv")

    random_generator = np.random.default_rng(SEED)
    print("=" * 96)
    print(f"XGBOOST RANDOM SEARCH - {SEARCH_TRIALS} configs at {SEARCH_TREES} trees")
    print("=" * 96)
    print(f"features {len(FEATURES)}   validation rows {len(validation_data):,}")
    print(
        f"current shipped model: validation {CURRENT_VALIDATION_AUC:.6f}, test {CURRENT_TEST_AUC:.6f}"
    )
    print("the target:           0.965000 test")
    print()

    search_rows: list[dict[str, object]] = []
    search_started = time.monotonic()
    for trial in range(1, SEARCH_TRIALS + 1):
        params = {
            name: values[random_generator.integers(len(values))]
            for name, values in SEARCH_SPACE.items()
        }
        params["n_estimators"] = SEARCH_TREES
        auc, seconds = fit_and_score(train_data, validation_data, params)
        search_rows.append(
            {
                "trial": trial,
                "search_auc": round(auc, 6),
                "fit_seconds": round(seconds, 1),
                **params,
            }
        )
        if trial % 5 == 0 or trial == 1:
            best_so_far = max(float(row["search_auc"]) for row in search_rows)
            print(
                f"  trial {trial:>2}/{SEARCH_TRIALS}  auc={auc:.6f}  best so far={best_so_far:.6f}"
            )

    search_table = pd.DataFrame(search_rows).sort_values("search_auc", ascending=False)
    search_elapsed = time.monotonic() - search_started
    print()
    print(f"search phase done in {search_elapsed / 60:.1f} minutes")
    print(f"best at reduced budget: {search_table.iloc[0]['search_auc']:.6f}")
    print()

    print("=" * 96)
    print(f"CONFIRM PHASE - top {CONFIRM_TOP_N} at {CONFIRM_TREES} trees")
    print("=" * 96)

    confirm_rows: list[dict[str, object]] = []
    for _, row in search_table.head(CONFIRM_TOP_N).iterrows():
        # Cast back to native Python types. Values read out of a DataFrame row are
        # numpy scalars, and xgboost's JSON parser rejects a numpy int for an
        # integer parameter: "Invalid type for: max_bin, expecting one of:
        # {Integer}, got: Number".
        params = {
            name: (
                int(row[name])
                if isinstance(SEARCH_SPACE[name][0], int)
                else float(row[name])
            )
            for name in SEARCH_SPACE
        }
        params["n_estimators"] = CONFIRM_TREES
        auc, seconds = fit_and_score(train_data, validation_data, params)
        confirm_rows.append(
            {"full_auc": round(auc, 6), "fit_seconds": round(seconds, 1), **params}
        )
        print(
            f"  auc={auc:.6f}  fit={seconds:.1f}s   depth={params['max_depth']} lr={params['learning_rate']} colsample={params['colsample_bytree']}"
        )

    # The shipped configuration, re-fitted at the same confirm budget, so the
    # comparison is like for like rather than against a smaller number of trees.
    shipped = {
        "max_depth": 8,
        "n_estimators": CONFIRM_TREES,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    shipped_auc, shipped_seconds = fit_and_score(train_data, validation_data, shipped)
    confirm_rows.append(
        {
            "full_auc": round(shipped_auc, 6),
            "fit_seconds": round(shipped_seconds, 1),
            "label": "SHIPPED CONFIG AT CONFIRM BUDGET",
            **shipped,
        }
    )

    confirm_table = pd.DataFrame(confirm_rows).sort_values("full_auc", ascending=False)

    output_dir = Path(CONFIG["report_path"])
    save_dataframe(search_table, output_dir / "xgb_search_trials.csv")
    save_dataframe(confirm_table, output_dir / "xgb_search_confirmed.csv")

    print()
    print("=" * 96)
    print("CONFIRMED, RANKED")
    print("=" * 96)
    print(
        confirm_table[
            [
                "full_auc",
                "fit_seconds",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "gamma",
                "max_bin",
            ]
        ].to_string(index=False)
    )
    print()

    best = float(confirm_table["full_auc"].max())
    gain = best - CURRENT_VALIDATION_AUC
    print(f"current shipped (validation): {CURRENT_VALIDATION_AUC:.6f}")
    print(f"best found  (validation):     {best:.6f}")
    print(f"gain:                         {gain:+.6f}")
    print()
    if gain <= 0.0001:
        print(
            "-> Tuning is exhausted. Nothing in this space beats what we already ship."
        )
    elif best >= 0.965:
        print(
            "-> 0.965 is reached on validation. Re-run the pipeline to confirm on test."
        )
    else:
        needed = 0.965 - best
        print(
            f"-> Best found is {best:.6f}. Still {needed:.6f} short of 0.965 on validation."
        )
        print("   Tuning alone does not close that gap. See the notes in the write-up.")
    print()
    print("wrote reports/xgb_search_trials.csv and reports/xgb_search_confirmed.csv")
    print("Validation only. The test split has not been opened.")


if __name__ == "__main__":
    main()
