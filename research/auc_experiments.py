"""Measure candidate improvements to ROC-AUC, one at a time, on validation.

Run with: python research/auc_experiments.py

`.lead/02-C` Step 11 is tuning, and it says one method at a time, on the validation
split, with the test split untouched. This script does exactly that: every variant
is scored on the same validation rows with the same seed, and the delta against the
current shipped feature set is what matters.

Nothing here is tuned against the test split. The test split is opened once, at the
end, by run_pipeline.py.

Each experiment is one line of reasoning that can be wrong. The point of this file
is to find out which ones are, using numbers instead of opinion.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.constants import SERVICE_RATING_COLUMNS
from src.utils.common import save_dataframe

SEED = 42
TARGET = "satisfaction"

# The feature list currently in config.yaml, read from there rather than copied,
# so this file cannot drift away from what actually ships.
import yaml

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
BASE_FEATURES: list[str] = list(CONFIG["features"])


def add_rating_extras(frame: pd.DataFrame) -> pd.DataFrame:
    """Add four candidate features built from the 13 ratings.

    Args:
        frame: Rows that already carry the 13 raw ratings.

    Returns:
        A new frame with four extra columns.

    Note:
        Each answers a different question:
          count_of_ratings_at_5        - how many things were excellent
          spread_of_ratings            - consistency, does the passenger rate
                                        everything the same
          count_of_zero_ratings        - the 0s may mean "not applicable" rather
                                        than "worst". See docs/column_dictionary.md
                                        Note 3. This counts them rather than
                                        guessing what they mean.
          mean_service_rating_x_class  - whether the class effect and the service
                                        effect multiply out
    """
    ratings = frame[SERVICE_RATING_COLUMNS]
    extra = pd.DataFrame(index=frame.index)
    extra["count_of_ratings_at_5"] = (ratings == 5).sum(axis=1)
    extra["spread_of_ratings"] = ratings.std(axis=1)
    extra["count_of_zero_ratings"] = (ratings == 0).sum(axis=1)
    extra["mean_service_rating_x_class"] = (
        ratings.mean(axis=1) * frame["Class_Business"]
    )
    return extra


def add_zero_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one indicator column per rating that ever holds a 0.

    Args:
        frame: Rows carrying the 13 raw ratings.

    Returns:
        A new frame with `<rating>_was_zero` columns for the ratings that contain 0.

    Note:
        If 0 means "this service was not offered" rather than "worst possible",
        then the 0 is a missing value wearing a number's clothes. An indicator lets
        the model use that distinction without deciding what 0 means - which is
        still open question 3.
    """
    indicators = pd.DataFrame(index=frame.index)
    for column in SERVICE_RATING_COLUMNS:
        if 0 in set(frame[column].dropna().unique().tolist()):
            indicators[f"{column}_was_zero"] = (frame[column] == 0).astype(int)
    return indicators


def score(
    features: list[str],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    label: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Fit HistGradientBoosting on train and score ROC-AUC on validation.

    Args:
        features: The exact feature list.
        train_data: The training rows.
        validation_data: The held-out rows.
        label: What this variant is called, for the output table.
        params: Hyperparameters, or None for the shipped defaults.

    Returns:
        One row of results, including the change against the baseline.
    """
    model = HistGradientBoostingClassifier(random_state=SEED, **(params or {}))
    started = time.monotonic()
    model.fit(train_data[features], train_data[TARGET])
    fit_seconds = time.monotonic() - started
    score = float(
        roc_auc_score(
            validation_data[TARGET],
            model.predict_proba(validation_data[features])[:, 1],
        )
    )
    return {
        "variant": label,
        "features": len(features),
        "roc_auc": round(score, 6),
        "fit_seconds": round(fit_seconds, 1),
    }


def main() -> None:
    """Run every candidate and print one table, biggest change first."""
    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    validation_data = pd.read_csv("artifacts/data_cleaning_encoding/validation.csv")

    # Build every candidate feature set once, on the train split, and carry the
    # same columns into validation. Anything computed must be computed the same way
    # for both, or it is leakage.
    extras = add_rating_extras(train_data)
    extras_validation = add_rating_extras(validation_data)
    zeros = add_zero_indicators(train_data)
    zeros_validation = add_zero_indicators(validation_data)

    train_with_extras = pd.concat([train_data, extras, zeros], axis=1)
    validation_with_extras = pd.concat(
        [validation_data, extras_validation, zeros_validation], axis=1
    )

    results: list[dict[str, object]] = []

    print("=" * 88)
    print(
        "FEATURE EXPERIMENTS - HistGradientBoosting, shipped defaults, validation split"
    )
    print("=" * 88)

    baseline = score(
        BASE_FEATURES, train_data, validation_data, "BASELINE: config.yaml as shipped"
    )
    results.append(baseline)
    print(
        f"  {baseline['variant']:<52} n={baseline['features']:>2}  auc={baseline['roc_auc']:.6f}"
    )

    candidates: list[tuple[str, list[str], pd.DataFrame, pd.DataFrame]] = [
        (
            "+ count_of_ratings_at_5",
            ["count_of_ratings_at_5"],
            train_with_extras,
            validation_with_extras,
        ),
        (
            "+ spread_of_ratings",
            ["spread_of_ratings"],
            train_with_extras,
            validation_with_extras,
        ),
        (
            "+ count_of_zero_ratings",
            ["count_of_zero_ratings"],
            train_with_extras,
            validation_with_extras,
        ),
        (
            "+ mean_service_rating_x_class",
            ["mean_service_rating_x_class"],
            train_with_extras,
            validation_with_extras,
        ),
        (
            "+ all four derived extras",
            [
                "count_of_ratings_at_5",
                "spread_of_ratings",
                "count_of_zero_ratings",
                "mean_service_rating_x_class",
            ],
            train_with_extras,
            validation_with_extras,
        ),
        (
            f"+ {len(zeros.columns)} per-rating zero indicators",
            list(zeros.columns),
            train_with_extras,
            validation_with_extras,
        ),
        (
            "+ zero indicators + count_of_zero_ratings",
            list(zeros.columns) + ["count_of_zero_ratings"],
            train_with_extras,
            validation_with_extras,
        ),
    ]

    for label, extra_columns, train_frame, validation_frame in candidates:
        row = score(BASE_FEATURES + extra_columns, train_frame, validation_frame, label)
        row["delta"] = round(float(row["roc_auc"]) - float(baseline["roc_auc"]), 6)
        results.append(row)
        print(
            f"  {row['variant']:<52} n={row['features']:>2}  auc={row['roc_auc']:.6f}  "
            f"delta={row['delta']:+.6f}"
        )

    print()
    print("=" * 88)
    print("HYPERPARAMETER EXPERIMENTS - on the shipped feature list")
    print("=" * 88)

    # `.lead/03-TRAIN-AND-TUNE.md` Step 3.4 lists these as the settings worth
    # tuning, in order of impact. These are hand-set values to size the prize
    # before deciding whether Optuna is worth installing.
    settings_candidates: list[tuple[str, dict[str, object]]] = [
        (
            "lower learning rate 0.02, more trees",
            {"learning_rate": 0.02, "max_iter": 900, "early_stopping": False},
        ),
        (
            "lower learning rate 0.03, more trees",
            {"learning_rate": 0.03, "max_iter": 700, "early_stopping": False},
        ),
        (
            "more leaves, 63",
            {"max_leaf_nodes": 63, "max_iter": 400, "early_stopping": False},
        ),
        (
            "fewer leaves, 15, more regularisation",
            {
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "max_iter": 600,
                "early_stopping": False,
            },
        ),
        (
            "smaller min_samples_leaf, 10",
            {"min_samples_leaf": 10, "max_iter": 500, "early_stopping": False},
        ),
        (
            "much deeper, min_samples_leaf 5",
            {
                "min_samples_leaf": 5,
                "max_leaf_nodes": 63,
                "max_iter": 800,
                "early_stopping": False,
            },
        ),
    ]

    for label, params in settings_candidates:
        row = score(BASE_FEATURES, train_data, validation_data, label, params=params)
        row["delta"] = round(float(row["roc_auc"]) - float(baseline["roc_auc"]), 6)
        results.append(row)
        print(
            f"  {row['variant']:<52} n={row['features']:>2}  auc={row['roc_auc']:.6f}  "
            f"delta={row['delta']:+.6f}"
        )

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    output = Path(CONFIG["report_path"]) / "auc_experiments.csv"
    save_dataframe(table, output)

    print()
    print("=" * 88)
    print("RANKED")
    print("=" * 88)
    print(table.to_string(index=False))
    print()
    print(f"wrote {output}")
    print()
    best = table.iloc[0]
    print(
        f"Best: {best['variant']} at {best['roc_auc']:.6f}, "
        f"{float(best['roc_auc']) - float(baseline['roc_auc']):+.6f} on the baseline."
    )
    print("Validation only. The test split has not been opened.")


if __name__ == "__main__":
    main()
