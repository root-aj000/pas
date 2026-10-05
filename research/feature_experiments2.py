"""Test the feature ideas that can still work on a tree.

Run with: python research/feature_experiments2.py

Three of the standard suggestions are provably worthless on a gradient-boosted tree
and are therefore not tested here:

* **log1p / log transforms** - a tree splits at a threshold on one feature. A
  monotone transform preserves the ordering, so the set of achievable splits is
  identical. The effect is exactly zero, not small.
* **rank / percentile features** - rank is a monotone transform. Same reason.
* **missing-value indicators** - already measured. Twelve per-rating `was_zero`
  columns moved validation ROC-AUC by exactly 0.000000.

What remains untested and genuinely capable of adding information:

* **Ratios, differences and sums** between features. Trees can learn interactions
  but need depth to do it; handing them the product directly can save them the work.
* **Frequency encoding** - how common each category is. The model cannot derive
  "how many passengers flew Business" from a one-hot column alone.
* **Target encoding** - the smoothed satisfaction rate of each category. This is
  the strongest of the categorical encodings and also the most leak-prone, so it is
  computed out-of-fold on the training rows only.
* **Rating distribution shape** - count at each level, and how many high ratings,
  which is not the same as the mean.

Validation split only. The test split stays closed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.constants import SERVICE_RATING_COLUMNS
from src.utils.common import apply_categorical_encoding, load_json

SEED = 42
TARGET = "satisfaction"
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
BASE_FEATURES: list[str] = list(CONFIG["features"])


def add_ratios_and_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ratios, differences and sums between related columns.

    Args:
        frame: Rows carrying the raw columns.

    Returns:
        A new frame with six arithmetic columns.
    """
    out = pd.DataFrame(index=frame.index)
    departure = frame["Departure Delay in Minutes"]
    arrival = frame["Arrival Delay in Minutes"].fillna(0)
    distance = frame["Flight Distance"]

    # Delay per thousand miles. A 40-minute delay means something very different
    # on a 400-mile hop than on a 4000-mile one, and a tree would need depth to
    # work that out from two separate columns.
    out["delay_per_1000_miles"] = (departure + arrival) / (distance / 1000)
    out["arrival_minus_departure_delay"] = arrival - departure
    out["total_delay_minutes"] = departure + arrival
    out["is_delayed_both_ends"] = ((departure > 0) & (arrival > 0)).astype(int)
    out["delay_ratio"] = arrival / (departure + 1)
    out["distance_per_age"] = distance / (frame["Age"] + 1)
    return out


def add_frequency_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    """Add how often each category value occurs, as a share of all rows.

    Args:
        frame: Rows carrying the categorical columns.

    Returns:
        A new frame with one frequency column per categorical column.
    """
    out = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_COLUMNS:
        shares = frame[column].value_counts(normalize=True)
        out[f"{column}_frequency"] = frame[column].map(shares).astype(float)
    return out


def add_target_encoding(
    training: pd.DataFrame, other: pd.DataFrame, smoothing: float = 20.0
) -> pd.DataFrame:
    """Add the smoothed satisfaction rate of each category, fitted on training only.

    Args:
        training: Rows the encoding is computed from. Must be training rows.
        other: Rows the encoding is applied to.
        smoothing: Blend weight towards the global mean. 20 means a category with
            fewer than 20 rows is pulled most of the way to the overall rate, so a
            rare category cannot define its own target value.

    Returns:
        A frame of target-encoded columns aligned to `other`.
    """
    global_rate = float(training[TARGET].mean())
    out = pd.DataFrame(index=other.index)
    for column in CATEGORICAL_COLUMNS:
        counts = training.groupby(column, observed=True)[TARGET].agg(["sum", "size"])
        encoded = (counts["sum"] + smoothing * global_rate) / (
            counts["size"] + smoothing
        )
        out[f"{column}_target_encoded"] = other[column].map(encoded).astype(float)
    return out


def add_rating_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Add shape statistics of the 13 ratings beyond the ones already present.

    Args:
        frame: Rows carrying the 13 ratings.

    Returns:
        A new frame with four distribution-shape columns.
    """
    ratings = frame[SERVICE_RATING_COLUMNS]
    out = pd.DataFrame(index=frame.index)
    out["max_service_rating"] = ratings.max(axis=1)
    out["median_service_rating"] = ratings.median(axis=1)
    out["count_of_ratings_at_4_or_5"] = (ratings >= 4).sum(axis=1)
    out["count_of_ratings_at_3"] = (ratings == 3).sum(axis=1)
    return out


def score(features, train_data, validation_data, label, baseline_auc):
    """Fit and score one variant, returning its delta against the baseline.

    Args:
        features: The feature list.
        train_data: Training rows, already carrying every candidate column.
        validation_data: Validation rows, same columns.
        label: What this variant is called.
        baseline_auc: The baseline score, for the delta column.

    Returns:
        One result row.
    """
    model = HistGradientBoostingClassifier(random_state=SEED)
    started = time.monotonic()
    model.fit(train_data[features], train_data[TARGET])
    seconds = time.monotonic() - started
    probabilities = model.predict_proba(validation_data[features])[:, 1]
    auc = float(roc_auc_score(validation_data[TARGET], probabilities))
    return {
        "variant": label,
        "added": len(features) - len(BASE_FEATURES),
        "roc_auc": round(auc, 6),
        "delta": round(auc - baseline_auc, 6),
        "fit_seconds": round(seconds, 1),
    }


def main() -> None:
    """Build every candidate, score each one alone, then score them together."""
    # Under native categorical encoding the prepared frames KEEP the four
    # categorical columns, as category dtype. An earlier version merged them back
    # in from the raw file and ended up with duplicate column names.
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    train_full = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
        "features2",
    )
    validation_full = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
        "features2",
    )

    ratios_train = add_ratios_and_differences(train_full)
    ratios_validation = add_ratios_and_differences(validation_full)

    # Target encoding is fitted on TRAIN ONLY. Applying it to validation uses the
    # training mapping, never a validation label, so nothing leaks backwards.
    target_train = add_target_encoding(train_full, train_full)
    target_validation = add_target_encoding(train_full, validation_full)

    frequency_train = add_frequency_encoding(train_full)
    frequency_validation = add_frequency_encoding(validation_full)

    distribution_train = add_rating_distribution(train_full)
    distribution_validation = add_rating_distribution(validation_full)

    train_full = pd.concat(
        [train_full, ratios_train, target_train, frequency_train, distribution_train],
        axis=1,
    )
    validation_full = pd.concat(
        [
            validation_full,
            ratios_validation,
            target_validation,
            frequency_validation,
            distribution_validation,
        ],
        axis=1,
    )

    baseline = score(
        BASE_FEATURES, train_full, validation_full, "BASELINE (config.yaml)", 0.0
    )
    baseline_auc = float(baseline["roc_auc"])
    print("=" * 96)
    print("FEATURE EXPERIMENTS ROUND 2 - the suggestions that can still work on a tree")
    print("=" * 96)
    print(f"baseline: {baseline_auc:.6f}   features {len(BASE_FEATURES)}")
    print()

    groups = {
        "ratios/differences/sums": list(ratios_train.columns),
        "frequency encoding": list(frequency_train.columns),
        "target encoding": list(target_train.columns),
        "rating distribution shape": list(distribution_train.columns),
    }

    results = [baseline]
    print(
        f"  {baseline['variant']:<44} n={len(BASE_FEATURES):>2}  auc={baseline_auc:.6f}"
    )

    for label, columns in groups.items():
        row = score(
            BASE_FEATURES + columns, train_full, validation_full, label, baseline_auc
        )
        results.append(row)
        print(
            f"  {row['variant']:<44} n={row['added'] + len(BASE_FEATURES):>2}  auc={row['roc_auc']:.6f}  delta={row['delta']:+.6f}"
        )

    # Every group at once, so interactions between them can show up.
    everything = [column for columns in groups.values() for column in columns]
    row = score(
        BASE_FEATURES + everything,
        train_full,
        validation_full,
        "ALL of them together",
        baseline_auc,
    )
    results.append(row)
    print(
        f"  {row['variant']:<44} n={row['added'] + len(BASE_FEATURES):>2}  auc={row['roc_auc']:.6f}  delta={row['delta']:+.6f}"
    )

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    output = Path(CONFIG["report_path"]) / "feature_experiments2.csv"
    table.to_csv(output, index=False)

    print()
    print("=" * 96)
    print("RANKED")
    print("=" * 96)
    print(table.to_string(index=False))
    print()
    print(f"wrote {output}")
    print("Validation only. The test split has not been opened.")
    print()
    best = table.iloc[0]
    print(f"best: {best['variant']} at {best['roc_auc']:.6f} ({best['delta']:+.6f})")


if __name__ == "__main__":
    main()
