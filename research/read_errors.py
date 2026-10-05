"""Read the errors of the winner. What is it still getting wrong?

Run with: python research/read_errors.py

`.lead/02-C-CHOOSING-THE-METHOD.md` Step 9 calls this the highest-yield hour in the
whole step: *"The second row is your next feature. That is how features get
invented — by reading errors, not by guessing. If all the errors look the same,
you have a feature idea. If they look random, you probably have a data problem."*

Two comparisons, and the order matters:

1. **Hard positives against easy positives.** Both rows are satisfied passengers.
   What makes one of them look dissatisfied to the model?
2. **Hard negatives against easy negatives.** Both rows are dissatisfied. What
   makes one of them look satisfied?

Then the question the file exists to answer: **what information distinguishes
these cases that the model does not already have?**

Model: the shipped model_4 (XGBoost, native categorical). Validation split only.
The test split stays closed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.utils.common import apply_categorical_encoding, load_json

SEED = 42

# A prediction this confident in the wrong direction is not noise. Something the
# model cannot see is deciding these rows.
HARD_POSITIVE_CUTOFF = 0.10
HARD_NEGATIVE_CUTOFF = 0.90

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
TARGET = str(CONFIG["target_column"])
MODEL_PATH = Path("models/model_4/model.pkl")
MODEL_FEATURES = Path("models/model_4/features.json")


def load_model_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Load the validation rows with the shipped model's own encoding.

    Returns:
        The validation rows and the feature list.

    Raises:
        FileNotFoundError: If the model or the prepared data is missing.
    """
    for path in (MODEL_PATH, MODEL_FEATURES):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the pipeline first.")
    saved = load_json(MODEL_FEATURES)
    frame = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        list(saved.get("categorical_columns", [])),
        str(saved.get("categorical_encoding", "one_hot")),
        "read_errors",
    )
    return frame, list(saved["features"])


def print_sorted_extremes(
    frame: pd.DataFrame, probabilities: np.ndarray, rows: int = 30
) -> None:
    """Print the lowest- and highest-scored rows, with their true answers.

    Args:
        frame: The scored rows, carrying the id and the label.
        probabilities: The model's probability per row.
        rows: How many rows to show from each end.
    """
    order = np.argsort(probabilities)
    print("LOWEST predictions (model is most sure these are NOT satisfied):")
    print(
        pd.DataFrame(
            {
                "id": frame["id"].iloc[order[:rows]].to_numpy(),
                "prediction": np.round(probabilities[order[:rows]], 4),
                "actual": frame[TARGET].iloc[order[:rows]].to_numpy().astype(int),
            }
        ).to_string(index=False)
    )
    print()
    print("HIGHEST predictions (model is most sure these ARE satisfied):")
    print(
        pd.DataFrame(
            {
                "id": frame["id"].iloc[order[-rows:]].to_numpy(),
                "prediction": np.round(probabilities[order[-rows:]], 4),
                "actual": frame[TARGET].iloc[order[-rows:]].to_numpy().astype(int),
            }
        ).to_string(index=False)
    )


def compare_groups(
    name: str,
    first: pd.DataFrame,
    second: pd.DataFrame,
    numeric_columns: list[str],
    categorical_roots: list[str],
) -> pd.DataFrame:
    """Compare two groups column by column, numeric and categorical.

    Args:
        name: What this comparison is called, for the printed heading.
        first: The first group.
        second: The second group.
        numeric_columns: Numeric columns to compare by mean.
        categorical_roots: Categorical column names, used for the mode share.

    Returns:
        One row per column: the group means and their absolute difference.
    """
    print()
    print(f"--- {name}")
    rows = []
    for column in numeric_columns:
        first_mean = float(first[column].mean())
        second_mean = float(second[column].mean())
        rows.append(
            {
                "column": column,
                "group_A_mean": round(first_mean, 3),
                "group_B_mean": round(second_mean, 3),
                "abs_difference": round(abs(first_mean - second_mean), 3),
            }
        )
    # Numeric means sort together. Categorical modes are printed separately below,
    # because mixing a float difference with a "see shares" string breaks sorting.
    numeric_table = pd.DataFrame(rows).sort_values("abs_difference", ascending=False)
    print(numeric_table.to_string(index=False))
    print()
    print("categorical modes, with the share holding that value:")
    share_rows = []
    for column in categorical_roots:
        first_mode = first[column].mode()
        second_mode = second[column].mode()
        first_label = (
            f"{first_mode.iloc[0]} ({(first[column] == first_mode.iloc[0]).mean():.0%})"
            if len(first_mode)
            else "n/a"
        )
        second_label = (
            f"{second_mode.iloc[0]} ({(second[column] == second_mode.iloc[0]).mean():.0%})"
            if len(second_mode)
            else "n/a"
        )
        share_rows.append(
            {
                "column": column,
                "group_A_mode": first_label,
                "group_B_mode": second_label,
            }
        )
    print(pd.DataFrame(share_rows).to_string(index=False))
    return numeric_table


def main() -> None:
    """Score validation with the shipped model and interrogate its mistakes."""
    frame, features = load_model_inputs()
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(frame[features])[:, 1]
    target = frame[TARGET].to_numpy().astype(int)

    print("=" * 92)
    print("READING THE ERRORS OF model_4 (XGBoost, native categorical)")
    print("=" * 92)
    print(f"validation AUC: {roc_auc_score(target, probabilities):.6f}")
    print()

    print_sorted_extremes(frame, probabilities)
    print()

    hard_positive = frame[(target == 1) & (probabilities < HARD_POSITIVE_CUTOFF)]
    easy_positive = frame[(target == 1) & (probabilities > HARD_NEGATIVE_CUTOFF)]
    hard_negative = frame[(target == 0) & (probabilities > HARD_NEGATIVE_CUTOFF)]
    easy_negative = frame[(target == 0) & (probabilities < HARD_POSITIVE_CUTOFF)]

    print("=" * 92)
    print("GROUP SIZES")
    print("=" * 92)
    print(
        f"  hard positives (satisfied, scored < {HARD_POSITIVE_CUTOFF})  {len(hard_positive):>7,}"
    )
    print(
        f"  easy positives (satisfied, scored > {HARD_NEGATIVE_CUTOFF}) {len(easy_positive):>7,}"
    )
    print(
        f"  hard negatives (not, scored > {HARD_NEGATIVE_CUTOFF})        {len(hard_negative):>7,}"
    )
    print(
        f"  easy negatives (not, scored < {HARD_POSITIVE_CUTOFF})        {len(easy_negative):>7,}"
    )

    numeric_columns = [
        c
        for c in features
        if c not in ("Gender", "Customer Type", "Type of Travel", "Class")
    ]
    categorical_roots = ["Gender", "Customer Type", "Type of Travel", "Class"]

    compare_groups(
        f"A: HARD POSITIVES vs B: EASY POSITIVES ({len(hard_positive):,} vs {len(easy_positive):,} rows)",
        hard_positive,
        easy_positive,
        numeric_columns,
        categorical_roots,
    )
    compare_groups(
        f"A: HARD NEGATIVES vs B: EASY NEGATIVES ({len(hard_negative):,} vs {len(easy_negative):,} rows)",
        hard_negative,
        easy_negative,
        numeric_columns,
        categorical_roots,
    )
    compare_groups(
        f"A: HARD POSITIVES vs B: HARD NEGATIVES ({len(hard_positive):,} vs {len(hard_negative):,} rows) - what the model cannot see",
        hard_positive,
        hard_negative,
        numeric_columns,
        categorical_roots,
    )

    # The rows that survive every filter: wrong with near-certainty. Print a few in
    # full, because an aggregate table hides what a single row reveals.
    worst_positive_index = np.where(target == 1)[0][
        np.argmin(probabilities[target == 1])
    ]
    worst_negative_index = np.where(target == 0)[0][
        np.argmax(probabilities[target == 0])
    ]
    print()
    print("=" * 92)
    print("THE TWO MOST CONFIDENT MISTAKES, IN FULL")
    print("=" * 92)
    print("Satisfied passenger the model was most sure was not:")
    row = frame.iloc[worst_positive_index]
    print(
        f"  id={row['id']}  predicted={probabilities[worst_positive_index]:.4f}  actual=1"
    )
    for column in features:
        print(f"    {column:<36} {row[column]}")
    print()
    print("Dissatisfied passenger the model was most sure was satisfied:")
    row = frame.iloc[worst_negative_index]
    print(
        f"  id={row['id']}  predicted={probabilities[worst_negative_index]:.4f}  actual=0"
    )
    for column in features:
        print(f"    {column:<36} {row[column]}")
    print()
    print("Validation only. The test split stays closed.")


if __name__ == "__main__":
    main()
