"""Shared harness for the five novel-technique experiments.

Run any of research/novel_*.py. All five measure the SAME thing on the SAME
split, so their numbers are directly comparable.

Design choices, and why:

* **A modest feature set, not the 178-column pipeline.** The question each
  experiment answers is "does this METHOD carry signal", not "are the engineered
  features good". Loading stage-2 artifacts would make every script a rerun of
  the existing pipeline and slow enough to discourage iteration. So: the 21 raw
  columns plus mean_service_rating, ~23 features.

* **A subsample, not all 699,635 rows.** These are research scripts that must
  finish in minutes on a CPU. A method that only works at full scale will look
  bad here, which is a real limitation and is stated in each script's output.

* **The same baseline in every run.** `baseline_auc()` fits one LightGBM on the
  identical split. Any method that cannot beat that number has not earned its
  complexity, and the comparison is what makes these scripts worth running.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
N_TRAIN = 80_000
N_EVAL = 20_000
TARGET = "satisfaction"

# Absolute, so these scripts run from any working directory. The rest of
# research/ assumes you are at the project root; that assumption broke these.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "train.csv"
DATA_TEST = PROJECT_ROOT / "data" / "test.csv"

RAW_COLUMNS = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

# Ordinal encoding for the categoricals, not one-hot: keeps the matrix numeric so
# every method here (including the distance-based one) can consume it directly.
CATEGORY_MAP = {
    "Gender": {"Female": 0, "Male": 1},
    "Customer Type": {"Loyal Customer": 0, "disloyal Customer": 1},
    "Type of Travel": {"Business travel": 0, "Personal Travel": 1},
    "Class": {"Business": 0, "Eco Plus": 1, "Eco": 2},
}


def load_split(
    n_train: int = N_TRAIN, n_eval: int = N_EVAL, with_unlabeled: bool = False
):
    """Return (X_train, y_train, X_eval, y_eval[, X_unlabeled]) on one fixed split.

    Args:
        n_train: Labelled training rows.
        n_eval: Held-out evaluation rows.
        with_unlabeled: Also return the competition test rows, unlabelled, for the
            transductive experiment. Loading them costs ~8s, so they are opt-in.

    Returns:
        Numeric float32 frames and int8 labels, plus feature names when unlabeled.
    """
    df = pd.read_csv(DATA)
    # 0 is a missing code in the rating columns, not a real rating.
    ratings = [c for c in RAW_COLUMNS if c not in CATEGORY_MAP]
    df[ratings] = df[ratings].replace(0, np.nan)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

    needed = n_train + n_eval
    df = df.sample(n=needed, random_state=SEED).reset_index(drop=True)

    X = encode(df)
    y = df[TARGET].astype("int8").to_numpy()

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=n_eval, random_state=SEED, stratify=y
    )

    if not with_unlabeled:
        return X_train, y_train, X_eval, y_eval

    comp = pd.read_csv(DATA_TEST, nrows=80_000)
    comp[ratings] = comp[ratings].replace(0, np.nan)
    comp["Arrival Delay in Minutes"] = comp["Arrival Delay in Minutes"].fillna(0)
    return X_train, y_train, X_eval, y_eval, encode(comp)


def encode(frame: pd.DataFrame) -> np.ndarray:
    """Return the raw columns plus mean_service_rating as float32."""
    out = frame[RAW_COLUMNS].copy()
    for column, mapping in CATEGORY_MAP.items():
        out[column] = out[column].map(mapping).astype("float32")
    ratings = [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
    ]
    out["mean_service_rating"] = out[ratings].mean(axis=1)
    return out.astype("float32").to_numpy()


def baseline_auc(X_train, y_train, X_eval, y_eval) -> float:
    """One LightGBM on this exact split. The bar every experiment must clear."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, verbosity=-1, n_jobs=8
    )
    model.fit(X_train, y_train)
    return float(roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1]))


def report(name: str, auc: float, base: float, seconds: float, note: str = "") -> None:
    """Print one result line and whether it beat the baseline."""
    delta = auc - base
    verdict = "BEATS baseline" if delta > 0 else "below baseline"
    print(f"\n{'=' * 72}")
    print(f"{name}")
    print(f"  AUC        {auc:.6f}")
    print(f"  baseline   {base:.6f}   (LightGBM, same split)")
    print(f"  delta      {delta:+.6f}  {verdict}")
    print(f"  time       {seconds:.1f}s")
    if note:
        print(f"  note       {note}")
    print("=" * 72, flush=True)


def banner(number: int, title: str, source: str) -> None:
    print("=" * 72, flush=True)
    print(f"NOVEL {number}: {title}", flush=True)
    print(f"based on: {source}", flush=True)
    print("=" * 72, flush=True)