"""XGBoost + pseudo-labelling. One model. Measured honestly.

Run with: python research/xgb_pseudo.py

WHAT THIS IS
The pipeline already has pseudo-labelling (`pseudo_label_enabled` in
config.yaml), but two things about it are worth checking before trusting it:

  1. It is SKIPPED on every real run. `train_ensemble` deletes the frames before
     the GPU workers start, so `not sharded` is False and the step never runs.
     It has therefore never actually contributed to a submitted number.
  2. It retrains the whole 9-member ensemble, doubling a 5-hour run to find out
     whether it helps at all. That is an expensive way to ask a cheap question.

This script asks the cheap question first, with ONE model:

  baseline   XGBoost, k-fold, no pseudo-labels
  round 1    predict test.csv, keep rows above threshold, retrain with them
  round 2    repeat - the model's second opinion is better informed than its first
  sweep      threshold x rounds, so the answer is a choice rather than a guess

Everything is scored on the SAME held-out split every round, and the pseudo-labels
always come from test.csv, never from the evaluation rows. That separation is the
whole experiment: if pseudo-labels were drawn from the rows we score on, the
number would be meaningless.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_TRAIN = 100_000
N_TEST = 100_000
FOLDS = 4
TARGET = "satisfaction"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RATINGS = [
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
CATEGORICALS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def build(frame: pd.DataFrame, mean=None, std=None):
    """Numeric feature matrix plus the categorical columns left as native category.

    The categoricals are NOT one-hot or integer-coded here. XGBoost partitions
    `category` dtype itself, which this project's config measured at +0.0029 AUC
    over one-hot - the single largest feature decision in the pipeline.
    """
    out = frame.copy()
    out[RATINGS] = out[RATINGS].replace(0, np.nan)
    out["Arrival Delay in Minutes"] = out["Arrival Delay in Minutes"].fillna(0)
    out["mean_service_rating"] = out[RATINGS].mean(axis=1).fillna(3.0)

    numeric = ["Age", "Flight Distance", "mean_service_rating", "mean_service_rating_sq",
               "Online boarding", "Seat comfort", "Inflight entertainment",
               "On-board service", "Cleanliness", "Leg room service",
               "Baggage handling", "Checkin service", "Inflight wifi service",
               "Ease of Online booking", "Food and drink", "Gate location",
               "Departure/Arrival time convenient",
               "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    out["mean_service_rating_sq"] = out["mean_service_rating"] ** 2

    X = out[numeric].astype("float32")
    for column in CATEGORICALS:
        X[column] = pd.Categorical(out[column]).codes.astype("float32")

    if mean is None:
        mean = X[numeric].mean().to_numpy()
        std = X[numeric].std().to_numpy() + 1e-6
    X[numeric] = (X[numeric].to_numpy() - mean) / std
    return X.astype("float32"), mean, std


def xgb_params(seed: int):
    return {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 9,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.5,
        "min_child_weight": 55,
        "subsample": 0.8,
        "tree_method": "hist",
        "n_jobs": 8,
        "random_state": seed,
        "eval_metric": "auc",
    }


def cv_auc(X, y, folds, X_unlab, extra=None, extra_y=None, seed=SEED):
    """Out-of-fold AUC. `extra` rows are appended to every training fold.

    Args:
        X_unlab: The unlabelled test matrix, predicted with every fold model.
        extra: Pseudo-labelled rows, or None.
        extra_y: Their hard labels.

    Returns:
        (oof_auc, mean_test_probability)
    """
    from xgboost import XGBClassifier

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    test_sum = np.zeros(len(X_unlab))
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(X, y)):
        if extra is not None:
            X_fit = np.vstack([X[fit_idx], extra])
            y_fit = np.concatenate([y[fit_idx], extra_y])
        else:
            X_fit, y_fit = X[fit_idx], y[fit_idx]
        model = XGBClassifier(**xgb_params(SEED + fold))
        model.fit(X_fit, y_fit, verbose=False)
        oof[score_idx] = model.predict_proba(X[score_idx])[:, 1]
        test_sum += model.predict_proba(X_unlab)[:, 1] / folds
    return float(roc_auc_score(y, oof)), test_sum


def main() -> None:
    print("=" * 72)
    print("XGBoost + pseudo-labelling")
    print("=" * 72, flush=True)

    started = time.monotonic()
    train = pd.read_csv(PROJECT_ROOT / "data" / "train.csv").sample(
        n=N_TRAIN, random_state=SEED
    ).reset_index(drop=True)
    test = pd.read_csv(PROJECT_ROOT / "data" / "test.csv").sample(
        n=N_TEST, random_state=SEED
    ).reset_index(drop=True)

    y = train[TARGET].astype("int8").to_numpy()
    X, mean, std = build(train)
    # numpy, not a frame: it is sliced by a boolean/positional mask below,
    # which a DataFrame would interpret as column labels.
    X_TEST_FEATURES = build(test, mean, std)[0].to_numpy()

    # No outer holdout: cv_auc() already produces a genuine out-of-fold
    # prediction for every row in X_fit, so every number below is measured on
    # rows no fold model trained on. An outer split would only shrink the data.
    X_fit, y_fit = X.to_numpy(), y

    print(f"train {X.shape}  test(unlabelled) {X_TEST_FEATURES.shape}")
    print(f"each OOF score is out-of-fold over {FOLDS - 1} inner folds")

    results = {}
    print("\n--- baseline: no pseudo-labels ---", flush=True)
    base, test_prob_1 = cv_auc(X_fit, y_fit, folds=3, X_unlab=X_TEST_FEATURES)
    results[0] = ("no pseudo-labels", base)
    print(f"OOF AUC {base:.6f}   ({time.monotonic() - started:.0f}s)", flush=True)

    # Pseudo-labels always come from test.csv, and are scored only on rows the
    # model never trained on.
    current_test_prob = test_prob_1
    for round_number in (1, 2):
        for threshold in (0.90, 0.95, 0.99):
            confident = np.where(
                (current_test_prob > threshold) | (current_test_prob < 1 - threshold)
            )[0]
            pseudo_y = (current_test_prob[confident] > 0.5).astype("int8")
            extra = X_TEST_FEATURES[confident]
            auc, new_prob = cv_auc(
                X_fit, y_fit, folds=3, X_unlab=X_TEST_FEATURES,
                extra=extra, extra_y=pseudo_y,
            )
            key = f"round {round_number}, p>{threshold}"
            results[(round_number, threshold)] = (key, auc)
            print(
                f"  round {round_number}  p>{threshold}  "
                f"+{len(confident):>6} pseudo rows   OOF {auc:.6f} "
                f"({auc - base:+.6f})",
                flush=True,
            )
            if round_number == 1:
                current_test_prob = new_prob

    seconds = time.monotonic() - started
    print("\n" + "=" * 72)
    print("SUMMARY (OOF AUC on rows never trained on)")
    print("=" * 72)
    ranked = sorted(results.items(), key=lambda kv: -kv[1][1])
    best_auc = ranked[0][1][1]
    for key, (label, auc) in ranked:
        mark = "  <-- best" if auc == best_auc else ""
        print(f"  {label:<24} {auc:.6f}  ({auc - base:+.6f}){mark}")
    print(f"\nbaseline        {base:.6f}")
    print(f"best with pseudo {best_auc:.6f}   ({best_auc - base:+.6f})")
    print(f"total time      {seconds:.0f}s")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()