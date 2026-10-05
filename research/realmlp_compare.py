"""RealMLP vs XGBoost on our features. GPU-aware.

Run locally with:  python research/realmlp_compare.py
Run on Kaggle with: python research/realmlp_compare.py  (GPU used automatically)
Smoke test with:   python research/realmlp_compare.py --quick

The question: their RealMLP v4 scores 0.961166 on THEIR features. Our XGBoost
scores 0.958868 on OURS. Is the gap the architecture or the features?

This answers it by running their architecture on our features. Same 22 native
categorical columns, same split, same seed, same metric. If RealMLP wins here,
the architecture matters. If it loses, their advantage was the features.

pytabkit detects CUDA on its own via PyTorch Lightning. No device flag is needed
and none is offered - a flag nobody has to set is a flag nobody can set wrong.

Validation only. The test split stays closed.
"""

import argparse
import logging
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

from src.utils.common import apply_categorical_encoding, load_json, save_dataframe

SEED = 42
TARGET = "satisfaction"

# The bar. Measured on this split, this seed, these features.
XGBOOST_NATIVE_AUC = 0.958868

logger = logging.getLogger("realmlp")


def load_features(
    quick: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the prepared splits as numpy arrays.

    Args:
        quick: If True, use a 30k/10k subsample so a smoke test is fast.

    Returns:
        Train features, train labels, validation features, validation labels,
        and the feature names.
    """
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    categorical = list(contract.get("categorical_columns", []))
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    train = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        categorical,
        encoding,
        "realmlp",
    )
    validation = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        categorical,
        encoding,
        "realmlp",
    )
    config = yaml.safe_load(Path("config.yaml").read_text())
    features: list[str] = list(config["features"])
    if quick:
        train = train.sample(n=30000, random_state=SEED)
        validation = validation.sample(n=10000, random_state=SEED)
    return (
        train[features].to_numpy(),
        train[TARGET].to_numpy(),
        validation[features].to_numpy(),
        validation[TARGET].to_numpy(),
        features,
    )


def run_realmlp(
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    ensemble_members: int,
    epochs: int,
) -> tuple[float, float]:
    """Fit RealMLP and return its validation ROC-AUC and fit seconds.

    Args:
        train_X: Training features.
        train_y: Training labels.
        validation_X: Validation features.
        validation_y: Validation labels.
        ensemble_members: How many networks to average.
        epochs: Training epochs per member.

    Returns:
        The validation ROC-AUC and the seconds taken.
    """
    import torch
    from pytabkit import RealMLP_TD_Classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device={device}", flush=True)
    if torch.cuda.is_available():
        print(
            f"  gpu={torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}",
            flush=True,
        )

    started = time.monotonic()
    model = RealMLP_TD_Classifier(
        n_ens=ensemble_members,
        n_epochs=epochs,
        random_state=SEED,
    )
    model.fit(train_X, train_y)
    probabilities = model.predict_proba(validation_X)[:, 1]
    seconds = time.monotonic() - started
    return float(roc_auc_score(validation_y, probabilities)), seconds


def main(quick: bool, ensemble_members: int, epochs: int) -> int:
    """Run the comparison and report.

    Args:
        quick: Smoke test mode.
        ensemble_members: Networks to average. 8 is their recipe.
        epochs: Epochs per member. 3 is their recipe.

    Returns:
        0 if RealMLP was scored.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    train_X, train_y, validation_X, validation_y, features = load_features(quick)
    print(
        f"mode={'QUICK' if quick else 'FULL'}  features={len(features)}  "
        f"train={len(train_X):,}  validation={len(validation_X):,}  "
        f"ensemble={ensemble_members}  epochs={epochs}"
    )

    auc, seconds = run_realmlp(
        train_X, train_y, validation_X, validation_y, ensemble_members, epochs
    )
    print(f"\nRealMLP:            {auc:.6f}  ({seconds:.0f}s)")
    print(f"XGBoost native:     {XGBOOST_NATIVE_AUC:.6f}")
    print(f"gap:                {auc - XGBOOST_NATIVE_AUC:+.6f}")
    print()
    if auc > XGBOOST_NATIVE_AUC:
        print(
            "-> RealMLP wins ON OUR FEATURES. The architecture matters, not just their features."
        )
    else:
        print(
            "-> RealMLP loses ON OUR FEATURES. Their advantage was the features, not the architecture."
        )

    output = Path("reports/realmlp_compare.csv")
    save_dataframe(
        pd.DataFrame(
            [
                {
                    "model": "RealMLP",
                    "ensemble_members": ensemble_members,
                    "epochs": epochs,
                    "roc_auc": round(auc, 6),
                    "fit_seconds": round(seconds, 1),
                    "xgboost_native_auc": XGBOOST_NATIVE_AUC,
                }
            ]
        ),
        output,
    )
    print(f"\nwrote {output}")
    print("Validation only. The test split stays closed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealMLP vs XGBoost on our features.")
    parser.add_argument(
        "--quick", action="store_true", help="Smoke test: subsample, 1 member, 1 epoch."
    )
    parser.add_argument(
        "--members", type=int, default=8, help="Ensemble members. Their recipe uses 8."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Epochs per member. Their recipe uses 3."
    )
    args = parser.parse_args()
    if args.quick:
        args.members, args.epochs = 1, 1
    sys.exit(main(quick=args.quick, ensemble_members=args.members, epochs=args.epochs))
