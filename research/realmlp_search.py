"""Optuna search over RealMLP. Every tunable parameter that can matter.

Run with:  python research/realmlp_search.py
Smoke with: python research/realmlp_search.py --quick
On Kaggle GPU: same command. CUDA is auto-detected, no flag needed.

73 parameters exist on RealMLP_TD_Classifier. This searches the ones that move
validation AUC, from three sources:

* The RealMLP paper (Holzmuller et al. 2024): lr, wd, hidden sizes, dropout,
  batch size, PLR embeddings and epochs are where the gains live.
* The reference notebook that reached 0.96147: flat_anneal schedule, v4 inputs,
  6 epochs, 8 ensemble members.
* Our own measurements: defaults beat the published recipe on our 22 features,
  so the search starts from defaults rather than from their recipe.

Deliberately NOT searched:

| Parameter | Why not |
|---|---|
| `wd_sched`, `p_drop_sched`, `ls_eps_sched`, `mom_sched`, `sq_mom_sched`, `opt_eps_sched` | Each builds a lambda inside pytabkit that joblib cannot serialize. Stage 3 would train fine and die saving. Only `lr_sched=flat_anneal` pickles clean. See the config.yaml comment |
| `calibration_method`, `sort_quantile_predictions` | Calibration does not change ranking, and AUC is ranking-only |
| `device`, `n_threads`, `tmp_folder`, `verbosity` | Infrastructure, not tuning |
| `predict_batch_size` | Speed only |

Time-boxed per .lead/03 Step 3.4. Validation only. Test stays closed.
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

from src.utils.common import apply_categorical_encoding, load_json

SEED = 42
TARGET = "satisfaction"
BUDGET_SECONDS = 3600

logger = logging.getLogger("realmlp_search")


def suggest_parameters(trial) -> dict:
    """Return one RealMLP configuration.

    Args:
        trial: The Optuna trial.

    Returns:
        Parameters covering architecture, optimization, regularization, PLR
        embeddings and budget. Every value here pickles clean - see module doc.
    """
    return {
        # Optimization. lr is the single biggest lever on any neural network.
        "lr": trial.suggest_float("lr", 0.005, 0.2, log=True),
        "lr_sched": trial.suggest_categorical("lr_sched", ["flat_anneal", "cos_log"]),
        "wd": trial.suggest_float("wd", 1e-4, 0.3, log=True),
        "sq_mom": trial.suggest_float("sq_mom", 0.9, 0.999),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        # Architecture. Wider and deeper, within what 3-6 epochs can train.
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes",
            [[128, 128], [256, 128], [512, 256, 128], [512, 256], [256, 256, 128]],
        ),
        "act": trial.suggest_categorical("act", ["silu", "gelu", "relu", "mish"]),
        "p_drop": trial.suggest_float("p_drop", 0.0, 0.3),
        # Categorical capacity. Our 4 categoricals have 2-3 values each.
        "embedding_size": trial.suggest_int("embedding_size", 2, 16),
        "max_one_hot_cat_size": trial.suggest_categorical(
            "max_one_hot_cat_size", [4, 8, 18]
        ),
        # PLR numerical embeddings. RealMLP-specific, and one of the reasons it
        # beats a basic MLP on tabular data. On or off, plus its two widths.
        "use_plr_embeddings": trial.suggest_categorical(
            "use_plr_embeddings", [True, False]
        ),
        "plr_hidden_1": trial.suggest_int("plr_hidden_1", 8, 32),
        "plr_hidden_2": trial.suggest_int("plr_hidden_2", 4, 16),
        "plr_sigma": trial.suggest_float("plr_sigma", 0.5, 10.0, log=True),
        # Label smoothing. Small values regularize; the schedule is excluded
        # because it breaks pickling (see module doc).
        "ls_eps": trial.suggest_float("ls_eps", 0.0, 0.05),
        # Budget. More members and epochs cost linearly and gain sublinearly.
        "n_ens": trial.suggest_categorical("n_ens", [4, 8, 12]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 8),
        # Fixed from the published recipe. Not searched: they interact with
        # everything and the recipe values are already good.
        # first_layer_lr_factor, plr_lr_factor, bias_init_mode, tfms stay default.
    }


def objective(
    trial,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
) -> float:
    """Fit one configuration and return validation ROC-AUC.

    Args:
        trial: The Optuna trial.
        train_X: Training features as numpy.
        train_y: Training labels.
        validation_X: Validation features.
        validation_y: Validation labels.

    Returns:
        Validation ROC-AUC.
    """
    from pytabkit import RealMLP_TD_Classifier

    params = suggest_parameters(trial)
    model = RealMLP_TD_Classifier(random_state=SEED, verbosity=0, **params)
    model.fit(train_X, train_y)
    probabilities = model.predict_proba(validation_X)[:, 1]
    return float(roc_auc_score(validation_y, probabilities))


def main(quick: bool, budget_seconds: int) -> int:
    """Run the search and save every trial.

    Args:
        quick: If True, 3 trials on a subsample. Tests the code path, not the space.
        budget_seconds: Time limit. Honoured, per .lead/03 Step 3.4.

    Returns:
        0 if the search completed within budget.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    categorical = list(contract.get("categorical_columns", []))
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    train = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        categorical,
        encoding,
        "realmlp_search",
    )
    validation = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        categorical,
        encoding,
        "realmlp_search",
    )
    config = yaml.safe_load(Path("config.yaml").read_text())
    features: list[str] = list(config["features"])
    if quick:
        train = train.sample(n=20000, random_state=SEED)
        validation = validation.sample(n=5000, random_state=SEED)
    train_X = train[features].to_numpy()
    train_y = train[TARGET].to_numpy()
    validation_X = validation[features].to_numpy()
    validation_y = validation[TARGET].to_numpy()

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="realmlp_tuning",
    )
    print(
        f"RealMLP search, budget {budget_seconds}s, "
        f"train={len(train_X):,}, validation={len(validation_X):,}",
        flush=True,
    )
    started = time.monotonic()

    def run_trial(trial) -> float:
        value = objective(trial, train_X, train_y, validation_X, validation_y)
        done = [
            t.value
            for t in study.trials
            if t.value is not None and t.number < trial.number
        ]
        best = max(done + [value])
        print(
            f"  trial {trial.number:>3}  auc={value:.6f}  best={best:.6f}  "
            f"{time.monotonic() - started:.0f}s",
            flush=True,
        )
        return value

    study.optimize(
        run_trial,
        timeout=None if quick else budget_seconds,
        n_trials=3 if quick else None,
    )
    elapsed = time.monotonic() - started

    frame = study.trials_dataframe().sort_values("value", ascending=False)
    output = Path("reports/realmlp_search_trials.csv")
    frame.to_csv(output, index=False)
    print(f"\n{len(study.trials)} trials in {elapsed / 60:.1f} min")
    print(f"best validation AUC: {study.best_value:.6f}")
    print("XGBoost native:      0.958868")
    print("RealMLP defaults:    0.959051")
    print(f"gain over defaults:  {study.best_value - 0.959051:+.6f}")
    print(f"\nsaved {output}")
    print("\nbest params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("\nValidation only. Test stays closed.")
    print("To ship the winner: copy best params into config.yaml model_params,")
    print("re-run the pipeline, confirm on test.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna search over RealMLP.")
    parser.add_argument("--quick", action="store_true", help="3 trials on a subsample.")
    parser.add_argument(
        "--budget", type=int, default=BUDGET_SECONDS, help="Seconds. Default 3600."
    )
    args = parser.parse_args()
    sys.exit(main(quick=args.quick, budget_seconds=args.budget))
