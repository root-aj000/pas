"""Optuna search for XGBoost. Nine parameters, no pruning, no extras.

Run with: python research/optuna_search.py

Set DEVICE = "cuda" to run on GPU.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.utils.common import apply_categorical_encoding, load_json

SEED = 42
BUDGET_SECONDS = 1200
DEVICE = "cpu"

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
TARGET = str(CONFIG["target_column"])


def read_encoding_contract() -> tuple[list[str], str]:
    """Read how stage 2 encoded the categoricals, from features.json.

    Returns:
        The categorical column names and the encoding name.
    """
    contract = load_json(Path("artifacts/data_cleaning_encoding/features.json"))
    return (
        list(contract.get("categorical_columns", [])),
        str(contract.get("categorical_encoding", "one_hot")),
    )


def main() -> None:
    """Search the nine parameters and save the trials."""
    categorical_columns, encoding = read_encoding_contract()
    train = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/train.csv"),
        categorical_columns,
        encoding,
        "optuna",
    )
    validation = apply_categorical_encoding(
        pd.read_csv("artifacts/data_cleaning_encoding/validation.csv"),
        categorical_columns,
        encoding,
        "optuna",
    )
    features = list(CONFIG["features"])

    def objective(trial: optuna.Trial) -> float:
        """Fit one configuration and return its validation ROC-AUC."""
        model = XGBClassifier(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 3000),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            max_bin=trial.suggest_categorical("max_bin", [128, 256, 512, 1024]),
            random_state=SEED,
            tree_method="hist",
            device=DEVICE,
            enable_categorical=True,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(train[features], train[TARGET], verbose=False)
        probabilities = model.predict_proba(validation[features])[:, 1]
        score = float(roc_auc_score(validation[TARGET], probabilities))
        done = [
            t.value
            for t in study.trials
            if t.value is not None and t.number < trial.number
        ]
        if trial.number % 10 == 0:
            best = max(done) if done else score
            print(
                f"  trial {trial.number:>3}  auc={score:.6f}  best={best:.6f}",
                flush=True,
            )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    print(
        f"Optuna search, 9 parameters, budget {BUDGET_SECONDS}s, device {DEVICE}",
        flush=True,
    )
    started = time.monotonic()
    study.optimize(objective, timeout=BUDGET_SECONDS)
    elapsed = time.monotonic() - started

    trials = study.trials_dataframe().sort_values("value", ascending=False)
    output = Path(CONFIG["report_path"]) / "optuna_trials.csv"
    trials.to_csv(output, index=False)

    print(f"\n{len(study.trials)} trials in {elapsed / 60:.1f} min")
    print(f"best validation AUC: {study.best_value:.6f}")
    print("shipped model      : 0.958844 validation, 0.957894 test")
    print(f"gain               : {study.best_value - 0.958844:+.6f}")
    print(f"\nsaved {output}")
    print("\ntop 5:")
    print(trials.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
