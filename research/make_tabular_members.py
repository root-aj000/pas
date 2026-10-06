"""
Generate OOF/test predictions for pytorch-tabular architectures, for stacking.

Why a separate script: pytorch-tabular has its own API (TabularModel with
DataConfig/ModelConfig/TrainerConfig) that does not fit the sklearn-style
build_estimator in make_members.py. So this mirrors that script's contract —
same 5-fold split, same output filenames — with the tabular API inside.

Honesty note: pytorch-tabular fits with early stopping on a validation split.
For each of the 5 folds, the 4/5 fit rows are split 90/10 into train/inner-val
for early stopping, and only then is the held-out 1/5 scored. The held-out rows
never influence training, not even through early stopping.

Run with: python -m research.make_tabular_members --only tabnet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from research.make_members import OUT_DIR

DEFAULT_ARTIFACTS = Path("artifacts/data_cleaning_encoding")

TABULAR_MODELS = [
    "tabnet",
    "node",
    "fttransformer",
    "tabtransformer",
    "autoint",
    "danet",
    "categoryembedding",
    "gandalf",
]


def build_configs(
    continuous: list[str], categorical: list[str], epochs: int, device: str
) -> dict[str, object]:
    """Return a TabularModel per architecture.

    Args:
        continuous: Numeric column names.
        categorical: Categorical column names.
        epochs: Maximum epochs; early stopping usually stops sooner.
        device: "cpu" or "cuda". Each chain process is pinned to one GPU via
            CUDA_VISIBLE_DEVICES, so "cuda" always means the process's own GPU.

    Returns:
        Architecture name to an unfitted TabularModel.
    """
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import (
        AutoIntConfig,
        CategoryEmbeddingModelConfig,
        DANetConfig,
        FTTransformerConfig,
        GANDALFConfig,
        NodeConfig,
        TabNetModelConfig,
        TabTransformerConfig,
    )

    data_config = DataConfig(
        target=["satisfaction"],
        continuous_cols=continuous,
        categorical_cols=categorical,
    )
    gpu = device == "cuda"
    trainer_config = TrainerConfig(
        max_epochs=epochs,
        early_stopping="valid_loss",
        early_stopping_patience=3,
        checkpoints=None,
        load_best=True,
        accelerator="gpu" if gpu else "cpu",
        devices=[0] if gpu else "auto",
        progress_bar="none",
        logger="none",
    )
    optimizer_config = OptimizerConfig()
    model_configs = {
        "tabnet": TabNetModelConfig(task="classification"),
        "node": NodeConfig(task="classification"),
        "fttransformer": FTTransformerConfig(task="classification"),
        "tabtransformer": TabTransformerConfig(task="classification"),
        "autoint": AutoIntConfig(task="classification"),
        "danet": DANetConfig(task="classification"),
        "categoryembedding": CategoryEmbeddingModelConfig(task="classification"),
        "gandalf": GANDALFConfig(task="classification"),
    }
    return {
        name: TabularModel(
            data_config=data_config,
            model_config=cfg,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        for name, cfg in model_configs.items()
    }


def run_member(
    name: str,
    model,
    frame: pd.DataFrame,
    Xc: pd.DataFrame,
    folds: int,
    seed: int,
) -> dict[str, object]:
    """Cross-validate one tabular model and write its predictions.

    Args:
        name: Architecture name, used for the output filenames.
        model: An unfitted TabularModel (rebuilt per fold from configs).
        frame: Training rows with the label.
        Xc: Competition rows without the label.
        folds: Number of folds.
        seed: Base random seed.

    Returns:
        A roster row with the OOF AUC and wall time.
    """
    y = frame["satisfaction"].to_numpy()
    oof = np.zeros(len(frame), dtype=np.float64)
    test = np.zeros(len(Xc), dtype=np.float64)
    started = time.time()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (fit_index, score_index) in enumerate(splitter.split(frame, y)):
        # Inner split for early stopping, so the held-out fold stays untouched.
        fit, inner = train_test_split(
            fit_index, test_size=0.1, random_state=seed + fold,
            stratify=y[fit_index],
        )
        fresh = model  # TabularModel refits cleanly; reuse the configured object
        fresh.fit(
            train=frame.iloc[fit].reset_index(drop=True),
            validation=frame.iloc[inner].reset_index(drop=True),
        )
        oof[score_index] = fresh.predict(
            frame.iloc[score_index].reset_index(drop=True), ret_logits=False
        )["satisfaction_1"].to_numpy()
        test += (
            fresh.predict(Xc.reset_index(drop=True), ret_logits=False)[
                "satisfaction_1"
            ].to_numpy()
            / folds
        )
        print(f"  {name} fold {fold + 1}/{folds} done ({time.time()-started:.0f}s)", flush=True)
    auc = float(roc_auc_score(y, oof))
    np.save(OUT_DIR / f"oof_{name}.npy", oof)
    np.save(OUT_DIR / f"test_{name}.npy", test)
    return {"member": name, "oof_auc": round(auc, 6), "seconds": round(time.time() - started, 1)}


def main() -> None:
    """Parse arguments, run each requested architecture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True, help="comma-separated architecture names")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                        help="directory holding train.csv and competition_test.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path("config.yaml").read_text())
    features = list(config["features"])
    target = str(config["target_column"])
    categorical = [
        c for c in ["Gender", "Customer Type", "Type of Travel", "Class"] if c in features
    ]
    continuous = [c for c in features if c not in categorical]

    artifacts = Path(args.artifacts)
    print(f"loading {len(features)} features from {artifacts}", flush=True)
    frame = pd.read_csv(
        artifacts / "train.csv", usecols=features + [target], low_memory=False
    )
    Xc = pd.read_csv(artifacts / "competition_test.csv", usecols=features, low_memory=False)
    for column in categorical:
        frame[column] = frame[column].astype("category")
        Xc[column] = Xc[column].astype("category")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roster_path = OUT_DIR / "roster.csv"
    rows = []
    models = build_configs(continuous, categorical, args.epochs, args.device)
    for name in args.only.split(","):
        name = name.strip()
        if name not in models:
            raise SystemExit(f"unknown {name}; have {sorted(models)}")
        print(f"\n=== {name} ===", flush=True)
        row = run_member(name, models[name], frame, Xc, args.folds, args.seed)
        rows.append(row)
        print(f"{name} OOF AUC {row['oof_auc']} ({row['seconds']}s)", flush=True)
        pd.DataFrame(rows).to_csv(
            roster_path,
            mode="a" if roster_path.exists() else "w",
            header=not roster_path.exists(),
            index=False,
        )


if __name__ == "__main__":
    main()