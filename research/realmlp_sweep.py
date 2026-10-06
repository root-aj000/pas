"""
Sweep model configs against one cached feature build, scored by 5-fold CV on train.

Why this exists: run_pipeline.py rebuilds every feature (about 35 minutes of
XGBoost fits for the auxiliary-task block) and then trains one model. Changing a
hyperparameter should not cost a feature rebuild, and selection should not touch
the validation or test splits. So this loads the stage-2 artifacts once, reuses
them for every config, and reports cross-validated AUC on train only.

The fold split is StratifiedKFold(5, shuffle=True, random_state=42), which is the
same split the reference notebook used, so the numbers are comparable with theirs.

Run with: python -m research.realmlp_sweep --configs base,tuned
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ARTIFACTS = Path("artifacts/data_cleaning_encoding")
OUT_DIR = Path("research/sweep")

# The tuned RealMLP recipe, verbatim from the reference notebook's REALMLP dict.
# Kept here as a named preset so a sweep can compare "what pytabkit does by
# default" against "what this dataset actually wants" in one run.
TUNED = {
    "batch_size": 256,
    "use_early_stopping": False,
    "lr": 0.053,
    "wd": 0.0236,
    "sq_mom": 0.988,
    "lr_sched": "flat_anneal",
    "wd_sched": "cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 5,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "expm4t",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": [
        "one_hot",
        "median_center",
        "robust_scale",
        "smooth_clip",
        "embedding",
        "l2_normalize",
    ],
}

XGB_PRESET = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5e-6,
    "reg_lambda": 0.0,
    "max_bin": 256,
    "subsample": 0.65,
    "colsample_bylevel": 0.9,
    "tree_method": "hist",
    "early_stopping_rounds": 150,
}


def load_frames(features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Read train.csv with only the columns the config needs.

    Args:
        features: The feature column names.
        target: The label column name.

    Returns:
        The feature frame and the label series, both float32 where possible.
    """
    dtypes = {c: "float32" for c in features}
    frame = pd.read_csv(
        ARTIFACTS / "train.csv",
        usecols=features + [target],
        dtype={**dtypes, target: "int8"},
        low_memory=False,
    )
    return frame[features], frame[target]


def build_model(kind: str, params: dict[str, object], seed: int):
    """Create one unfitted estimator.

    Args:
        kind: realmlp, xgboost, lightgbm or catboost.
        params: Keyword arguments for the estimator.
        seed: Random seed.

    Returns:
        The estimator.
    """
    if kind == "realmlp":
        from pytabkit import RealMLP_TD_Classifier

        return RealMLP_TD_Classifier(**params, random_state=seed, verbosity=0)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, random_state=seed)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, random_state=seed, verbosity=-1)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params, random_seed=seed, verbose=0)
    raise ValueError(f"unknown model kind: {kind}")


def cross_validate(
    kind: str,
    params: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
    categorical: list[str],
) -> tuple[np.ndarray, float]:
    """Score one config by stratified k-fold CV on the training rows.

    Args:
        kind: Which estimator to build.
        params: Its keyword arguments.
        X: Feature frame.
        y: Labels.
        folds: Number of folds.
        seed: Random seed, used for the split and for each fit.
        categorical: Columns to declare categorical, for the estimators that
            take such an argument.

    Returns:
        The out-of-fold probabilities, and the AUC they score.
    """
    oof = np.zeros(len(y), dtype=np.float64)
    started = time.time()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (fit_index, score_index) in enumerate(splitter.split(X, y)):
        model = build_model(kind, params, seed + fold)
        extra = {"cat_col_names": categorical} if kind == "realmlp" else {}
        try:
            model.fit(X.iloc[fit_index], y.iloc[fit_index], **extra)
        except TypeError:
            # XGBoost wants eval_set for early stopping, and does not take
            # cat_col_names. Split the arguments per estimator instead of
            # branching on the kind everywhere above.
            if kind == "xgboost":
                model.set_params(early_stopping_rounds=None)
                model.fit(X.iloc[fit_index], y.iloc[fit_index])
            else:
                raise
        oof[score_index] = model.predict_proba(X.iloc[score_index])[:, 1]
        del model
    return oof, time.time() - started


def main() -> None:
    """Parse arguments, run each config, print a leaderboard."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma-separated preset names")
    parser.add_argument("--kind", default="realmlp")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--members", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    config = yaml.safe_load(Path("config.yaml").read_text())
    features = list(config["features"])
    target = str(config["target_column"])
    categorical = [
        c for c in config.get("categorical_columns", []) if c in features
    ] or [
        c
        for c in features
        if c.endswith("_cat_")
        or c in ("Gender", "Customer Type", "Type of Travel", "Class")
    ]

    X, y = load_frames(features, target)
    print(
        f"loaded {X.shape[0]} rows x {X.shape[1]} features, {len(categorical)} categorical",
        flush=True,
    )

    presets: dict[str, dict[str, object]] = {
        "base": {"n_ens": args.members, "n_epochs": args.epochs, "device": "cpu"},
        "tuned": {
            "n_ens": args.members,
            "n_epochs": args.epochs,
            "device": "cpu",
            **TUNED,
        },
    }
    if args.kind == "xgboost":
        presets = {"base": {**XGB_PRESET, "device": "cpu"}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.configs.split(","):
        name = name.strip()
        if name not in presets:
            raise SystemExit(f"unknown preset {name}; have {sorted(presets)}")
        params = presets[name]
        print(f"\n=== {name} ===\n{json.dumps(params, default=str)}", flush=True)
        oof, seconds = cross_validate(
            args.kind, params, X, y, args.folds, args.seed, categorical
        )
        auc = roc_auc_score(y, oof)
        np.save(OUT_DIR / f"oof_{args.kind}_{name}.npy", oof)
        print(f"OOF AUC {auc:.6f}   ({seconds:.0f}s)", flush=True)
        results.append((name, auc, seconds))

    print("\n=== leaderboard ===")
    for name, auc, seconds in sorted(results, key=lambda r: -r[1]):
        print(f"{name:28s} {auc:.6f}  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
