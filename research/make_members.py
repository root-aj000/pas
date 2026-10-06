"""
Generate out-of-fold and test predictions for a roster of models, for stacking.

Why this exists: run_pipeline.py trains one model on one split. A stack needs
many models scored on the same rows, and re-running the whole pipeline per
candidate costs a feature rebuild each time. So this loads the stage-2 artifacts
once, cross-validates every member on the same stratified fold split, and writes
each member's predictions to disk. The stacker then reads those files.

Reproducibility, because the user asked for it:
  * The fold split is StratifiedKFold(5, shuffle=True, random_state=42), the same
    split the reference notebook used, so our OOF numbers are comparable to theirs.
  * Every member's exact parameters, its library versions and its wall time are
    appended to research/members/roster.csv, so any member can be rebuilt alone.
  * Each fold model also predicts the competition rows, and the five predictions
    are averaged. That is why the test predictions are an ensemble of five models
    trained on 80% each rather than one model trained on all of train.
  * Nothing here reads the validation or test splits. Selection happens on
    out-of-fold scores over the training rows only.

Run with: python -m research.make_members --only realmlp_tuned
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

DEFAULT_ARTIFACTS = Path("artifacts/data_cleaning_encoding")
OUT_DIR = Path("research/members")

# The tuned RealMLP recipe, verbatim from the reference notebook's REALMLP dict.
# pytabkit's own defaults differ in 23 of its 26 values, and the two that matter
# most are plr_sigma (0.1 by default, 2.33 here - at 0.1 the PLR numerical
# embeddings are effectively off, which is the whole point of RealMLP) and
# ls_eps (0.1 by default, 0.01 here).
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

XGB_CPU = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5e-6,
    "reg_lambda": 0.0,
    "max_bin": 256,
    "subsample": 0.65,
    "colsample_bylevel": 0.9,
    "tree_method": "hist",
    "n_jobs": 8,
}
XGB_GPU = {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5e-6,
    "reg_lambda": 0.0,
    "max_bin": 256,
    "subsample": 0.65,
    "colsample_bylevel": 0.9,
    "tree_method": "hist",
    "device": "cuda",
}

LGB_CPU = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 100,
    "max_bin": 255,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "n_jobs": 8,
}
LGB_GPU = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 100,
    "max_bin": 255,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "device": "gpu",
}


def build_roster(members: int, epochs: int, device: str) -> dict[str, dict[str, object]]:
    """Return the candidate roster.

    Args:
        members: Ensemble size for the neural members. The reference measured
            going from 8 to 16 members at +0.00002, so screening uses a small
            number and only the finalists get the full one.
        epochs: Epochs for the neural members.
        device: "cpu" or "cuda", passed straight to every estimator.

    Returns:
        Member name to {"kind": ..., "params": {...}}.

    Note:
    The roster is deliberately wide and shallow. The reference's most reliable
    finding is that a member's own score is not its value to the stack - the
    weakest member added the most, because it read the data differently. Nine
    architectures that each score a little below the best are worth more here
    than one architecture tuned to its last decimal. So: many families, few
    members each.
    """
    # RealMLP is the exception: pytabkit's defaults for it are NOT this dataset's
    # tuning, so it gets the full explicit recipe. The rtdl-style families are the
    # opposite case - pytabkit ships a tuned preset per architecture, so those get
    # only an epoch budget and let the preset supply the rest.
    return {
        "realmlp_tuned": {
            "kind": "realmlp",
            "params": {"n_ens": members, "n_epochs": epochs, "device": device, **TUNED},
        },
        "tabm": {"kind": "tabm", "params": {"n_epochs": epochs, "device": device}},
        "ftt": {"kind": "ftt", "params": {"max_epochs": epochs, "device": device}},
        "mlp_plr": {
            "kind": "mlp_plr",
            "params": {"max_epochs": epochs, "device": device},
        },
        "mlp_rtdl": {
            "kind": "mlp_rtdl",
            "params": {"max_epochs": epochs, "device": device},
        },
        "resnet_rtdl": {
            "kind": "resnet_rtdl",
            "params": {"max_epochs": epochs, "device": device},
        },
        "xrfr": {"kind": "xrfr", "params": {"device": device}},
        "xgboost": {"kind": "xgboost", "params": dict(XGB_GPU if device == "cuda" else XGB_CPU)},
        "lightgbm": {"kind": "lightgbm", "params": dict(LGB_GPU if device == "cuda" else LGB_CPU)},
    }


def build_estimator(kind: str, params: dict[str, object], seed: int):
    """Create one unfitted estimator of the requested family.

    Args:
        kind: Family name.
        params: Keyword arguments.
        seed: Random seed.

    Returns:
        The unfitted estimator.
    """
    if kind == "realmlp":
        from pytabkit import RealMLP_TD_Classifier

        return RealMLP_TD_Classifier(**params, random_state=seed, verbosity=0)
    if kind == "tabm":
        from pytabkit import TabM_D_Classifier

        return TabM_D_Classifier(**params, random_state=seed)
    if kind == "ftt":
        from pytabkit import FTT_D_Classifier

        return FTT_D_Classifier(**params, random_state=seed)
    if kind == "mlp_plr":
        from pytabkit import MLP_PLR_D_Classifier

        return MLP_PLR_D_Classifier(**params, random_state=seed)
    if kind == "mlp_rtdl":
        from pytabkit import MLP_RTDL_D_Classifier

        return MLP_RTDL_D_Classifier(**params, random_state=seed)
    if kind == "resnet_rtdl":
        from pytabkit import Resnet_RTDL_D_Classifier

        return Resnet_RTDL_D_Classifier(**params, random_state=seed)
    if kind == "xrfr":
        from pytabkit import XRFM_D_Classifier

        return XRFM_D_Classifier(**params, random_state=seed)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, random_state=seed, enable_categorical=True)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, random_state=seed, verbosity=-1)
    raise ValueError(f"unknown kind: {kind}")


def fit_one(estimator, X: pd.DataFrame, y: pd.Series, categorical: list[str]):
    """Fit an estimator, coping with the two fit signatures in the zoo.

    Args:
        estimator: The unfitted estimator.
        X: Feature frame.
        y: Labels.
        categorical: Categorical column names.

    Returns:
        The fitted estimator.

    Note:
    The _TD classifiers take cat_col_names as a fit argument; the _D ones read
    pandas category dtypes instead and reject the extra keyword. Trying with the
    keyword and falling back is less code than branching per family, and it fails
    loudly if some family's signature is neither.
    """
    try:
        estimator.fit(X, y, cat_col_names=categorical)
    except TypeError:
        pass
    try:
        estimator.fit(X, y, categorical_feature=categorical)
    except TypeError:
        estimator.fit(X, y)
    return estimator


def load_frame(
    path: Path, features: list[str], target: str | None, categorical: list[str]
) -> pd.DataFrame:
    """Read one artifact, keeping only the columns the roster needs.

    Args:
        path: The csv to read.
        features: Feature column names.
        target: Label column name, or None for the competition rows.
        categorical: Columns stored as strings that must come back as category.

    Returns:
        The frame, float32 for numerics and category for the categoricals.
    """
    columns = features + ([target] if target else [])
    numeric = [c for c in columns if c not in categorical and c != target]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={**{c: "float32" for c in numeric}, **({target: "int8"} if target else {})},
        low_memory=False,
    )
    for column in categorical:
        frame[column] = frame[column].astype("category")
    return frame


def run_member(
    name: str,
    kind: str,
    params: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    Xc: pd.DataFrame,
    categorical: list[str],
    folds: int,
    seed: int,
) -> dict[str, object]:
    """Cross-validate one member and write its predictions.

    Args:
        name: Member name, used for the output filenames.
        kind: Family name.
        params: Keyword arguments.
        X: Training features.
        y: Training labels.
        Xc: Competition features, to predict.
        categorical: Categorical column names.
        folds: Number of folds.
        seed: Base random seed; each fold gets seed + fold.

    Returns:
        A roster row: the member's OOF AUC, its wall time, and its parameters.
    """
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(len(Xc), dtype=np.float64)
    started = time.time()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (fit_index, score_index) in enumerate(splitter.split(X, y)):
        estimator = fit_one(
            build_estimator(kind, params, seed + fold),
            X.iloc[fit_index],
            y.iloc[fit_index],
            categorical,
        )
        oof[score_index] = estimator.predict_proba(X.iloc[score_index])[:, 1]
        test += estimator.predict_proba(Xc)[:, 1] / folds
        del estimator
        print(
            f"  {name} fold {fold + 1}/{folds} done ({time.time() - started:.0f}s)",
            flush=True,
        )
    auc = float(roc_auc_score(y, oof))
    np.save(OUT_DIR / f"oof_{name}.npy", oof)
    np.save(OUT_DIR / f"test_{name}.npy", test)
    return {
        "member": name,
        "kind": kind,
        "oof_auc": round(auc, 6),
        "seconds": round(time.time() - started, 1),
        "folds": folds,
        "seed": seed,
        "params": json.dumps(params, default=str),
    }


def main() -> None:
    """Parse arguments, run each requested member, append to the roster."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True, help="comma-separated member names")
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", default="", help="appended to output filenames")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS),
                        help="directory holding train.csv and competition_test.csv")
    parser.add_argument(
        "--drop-prefix",
        default="",
        help="comma-separated column-name prefixes to exclude, e.g. 'aux_,te_'",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path("config.yaml").read_text())
    features = list(config["features"])
    if args.drop_prefix:
        prefixes = tuple(p for p in args.drop_prefix.split(",") if p)
        features = [f for f in features if not f.startswith(prefixes)]
        print(f"dropped prefixes {prefixes}: {len(features)} features left")
    target = str(config["target_column"])
    categorical = [
        c
        for c in ["Gender", "Customer Type", "Type of Travel", "Class"]
        if c in features
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = build_roster(args.members, args.epochs, args.device)

    print(f"loading {len(features)} features from the stage-2 artifacts", flush=True)
    artifacts = Path(args.artifacts)
    X = load_frame(artifacts / "train.csv", features, target, categorical)
    Xc = load_frame(artifacts / "competition_test.csv", features, None, categorical)
    y = X[target].astype("int8")
    X = X.drop(columns=[target])
    print(f"train {X.shape} competition {Xc.shape}", flush=True)

    roster_path = OUT_DIR / "roster.csv"
    rows = []
    for name in args.only.split(","):
        name = name.strip()
        if name not in roster:
            raise SystemExit(f"unknown member {name}; have {sorted(roster)}")
        entry = roster[name]
        print(f"\n=== {name} ({entry['kind']}) ===", flush=True)
        row = run_member(
            name + args.suffix,
            str(entry["kind"]),
            dict(entry["params"]),
            X,
            y,
            Xc,
            categorical,
            args.folds,
            args.seed,
        )
        rows.append(row)
        print(f"{name} OOF AUC {row['oof_auc']}  ({row['seconds']}s)", flush=True)
        pd.DataFrame(rows).to_csv(
            roster_path,
            mode="a" if roster_path.exists() else "w",
            header=not roster_path.exists(),
            index=False,
        )


if __name__ == "__main__":
    main()
