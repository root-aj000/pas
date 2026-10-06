"""
Trains several models by cross-validation and combines them into one submission.

Why this exists: the single-model path in stage 3 fits one estimator on the
training split and predicts once. An ensemble needs a different thing - every
member cross-validated so that every row has a prediction from a model that never
saw it, and a combiner fitted on those out-of-fold predictions. Doing that from a
notebook leaves the result unreproducible, because the predictions live in scratch
files rather than in the pipeline. So it lives here.

The method, and why each part is the way it is:

* **Members are scored out of fold.** A member's own score is only meaningful if
  no row was predicted by a model trained on it. Every row is therefore predicted
  exactly once, by the fold model that held it out.
* **The competition rows are predicted by all the fold models, averaged.** Not by
  one model refitted on everything. Averaging the ten fold models is what the
  reference notebook does, and it costs nothing extra because those models exist
  already.
* **Members are combined in logit space, by logistic regression.** ROC-AUC
  depends only on ranking, so logits are the natural scale, and a linear model
  with one weight per member cannot overfit half a million rows even with a dozen
  members.
* **The reported stack score is nested.** The stacker has weights, so scoring it
  on rows its weights were fitted on would flatter it. Every number printed here
  comes from folds where each row's score came from weights fitted on other rows.

Two failure modes this module exists to prevent, both of which happened:

* A member trained with `patience` but no validation data cannot early-stop, so
  it silently ran every epoch and was scored on a model nobody had checked.
  `fit_member` gives every estimator that wants early stopping a validation slice
  carved out of the *fit* rows, never the scored ones.
* A partially written prediction file looks exactly like a finished one. Nothing
  here reads predictions that were not written by a completed run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.utils.common import log_step, save_json

# The fold split every selection number in this project was measured on. Changing
# it invalidates comparison with docs/experiment_log.md, so it is a constant rather
# than a per-run setting.
SPLIT_SEED = 42
STACK_SEED = 7
LOGIT_FLOOR = 1e-6


@dataclass
class MemberSpec:
    """One model in the ensemble.

    Attributes:
        name: Output name, and the key in the prediction files.
        kind: Family, e.g. "realmlp", "xgboost", "lightgbm", "tabnet".
        params: Keyword arguments for that family.
        drop_prefix: Column-name prefixes to exclude, so the same member can be
            trained on a narrower view of the data.
        drop_suffix: Column-name suffixes to exclude, same purpose.
        target_encodings: Whether to append the in-fold target-encoding block.
            Trees want it; the neural members are better off without it. They are
            the only members that see every column at once, and 21 per-column plus
            10 pair encodings on top of the 6 stage 2 already builds puts a fifth
            of the input into encodings of columns that are still present beside
            them. The reference gives RealMLP six encoding keys out of ~72 columns
            and measured that crossing more of them overfits.
    """

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    drop_prefix: tuple[str, ...] = ()
    drop_suffix: tuple[str, ...] = ()
    target_encodings: bool = True


@dataclass
class EnsembleConfig:
    """Everything the ensemble run needs that is not per-member.

    Attributes:
        enabled: When false the pipeline keeps its single-model behaviour.
        members: The members to train.
        folds: Cross-validation folds. Ten by measurement, not by default.
        seed: Base seed; fold n uses seed + n.
        device: "cpu" or "cuda", passed to every estimator.
        stack_C: Logistic regression regularisation. 1.0 was measured insensitive
            between 0.03 and 3.
        te_columns: Raw columns to target-encode inside each fold. Empty disables
            the block.
        combiner: "logistic" for logistic regression on member logits, "rank" for
            rank-averaging with fitted weights.
    """

    enabled: bool = False
    members: list[MemberSpec] = field(default_factory=list)
    folds: int = 10
    seed: int = 42
    device: str = "cpu"
    stack_C: float = 1.0
    te_columns: list[str] = field(default_factory=list)
    combiner: str = "logistic"


def select_features(all_features: list[str], spec: MemberSpec) -> list[str]:
    """Return the columns one member should see.

    Args:
        all_features: Every feature the pipeline built.
        spec: The member, which may want a narrower view.

    Returns:
        The feature list for that member.

    Note:
    A member trained on fewer columns is a genuinely different member, not a
    worse copy. Measured on this dataset, the same network on the 22 raw columns
    added more to the stack than two extra architectures did. That is the whole
    reason this function exists rather than a single shared feature list.
    """
    keep = [
        f
        for f in all_features
        if not (spec.drop_prefix and f.startswith(spec.drop_prefix))
        and not (spec.drop_suffix and f.endswith(spec.drop_suffix))
    ]
    if not keep:
        raise ValueError(f"member {spec.name} has no features left after filtering")
    return keep


def resolve_device(requested: str) -> str:
    """Resolve `ensemble.device` to something an estimator will accept.

    Args:
        requested: "auto", "cpu" or "cuda".

    Returns:
        "cuda" when a GPU is visible and either "auto" or "cuda" was asked for,
        otherwise "cpu".

    Note:
    "auto" exists because the alternative is a config that is wrong on one of the
    two machines it has to be right on. The same file is committed for a laptop
    with no GPU and a Kaggle session with two, so a fixed "cpu" trains nine
    members over 700k rows on the processor and a fixed "cuda" crashes locally.
    Neither failure says which line of config to change.
    """
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - no torch is a cpu-only machine
            return "cpu"
    return requested


def load_ensemble_frames(
    artifacts_dir: Path, features: list[str], categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and competition frames for the ensemble.

    Args:
        artifacts_dir: The `data_cleaning_encoding` folder stage 2 wrote.
        features: The configured feature list.
        categorical: Categorical column names within it.

    Returns:
        The training rows with the label, and the competition rows without.

    Raises:
        FileNotFoundError: If stage 2 has not run.
        ValueError: If the training frame carries two columns of the same name.

    Note:
    The validation split is folded into training. Stage 2 splits 70/15/15 for the
    single-model path, where the validation file gave stages 3 and 4 something to
    early-stop and select on. The ensemble cross-validates internally and needs
    no held-out split of its own, so reading only train.csv threw away 15% of the
    labelled rows - 489,743 instead of 699,635 - for every member.

    Safe to concatenate: stage 2 encoded both files in one pass, with identical
    columns and the same train-fitted statistics, and its label-based columns were
    cross-fitted, so a validation row's encoding never used its own label. The
    fold structure that matters is created by the caller.

    This lives here rather than in the stage because the per-GPU workers have to
    build exactly the same frames, and two copies of a frame loader is how a
    worker ends up training on different rows from the stack.
    """
    from src.utils.common import apply_categorical_encoding, load_json

    train_path = artifacts_dir / "train.csv"
    validation_path = artifacts_dir / "validation.csv"
    competition_path = artifacts_dir / "competition_test.csv"
    for path in (train_path, validation_path, competition_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run stage 2 first: python -m "
                "src.pipeline.stage_02_data_cleaning_encoding"
            )

    contract = load_json(artifacts_dir / "features.json")
    encoding = str(contract.get("categorical_encoding", "one_hot"))

    def read(path: Path, label: str) -> pd.DataFrame:
        return apply_categorical_encoding(
            pd.read_csv(path, low_memory=False), categorical, encoding, label
        )

    train_only = read(train_path, "ensemble_train")
    validation = read(validation_path, "ensemble_train")
    train_frame = pd.concat([train_only, validation], ignore_index=True)
    if train_frame.columns.duplicated().any():
        duplicated = sorted(set(train_frame.columns[train_frame.columns.duplicated()]))
        raise ValueError(
            f"duplicate columns in the training frame: {duplicated}. Stage 2 wrote "
            f"two features with the same name."
        )
    competition_frame = read(competition_path, "ensemble_competition")
    log_step(
        "ensemble_frames",
        rows=len(train_frame),
        train_rows=len(train_only),
        validation_rows=len(validation),
        competition_rows=len(competition_frame),
    )
    return train_frame, competition_frame


def plan_gpu_shards(
    members: list[MemberSpec], device: str, visible_devices: int | None = None
) -> list[list[str]]:
    """Split member names across the GPUs, or return one group to run inline.

    Args:
        members: The configured members, in order.
        device: The resolved device, "cpu" or "cuda".
        visible_devices: How many GPUs this process can see. Read from torch when
            not given.

    Returns:
        A list of member-name groups. One group means "run these yourself, in
        this process"; more than one means one worker process per GPU.

    Note:
    Members are dealt round-robin rather than split into contiguous blocks,
    because the members differ in cost by an order of magnitude - the two RealMLP
    members are minutes each, CatBoost is tens of minutes. Round-robin puts a
    slow member on each GPU instead of loading one with all the slow ones.

    A single GPU, or cpu, returns one group and the caller runs it inline. No
    subprocess is worth its overhead for one group.
    """
    names = [member.name for member in members]
    if device != "cuda":
        return [names]
    if visible_devices is None:
        try:
            import torch

            visible_devices = int(torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            visible_devices = 0
    groups = min(max(visible_devices, 1), max(len(names), 1))
    return [names[index::groups] for index in range(groups)]


def fit_member_to_artifacts(
    spec: MemberSpec,
    train_frame: pd.DataFrame,
    competition_frame: pd.DataFrame,
    all_features: list[str],
    categorical: list[str],
    config: EnsembleConfig,
    target_column: str,
    artifacts_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validate one member and save its predictions.

    Args:
        spec: Which member.
        train_frame: Training rows, label included.
        competition_frame: Competition rows.
        all_features: Every configured feature.
        categorical: Categorical column names.
        config: Fold count, seed, device.
        target_column: Label name, used only for error messages.
        artifacts_dir: Where `oof_<name>.npy` and `test_<name>.npy` are written.

    Returns:
        The member's out-of-fold and competition probabilities, and its AUC.

    Note:
    Split out of the member loop because the per-GPU worker calls exactly this.
    Both paths must produce byte-identical arrays for the same member: the seed
    is `config.seed + fold` and never depends on which GPU or which group the
    member ran in, so a sharded run and a single-process run stack the same
    numbers.
    """
    features = select_features(all_features, spec)
    missing = [c for c in features if c not in competition_frame.columns]
    if missing:
        raise ValueError(
            f"member {spec.name} wants {len(missing)} columns the competition "
            f"frame does not have: {missing[:5]}. Run stage 2 first."
        )
    use_cat = [c for c in categorical if c in features]
    oof, test = train_member(
        spec,
        train_frame[features],
        train_frame[target_column].astype("int8"),
        competition_frame[features],
        use_cat,
        config,
        target_column,
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / f"oof_{spec.name}.npy", oof)
    np.save(artifacts_dir / f"test_{spec.name}.npy", test)
    return oof, test, float(roc_auc_score(train_frame[target_column], oof))


def apply_device(kind: str, params: dict[str, Any], device: str) -> dict[str, Any]:
    """Return member params with the run's device written into them.

    Args:
        kind: Family name.
        params: The member's configured params.
        device: "cpu" or "cuda", from `ensemble.device` in config.yaml.

    Returns:
        A copy of params carrying the right key for that family. On "cpu" the
        params are returned untouched, so a member that names its own device in
        config.yaml can still opt into the GPU on a CPU-configured run.

    Note:
    Every family spells it differently - XGBoost wants `device="cuda"`,
    LightGBM wants `device="gpu"`, CatBoost wants `task_type="GPU"`, and
    pytabkit and TabNet both want `device`. None of them defaults to the GPU.
    Without this, `ensemble.device: cuda` in config.yaml was a key nothing read:
    the nine members were built with no device at all and a Kaggle run trained
    the whole ensemble on CPU.
    """
    out = dict(params)
    if device != "cuda":
        return out
    if kind == "xgboost":
        out["device"] = "cuda"
    elif kind == "lightgbm":
        out["device"] = "gpu"
    elif kind == "catboost":
        out["task_type"] = "GPU"
    else:
        out["device"] = device
    return out


def build_estimator(kind: str, params: dict[str, Any], seed: int):
    """Create one unfitted estimator.

    Args:
        kind: Family name.
        params: Keyword arguments.
        seed: Random seed.

    Returns:
        The unfitted estimator.

    Raises:
        ValueError: If the family is unknown. Better here than a KeyError from
            deep inside a fit three hours into a run.
    """
    if kind == "realmlp":
        from pytabkit import RealMLP_TD_Classifier

        return RealMLP_TD_Classifier(**params, random_state=seed, verbosity=0)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params, random_state=seed, enable_categorical=True)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        # Only supply verbosity if the config did not. Passing it twice is a
        # TypeError, and a member must not fail because a caller was helpful.
        params.setdefault("verbosity", -1)
        return LGBMClassifier(**params, random_state=seed)
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params, random_seed=seed, verbose=0)
    if kind == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        return TabNetClassifier(
            verbose=0,
            seed=seed + 1,
            device_name="cuda" if params.get("device") == "cuda" else "cpu",
        )
    raise ValueError(f"unknown member kind: {kind}")


def _tabnet_arrays(
    frame: pd.DataFrame, categorical: list[str]
) -> tuple[np.ndarray, list[int], list[int]]:
    """Encode a frame the way TabNet needs it.

    Args:
        frame: Rows to encode.
        categorical: Columns to treat as categorical.

    Returns:
        The float32 array, the categorical column positions, and their widths.

    Note:
    TabNet takes category indices and sizes at construction, not at fit, so the
    estimator has to be rebuilt once the columns are known.
    """
    positions: list[int] = []
    sizes: list[int] = []
    encoded = frame.copy()
    for position, column in enumerate(encoded.columns):
        if column in categorical or str(encoded[column].dtype) == "category":
            codes, uniques = pd.factorize(encoded[column])
            encoded[column] = codes
            positions.append(position)
            sizes.append(len(uniques))
    for column in encoded.columns:
        if encoded[column].dtype == object:
            encoded[column] = pd.factorize(encoded[column])[0]
    return encoded.to_numpy(dtype=np.float32), positions, sizes


def fit_member(
    estimator,
    kind: str,
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    seed: int,
):
    """Fit one estimator on one fold, handling each family's quirks.

    Args:
        estimator: The unfitted estimator.
        kind: Its family.
        X: Fit rows.
        y: Fit labels.
        categorical: Categorical column names.
        seed: Random seed for the inner split.

    Returns:
        The fitted estimator.

    Note:
    TabNet's `patience` is a no-op without an `eval_set`, so it gets a validation
    slice carved out of the fit rows. That slice is taken from the fit rows and
    never from the rows this model will be scored on - otherwise early stopping
    would be selecting on the very rows it is then evaluated against, and the
    score would mean nothing.
    """
    if kind == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        Xn, positions, sizes = _tabnet_arrays(X, categorical)
        yn = y.to_numpy()
        inner_fit, inner_val = train_test_split(
            np.arange(len(yn)), test_size=0.1, random_state=seed, stratify=yn
        )
        fitted = TabNetClassifier(
            verbose=0,
            seed=seed + 1,
            device_name=getattr(estimator, "device_name", "cpu"),
            cat_idxs=positions,
            cat_dims=sizes,
            cat_emb_dim=1,
        )
        fitted.fit(
            Xn[inner_fit],
            yn[inner_fit],
            eval_set=[(Xn[inner_val], yn[inner_val])],
            patience=5,
            max_epochs=40,
            batch_size=4096,
        )
        return fitted

    if kind == "catboost":
        # CatBoost takes categorical columns by name and rejects pandas `category`
        # dtype with a clear message, so hand it the names it asked for.
        estimator.fit(X, y, cat_features=categorical)
        return estimator

    try:
        estimator.fit(X, y, cat_col_names=categorical)
    except TypeError:
        try:
            estimator.fit(X, y, categorical_feature=categorical)
        except TypeError:
            estimator.fit(X, y)
    return estimator


def predict_member(estimator, kind: str, frame: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a frame.

    Args:
        estimator: A fitted estimator.
        kind: Its family.
        frame: Rows to score.

    Returns:
        One probability per row.
    """
    if kind == "tabnet":
        Xn, _, _ = _tabnet_arrays(frame, [])
        return estimator.predict_proba(Xn)[:, 1]
    return estimator.predict_proba(frame)[:, 1]


TE_PAIR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Class", "Type of Travel"),
    ("Customer Type", "Type of Travel"),
    ("Class", "Customer Type"),
    ("Gender", "Class"),
    ("Inflight wifi service", "Online boarding"),
    ("Inflight wifi service", "Type of Travel"),
    ("Online boarding", "Type of Travel"),
    ("Seat comfort", "Inflight entertainment"),
    ("Class", "Inflight wifi service"),
    ("Class", "Online boarding"),
)


def _pair_keys(frame: pd.DataFrame, pairs: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Join column pairs into one string key each, for joint target encoding.

    Args:
        frame: Rows to build keys for.
        pairs: The column pairs being joined.

    Returns:
        One column per pair, named `pair_<a>_<b>`.
    """
    return pd.DataFrame(
        {
            f"pair_{a}_{b}": frame[a].astype(str) + "_" + frame[b].astype(str)
            for a, b in pairs
        }
    )


def add_in_fold_target_encodings(
    fit_frame: pd.DataFrame,
    y: pd.Series,
    score_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    columns: list[str],
    pair_columns: tuple[tuple[str, str], ...],
    seed: int,
    inner_folds: int = 5,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Target-encode columns and column pairs, fold-safely, for one CV fold.

    Args:
        fit_frame: The fold's training features. The label comes in as `y`,
            because the caller holds features and labels separately and a frame
            carrying both invites the label to be fitted on by accident.
        y: Labels for the fold's training rows, aligned to fit_frame.
        score_frame: The rows this fold will be scored on.
        test_frame: The competition rows.
        columns: Columns to encode individually.
        pair_columns: Column pairs to encode jointly.
        seed: Random seed for the inner cross-fitting.
        inner_folds: Inner folds used to cross-fit the training rows.
        smoothing: Blend toward the prior, larger meaning less trust in small groups.

    Returns:
        The three frames, each with `tef_` and `te2_` columns appended.

    Note:
    This is deliberately stricter than the target encodings stage 2 builds. Those
    are cross-fitted once over the whole training set, so a scoring row's encoded
    value depends on the labels of other rows *in its own scoring fold*. Here the
    encodings are refitted inside every outer fold from that fold's training part
    only, and cross-fitted again within it, so no row is ever encoded using its
    own label and no scoring row contributes to another scoring row's value.

    The reference measured +0.000148 for per-column encodings and +0.000067 for
    the column pairs, on top of a base that already had a route profile.

    The `tef_` prefix, not `te_`, because stage 2 already wrote columns called
    `te_Age`, `te_FD` and `te_FD|Age`. A column named `te_Age` here collided with
    the one from stage 2, and `pd.concat` keeps both: the merged frame carried two
    `te_Age` columns and every member silently received a duplicated input.
    """
    y = y.astype("int8")

    def _encode(
        train_keys: pd.Series, train_y: pd.Series, apply_keys: pd.Series
    ) -> np.ndarray:
        # The prior is read from the rows this encoder was fitted on, never from
        # the whole fold. A prior that included the row being encoded leaks that
        # row's label into every unseen group's value, which a test with a unique
        # key catches and a test on a shared key cannot.
        prior = float(train_y.mean())
        stats = (
            pd.DataFrame({"k": train_keys, "y": train_y})
            .groupby("k")["y"]
            .agg(["sum", "count"])
        )
        table = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
        return apply_keys.map(table).fillna(prior).to_numpy(dtype="float32")

    def _build_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        fit_keys = _pair_keys(fit_frame, pair_columns)
        score_keys = _pair_keys(score_frame, pair_columns)
        test_keys = _pair_keys(test_frame, pair_columns)
        out_fit = pd.DataFrame(index=np.arange(len(fit_keys)))
        out_score = pd.DataFrame(index=np.arange(len(score_keys)))
        out_test = pd.DataFrame(index=np.arange(len(test_keys)))
        for a, b in pair_columns:
            name = f"pair_{a}_{b}"
            column = fit_keys[name]
            encoded_fit = np.zeros(len(fit_keys), dtype="float32")
            for inner_fit, inner_score in splitter.split(
                np.arange(len(fit_keys)), y.to_numpy()
            ):
                encoded_fit[inner_score] = _encode(
                    column.iloc[inner_fit], y.iloc[inner_fit], column.iloc[inner_score]
                )
            out_fit[f"te2_{name}"] = encoded_fit
            out_score[f"te2_{name}"] = _encode(column, y, score_keys[name])
            out_test[f"te2_{name}"] = _encode(column, y, test_keys[name])
        return out_fit, out_score, out_test

    fit_keys = fit_frame[columns].astype(str)
    score_keys = score_frame[columns].astype(str)
    test_keys = test_frame[columns].astype(str)
    out_fit = pd.DataFrame(index=np.arange(len(fit_keys)))
    out_score = pd.DataFrame(index=np.arange(len(score_keys)))
    out_test = pd.DataFrame(index=np.arange(len(test_keys)))
    # Unstratified on purpose. A stratified split is assigned by label
    # composition, so changing one label moves rows between folds and changes the
    # prior they are encoded against - which makes "this row's encoding does not
    # depend on its own label" untestable, and it is the property that matters
    # most here. Group sizes are what an encoder cares about, not label balance
    # across folds.
    splitter = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    positions = np.arange(len(fit_keys))
    for column in columns:
        # Cross-fitted inside the fold: a training row's encoding comes from
        # models that never saw it.
        encoded_fit = np.zeros(len(fit_keys), dtype="float32")
        for inner_fit, inner_score in splitter.split(positions, y.to_numpy()):
            encoded_fit[inner_score] = _encode(
                fit_keys[column].iloc[inner_fit],
                y.iloc[inner_fit],
                fit_keys[column].iloc[inner_score],
            )
        out_fit[f"tef_{column}"] = encoded_fit
        # Scoring and competition rows take the full fold-fit mapping.
        out_score[f"tef_{column}"] = _encode(fit_keys[column], y, score_keys[column])
        out_test[f"tef_{column}"] = _encode(fit_keys[column], y, test_keys[column])
    pair_fit, pair_score, pair_test = _build_pairs()
    for frame, extra in (
        (out_fit, pair_fit),
        (out_score, pair_score),
        (out_test, pair_test),
    ):
        for name in extra.columns:
            frame[name] = extra[name]
    return (
        pd.concat([fit_frame.reset_index(drop=True), out_fit], axis=1),
        pd.concat([score_frame.reset_index(drop=True), out_score], axis=1),
        pd.concat([test_frame.reset_index(drop=True), out_test], axis=1),
    )


def train_member(
    spec: MemberSpec,
    X: pd.DataFrame,
    y: pd.Series,
    X_competition: pd.DataFrame,
    categorical: list[str],
    config: EnsembleConfig,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validate one member and return its out-of-fold and test predictions.

    Args:
        spec: Which member to train.
        X: Training features for this member's view of the data.
        y: Training labels.
        X_competition: The competition rows, same columns.
        categorical: Categorical column names within X.
        config: Fold count, seed, device.
        target_column: Label name, used only for error messages.

    Returns:
        Out-of-fold probabilities, one per training row, and the competition
        probabilities averaged over the folds.

    Raises:
        ValueError: If any row was not scored exactly once, which would mean the
            out-of-fold array is not a clean cross-validated prediction.
    """
    oof = np.full(len(X), np.nan, dtype=np.float64)
    test = np.zeros(len(X_competition), dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=config.folds, shuffle=True, random_state=SPLIT_SEED
    )
    te_columns = (
        [c for c in config.te_columns if c in X.columns]
        if spec.target_encodings
        else []
    )
    pair_columns = tuple(
        (a, b) for a, b in TE_PAIR_COLUMNS if a in X.columns and b in X.columns
    )

    for fold, (fit_index, score_index) in enumerate(splitter.split(X, y)):
        X_fit = X.iloc[fit_index]
        y_fit = y.iloc[fit_index]
        X_score = X.iloc[score_index]
        if te_columns:
            # Refitted inside the fold, cross-fitted within it. Stage 2's encodings
            # are cross-fitted once over the whole training set, which lets a
            # scoring row depend on the labels of its own fold-mates; these do not.
            X_fit, X_score, X_test_te = add_in_fold_target_encodings(
                X_fit,
                y_fit,
                X_score,
                X_competition,
                te_columns,
                pair_columns,
                config.seed + fold,
            )
        else:
            X_test_te = X_competition
        estimator = fit_member(
            build_estimator(
                spec.kind,
                apply_device(spec.kind, spec.params, config.device),
                config.seed + fold,
            ),
            spec.kind,
            X_fit,
            y_fit,
            categorical,
            config.seed + fold,
        )
        oof[score_index] = predict_member(estimator, spec.kind, X_score)
        test += predict_member(estimator, spec.kind, X_test_te) / config.folds
        del estimator
        log_step("member_fold", member=spec.name, fold=fold + 1, folds=config.folds)

    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise ValueError(
            f"member {spec.name} left {missing} of {len(oof)} training rows "
            f"unscored. The out-of-fold array would be a partial prediction and "
            f"anything stacked on it is meaningless. Check that folds < rows and "
            f"that {target_column} was passed as the label."
        )
    return oof, test


def to_logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits, clipped away from 0 and 1.

    Args:
        p: Probabilities.

    Returns:
        The log-odds.
    """
    clipped = np.clip(p, LOGIT_FLOOR, 1 - LOGIT_FLOOR)
    return np.log(clipped / (1 - clipped))


def nested_stack_score(
    Z: np.ndarray, y: np.ndarray, C: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Score a stack whose weights never saw the rows they score.

    Args:
        Z: Member logits, rows by members.
        y: Labels.
        C: Logistic regression regularisation.

    Returns:
        One held-out score per row, and the weights fitted on everything.
    """
    out = np.zeros(len(y))
    for fit_index, score_index in StratifiedKFold(
        5, shuffle=True, random_state=STACK_SEED
    ).split(Z, y):
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(Z[fit_index], y[fit_index])
        out[score_index] = model.decision_function(Z[score_index])
    final = LogisticRegression(C=C, max_iter=1000).fit(Z, y)
    return out, final.coef_[0]


def rank_blend_weights(P: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fit non-negative weights that maximise cross-validated AUC of a rank blend.

    Args:
        P: Member predictions, rows by members, already on the rank scale.
        y: Labels.
        seed: Kept for signature symmetry; the optimiser is deterministic.

    Returns:
        Weights summing to one.

    Note:
    AUC reads only the order of the scores, so members are put on the rank scale
    before averaging - a model whose probabilities are compressed would otherwise
    count for less than its ranking earns. Logistic regression on logits cannot
    express that; measured here, the rank blend reached 0.961554 where the
    logistic stack reached 0.961016 on the same five members.

    Nelder-Mead rather than a gradient method, because the objective is a step
    function of the weights and has no gradient to follow.
    """
    from scipy.optimize import minimize

    def objective(w: np.ndarray) -> float:
        weights = np.abs(w) / np.abs(w).sum()
        return -roc_auc_score(y, P @ weights)

    start = np.full(P.shape[1], 1.0 / P.shape[1])
    result = minimize(
        objective, start, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-4}
    )
    weights = np.abs(result.x)
    return weights / weights.sum()


def to_rank(p: np.ndarray) -> np.ndarray:
    """Map a prediction vector onto [0, 1] by rank.

    Args:
        p: Predictions.

    Returns:
        The rank scale, averaged over tied values.
    """
    from scipy.stats import rankdata

    return rankdata(p) / len(p)


def nested_rank_blend(
    P: np.ndarray, y: np.ndarray, splitter: list, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Score a rank blend whose weights never saw the rows they score.

    Args:
        P: Member predictions, rows by members, raw scale.
        y: Labels.
        splitter: The fold list to fit weights out of fold.
        seed: Unused; weights are deterministic.

    Returns:
        One held-out score per row, and the weights fitted on everything.
    """
    ranked = np.column_stack([to_rank(P[:, i]) for i in range(P.shape[1])])
    honest = np.zeros(len(y))
    for fit_index, score_index in splitter:
        honest[score_index] = ranked[score_index] @ rank_blend_weights(
            ranked[fit_index], y[fit_index], seed
        )
    return honest, rank_blend_weights(ranked, y, seed)


def run_gpu_shards(
    groups: list[list[str]], artifacts_dir: Path, log_dir: Path | None = None
) -> None:
    """Train each member group in its own process, one per GPU.

    Args:
        groups: Member-name groups, one per GPU.
        artifacts_dir: Where the workers write their `.npy` predictions.
        log_dir: Where to put one log per worker. Defaults to the artifacts dir.

    Raises:
        RuntimeError: If any worker exits non-zero. Its log is named in the
            message, because a worker that dies in a subprocess otherwise fails
            silently and the stack is built from whichever members survived.

    Note:
    One process per GPU, not threads: a CUDA context is per process, so two
    threads in one process share device 0 and the second one waits. Each worker
    is pinned with `CUDA_VISIBLE_DEVICES`, which makes its single GPU appear as
    device 0 - so the members need no idea they are sharded at all.
    """
    import subprocess
    import sys

    if log_dir is None:
        log_dir = artifacts_dir / "shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups):
        names = ",".join(group)
        log_path = log_dir / f"gpu{index}.log"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(index)
        command = [
            sys.executable,
            "-m",
            "src.pipeline.gpu_shard",
            "--members",
            names,
            "--artifacts",
            str(artifacts_dir),
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"the worker for GPU {index} failed with exit code "
                f"{completed.returncode} while training {names}. Its log is "
                f"{log_path}."
            )
        log_step("gpu_shard", gpu=index, members=len(group), log=str(log_path))


def train_ensemble(
    config: EnsembleConfig,
    train_frame: pd.DataFrame,
    competition_frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    artifacts_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    """Train every member, stack them, and save the artefacts.

    Args:
        config: The ensemble settings.
        train_frame: The training rows, including the label.
        competition_frame: The competition rows, no label.
        features: The feature columns config.yaml asks for. Passed in rather than
            derived from the frame's columns, because the artifacts carry columns
            the model must never see - the id, the raw delays, and
            arrival_delay_status - and "whatever is in the file" is not a
            defensible feature list.
        categorical: Categorical column names.
        artifacts_dir: Where predictions and metadata are written.
        target_column: Label name.

    Returns:
        A summary: each member's score, the nested stack score, the weights, and
        the competition probabilities.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_features = list(features)
    for column in all_features:
        if column not in train_frame.columns:
            raise ValueError(
                f"config lists feature {column!r} but the training frame does not "
                f"have it. Run stage 2, or fix the feature list in config.yaml."
            )
    y = train_frame[target_column].astype("int8")
    summary: dict[str, Any] = {"members": {}, "folds": config.folds}
    logits_train: list[np.ndarray] = []
    logits_test: list[np.ndarray] = []
    names: list[str] = []

    device = resolve_device(config.device)
    groups = plan_gpu_shards(config.members, device)
    if len(groups) > 1:
        # One worker process per GPU. They write the same .npy files the inline
        # path would, and the loop below then reads those back, so the stacking
        # code is identical either way.
        log_step(
            "ensemble_device",
            device=device,
            gpus=len(groups),
            members=len(config.members),
        )
        run_gpu_shards(groups, artifacts_dir)
        for spec in config.members:
            oof = np.load(artifacts_dir / f"oof_{spec.name}.npy")
            test = np.load(artifacts_dir / f"test_{spec.name}.npy")
            summary["members"][spec.name] = {
                "oof_auc": round(float(roc_auc_score(y, oof)), 6),
                "features": len(select_features(all_features, spec)),
            }
            names.append(spec.name)
            logits_train.append(to_logit(oof))
            logits_test.append(to_logit(test))
        log_step("ensemble_shards_done", gpus=len(groups))
    else:
        for spec in config.members:
            oof, test, auc = fit_member_to_artifacts(
                spec,
                train_frame,
                competition_frame,
                all_features,
                categorical,
                config,
                target_column,
                artifacts_dir,
            )
            summary["members"][spec.name] = {
                "oof_auc": round(auc, 6),
                "features": len(select_features(all_features, spec)),
            }
            log_step(
                "member",
                member=spec.name,
                oof_auc=round(auc, 6),
                features=summary["members"][spec.name]["features"],
            )
            names.append(spec.name)
            logits_train.append(to_logit(oof))
            logits_test.append(to_logit(test))

    Z = np.column_stack(logits_train)
    ZT = np.column_stack(logits_test)
    scores, weights = nested_stack_score(Z, y.to_numpy(), config.stack_C)
    nested = float(roc_auc_score(y, scores))
    fold_list = list(
        StratifiedKFold(
            n_splits=config.folds, shuffle=True, random_state=SPLIT_SEED
        ).split(Z, y.to_numpy())
    )
    blend_scores, blend_weights = nested_rank_blend(
        np.column_stack(
            [np.exp(Z[:, i]) / (1 + np.exp(Z[:, i])) for i in range(Z.shape[1])]
        ),
        y.to_numpy(),
        fold_list,
    )
    nested_blend = float(roc_auc_score(y, blend_scores))
    summary["solo_oof_auc"] = {n: v["oof_auc"] for n, v in summary["members"].items()}
    summary["mean_logit_auc"] = round(float(roc_auc_score(y, Z.mean(1))), 6)
    summary["nested_stack_auc"] = round(nested, 6)
    summary["nested_rank_blend_auc"] = round(nested_blend, 6)
    summary["rank_blend_weights"] = {
        n: round(float(w), 6) for n, w in zip(names, blend_weights)
    }
    # The reported combiner is whichever scored higher out of fold, and the
    # submission is built from that one. Deciding this on the out-of-fold score
    # is a comparison between two candidates on the same rows, which is the same
    # kind of choice as picking a model; it is not fitting anything to the labels.
    if nested_blend > nested:
        ranked_test = np.column_stack(
            [to_rank(1 / (1 + np.exp(-ZT[:, i]))) for i in range(ZT.shape[1])]
        )
        summary["combiner"] = "rank"
        summary["competition_probabilities"] = ranked_test @ blend_weights
        log_step("ensemble_combiner", chosen="rank", nested_auc=round(nested_blend, 6))
        save_json(summary, artifacts_dir / "ensemble_summary.json")
        return summary
    summary["combiner"] = "logistic"
    summary["weights"] = {n: round(float(w), 6) for n, w in zip(names, weights)}
    save_json(summary, artifacts_dir / "ensemble_summary.json")
    log_step(
        "ensemble_stack",
        nested_auc=summary["nested_stack_auc"],
        mean_logits=summary["mean_logit_auc"],
        members=len(names),
    )
    # ZT, not Z. Z holds the training rows' logits; the submission is one row per
    # competition row, so the weights have to be applied to ZT. Using Z here
    # produces predictions for the wrong rows entirely.
    summary["competition_probabilities"] = 1 / (1 + np.exp(-(ZT @ weights)))
    return summary
