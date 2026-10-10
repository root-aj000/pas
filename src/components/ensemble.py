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

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.utils.common import log_step, save_json

# The fold split every selection number in this project was measured on. Changing
# it invalidates comparison with docs/experiment_log.md, so it is a constant rather
# than a per-run setting.
SPLIT_SEED = 42
STACK_SEED = 42
LOGIT_FLOOR = 1e-6

# Prefix for the pseudo-label retrain's fold checkpoints, so the second training
# pass keeps its own files instead of colliding with the first's.
#
# Without it both passes write `oof_<member>.fold.npy`, and a retrain resuming
# from a first-pass fold would be scored against labels it never saw. The tag goes
# into the file name AND the fingerprint, so the two cannot be confused even if the
# names were ever made to collide.
PSEUDO_CHECKPOINT_TAG = "pseudo__"


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
        keep_prefixes: When set, ONLY columns starting with one of these are kept,
            and `drop_prefix`/`drop_suffix` are ignored. This exists because the
            drop-based filter cannot express the views that matter most.
            `drop_prefix: ["Age"]` also removes `Age_d0`, `Age_squared`,
            `Age_is_outlier` and `Age_cat_`, but not `age_over_distance` -
            case-sensitive, so an exclusion list for "the 13 ratings and nothing
            else" would be a hundred prefixes and would silently change meaning
            the day someone adds a feature. A keep-list says what it means.
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
    keep_prefixes: tuple[str, ...] = ()
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
        pseudo_label_enabled: Whether to add confident competition predictions as
            pseudo-labelled training rows and retrain. Skipped on the sharded path,
            which no longer holds the frames.
        pseudo_label_high: Upper confidence threshold for pseudo-labeling.
        pseudo_label_low: Lower confidence threshold for pseudo-labeling.
    """

    enabled: bool = False
    members: list[MemberSpec] = field(default_factory=list)
    folds: int = 10
    seed: int = 42
    stack_C: float = 1.0
    te_columns: list[str] = field(default_factory=list)
    combiner: str = "logistic"
    pseudo_label_enabled: bool = False
    pseudo_label_high: float = 0.95
    pseudo_label_low: float = 0.05
    # When true, each member is fitted a second time on all training rows and its
    # weights are written to artifacts/ensemble/models/model_<name>.pkl. Without
    # it a run keeps only predictions, so predicting on any new row means
    # retraining the whole ensemble.
    save_models: bool = False
    # On CPU, train one member per subprocess so peak memory is one member's
    # rather than the parent's frames plus a member's. Ignored on GPU, where
    # `run_gpu_shards` already splits members across processes.
    cpu_one_process_per_member: bool = True
    # Write a checkpoint after every completed fold and resume from it when the
    # member is restarted. A 5-fold member on CPU takes an hour or more, and
    # without this an OOM or a laptop sleep at fold 4 discards all of it.
    resume: bool = True
    device: str = ""
    """config.yaml's `device` key, passed through unchanged.

    An empty default rather than "cpu" or "cuda" on purpose. A real value as the
    default would be a second, silent source: a config missing the key would
    quietly train on one device and report success. Empty fails
    `resolve_device`'s check instead, naming config.yaml as the thing to fix.
    """


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

    `keep_prefixes` wins over `drop_prefix` when it is set, and `drop_suffix`
    applies to whatever survives it. That combination is what a "the 13 ratings,
    nothing else" view needs: `keep_prefixes` cannot exclude the `*_cat_` twins
    on their own, because "Inflight wifi service" matches both the rating column
    and its twin, and a tree reads the twin as a duplicate of the rating rather
    than as extra information.

    A keep-list and a drop-list that disagree have no sensible resolution, so
    `drop_prefix` is ignored rather than combined when `keep_prefixes` is set.
    Picking one silently is how a view ends up being something other than what its
    config says.
    """
    if spec.keep_prefixes:
        keep = [
            f
            for f in all_features
            if f.startswith(spec.keep_prefixes)
            and not (spec.drop_suffix and f.endswith(spec.drop_suffix))
        ]
    else:
        keep = [
            f
            for f in all_features
            if not (spec.drop_prefix and f.startswith(spec.drop_prefix))
            and not (spec.drop_suffix and f.endswith(spec.drop_suffix))
        ]
    if not keep:
        raise ValueError(
            f"member {spec.name} has no features left after filtering. "
            f"keep_prefixes={list(spec.keep_prefixes)} matched nothing in the "
            f"{len(all_features)} configured features."
        )
    return keep


def resolve_device(requested: str) -> str:
    """Return the device to actually train on, given config.yaml's request.

    Args:
        requested: "cpu" or "cuda", straight from config.yaml's `device` key.

    Returns:
        "cuda" if config asked for cuda and a GPU is visible, otherwise "cpu".

    Raises:
        ValueError: If `requested` is neither "cpu" nor "cuda".

    Note:
    `device: cuda` in config means "use the GPU if this machine has one", not a
    promise that it does. Kaggle sessions have two T4s and a laptop has none, and
    one committed config has to be right on both - so cuda falls back to cpu when
    no GPU is visible, which is what lets the same file run unmodified in both
    places instead of failing halfway through a fit.

    The fallback only ever downgrades, never upgrades. `device: cpu` stays cpu on
    a machine that has a GPU, because the setting exists to be obeyed and the only
    question this answers is whether obeying it is possible.

    Anything other than cpu or cuda is rejected rather than guessed at, so a typo
    in that one line is a message naming config.yaml instead of an estimator error
    thirty minutes into a run.
    """
    if requested not in ("cpu", "cuda"):
        raise ValueError(
            f"device must be 'cpu' or 'cuda', got {requested!r}. config.yaml's "
            f"`device` key is the only place this is set."
        )
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - no torch means no GPU, which means cpu
        return "cpu"


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
    All three labelled splits are folded into training - train, validation and
    the frozen test split. Stage 2 splits 70/15/15 for the single-model path,
    where the validation file gave stages 3 and 4 something to early-stop and
    select on and the test file gave stage 4 something to report a final number
    against. The ensemble cross-validates internally and reads neither, so both
    were pure loss: reading only train.csv trained every member on 489,743 of the
    699,635 labelled rows, and a first fix that added validation still left
    104,946 behind at 594,689.

    Safe to concatenate: stage 2 encoded all three in one pass, with identical
    columns and the same train-fitted statistics, and its label-based columns were
    cross-fitted, so no row's encoding used its own label. The fold structure that
    matters is created by the caller.

    Nothing is lost as an evaluation set by this: `nested_stack_auc` scores every
    row from a combiner fitted on other rows, which is the same guarantee the
    frozen split was there to provide.

    This lives here rather than in the stage because the per-GPU workers have to
    build exactly the same frames, and two copies of a frame loader is how a
    worker ends up training on different rows from the stack.
    """
    from src.utils.common import apply_categorical_encoding, load_json

    train_path = artifacts_dir / "train.csv"
    validation_path = artifacts_dir / "validation.csv"
    test_path = artifacts_dir / "test.csv"
    competition_path = artifacts_dir / "competition_test.csv"
    for path in (train_path, validation_path, test_path, competition_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run stage 2 first: python -m "
                "src.pipeline.stage_02_data_cleaning_encoding"
            )

    contract = load_json(artifacts_dir / "features.json")
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    categorical_columns = list(contract.get("categorical_columns", []))
    category_maps = contract.get("category_maps") or {}
    # The contract, not the caller's list, decides what gets typed - it is what
    # stage 2 actually pinned. The caller's list is cross-checked against it so a
    # config edited after stage 2 last ran is caught here rather than by an
    # estimator three hours later.
    stale = set(categorical) - set(categorical_columns)
    if stale:
        raise ValueError(
            f"config.yaml names {sorted(stale)} as categorical but stage 2's "
            f"features.json does not list them. The artifacts predate the config; "
            f"re-run stage 2."
        )

    def read(path: Path, label: str) -> pd.DataFrame:
        return apply_categorical_encoding(
            pd.read_csv(path, low_memory=False),
            categorical_columns,
            encoding,
            label,
            category_maps,
        )

    train_only = read(train_path, "ensemble_train")
    validation = read(validation_path, "ensemble_train")
    test_split = read(test_path, "ensemble_train")
    train_frame = pd.concat([train_only, validation, test_split], ignore_index=True)
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
        test_split_rows=len(test_split),
        competition_rows=len(competition_frame),
    )
    return train_frame, competition_frame


def read_categorical_columns(artifacts_dir: Path, features: list[str]) -> list[str]:
    """Return the columns stage 2 pinned as categorical, in feature order.

    Args:
        artifacts_dir: The `data_cleaning_encoding` folder stage 2 wrote.
        features: The configured feature list, which sets the order.

    Returns:
        The intersection of the contract's categorical columns with `features`.

    Note:
    The contract, not config.yaml, is the authority here - and that is the whole
    function. Stage 2 pins every column it types as `category`: the four short
    categoricals plus the 20 `_cat_` twins, 24 in total, all recorded in
    features.json.

    Reading config.yaml's `categorical_columns` instead gave members a 4-item list
    against a frame carrying 24 `category` columns. XGBoost and LightGBM accepted
    it because `enable_categorical` infers the set from the frame; CatBoost
    rejected it with `column 'Online boarding_cat_' has dtype 'category' but is
    not in cat_features list`, which took out one member and, because the members
    run sequentially, most of a run.

    A member that drops columns - a stage 6 view - narrows this list by
    intersection in `fit_member_to_artifacts`, so a view never asks CatBoost to
    categorise a column it cannot see.
    """
    from src.utils.common import load_json

    contract = load_json(artifacts_dir / "features.json")
    recorded = set(contract.get("categorical_columns") or [])
    return [column for column in features if column in recorded]


def load_competition_frame(
    artifacts_dir: Path, features: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Load only the competition rows, for writing a submission.

    Args:
        artifacts_dir: The `data_cleaning_encoding` folder stage 2 wrote.
        features: The configured feature list.
        categorical: Categorical column names within it.

    Returns:
        The competition rows, encoded exactly as `load_ensemble_frames` encodes
        them.

    Note:
    Separate from `load_ensemble_frames` so a caller that is about to hand the
    frames to a subprocess trainer does not have to hold them just to get the
    competition rows back afterwards. The three labelled splits are 699,635 rows;
    the competition file is a separate file, so skipping it is the whole saving.
    """
    from src.utils.common import apply_categorical_encoding, load_json

    competition_path = artifacts_dir / "competition_test.csv"
    if not competition_path.exists():
        raise FileNotFoundError(
            f"{competition_path} not found. Run stage 2 first: python -m "
            "src.pipeline.stage_02_data_cleaning_encoding"
        )
    contract = load_json(artifacts_dir / "features.json")
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    frame = apply_categorical_encoding(
        pd.read_csv(competition_path, low_memory=False),
        list(contract.get("categorical_columns", [])),
        encoding,
        "ensemble_competition",
        contract.get("category_maps") or {},
    )
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(
            f"the competition frame is missing {len(missing)} configured columns: "
            f"{missing[:5]}. Run stage 2 first."
        )
    return frame


def train_ensemble_drops_frames(config: EnsembleConfig) -> bool:
    """Whether `train_ensemble` frees the frames before training any member.

    Args:
        config: The ensemble settings.

    Returns:
        True when every member trains in a subprocess, so this process never needs
        the frames while they train.

    Note:
    Exists so a caller can decide not to load the training frames at all. The
    reason the frames were being held was `train_ensemble`'s own `del`, and `del`
    inside a function frees nothing while the caller still has the object bound to
    one of its own names - the ~2 GB stayed resident through every fold. The
    caller has to be the one that never takes them.

    True on cuda for every member count: with one GPU `train_ensemble` runs the
    members inline and does need them, but its worker path and its single-GPU
    inline path differ in nothing the caller can act on, and the cost of this
    answer being wrong is a re-read of three CSVs.
    """
    if resolve_device(config.device) == "cuda":
        return True
    return bool(config.cpu_one_process_per_member)


def config_payload(config: EnsembleConfig) -> dict[str, Any]:
    """Return `config` as plain JSON-serialisable data.

    Args:
        config: The ensemble settings.

    Returns:
        A dict holding every member spec and every run-level setting.

    Note:
    A subprocess has to train the member the caller asked for, not the member
    config.yaml happens to name. Reading config.yaml in the child made
    `train_ensemble` a function of its arguments in name only: a caller passing a
    two-member synthetic config got the nine members from the file, or an error
    naming them, and the difference between the two was invisible from outside.
    """
    return {
        "folds": config.folds,
        "seed": config.seed,
        "stack_C": config.stack_C,
        "te_columns": list(config.te_columns),
        "combiner": config.combiner,
        "pseudo_label_enabled": config.pseudo_label_enabled,
        "pseudo_label_high": config.pseudo_label_high,
        "pseudo_label_low": config.pseudo_label_low,
        "save_models": config.save_models,
        "cpu_one_process_per_member": config.cpu_one_process_per_member,
        "resume": config.resume,
        "device": config.device,
        "members": [
            {
                "name": m.name,
                "kind": m.kind,
                "params": m.params,
                "drop_prefix": list(m.drop_prefix),
                "drop_suffix": list(m.drop_suffix),
                "keep_prefixes": list(m.keep_prefixes),
                "target_encodings": m.target_encodings,
            }
            for m in config.members
        ],
    }


def config_from_payload(payload: dict[str, Any]) -> EnsembleConfig:
    """Rebuild an `EnsembleConfig` from `config_payload`.

    Args:
        payload: What `config_payload` wrote.

    Returns:
        The ensemble settings, member for member.
    """
    return EnsembleConfig(
        folds=int(payload["folds"]),
        seed=int(payload["seed"]),
        stack_C=float(payload["stack_C"]),
        te_columns=[str(c) for c in payload.get("te_columns", [])],
        combiner=str(payload.get("combiner", "logistic")),
        pseudo_label_enabled=bool(payload.get("pseudo_label_enabled", False)),
        pseudo_label_high=float(payload.get("pseudo_label_high", 0.95)),
        pseudo_label_low=float(payload.get("pseudo_label_low", 0.05)),
        save_models=bool(payload.get("save_models", False)),
        cpu_one_process_per_member=bool(
            payload.get("cpu_one_process_per_member", True)
        ),
        resume=bool(payload.get("resume", True)),
        device=str(payload.get("device", "")),
        members=[
            MemberSpec(
                name=str(m["name"]),
                kind=str(m["kind"]),
                params=dict(m.get("params") or {}),
                drop_prefix=tuple(m.get("drop_prefix") or ()),
                drop_suffix=tuple(m.get("drop_suffix") or ()),
                keep_prefixes=tuple(m.get("keep_prefixes") or ()),
                target_encodings=bool(m.get("target_encodings", True)),
            )
            for m in payload.get("members", [])
        ],
    )


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
) -> tuple[np.ndarray, np.ndarray, float]:
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
        artifacts_dir,
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / f"oof_{spec.name}.npy", oof)
    np.save(artifacts_dir / f"test_{spec.name}.npy", test)
    # The member is finished, so its fold checkpoints are stale. Leaving them
    # would let a later run resume from them and skip refitting a member whose
    # config had since changed.
    _clear_fold_checkpoint(spec.name, artifacts_dir)

    if config.save_models:
        save_member_weights(
            spec,
            train_frame,
            competition_frame,
            features,
            use_cat,
            config,
            target_column,
            artifacts_dir,
        )

    return oof, test, float(roc_auc_score(train_frame[target_column], oof))


def save_member_weights(
    spec: MemberSpec,
    train_frame: pd.DataFrame,
    competition_frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    config: EnsembleConfig,
    target_column: str,
    artifacts_dir: Path,
) -> Path:
    """Fit one member on ALL training rows and write its weights to disk.

    Args:
        spec: Which member.
        train_frame: Training rows, label included.
        competition_frame: Competition rows. Used only for the column check, so a
            member cannot be fitted against features the competition frame lacks.
        features: The columns this member sees.
        categorical: Categorical column names within them.
        config: Seed, device, and the stack settings.
        target_column: Label name.
        artifacts_dir: Where `model_<name>.pkl` is written.

    Returns:
        The path written.

    Note:
    This is a SECOND fit, on every row, after the cross-validated one that
    produced oof_<name>.npy. That is deliberate and not redundant. The
    cross-validated fit exists to produce an honest score: each row is predicted
    by a model that never saw it. Those fold models are therefore collectively
    weaker than any single one of them, and none of them is trained on all the
    data. This fit is the deployable artifact - it is what you load to predict on
    rows that were never in train.csv, which the .npy prediction files cannot do.

    One model per member, not ten. Saving all the fold models would multiply the
    disk cost by `folds` and still leave you needing an ensemble at inference,
    which is the thing this project is trying to avoid.

    `competition_frame` is here to be checked, not fitted on. A saved model is a
    promise that this member can score a competition row, and that promise is
    unverifiable if the frame it will be asked about is missing a column. The
    check was documented but not performed, so a member could be saved that fails
    on every row it was supposed to be for - and the failure would surface later,
    in a different process, as a KeyError from inside an estimator.

    Size matters on Kaggle, where /kaggle/working is capped. The size is logged
    per member so a run that fills the disk says which member did it.
    """
    import cloudpickle

    missing = [c for c in features if c not in competition_frame.columns]
    if missing:
        raise ValueError(
            f"member {spec.name} wants {len(missing)} columns the competition "
            f"frame does not have: {missing[:5]}. Re-run stage 2 before saving "
            f"member weights - a model fitted on them could not score a "
            f"competition row."
        )

    estimator = build_estimator(
        spec.kind,
        apply_device(spec.kind, spec.params, config.device),
        config.seed,
        config.device,
    )
    fitted = fit_member(
        estimator,
        spec.kind,
        train_frame[features],
        train_frame[target_column].astype("int8"),
        categorical,
        config.seed,
        spec.params,
    )

    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"model_{spec.name}.pkl"
    # cloudpickle, not joblib, for the same reason stage 3 uses it: the tuned
    # RealMLP recipe stores wd_sched, p_drop_sched and ls_eps_sched as compiled
    # lambdas, and plain pickle cannot serialise those. cloudpickle stores them by
    # value, so the file loads back without the defining module existing.
    #
    # The comment was right and the call was wrong. `joblib.dump` pickles with the
    # standard library, so the RealMLP member died on its way to disk with
    #
    #   PicklingError: Can't pickle <function get_schedule.<locals>.<lambda>>:
    #   it's not found as pytabkit.models.training.scheduling...
    #
    # after training all five folds successfully - 39 minutes of fitting discarded
    # at the last step, and because the member's predictions had already been
    # written the run still counted it as failed. `cloudpickle.dump` wants a file
    # object rather than a Path, so the handle is opened here.
    with path.open("wb") as handle:
        cloudpickle.dump(fitted, handle)

    size_mb = path.stat().st_size / 1024 / 1024
    log_step(
        "member_weights",
        member=spec.name,
        wrote=str(path),
        size_mb=round(size_mb, 1),
        features=len(features),
    )
    return path


def apply_device(kind: str, params: dict[str, Any], device: str) -> dict[str, Any]:
    """Return member params with the run's device written into them.

    Args:
        kind: Family name.
        params: The member's configured params.
        device: "cpu" or "cuda". Must already be resolved by `resolve_device`.

    Returns:
        A copy of params carrying the right key for that family. On "cpu" the
        params are returned untouched, so a member that names its own device in
        config.yaml can still opt into the GPU on a CPU-configured run.

    Raises:
        ValueError: If `device` is neither "cpu" nor "cuda". This guard exists
            because the alternative was silent: `ensemble.device: auto` reached
            this function unresolved, so no family got a device param, none of
            them defaults to the GPU, and a Kaggle run trained all nine members on
            the processor while reporting no error and looking like it had
            sharded across two cards.

    Note:
    Every family spells it differently - XGBoost wants `device="cuda"`,
    LightGBM wants `device="gpu"`, CatBoost wants `task_type="GPU"`, and
    pytabkit and TabNet both want `device`. None of them defaults to the GPU.
    Without this, `ensemble.device: cuda` in config.yaml was a key nothing read:
    the nine members were built with no device at all and a Kaggle run trained
    the whole ensemble on CPU.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(
            f"device must be 'cpu' or 'cuda', got {device!r}. If it came from "
            f"config.yaml it was not resolved - call resolve_device() first. An "
            f"unresolved 'auto' trains everything on CPU without complaining."
        )
    out = dict(params)
    if device != "cuda":
        return out
    # Every member in config.yaml asks for n_jobs=8. On a laptop that is right.
    # On a 2x T4 Kaggle session there are four cores for *both* workers, so two
    # workers each grabbing eight threads is 16 threads on four cores - the
    # oversubscription shows up as a maxed-out CPU next to a mostly idle GPU,
    # because every histogram build and frame conversion is waiting on a thread
    # that is queued behind seven others. Split the cores instead of doubling
    # them up. This is the ceiling of a naive split, not a scheduler: if the
    # member counts per shard are unbalanced this still oversubscribes.
    workers = max(int(os.environ.get("PAS_SHARD_COUNT", "1")), 1)
    if workers > 1 and int(out.get("n_jobs", 0)) > 1:
        out["n_jobs"] = max(int(out["n_jobs"]) // workers, 1)
    if kind == "xgboost":
        out["device"] = "cuda"
    elif kind == "lightgbm":
        out["device"] = "gpu"
    elif kind == "catboost":
        out["task_type"] = "GPU"
    else:
        out["device"] = device
    return out


def build_estimator(
    kind: str, params: dict[str, Any], seed: int, device: str = "cpu"
):
    """Create one unfitted estimator.

    Args:
        kind: Family name.
        params: Keyword arguments.
        seed: Random seed.
        device: The device already resolved by `resolve_device`. TabNet reads this
            rather than `params["device"]`, because `apply_device` returns params
            untouched on the cpu branch - so a config saying `cuda` on a machine
            with no GPU would otherwise hand TabNet `device_name="cuda"` and crash
            in its constructor.

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

        # The architecture comes from config.yaml. `device` is dropped rather than
        # forwarded (TabNet spells that device_name), and so are the three fit-time
        # arguments - patience, max_epochs and batch_size belong to fit(), not to
        # the constructor, and passing them here raises TypeError.
        options = {
            key: value
            for key, value in params.items()
            if key not in ("device", "patience", "max_epochs", "batch_size")
        }
        return TabNetClassifier(
            **options,
            verbose=0,
            seed=seed + 1,
            device_name=device,
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
    params: dict[str, Any] | None = None,
):
    """Fit one estimator on one fold, handling each family's quirks.

    Args:
        estimator: The unfitted estimator.
        kind: Its family.
        X: Fit rows.
        y: Fit labels.
        categorical: Categorical column names.
        seed: Random seed for the inner split.
        params: The member's configured params. TabNet reads its three fit-time
            arguments from here, because they are not constructor arguments and so
            cannot be recovered from the estimator.

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

        options = params or {}
        Xn, positions, sizes = _tabnet_arrays(X, categorical)
        yn = y.to_numpy()
        inner_fit, inner_val = train_test_split(
            np.arange(len(yn)), test_size=0.1, random_state=seed, stratify=yn
        )
        fitted = TabNetClassifier(
            **{
                key: value
                for key, value in estimator.get_params().items()
                if key not in ("cat_idxs", "cat_dims", "cat_emb_dim")
            },
            cat_idxs=positions,
            cat_dims=sizes,
            cat_emb_dim=1,
        )
        fitted.fit(
            Xn[inner_fit],
            yn[inner_fit],
            eval_set=[(Xn[inner_val], yn[inner_val])],
            patience=options.get("patience", 5),
            max_epochs=options.get("max_epochs", 40),
            batch_size=options.get("batch_size", 4096),
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


def _fold_checkpoint_paths(name: str, artifacts_dir: Path) -> tuple[Path, Path]:
    """Return the two checkpoint paths for one member."""
    folder = artifacts_dir / "checkpoints"
    return folder / f"oof_{name}.fold.npy", folder / f"test_{name}.fold.npy"


def member_fingerprint(
    spec: MemberSpec,
    config: EnsembleConfig,
    features: list[str],
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a stable id for everything about a member that changes its output.

    Args:
        spec: The member.
        config: The run's fold count, seed, device and target-encoding columns.
        features: The columns this member sees, in order.
        extra: Anything else that changes the result and is not in `spec` or
            `config`. The pseudo-label retrain uses this for the number of rows it
            added and the thresholds that selected them.

    Returns:
        A hex digest. Equal digests mean resuming from a checkpoint is safe.

    Note:
    A checkpoint is only reusable if the predictions in it would have been the
    same ones this run would produce. That is a function of the member's family,
    every hyperparameter, its column filters, whether it gets the in-fold
    encodings, the columns it sees, the fold count, the seed and the device.

    Keying only on the fold count - which is what this did before - meant that
    editing `n_estimators` in config.yaml, losing power at fold 3 and re-running
    resumed the *old* parameters' folds and reported a complete member. Nothing
    in the output said so, and the AUC in ensemble_summary.json belonged to a
    model that no longer existed. A hash is the only way to make "the config
    changed" detectable without asking the user to remember to delete a folder.
    """
    payload = {
        "name": spec.name,
        "kind": spec.kind,
        "params": spec.params,
        "drop_prefix": list(spec.drop_prefix),
        "drop_suffix": list(spec.drop_suffix),
        "keep_prefixes": list(spec.keep_prefixes),
        "target_encodings": spec.target_encodings,
        "features": features,
        "folds": config.folds,
        "seed": config.seed,
        "device": config.device,
        "te_columns": config.te_columns,
        "split_seed": SPLIT_SEED,
        # Sorted so the digest does not depend on insertion order, and repr'd
        # because a value here is usually a count or a threshold.
        "extra": {key: str(extra[key]) for key in sorted(extra)} if extra else {},
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_fold_checkpoint(
    name: str,
    oof: np.ndarray,
    test: np.ndarray,
    folds: int,
    artifacts_dir: Path,
    fingerprint: str = "",
) -> int | None:
    """Restore a member's partial folds in place. Return the first missing fold.

    Args:
        name: Member name.
        oof, test: Preallocated destination arrays, filled in place.
        folds: How many folds this member has.
        artifacts_dir: Where checkpoints live.
        fingerprint: What the member is now configured to be. A checkpoint
            written for a different member is rejected.

    Returns:
        The index of the first fold still to compute, `folds` when the checkpoint
        already holds every fold, or None when no usable checkpoint exists.

    Note:
    The checkpoint stores `folds_completed` plus the summed competition
    predictions, NOT the running mean. Dividing a stored mean by a larger fold
    count on the next run would be wrong, so the sum is what gets written.

    A checkpoint is rejected when the fold count, the row counts, or the member's
    own configuration differ from this run's. Each of those means the stored
    partial sums belong to a different computation, and resuming anyway would
    silently corrupt the result - so refitting from scratch is the only safe
    answer, and it is strictly better than producing a plausible wrong number.

    `folds` is a legal return value and is not a rejection: a member that
    finished all its folds and then died before writing its final `.npy` files is
    exactly the case checkpointing exists for, and refitting an hour of work
    because of a lost rename would defeat the point.
    """
    oof_path, test_path = _fold_checkpoint_paths(name, artifacts_dir)
    meta_path = artifacts_dir / "checkpoints" / f"{name}.fold.json"
    if not (oof_path.exists() and test_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if int(meta["folds"]) != int(folds):
            log_step(
                "checkpoint_rejected",
                member=name,
                reason="fold count changed",
                saved=meta["folds"],
                wanted=folds,
            )
            return None
        if fingerprint and str(meta.get("fingerprint", "")) != fingerprint:
            # Nothing is deleted here. Leaving it in place means the rejection is
            # visible in the log on the next run too, and the user can see which
            # member it was rather than inferring it from a missing number.
            log_step(
                "checkpoint_rejected",
                member=name,
                reason="member config changed since it was written",
            )
            return None
        completed = int(meta["folds_completed"])
        saved_oof = np.load(oof_path)
        saved_test = np.load(test_path)
    except Exception as error:  # noqa: BLE001 - a bad checkpoint must not kill the run
        log_step("checkpoint_rejected", member=name, reason=type(error).__name__)
        return None

    if len(saved_oof) != len(oof) or len(saved_test) != len(test):
        log_step("checkpoint_rejected", member=name, reason="row count changed")
        return None
    if not 0 <= completed <= folds:
        return None

    # Only the rows the completed folds actually scored are trustworthy; the rest
    # are the NaN sentinel the loop preallocates.
    scored = ~np.isnan(saved_oof[: len(oof)])
    oof[:] = np.where(scored, saved_oof[: len(oof)], np.nan)
    test[:] = saved_test[: len(test)]
    log_step("checkpoint_resumed", member=name, folds_completed=completed, folds=folds)
    return completed


def _save_fold_checkpoint(
    name: str,
    oof: np.ndarray,
    test: np.ndarray,
    folds: int,
    folds_completed: int,
    artifacts_dir: Path,
    fingerprint: str = "",
) -> None:
    """Persist partial predictions so an interrupted member can resume.

    Note:
    The arrays are written before the metadata, and the metadata names the fold
    count they hold. A crash between the two leaves a checkpoint describing
    FEWER folds than its arrays contain, which costs at most one fold of refitting;
    the reverse order would let a torn `oof_*.npy` be believed, which costs
    correctness. So the safe ordering is the one used here.
    """
    oof_path, test_path = _fold_checkpoint_paths(name, artifacts_dir)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, oof)
    np.save(test_path, test)
    (oof_path.parent / f"{name}.fold.json").write_text(
        json.dumps(
            {
                "folds": int(folds),
                "folds_completed": int(folds_completed),
                "fingerprint": fingerprint,
            }
        )
    )


def _clear_pseudo_checkpoints(config: EnsembleConfig, artifacts_dir: Path) -> None:
    """Remove every pseudo-label retrain checkpoint.

    Args:
        config: Supplies the member names.
        artifacts_dir: Where the checkpoints live.

    Note:
    Called once the retrain has been decided either way. Until then they are
    resumable; afterwards they are spent folds. A later run that reproduces the
    same selection will reuse them and reach the same conclusion in seconds, which
    is why they are cleared here rather than on the next run.
    """
    for spec in config.members:
        _clear_fold_checkpoint(f"{PSEUDO_CHECKPOINT_TAG}{spec.name}", artifacts_dir)


def _clear_fold_checkpoint(name: str, artifacts_dir: Path) -> None:
    """Remove a member's checkpoint once it has completed."""
    oof_path, test_path = _fold_checkpoint_paths(name, artifacts_dir)
    for path in (oof_path, test_path, oof_path.parent / f"{name}.fold.json"):
        path.unlink(missing_ok=True)


def train_member(
    spec: MemberSpec,
    X: pd.DataFrame,
    y: pd.Series,
    X_competition: pd.DataFrame,
    categorical: list[str],
    config: EnsembleConfig,
    target_column: str,
    artifacts_dir: Path | None = None,
    checkpoint_tag: str = "",
    fingerprint_extra: dict[str, Any] | None = None,
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
        artifacts_dir: Where fold checkpoints live. None disables checkpointing
            entirely, so `train_member` still works as a pure function.
        checkpoint_tag: Distinguishes this member's checkpoints from another's when
            the same member is trained more than once in a run - the pseudo-label
            retrain does exactly that. Appended to the file names, so the two
            training passes cannot read or overwrite each other's state.
        fingerprint_extra: Extra facts folded into the fingerprint, for anything
            outside `spec`/`config` that changes the result. The retrain passes the
            pseudo-label selection, so a checkpoint is only reused when the same
            rows would have been added.

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

    start_fold = 0
    # What this member is now configured to be. Checkpoints written for anything
    # else are refused, so editing a member's params and re-running cannot
    # silently blend old folds with new ones.
    # The tag goes into the file names, and into the fingerprint so a first-pass
    # checkpoint is never handed to the retrain even if both were somehow under
    # the same key.
    checkpoint_key = f"{checkpoint_tag}{spec.name}" if checkpoint_tag else spec.name
    fingerprint = member_fingerprint(spec, config, list(X.columns), fingerprint_extra)
    if checkpoint_tag:
        fingerprint = hashlib.sha256(f"{fingerprint}:{checkpoint_tag}".encode()).hexdigest()
    if config.resume and artifacts_dir is not None:
        resumed = _load_fold_checkpoint(
            checkpoint_key, oof, test, config.folds, artifacts_dir, fingerprint
        )
        if resumed is not None:
            start_fold = resumed

    for fold, (fit_index, score_index) in enumerate(splitter.split(X, y)):
        if fold < start_fold:
            continue
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
                config.device,
            ),
            spec.kind,
            X_fit,
            y_fit,
            categorical,
            config.seed + fold,
            spec.params,
        )
        oof[score_index] = predict_member(estimator, spec.kind, X_score)
        test += predict_member(estimator, spec.kind, X_test_te) / config.folds
        del estimator
        log_step("member_fold", member=spec.name, fold=fold + 1, folds=config.folds)
        if config.resume and artifacts_dir is not None:
            _save_fold_checkpoint(
                checkpoint_key, oof, test, config.folds, fold + 1, artifacts_dir,
                fingerprint,
            )

    # The loop above skips every fold a checkpoint already held, so reaching here
    # with a complete `oof` means the member needed no fitting at all. That is the
    # normal path for a member that finished and lost only its final file write,
    # and re-running the fits would throw away an hour to recompute an identical
    # answer.
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


def newton_logistic_fit(
    Z: np.ndarray, y: np.ndarray, C: float, max_iter: int = 50, tol: float = 1e-8
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit L2-regularised logistic regression by Newton's method.

    Args:
        Z: Member logits, rows by members.
        y: Labels, 0 or 1.
        C: Regularisation. The objective is mean log loss + ||w||^2/(2*C*n), so
            larger C means weaker regularisation - the same sense as
            scikit-learn's parameter.
        max_iter: Iteration cap. Newton's method converges in a handful of steps
            on a well-conditioned convex problem; this only bounds a pathological
            case.
        tol: Convergence threshold on the maximum absolute gradient.

    Returns:
        The weights, the unpenalised bias, and a record of the fit.

    Note:
    Replaces `sklearn.linear_model.LogisticRegression` for the stacker, and the
    reason is that "converged" stops meaning what it should. L-BFGS returns
    whatever it had when `max_iter` ran out, with no statement about the gradient -
    so a stack built on four members and a stack built on a hundred and thirty-six
    differ in whether the solver finished, and nothing in the output says so.
    Measured on this problem, L-BFGS emits `ConvergenceWarning` on the rating
    columns while still reaching a usable answer, which is exactly the situation
    where you cannot tell a converged fit from a truncated one.

    Newton's method computes the exact Hessian and solves for the step directly, so
    convergence is a checkable fact rather than a hope: the gradient is asserted to
    be below `tol`, and if it is not the fit raises. That makes a failure loud
    instead of silent, which is the same reason this project checks that every
    training row was scored.

    The bias is left unpenalised. Penalising it is a modelling choice, not a
    numerical necessity, and shifting the regularisation between weights and bias
    changes the answer; `1/(C*n)` is applied to the weights only, with a zero in the
    bias slot.

    float64 throughout, and the Hessian is formed as `X^T diag(p(1-p)) X` with
    `p(1-p)` clamped away from zero. With 699,635 rows and 136 members that is a
    136x136 solve per iteration - about four of them - so cost is irrelevant
    next to the training it sits on top of.

    Lifted in substance from the reference solution in `sub/download/newton_stack.py`,
    which is where the 0.9621 leaderboard number comes from. Restructured to take
    numpy rather than assuming a CUDA device, to return numpy rather than torch
    tensors, and to raise rather than return a flag on non-convergence.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if Z.shape[0] != labels.shape[0]:
        raise ValueError(
            f"newton_logistic_fit got {Z.shape[0]} rows but {labels.shape[0]} labels."
        )
    n_rows, n_columns = Z.shape
    design = np.column_stack([Z, np.ones(n_rows)])

    weights = np.zeros(n_columns + 1, dtype=np.float64)
    # Start from a constant model at the base rate, with small weights. Newton
    # needs a full-rank Hessian; starting from all-zero weights leaves p(1-p)=0.25
    # everywhere, which is well conditioned, and the base rate puts the linear
    # predictor where the data actually is.
    weights[-1] = float(np.log(max(labels.mean(), 1e-6) / max(1 - labels.mean(), 1e-6)))
    weights[:n_columns] = 1.0 / max(n_columns, 1)

    penalty = np.full(n_columns + 1, 1.0 / (float(C) * n_rows), dtype=np.float64)
    penalty[-1] = 0.0

    def objective(at: np.ndarray) -> float:
        z = design @ at
        # log(1+exp(z)) computed stably, and log1p(-y) folded in as -y*z
        return float(
            (np.logaddexp(0.0, z) - labels * z).mean()
            + 0.5 * (penalty * at**2).sum()
        )

    iterations = 0
    for iterations in range(1, max_iter + 1):
        z = design @ weights
        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        gradient = design.T @ (probability - labels) / n_rows + penalty * weights
        gradient_inf = float(np.abs(gradient).max())
        if gradient_inf < tol:
            break
        variance = np.clip(probability * (1.0 - probability), 1e-12, None)
        hessian = (design.T * variance) @ design / n_rows + np.diag(penalty)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"newton_logistic_fit: the Hessian is singular at C={C}. That "
                f"means two member logits are identical, so the stack cannot "
                f"separate them. Drop a duplicated member."
            ) from error
        if not np.all(np.isfinite(direction)):
            raise ValueError(
                f"newton_logistic_fit: the Newton direction is not finite at C={C}."
            )

        # Backtracking line search. Newton on a convex problem is globally
        # convergent without one, but a step this size on 699,635 rows can
        # overshoot if the Hessian is ill-conditioned at small C.
        before = objective(weights)
        slope = float(gradient @ direction)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError(
                f"newton_logistic_fit: no descent direction at C={C} "
                f"(gradient.direction = {slope:.3g})."
            )
        step = 1.0
        accepted = False
        for _ in range(30):
            candidate = weights - step * direction
            if objective(candidate) <= before - 1e-4 * step * slope + 1e-14:
                weights = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ValueError(
                f"newton_logistic_fit: the line search failed to improve the "
                f"objective at C={C}. Halving 30 times is not a numerical detail."
            )

    z = design @ weights
    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    final_gradient = float(
        np.abs(design.T @ (probability - labels) / n_rows + penalty * weights).max()
    )
    if final_gradient >= tol:
        raise ValueError(
            f"newton_logistic_fit did not converge at C={C} in {max_iter} "
            f"iterations: max gradient {final_gradient:.3g} >= {tol:.3g}. "
            f"Raise max_iter, or pick a larger C."
        )
    record = {
        "C": float(C),
        "iterations": iterations,
        "gradient_inf": final_gradient,
        "loss": objective(weights),
        "rows": int(n_rows),
        "columns": int(n_columns),
        "converged": True,
    }
    return weights[:-1], float(weights[-1]), record


def newton_stack_score(
    Z: np.ndarray, y: np.ndarray, C: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Score a stack whose weights never saw the rows they score.

    Args:
        Z: Member logits, rows by members.
        y: Labels.
        C: Logistic regression regularisation.

    Returns:
        One held-out score per row, the weights fitted on everything, and a record
        of the final fit.

    Note:
    The nested structure is unchanged from the scikit-learn version it replaces -
    same 5 folds, same `STACK_SEED`, same "weights never saw the rows they score".
    Only the solver differs. That is deliberate: the honest-ness of this number
    comes from the fold structure, and changing the solver does not change it.

    `decision_function` was the score before. This returns `Z @ w + b`, the same
    quantity, since AUC is invariant to the affine map that turns a decision value
    into a probability.
    """
    out = np.zeros(len(y))
    for fit_index, score_index in StratifiedKFold(
        5, shuffle=True, random_state=STACK_SEED
    ).split(Z, y):
        weights, bias, _ = newton_logistic_fit(Z[fit_index], y[fit_index], C)
        out[score_index] = Z[score_index] @ weights + bias
    weights, bias, record = newton_logistic_fit(Z, y, C)
    del bias
    return out, weights, record


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
    started: list[tuple[int, str, Any, Any, Path]] = []
    for index, group in enumerate(groups):
        names = ",".join(group)
        log_path = log_dir / f"gpu{index}.log"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(index)
        # So `apply_device` splits `n_jobs` between the workers instead of letting
        # each one claim every core.
        environment["PAS_SHARD_COUNT"] = str(len(groups))
        command = [
            sys.executable,
            "-m",
            "src.pipeline.gpu_shard",
            "--members",
            names,
            "--artifacts",
            str(artifacts_dir),
        ]
        # Started, not waited on. `subprocess.run` blocks, and blocking here made
        # the workers strictly sequential: GPU 0 trained every one of its members
        # while GPU 1 sat at 0% until the first worker exited, so a two-GPU
        # session delivered one GPU's throughput and looked like the second card
        # was broken.
        handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
        except Exception:
            handle.close()
            raise
        started.append((index, names, process, handle, log_path))
        log_step("gpu_shard", gpu=index, members=len(group), log=str(log_path))

    failures: list[str] = []
    for index, names, process, handle, log_path in started:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            failures.append(
                f"GPU {index} (members {names}) exited {return_code}; log {log_path}"
            )
    if failures:
        raise RuntimeError(
            "ensemble workers failed: " + "; ".join(failures) + ". The surviving "
            "workers' .npy files are on disk, but they must not be stacked against "
            "a member that is missing."
        )


def train_ensemble(
    config: EnsembleConfig,
    train_frame: pd.DataFrame | None,
    competition_frame: pd.DataFrame | None,
    features: list[str],
    categorical: list[str],
    artifacts_dir: Path,
    target_column: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Train every member, stack them, and save the artefacts.

    Args:
        config: The ensemble settings.
        train_frame: The training rows, including the label. May be None, in
            which case they are read from `data_dir` - see the note below.
        competition_frame: The competition rows, no label. Also optional, for the
            same reason.
        features: The feature columns config.yaml asks for. Passed in rather than
            derived from the frame's columns, because the artifacts carry columns
            the model must never see - the id, the raw delays, and
            arrival_delay_status - and "whatever is in the file" is not a
            defensible feature list.
        categorical: Categorical column names.
        artifacts_dir: Where predictions and metadata are written.
        target_column: Label name.
        data_dir: The `data_cleaning_encoding` folder to read frames from when
            `train_frame` is None.

    Returns:
        A summary: each member's score, the nested stack score, the weights, and
        the competition probabilities.

    Note:
    Passing `train_frame=None` is how a caller keeps its own copy of the frames
    from existing. On the CPU subprocess path nothing in this process needs the
    frames while the members train, and `del train_frame` inside this function
    cannot free memory that the *caller* still holds a reference to - which is the
    entire memory the path exists to save. So the caller passes None and this
    function reads them itself, holds them in a local it controls, and drops them
    before the first member starts.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    all_features = list(features)
    if train_frame is None or competition_frame is None:
        if data_dir is None:
            raise ValueError(
                "train_ensemble was given no frames and no data_dir to read them "
                "from. Pass both, or pass data_dir alone."
            )
        train_frame, competition_frame = load_ensemble_frames(
            data_dir, all_features, categorical
        )
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
    # Write the resolved device back into the config. `train_member` reads
    # `config.device` to hand to `apply_device`, and that call is what actually
    # puts `device="cuda"` on an estimator. Resolving into a local for the
    # sharding decision only - which is what happened first - left the members
    # reading the raw "auto", so no member got a device and all nine trained on
    # the processor.
    config = replace(config, device=device)
    groups = plan_gpu_shards(config.members, device)
    # `sharded` is true only when there is more than one group, i.e. more than one
    # GPU. On CPU `plan_gpu_shards` returns a single group, so the CPU branch below
    # is keyed on `device` directly and NOT on `sharded`.
    sharded = len(groups) > 1
    cpu_split = device == "cpu" and config.cpu_one_process_per_member
    if cpu_split:
        # On CPU there is one device, so without this every member would train
        # sequentially inside THIS process - holding the parent's ~2 GB of frames
        # for the whole run while each fold adds its own copy plus the in-fold
        # target-encoding block. One member per subprocess returns that memory to
        # the OS before the next one allocates. Same `fit_member_to_artifacts`,
        # same seeds, same `.npy` files.
        #
        # The config is handed to the workers rather than re-read from
        # config.yaml inside them, so a member trained here is the member this
        # function was given.
        from src.pipeline.cpu_member import run_cpu_members

        # Both frames are unreferenced for the rest of the member loop, and this
        # local is the only reference on this path when the caller passed None.
        # Pseudo-labeling reloads them below rather than keeping them pinned.
        #
        # Assigned rather than `del`. `del` unbinds the name, so the later
        # `if train_frame is None` raises UnboundLocalError instead of taking the
        # reload branch - which is how this path ended up crashing at the very end
        # of a full run. Setting it to None drops the same last reference and
        # leaves the name readable.
        #
        # Only released when they can be read again. A caller that passes frames
        # but no `data_dir` has no way to get them back, so holding them is what
        # keeps pseudo-labeling possible; releasing them would turn a configured
        # feature into an error.
        if data_dir is not None:
            train_frame = competition_frame = None
        log_step(
            "ensemble_device",
            device=device,
            mode="cpu_one_process_per_member",
            members=len(config.members),
        )
        if data_dir is None:
            # The frames were passed in and are being held for pseudo-labeling,
            # so the workers cannot be told where to read from. Say so rather than
            # letting a worker fail on a None path and taking the run with it.
            raise ValueError(
                "cpu_one_process_per_member is on but no data_dir was given, so the "
                "per-member workers have nowhere to read the prepared frames from. "
                "Pass data_dir, or set cpu_one_process_per_member: false to train "
                "the members in this process."
            )
        run_cpu_members(
            [m.name for m in config.members],
            config,
            artifacts_dir,
            data_dir,
            features,
            categorical,
            target_column,
        )
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
        log_step("ensemble_cpu_members_done", members=len(config.members))
    elif sharded:
        # One worker process per GPU. They write the same .npy files the inline
        # path would, and the loop below then reads those back, so the stacking
        # code is identical either way.
        log_step(
            "ensemble_device",
            device=device,
            gpus=len(groups),
            members=len(config.members),
        )
        # The parent does no training on this path - every worker builds and holds
        # its own copy of the frames - so the parent's copy is dead weight. At
        # 699,635 rows by 160 columns that is ~2 GB per process competing with
        # the workers for the same RAM and cores. Only the label survives, and it
        # is needed below to score the .npy files the workers wrote. Assigned to
        # None rather than deleted, so the pseudo-label block below can still read
        # the name to find out there are no frames.
        train_frame = competition_frame = None
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
    scores, weights, stack_record = newton_stack_score(Z, y.to_numpy(), config.stack_C)
    log_step(
        "stack_solver",
        method="newton",
        C=config.stack_C,
        iterations=stack_record["iterations"],
        gradient_inf=stack_record["gradient_inf"],
        rows=stack_record["rows"],
        members=stack_record["columns"],
        converged=stack_record["converged"],
        note=(
            "gradient below tolerance is a checked fact, not an iteration cap "
            "that ran out; sklearn's L-BFGS reported the latter"
        ),
    )
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
    else:
        summary["combiner"] = "logistic"
        summary["weights"] = {n: round(float(w), 6) for n, w in zip(names, weights)}
        log_step(
            "ensemble_stack",
            nested_auc=summary["nested_stack_auc"],
            mean_logits=summary["mean_logit_auc"],
            members=len(names),
        )
        summary["competition_probabilities"] = 1 / (1 + np.exp(-(ZT @ weights)))

    # Pseudo-labeling needs the frames, and it retrains every member, so it is a
    # second full run rather than a cheap step. On the sharded path the frames are
    # not available - the workers hold their own and the parent's copy is dropped
    # to keep ~2 GB out of their way - so it is skipped there, and says so.
    #
    # On the CPU subprocess path the frames ARE loadable, so rather than leaving a
    # configured feature silently off, they are read back here. That is the
    # deliberate trade: peak memory returns for this one step, in exchange for the
    # setting in config.yaml actually doing what it says. The alternative -
    # keeping the frames pinned for the whole run - costs ~2 GB on every fold of
    # every member instead, which is the thing the subprocess path exists to fix.
    if config.pseudo_label_enabled and not sharded:
        if train_frame is None or competition_frame is None:
            if data_dir is None:
                raise ValueError(
                    "pseudo_label_enabled is true but the frames are gone and no "
                    "data_dir was given to reload them from."
                )
            log_step(
                "pseudo_label_reload_frames",
                reason="frames were released for the subprocess member loop",
            )
            train_frame, competition_frame = load_ensemble_frames(
                data_dir, all_features, categorical
            )
        summary = _apply_pseudo_labeling(
            summary,
            train_frame,
            competition_frame,
            all_features,
            categorical,
            config,
            artifacts_dir,
            target_column,
            y,
            names,
            stacked_auc=(
                nested_blend if summary["combiner"] == "rank" else nested
            ),
            Z=Z,
            ZT=ZT,
            weights=weights,
            splitter=fold_list,
        )
    elif config.pseudo_label_enabled:
        log_step(
            "pseudo_label",
            skipped=True,
            reason="sharded GPU path does not hold the frames",
        )
    save_json(summary, artifacts_dir / "ensemble_summary.json")
    return summary


def _apply_pseudo_labeling(
    summary, train_frame, competition_frame, features, categorical,
    config, artifacts_dir, target_column, y, names, stacked_auc, Z, ZT, weights,
    splitter,
):
    """Add confident competition predictions as pseudo-labelled training rows.

    Args:
        summary: What the first pass produced. Mutated and returned.
        train_frame: Training rows, label included.
        competition_frame: Competition rows.
        features: Every configured feature; each member narrows it with
            `select_features`, exactly as the first pass did.
        categorical: Categorical column names.
        config: The ensemble settings.
        artifacts_dir: Where the summary is rewritten.
        target_column: Label name.
        y: Labels for the original training rows.
        names: Member names, in stack order.
        stacked_auc: The out-of-fold score of whichever combiner was actually
            chosen for the submission. This is the number the retrained stack has
            to beat.
        Z: Member logits on the training rows, rows by members.
        ZT: Member logits on the competition rows, rows by members. This is what
            the confidence mask is computed from - the rows being pseudo-labelled.
        weights: The fitted logistic stack weights. Turn `ZT` into the calibrated
            probabilities the thresholds are applied to.
        splitter: The fold list the first pass used to score its combiners, so the
            retrained ones are measured on the same structure.

    Returns:
        The summary, with either the pseudo-labelled predictions or the original
        ones.

    Note:
    Four things had to be right here and were not.

    *The probabilities must be the COMPETITION rows'.* The mask selects competition
    rows to pseudo-label, so the confidence has to be the stack's prediction for
    those rows. It was computed as `expit(Z @ weights)`, where `Z` is the member
    logits on the 699,635 TRAINING rows. That is a different set of rows from the
    299,844 the mask is applied to, and it does not fail loudly either: numpy
    happily builds a 699,635-long boolean array and `competition_frame[mask]`
    raises `ValueError: Item wrong length 699635 instead of 299844` - after every
    member has already been retrained once, which is hours in. So `pseudo_label:
    true` in config.yaml could never have completed.

    The fix is not only to use `ZT`, but to use it *with the same weights the
    stack was fitted with*, and to note that those weights are fitted on all
    training rows while `ZT` comes from fold models - so the competition
    predictions are honest and only the weight vector is in-sample. That is the
    same arrangement the submission itself uses.

    *The confidence thresholds need probabilities, not ranks.* When the rank blend
    wins, the submission values are rank-scale numbers in [0, 1] - a rank, not a
    probability. Thresholding those at 0.95 selects the top 5% *by rank*, which
    is a statement about the blend's ordering and not about any model's confidence,
    and then labelling them by `> 0.5` labels the top half of the competition set
    positive. So the mask always comes from the logistic member stack on
    `ZT`, which is calibrated and on a probability scale, whichever combiner
    builds the submission.

    *The comparison has to be against the chosen combiner.* `nested_stack_auc` is
    the logistic stack's score. The rank blend is chosen precisely when it beats
    that, and the rank blend's score is often the lower of the two. Comparing a
    retrained logistic stack against the number for a submission that was not
    logistic accepted the retrain on a criterion the submission never met. The
    caller passes the chosen combiner's score as `stacked_auc`.

    *The retrained stack has to be scored the way the submission is built.* The
    first pass picks between a logistic stack and a rank blend by out-of-fold
    score; the retrain evaluated only the logistic one and, if it won, rebuilt the
    submission from it - so the combiner decision was silently reversed by a
    second, unexamined choice. Both are now computed on the retrained members and
    the same rule is applied to both.

    *Each member has to see its own columns.* The first pass narrows features per
    member with `select_features`; retraining with the full list gives a member
    with `drop_prefix` set columns it never had, so the retrained member is a
    different model than the one being replaced, and the comparison is between
    two different things.

    *The augmented frame is not a cross-validated training set.* Pseudo-labelled
    competition rows are in `augmented_train`, so a row can be scored by a model
    that trained on a near-duplicate of a competition row it also scored. The
    nested score is therefore optimistic in absolute terms. It is still the right
    thing to compare, because both numbers are computed the same way, and the
    alternative - a second held-out split - costs as much as the thing it measures.
    """
    from scipy.special import expit

    # Confidence for the rows being pseudo-labelled, from the logistic stack on
    # `ZT`. Never from `summary["competition_probabilities"]`, which is rank-scale
    # whenever the rank blend won and would make the 0.95/0.05 thresholds a
    # statement about ordering rather than about confidence. `expit` rather than
    # `1/(1+exp(-x))` because the latter overflows to 0 on large logits silently.
    #
    # `len(probs)` is asserted against the competition rows before anything is
    # masked, because this was the exact mismatch that made the whole step
    # unusable: a mask built on training-row probabilities was applied to
    # competition rows and raised hours later, after the retraining.
    probs = expit(ZT @ weights)
    if len(probs) != len(competition_frame):
        raise ValueError(
            f"pseudo-label confidence has {len(probs)} values but the competition "
            f"frame has {len(competition_frame)} rows. The mask is applied to the "
            f"competition rows, so the two must match."
        )
    confident_mask = (probs > config.pseudo_label_high) | (probs < config.pseudo_label_low)
    n_confident = int(confident_mask.sum())
    if n_confident == 0:
        log_step(
            "pseudo_label",
            added=0,
            reason=(
                f"no competition row fell outside "
                f"[{config.pseudo_label_low}, {config.pseudo_label_high}]"
            ),
            prob_min=round(float(probs.min()), 6),
            prob_max=round(float(probs.max()), 6),
            prob_median=round(float(np.median(probs)), 6),
        )
        return summary

    # How confident, and in which direction. Logged because a threshold that
    # silently selects 3% of the competition set looks identical to one that
    # selects 40% unless the count is written down.
    positive = int((probs[confident_mask] > config.pseudo_label_high).sum())
    log_step(
        "pseudo_label_confident",
        rows=len(probs),
        selected=n_confident,
        selected_share=round(n_confident / len(probs), 4),
        pseudo_positive=positive,
        pseudo_negative=n_confident - positive,
        high=config.pseudo_label_high,
        low=config.pseudo_label_low,
    )

    pseudo_labels = (probs[confident_mask] > 0.5).astype("int8")
    pseudo_frame = competition_frame[confident_mask].copy()
    pseudo_frame[target_column] = pseudo_labels

    augmented_train = pd.concat([train_frame, pseudo_frame], ignore_index=True)
    log_step("pseudo_label", added=n_confident, total=len(augmented_train))

    # Retrain on augmented data
    y_aug = augmented_train[target_column].astype("int8")
    logits_train_aug = []
    logits_test_aug = []
    # Checkpointed, because this is roughly half of stage 5's wall clock and it is
    # the half that used to have no protection at all: `train_member` was called
    # without an `artifacts_dir`, which is exactly what turns checkpointing off.
    # Losing power at member 3 of 4 here used to cost the whole retrain.
    #
    # The tag keeps these checkpoints in separate files from the first pass's, so a
    # first-pass fold can never be read as a retrain fold. The fingerprint carries
    # the selection that produced this augmented set: the same rows, added at the
    # same thresholds. If the first pass scores differently next run and selects a
    # different number of rows, the retrain restarts rather than finishing a fold
    # trained on a set that no longer exists.
    pseudo_extra = {
        "pseudo_rows_added": n_confident,
        "pseudo_high": config.pseudo_label_high,
        "pseudo_low": config.pseudo_label_low,
        "pseudo_positive": int(positive),
        "augmented_total": len(augmented_train),
    }
    log_step(
        "pseudo_label_retrain_checkpoints",
        enabled=bool(artifacts_dir),
        tag=PSEUDO_CHECKPOINT_TAG,
        directory=str(artifacts_dir) if artifacts_dir else None,
        fingerprint_basis=sorted(pseudo_extra),
    )
    for spec in config.members:
        member_features = select_features(features, spec)
        oof_aug, test_aug = train_member(
            spec,
            augmented_train[member_features],
            y_aug,
            competition_frame[member_features],
            [c for c in categorical if c in member_features],
            config,
            target_column,
            artifacts_dir,
            checkpoint_tag=PSEUDO_CHECKPOINT_TAG,
            fingerprint_extra=pseudo_extra,
        )
        logits_train_aug.append(to_logit(oof_aug))
        logits_test_aug.append(to_logit(test_aug))

    Z_aug = np.column_stack(logits_train_aug)
    ZT_aug = np.column_stack(logits_test_aug)

    # Scored on the REAL rows only - the first `len(y)` of the augmented set, which
    # is exactly the original training frame because `augmented_train` concatenates
    # the pseudo rows on the end.
    #
    # This is the difference between "the retrain looks better" and a number worth
    # believing. Scoring on `y_aug` lets a competition row be scored by a fold model
    # that trained on a near-identical pseudo-labelled row, so the pseudo rows vote
    # for their own predictions; and it compares a score over 279k rows against a
    # baseline over 699k, which is not a comparison at all. Both sides of the
    # comparison are now the same rows - the genuine ones - so a difference means
    # the added data changed how those rows are ranked.
    #
    # The members are still TRAINED on the augmented set. That is the point of the
    # step; only the measurement is restricted.
    real_rows = len(y)
    y_real = y.to_numpy()
    Z_real = Z_aug[:real_rows]
    scores_real, weights_real, real_record = newton_stack_score(
        Z_real, y_real, config.stack_C
    )
    log_step(
        "pseudo_label_stack_solver",
        method="newton",
        C=config.stack_C,
        iterations=real_record["iterations"],
        gradient_inf=real_record["gradient_inf"],
        converged=real_record["converged"],
    )
    nested_real = float(roc_auc_score(y_real, scores_real))
    log_step(
        "pseudo_label_score_on_real_rows",
        rows=real_rows,
        dropped_pseudo_rows=int(len(y_aug) - real_rows),
        nested_auc=round(nested_real, 6),
        note=(
            "scored on genuine labels only; scoring the augmented frame would let "
            "the pseudo rows score themselves"
        ),
    )
    # Both combiners are scored on the retrained members, by the same rule the
    # first pass used. Scoring only the logistic one and then rebuilding the
    # submission from it reversed the combiner decision by default, without
    # anything reporting that it had.
    # The augmented-frame figures are computed only to be logged, so a reader can
    # see how much self-scoring inflates them. Nothing is accepted on them.
    blend_scores_aug, _ = nested_rank_blend(
        _to_probabilities(Z_aug), y_aug.to_numpy(), splitter
    )
    scores_aug, _, _ = newton_stack_score(Z_aug, y_aug.to_numpy(), config.stack_C)
    nested_aug = float(roc_auc_score(y_aug, scores_aug))
    # The rank blend's ACCEPTANCE score, on the real rows for the same reason.
    # Fitted on the real-row subset so its weights are not influenced by the pseudo
    # rows, and scored on the same rows as the logistic stack.
    blend_real_scores, blend_real_weights = nested_rank_blend(
        _to_probabilities(Z_real), y_real, splitter
    )
    nested_blend_real = float(roc_auc_score(y_real, blend_real_scores))
    nested_blend_aug = float(roc_auc_score(y_aug, blend_scores_aug))
    log_step(
        "pseudo_label_rank_candidates",
        on_real_rows=round(nested_blend_real, 6),
        on_augmented_rows=round(nested_blend_aug, 6),
        note="acceptance uses the real-rows figure",
    )
    log_step(
        "pseudo_label_candidates",
        logistic_auc_on_real=round(nested_real, 6),
        rank_auc_on_real=round(nested_blend_real, 6),
        logistic_auc_on_augmented=round(nested_aug, 6),
        rank_auc_on_augmented=round(nested_blend_aug, 6),
        had_to_beat=round(float(stacked_auc), 6),
        note=(
            "every figure below is on the genuine rows; the augmented-frame "
            "numbers are logged only to show how much self-scoring inflates them"
        ),
    )

    # Acceptance compares like with like: a retrained combiner's score on the
    # genuine rows against the first pass's score on the genuine rows.
    if nested_blend_real > stacked_auc or nested_real > stacked_auc:
        use_rank = nested_blend_real >= nested_real
        if use_rank:
            summary["nested_rank_blend_auc"] = round(nested_blend_real, 6)
            summary["combiner"] = "rank"
            summary["rank_blend_weights"] = {
                n: round(float(w), 6) for n, w in zip(names, blend_real_weights)
            }
            # Rank scale, on the competition rows, exactly as the first pass does.
            summary["competition_probabilities"] = np.column_stack(
                [to_rank(1 / (1 + np.exp(-ZT_aug[:, i]))) for i in range(ZT_aug.shape[1])]
            ) @ blend_real_weights
            accepted = nested_blend_real
        else:
            summary["nested_stack_auc"] = round(nested_real, 6)
            summary["combiner"] = "logistic"
            summary["weights"] = {
                n: round(float(w), 6) for n, w in zip(names, weights_real)
            }
            summary["competition_probabilities"] = expit(ZT_aug @ weights_real)
            accepted = nested_real
        # Per-member scores on the genuine rows, for the same reason.
        summary["solo_oof_auc"] = {
            n: round(float(roc_auc_score(y_real, Z_aug[:real_rows, i])), 6)
            for i, n in enumerate(names)
        }
        summary["pseudo_label_applied"] = True
        summary["pseudo_label_added"] = n_confident
        # Both numbers kept. `nested_stack_auc`/`nested_rank_blend_auc` are
        # overwritten above, so without these the summary would report only the
        # after-state and nothing would show what pseudo-labeling was worth.
        summary["pseudo_label_before"] = round(float(stacked_auc), 6)
        summary["pseudo_label_after"] = round(float(accepted), 6)
        summary["pseudo_label_scored_on"] = f"{real_rows} genuine rows"
        save_json(summary, artifacts_dir / "ensemble_summary.json")
        _clear_pseudo_checkpoints(config, artifacts_dir)
        log_step(
            "pseudo_label_applied",
            combiner=summary["combiner"],
            new_auc_on_real=round(float(accepted), 6),
            beat=round(float(stacked_auc), 6),
            added=n_confident,
        )
    else:
        summary["pseudo_label_applied"] = False
        summary["pseudo_label_added"] = 0
        summary["pseudo_label_before"] = round(float(stacked_auc), 6)
        summary["pseudo_label_after"] = round(
            float(max(nested_real, nested_blend_real)), 6
        )
        summary["pseudo_label_scored_on"] = f"{real_rows} genuine rows"
        # Cleared on reject too. The retrain is finished either way, so its folds
        # are spent; leaving them means the NEXT run resumes a retrain it has
        # already decided against, which is the same stale-checkpoint class of bug
        # as a changed member config.
        _clear_pseudo_checkpoints(config, artifacts_dir)
        log_step(
            "pseudo_label_rejected",
            logistic_auc_on_real=round(nested_real, 6),
            rank_auc_on_real=round(nested_blend_real, 6),
            had_to_beat=round(float(stacked_auc), 6),
        )

    return summary


def _to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Return member logits as probabilities.

    Args:
        logits: Rows by members, on the logit scale.

    Returns:
        The same shape, squashed into [0, 1].

    Note:
    `nested_rank_blend` documents its input as "raw scale" and rank-transforms it
    internally, so any monotone scale gives the identical ranks. This exists to
    make that explicit at the call site and to use `expit`, which does not
    overflow on large logits where `1/(1+exp(-x))` silently returns 0.
    """
    from scipy.special import expit

    return expit(logits)
