"""
Stage 6: test-time augmentation by feature view.

Run with: python -m src.pipeline.stage_06_feature_views

Input:  artifacts/data_cleaning_encoding/{train,competition_test}.csv,
        artifacts/ensemble/{oof,test}_<member>.npy from stage 5
Output: artifacts/views/view_summary.json,
        reports/view_submission.csv when a view is accepted

WHAT THIS IS
Stage 5 builds four members, and all four see the same 178 columns. Their errors
are therefore correlated, and the stack shows it: it scores 0.9582 nested while
its best single member scores higher alone. Averaging near-identical models
cancels the little independent signal they have.

This stage fixes that by training the same model on DIFFERENT VIEWS of the
columns, then checking whether the ensemble improves once a view is added. A
view is not a new architecture - it is the same estimator looking at a narrower
piece of the data.

WHY VIEWS AND NOT MORE ARCHITECTURES
Measured on this dataset, out of fold and with one model held constant:

    the 13 raw ratings only        0.952410
    ratings + demographics         0.957730
    all 178 configured features    0.959139

A +0.0067 spread between views of the same rows. That spread is what makes a
stack worth having: if every member scored 0.9591 the stack would have nothing
to combine. So the views are already diverse - the pipeline just never uses them.
`MemberSpec.drop_prefix` and `drop_suffix` have existed for that purpose and no
member has ever set them.

HOW IT DECIDES
Every candidate is gated on the NESTED stack AUC over genuine labels, not on its
own score. A view that scores well alone but does not improve the stack is
rejected, because the problem being solved is correlation, not accuracy.

The accept test also requires the gain to clear the noise floor. The nested score
moves by about +/-0.0003 between fold seeds on this dataset, so a gain smaller
than that is indistinguishable from re-drawing the folds. `min_gain` defaults to
0.0004 for that reason: slightly above the measured noise, so a real effect passes
and a lucky seed does not.

Note on the ceiling
`research/noise_ceiling.py` measured the achievable out-of-fold AUC on these
features at 0.958821 against 0.959139 achieved, with the gap inside one standard
deviation. If p(x) is already fully recovered then NO view can help, and this
stage will report every candidate rejected. That is a legitimate and useful
outcome: it would confirm the ceiling rather than waste hours on it. The reason to
run it anyway is that the ceiling was measured with the 178-column model, and the
claim that views cannot beat it is a prediction, not a result.

Cost, measured on an 8-core CPU box with no GPU:

    one view, xgboost, 1500 trees depth 9, 5 folds   ~13 min
    one view, lightgbm, 1500 trees 255 leaves        ~10 min

Views are trained through the same per-member subprocess path stage 5 uses, so
they are sequential and memory-bounded, and each is checkpointed per fold. An
interrupted run resumes rather than restarting.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import roc_auc_score

from src.components.ensemble import (
    EnsembleConfig,
    MemberSpec,
    load_competition_frame,
    load_ensemble_frames,
    newton_stack_score,
    read_categorical_columns,
    select_features,
    to_logit,
    train_member,
)
from src.components.model_evaluation import write_submission
from src.config.configuration import PipelineConfigReader, ViewConfig
from src.utils.common import get_logger, log_step, save_json, setup_logging

STAGE_NAME = "stage_06_feature_views"
# Same constant the stack uses, so stage 6 and stage 5 split rows identically and
# their nested scores are comparable.
STACK_SEED = 42


def _nested_auc(
    logits: dict[str, Any], y: np.ndarray, C: float, names: list[str]
) -> float:
    """Return the nested out-of-fold AUC of the Newton stack over `names`.

    Args:
        logits: Member name to its out-of-fold logits, rows by members.
        y: Labels.
        C: Stack regularisation.
        names: Which members to stack, in order.

    Returns:
        The pooled nested AUC.
    """
    Z = np.column_stack([logits[name] for name in names])
    scores, _, _ = newton_stack_score(Z, y, C)
    return float(roc_auc_score(y, scores))


def run_pipeline() -> str | None:
    """Search feature views and submit one if it improves the stack.

    Returns:
        The submission path written, or None when nothing was accepted.

    Raises:
        FileNotFoundError: If stage 2 has not run.
        ValueError: If stage 5 has not run, or a view matches no columns.
    """
    setup_logging()
    logger = get_logger()
    logger.info("[%s] starting feature-view search", STAGE_NAME)

    reader = PipelineConfigReader()
    config = reader.create_ensemble_config()
    view_config: ViewConfig = reader.create_view_config()
    if not view_config.enabled:
        log_step(STAGE_NAME, skipped=True, reason="feature_views.enabled is false")
        return None

    pipeline = reader.create_pipeline_config()
    target_column = str(pipeline.training.target_column)
    features = list(pipeline.training.features)
    data_dir = pipeline.artifacts_root / "data_cleaning_encoding"
    # From the contract, for the same reason stage 5 does: it names all 24 columns
    # stage 2 pinned, where config.yaml names 4, and CatBoost refuses a frame
    # carrying a `category` column that is not in its cat_features list.
    categorical = read_categorical_columns(data_dir, features)
    ensemble_dir = pipeline.artifacts_root / "ensemble"
    views_dir = pipeline.artifacts_root / "views"

    # The baseline members must already exist. Stage 6 adds to stage 5, it does not
    # replace it - the views are only worth anything as extra members.
    member_names = [m.name for m in config.members]
    missing = [
        name
        for name in member_names
        if not (ensemble_dir / f"oof_{name}.npy").exists()
    ]
    if missing:
        raise ValueError(
            f"stage 6 needs stage 5's predictions first, and these are missing: "
            f"{missing}. Run python -m src.pipeline.stage_05_ensemble first."
        )

    train_frame, competition_frame = load_ensemble_frames(
        data_dir, features, categorical
    )
    y = train_frame[target_column].astype("int8").to_numpy()
    log_step(
        "view_stage_ready",
        members=len(member_names),
        views=len(view_config.views),
        min_gain=view_config.min_gain,
        rows=len(y),
    )

    base_logits: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    for name in member_names:
        base_logits[name] = to_logit(np.load(ensemble_dir / f"oof_{name}.npy"))
        base_test[name] = to_logit(np.load(ensemble_dir / f"test_{name}.npy"))

    baseline = _nested_auc(base_logits, y, config.stack_C, member_names)
    log_step(
        "view_baseline",
        nested_auc=round(baseline, 6),
        members=len(member_names),
        note="stage 5's members, stacked, on genuine labels",
    )

    views_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for entry in view_config.views:
        name = entry["name"]
        spec = MemberSpec(
            name=name,
            kind=str(entry.get("kind", "xgboost")),
            params=dict(entry.get("params") or {}),
            drop_prefix=tuple(entry.get("drop_prefix") or ()),
            drop_suffix=tuple(entry.get("drop_suffix") or ()),
            keep_prefixes=tuple(entry.get("keep_prefixes") or ()),
            target_encodings=bool(entry.get("target_encodings", True)),
        )
        view_features = select_features(features, spec)
        dropped = len(features) - len(view_features)
        log_step(
            "view_start",
            view=name,
            kind=spec.kind,
            columns=len(view_features),
            dropped=dropped,
            target_encodings=spec.target_encodings,
        )

        oof, test, _solo = fit_view(
            spec,
            train_frame,
            competition_frame,
            features,
            categorical,
            replace(config, members=[spec]),
            target_column,
            views_dir,
            name,
        )
        solo_auc = float(roc_auc_score(y, oof))

        view_logits = to_logit(oof)
        view_test_logits = to_logit(test)
        with_view = _nested_auc(
            {**base_logits, name: view_logits}, y, config.stack_C, [*member_names, name]
        )
        gain = with_view - baseline
        verdict = "ACCEPT" if gain >= view_config.min_gain else "reject"

        record = {
            "view": name,
            "kind": spec.kind,
            "columns": len(view_features),
            "dropped": dropped,
            "solo_auc": round(solo_auc, 6),
            "nested_with_view": round(with_view, 6),
            "nested_baseline": round(baseline, 6),
            "gain": round(gain, 6),
            "min_gain": view_config.min_gain,
            "verdict": verdict,
        }
        results.append(record)
        log_step("view_result", **{k: v for k, v in record.items() if k != "verdict"},
                 verdict=verdict)

        if verdict == "ACCEPT" and (accepted is None or gain > accepted["gain"]):
            accepted = {
                **record,
                "name": name,
                "oof": view_logits,
                "test": view_test_logits,
            }

    summary = {
        "baseline_nested_auc": round(baseline, 6),
        "stack_C": config.stack_C,
        "min_gain": view_config.min_gain,
        "base_members": member_names,
        "views": results,
        "accepted": accepted["view"] if accepted else None,
    }
    save_json(summary, views_dir / "view_summary.json")

    if accepted is None:
        log_step(
            STAGE_NAME,
            submitted=False,
            reason="no view cleared min_gain",
            baseline=round(baseline, 6),
            best_gain=max((r["gain"] for r in results), default=0.0),
        )
        logger.info("[%s] no view improved the stack; nothing submitted", STAGE_NAME)
        return None

    submission_path = write_view_submission(
        accepted,
        base_test,
        member_names,
        y,
        config,
        pipeline,
        features,
        categorical,
        data_dir,
    )
    log_step(
        STAGE_NAME,
        submitted=True,
        view=accepted["view"],
        gain=accepted["gain"],
        nested_auc=accepted["nested_with_view"],
        wrote=str(submission_path),
    )
    logger.info("[%s] finished", STAGE_NAME)
    return str(submission_path)


def fit_view(
    spec: MemberSpec,
    train_frame: pd.DataFrame,
    competition_frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    config: EnsembleConfig,
    target_column: str,
    views_dir: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cross-validate one view and save its predictions.

    Args:
        spec: The view, as a member spec carrying its column filters.
        train_frame: Training rows with the label.
        competition_frame: Competition rows.
        features: Every configured feature; the view narrows this.
        categorical: Categorical column names.
        config: The ensemble settings, with `members` set to just this view.
        target_column: Label name.
        views_dir: Where this view's checkpoints and `.npy` files go.
        name: The view's name.

    Returns:
        Out-of-fold probabilities, competition probabilities, and the view's solo
        AUC.

    Note:
    Checkpointed per fold, under `views_dir` rather than the ensemble's own
    checkpoint folder, so a view interrupted half-way cannot be confused with a
    stage 5 member. The fingerprint mechanism is stage 5's, so changing a view's
    params rejects its own stale checkpoints rather than resuming them.
    """
    view_features = select_features(features, spec)
    use_cat = [c for c in categorical if c in view_features]
    oof, test = train_member(
        spec,
        train_frame[view_features],
        train_frame[target_column].astype("int8"),
        competition_frame[view_features],
        use_cat,
        config,
        target_column,
        views_dir,
    )
    np.save(views_dir / f"oof_{name}.npy", oof)
    np.save(views_dir / f"test_{name}.npy", test)
    return (
        oof,
        test,
        float(roc_auc_score(train_frame[target_column].astype("int8"), oof)),
    )


def write_view_submission(
    accepted: dict[str, Any],
    base_test: dict[str, np.ndarray],
    member_names: list[str],
    y: np.ndarray,
    config: EnsembleConfig,
    pipeline: Any,
    features: list[str],
    categorical: list[str],
    data_dir: Path,
) -> Path:
    """Stack stage 5's members plus the accepted view and write a submission.

    Args:
        accepted: The winning view's record, with its logits.
        base_test: Stage 5 members' competition logits.
        member_names: Stage 5's member names, in order.
        y: Training labels.
        config: The ensemble settings, for `stack_C`.
        pipeline: The pipeline config, for paths.
        features: Every configured feature.
        categorical: Categorical column names.
        data_dir: Where the prepared competition frame is.

    Returns:
        The submission path written.

    Note:
    The stack weights are fitted on the OUT-OF-FOLD logits, which is what the
    nested score measured, and then applied to the competition logits. The weights
    therefore never saw the rows they are scored on, and the submission is built
    from the same fit that was accepted.
    """
    names = [*member_names, accepted["name"]]
    train_logits = np.column_stack(
        [np.load(data_dir.parent / "ensemble" / f"oof_{n}.npy") for n in member_names]
    )
    train_logits = to_logit(train_logits)
    view_oof = np.load(data_dir.parent / "views" / f"oof_{accepted['name']}.npy")
    train_logits = np.column_stack([train_logits, to_logit(view_oof)])
    test_logits = np.column_stack(
        [base_test[n] for n in member_names] + [accepted["test"]]
    )

    _, weights, _ = newton_stack_score(train_logits, y, config.stack_C)
    from scipy.special import expit

    probabilities = expit(test_logits @ weights)
    log_step(
        "view_submission",
        members=names,
        weights={n: round(float(w), 4) for n, w in zip(names, weights)},
    )

    competition_frame = load_competition_frame(data_dir, features, categorical)
    output_dir = pipeline.report_root / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission_views.csv"
    write_submission(
        competition_frame,
        probabilities,
        pipeline.evaluation.sample_submission_path,
        output_path,
    )
    return output_path


if __name__ == "__main__":
    run_pipeline()