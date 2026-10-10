"""
Stage 5: train the ensemble members, stack them, and write the submission.

Run with: python -m src.pipeline.stage_05_ensemble

Input:  artifacts/data_cleaning_encoding/{train,validation,competition_test}.csv
Output: artifacts/ensemble/oof_<member>.npy, test_<member>.npy,
        ensemble_summary.json, reports/submissions/submission_ensemble.csv

Why this is a separate stage rather than part of stage 3: stage 3 fits one model
and stage 4 scores it on the frozen test split. An ensemble is a different shape
of work - every member is cross-validated, and the combiner is fitted on those
out-of-fold predictions. Folding it into stage 3 would change that stage's
contract and its tests for no benefit.

The frozen test split is not held back from training: this stage cross-validates
internally, so it needs no split of its own and folds train, validation and test
into the training rows. Every number it reports is measured out of fold. See
`load_ensemble_frames`.
"""

from src.components.ensemble import (
    load_competition_frame,
    load_ensemble_frames,
    read_categorical_columns,
    train_ensemble,
    train_ensemble_drops_frames,
)
from src.components.model_evaluation import write_submission
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step, setup_logging

STAGE_NAME = "stage_05_ensemble"


def run_pipeline() -> str | None:
    """Train the ensemble and write its submission.

    Returns:
        The submission path that was written, or None if the ensemble is off.

    Raises:
        FileNotFoundError: If stage 2 has not run.
        ValueError: If a member wants a column the artifacts do not have.
    """
    setup_logging()
    logger = get_logger()
    logger.info("[%s] starting ensemble", STAGE_NAME)

    reader = PipelineConfigReader()
    config = reader.create_ensemble_config()
    if not config.enabled:
        log_step(STAGE_NAME, skipped=True, reason="ensemble.enabled is false")
        return None

    pipeline = reader.create_pipeline_config()
    target_column = str(pipeline.training.target_column)
    features = list(pipeline.training.features)
    data_dir = pipeline.artifacts_root / "data_cleaning_encoding"
    # The categorical list comes from stage 2's CONTRACT, not from config.yaml's
    # four names.
    #
    # Stage 2 pins every column it typed as `category` - the four short categoricals
    # and the 20 `*_cat_` twins - and records all of them in features.json. The
    # contract is the authority on what is categorical, because it is what the CSVs
    # were written from. Reading config.yaml's four names instead produced a frame
    # where `Online boarding_cat_` was `category` dtype but absent from the list
    # handed to CatBoost, and CatBoost refuses that:
    #
    #   features data: column 'Online boarding_cat_' has dtype 'category' but is
    #   not in cat_features list
    #
    # XGBoost and LightGBM did not catch it because `enable_categorical` infers
    # the set from the frame, so only CatBoost failed - and it failed after an
    # hour and a half of training the other three members.
    categorical = read_categorical_columns(data_dir, features)

    # The frame assembly - including folding the validation and test splits back
    # into training - lives in the components layer because the per-GPU workers
    # have to build exactly the same frames.
    #
    # When the members train in subprocesses this stage passes None instead of the
    # frames. `train_ensemble` drops its own references before the first member
    # starts, but that frees nothing while the frames are still bound to a name in
    # THIS frame - so the ~2 GB the subprocess path exists to keep out of the
    # workers' way stayed resident through every fold of every member. The only
    # frame read back afterwards is the competition one, for the submission.
    if train_ensemble_drops_frames(config):
        competition_frame = None
        log_step(
            "stage_05_frames",
            loaded_by="train_ensemble",
            reason="members train in subprocesses",
        )
    else:
        train_frame, competition_frame = load_ensemble_frames(
            data_dir, features, categorical
        )

    summary = train_ensemble(
        config,
        None if competition_frame is None else train_frame,
        competition_frame,
        features,
        categorical,
        pipeline.artifacts_root / "ensemble",
        target_column,
        data_dir,
    )

    if competition_frame is None:
        # Read back for the submission only. `write_submission` needs the ids and
        # the row order; it does not need the label, which competition rows have
        # no column for.
        competition_frame = load_competition_frame(data_dir, features, categorical)

    output_dir = pipeline.report_root / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission_ensemble.csv"
    write_submission(
        competition_frame,
        summary["competition_probabilities"],
        pipeline.evaluation.sample_submission_path,
        output_path,
    )
    log_step(
        STAGE_NAME,
        nested_auc=summary["nested_stack_auc"],
        best_single=max(summary["solo_oof_auc"].values()),
        wrote=str(output_path),
    )
    logger.info("[%s] finished", STAGE_NAME)
    return str(output_path)


if __name__ == "__main__":
    run_pipeline()
