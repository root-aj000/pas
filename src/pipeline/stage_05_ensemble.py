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

from src.components.ensemble import load_ensemble_frames, train_ensemble
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
    cleaning = reader.create_cleaning_config()
    features = list(pipeline.training.features)
    categorical = [c for c in cleaning.categorical_columns if c in features]

    # The frame assembly - including folding the validation and test splits back
    # into training - lives in the components layer because the per-GPU workers
    # have to build exactly the same frames.
    train_frame, competition_frame = load_ensemble_frames(
        pipeline.artifacts_root / "data_cleaning_encoding", features, categorical
    )

    summary = train_ensemble(
        config,
        train_frame,
        competition_frame,
        features,
        categorical,
        pipeline.artifacts_root / "ensemble",
        target_column,
    )

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
