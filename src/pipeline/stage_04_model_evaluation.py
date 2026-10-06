"""
Stage 4 of the pipeline: score the model on the frozen test split, write the
submission.

Run with: python -m src.pipeline.stage_04_model_evaluation

A stage contains ordering and logging only. The metrics, the confusion matrix,
the error reading and the submission file all live in
src/components/model_evaluation.py. See .lead/02-A-ARCHITECTURE.md section 5.
"""

from src.components.model_evaluation import run_model_evaluation
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step, next_model_version

STAGE_NAME = "stage_04_model_evaluation"


def run_pipeline(model_version: int | None = None):
    """Score the trained model and write the submission file.

    Args:
        model_version: Which trained model to score. None means the newest one.

    Returns:
        The evaluation artifact, carrying the exact report and submission paths.
        Returned (not re-discovered by globbing) because lexicographic sorting
        puts submission_model_8.csv after submission_model_10.csv, which once
        made the run log name the wrong file.

    Raises:
        FileNotFoundError: If the model or the prepared data is missing.
        ValueError: If the saved feature list does not match the prepared data, or
            the submission row count does not match the template.
    """
    logger = get_logger()
    logger.info("[%s] starting model evaluation", STAGE_NAME)

    reader = PipelineConfigReader()
    version = (
        model_version
        if model_version is not None
        else (next_model_version(reader.create_pipeline_config().models_root) - 1)
    )
    if version < 1:
        raise ValueError(
            "No trained model found. Run stage 3 first: "
            "python -m src.pipeline.stage_03_model_training"
        )

    evaluation_config = reader.create_evaluation_config(version)
    artifact = run_model_evaluation(evaluation_config)

    log_step(
        STAGE_NAME,
        rows_out=artifact.submission_rows,
        test_roc_auc=round(artifact.test_metrics.roc_auc, 6),
        test_accuracy=round(artifact.test_metrics.accuracy, 6),
        report=str(artifact.report_path),
        submission=str(artifact.submission_path),
    )
    logger.info("[%s] finished", STAGE_NAME)
    return artifact


if __name__ == "__main__":
    run_pipeline()
