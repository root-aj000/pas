"""
Stage 3 of the pipeline: train the model and save the bundle.

Run with: python -m src.pipeline.stage_03_model_training

A stage contains ordering and logging only. The estimator, the registry, the
overfitting check and the model card all live in
src/components/model_training.py. See .lead/02-A-ARCHITECTURE.md section 5.
"""

from src.components.model_training import run_model_training
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step, next_model_version, setup_logging

STAGE_NAME = "stage_03_model_training"


def run_pipeline(model_version: int | None = None) -> int:
    """Train the model named in config.yaml and save it.

    Args:
        model_version: Which model folder to write into. One more than the
            highest already present, so a trained model is never overwritten. None
            means work it out.

    Returns:
        The model version that was written.

    Raises:
        FileNotFoundError: If stage 2 has not run.
        KeyError: If model_name is not in MODEL_REGISTRY.
        ValueError: If a configured feature is missing or training overruns its
            time budget.
        AssertionError: If the model cannot memorise 16 rows.
    """
    setup_logging()
    logger = get_logger()
    logger.info("[%s] starting model training", STAGE_NAME)

    reader = PipelineConfigReader()
    version = (
        model_version
        if model_version is not None
        else next_model_version(reader.create_pipeline_config().models_root)
    )
    training_config = reader.create_training_config(version)
    validation_path = training_config.train_data_path.parent / "validation.csv"

    artifact = run_model_training(training_config, validation_data_path=validation_path)

    log_step(
        STAGE_NAME,
        rows_out=artifact.validation_metrics.get("accuracy"),
        model_version=artifact.model_version,
        roc_auc=round(artifact.validation_metrics.get("roc_auc", 0.0), 6),
        fit_seconds=round(artifact.training_seconds, 1),
        wrote=str(artifact.model_path),
    )
    logger.info("[%s] finished", STAGE_NAME)
    return artifact.model_version


if __name__ == "__main__":
    run_pipeline()
