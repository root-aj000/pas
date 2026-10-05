"""
Stage 2 of the pipeline: split the data, derive features, encode categories.

Run with: python -m src.pipeline.stage_02_data_cleaning_encoding

A stage contains ordering and logging only. The split, the derived columns and
the encoding all live in src/components/data_cleaning_encoding.py.
See .lead/02-A-ARCHITECTURE.md section 5.
"""

from src.components.data_cleaning_encoding import run_data_cleaning_encoding
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step

STAGE_NAME = "stage_02_data_cleaning_encoding"


def run_pipeline() -> None:
    """Prepare the three splits and save them for stage 3.

    Raises:
        FileNotFoundError: If stage 1 has not run.
        ValueError: If the split is degenerate, the base rates diverge, or the
            feature list does not match the data.
    """
    logger = get_logger()
    logger.info("[%s] starting data cleaning and encoding", STAGE_NAME)

    reader = PipelineConfigReader()
    ingestion_config = reader.create_ingestion_config()
    cleaning_config = reader.create_cleaning_config()

    raw_dir = ingestion_config.artifacts_dir / "data_ingestion"
    artifact = run_data_cleaning_encoding(
        raw_train_path=raw_dir / "raw_train.csv",
        raw_competition_path=raw_dir / "raw_test.csv",
        config=cleaning_config,
    )

    log_step(
        STAGE_NAME,
        rows_out=artifact.train_rows,
        validation_rows=artifact.validation_rows,
        test_rows=artifact.test_rows,
        features=len(artifact.features),
        wrote=str(artifact.train_path),
    )
    logger.info("[%s] finished", STAGE_NAME)


if __name__ == "__main__":
    run_pipeline()
