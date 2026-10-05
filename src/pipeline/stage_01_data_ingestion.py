"""
Stage 1 of the pipeline: load the raw data and check it.

Run with: python -m src.pipeline.stage_01_data_ingestion

A stage contains ordering and logging only. If this file starts containing real
logic, that logic belongs in src/components/data_ingestion.py.
See .lead/02-A-ARCHITECTURE.md section 5.
"""

from src.components.data_ingestion import run_data_ingestion
from src.config.configuration import PipelineConfigReader
from src.utils.common import get_logger, log_step

STAGE_NAME = "stage_01_data_ingestion"


def run_pipeline() -> None:
    """Load the raw data and save it for stage 2.

    Raises:
        FileNotFoundError: If a raw file is missing.
        ValueError: If the data is empty, a column is missing, a rating is out of
            range, an id repeats, or a column is mostly blank.
    """
    logger = get_logger()
    logger.info("[%s] starting data ingestion", STAGE_NAME)

    config = PipelineConfigReader().create_ingestion_config()
    artifact = run_data_ingestion(config)

    log_step(
        STAGE_NAME,
        rows_out=artifact.row_count,
        columns=len(artifact.column_names),
        wrote=str(artifact.raw_data_path),
    )
    logger.info("[%s] finished", STAGE_NAME)


if __name__ == "__main__":
    run_pipeline()
