"""
Loads the raw data, checks it, and saves an unchanged copy for the next stage.

Input:  data/train.csv, data/test.csv
Output: artifacts/data_ingestion/raw_train.csv, artifacts/data_ingestion/raw_test.csv

Run with: python -m src.pipeline.stage_01_data_ingestion

This file does one job: read the files and prove they are what we think they are.
It does not clean anything and it does not know what comes next. The quality
checks in .lead/01-DATA.md Step 1.6 live here, because they must run on every run
rather than once by hand.
"""

import pandas as pd

from src.constants import MAX_SERVICE_RATING, MIN_SERVICE_RATING, SERVICE_RATING_COLUMNS
from src.entity.config_entity import DataIngestionArtifact, DataIngestionConfig
from src.utils.common import check_columns, log_step, save_dataframe

# A file that is more than 40% blank in a column has failed to load. From
# .lead/01-DATA.md Step 1.6, whose worked example of a bad load is 60% missing in
# a key column. 40% catches it with margin.
MAX_MISSING_SHARE = 0.40


def check_required_columns(
    name: str, frame: pd.DataFrame, expected: set[str], target_column: str
) -> None:
    """Stop the run if a column the pipeline needs is missing.

    Args:
        name: Which file is being checked, used in the error message.
        frame: The loaded rows.
        expected: Every column the pipeline is allowed to use, including the label.
        target_column: The label's name, which is allowed to be absent from the
            competition test file.

    Raises:
        ValueError: If an unexpected column is missing from the training file, or
            if a column is missing from the competition test file.
    """
    actual = set(frame.columns)
    if target_column in actual:
        check_columns(f"{name}.train", actual, expected)
        return
    # The competition test file has no label, so everything except it must exist.
    check_columns(f"{name}.test", actual, expected - {target_column})


def check_service_ratings_in_range(name: str, frame: pd.DataFrame) -> None:
    """Stop the run if a service rating falls outside 0-5.

    Args:
        name: Which file is being checked, used in the error message.
        frame: The loaded rows.

    Raises:
        ValueError: If any rating falls outside the documented scale. A value of
            7 means either a different scale or a corrupted load, and training on
            it silently would be worse than stopping.
    """
    for column in SERVICE_RATING_COLUMNS:
        if column not in frame.columns:
            continue
        values = set(frame[column].dropna().unique().tolist())
        outside = sorted(
            value
            for value in values
            if not MIN_SERVICE_RATING <= value <= MAX_SERVICE_RATING
        )
        if outside:
            raise ValueError(
                f"{name}: column '{column}' holds values outside "
                f"{MIN_SERVICE_RATING}-{MAX_SERVICE_RATING}: {outside}. "
                "Either the survey scale changed or the file is corrupted. "
                "Check the file against the hash in docs/data_inventory.md."
            )


def check_missing_values_are_reasonable(name: str, frame: pd.DataFrame) -> None:
    """Stop the run if any column is mostly blank, which means a failed load.

    Args:
        name: Which file is being checked, used in the error message.
        frame: The loaded rows.

    Raises:
        ValueError: If any column is more than MAX_MISSING_SHARE blank.

    Note:
        This is a threshold, not a demand for zero blanks. The dataset has 292
        blank arrival delays out of 699,635 rows, which is an ordinary missing
        value problem and not a broken load. Those are handled downstream as
        their own group - see docs/column_dictionary.md Note 1.
    """
    missing_share = frame.isna().mean()
    too_missing = missing_share[missing_share > MAX_MISSING_SHARE]
    if not too_missing.empty:
        detail = {
            column: round(float(share), 4) for column, share in too_missing.items()
        }
        raise ValueError(
            f"{name}: these columns are more than {MAX_MISSING_SHARE:.0%} blank: "
            f"{detail}. That is the signature of a failed load rather than real "
            "missing data. Stopping instead of training on it."
        )


def check_ids_are_unique(name: str, frame: pd.DataFrame, id_column: str) -> None:
    """Stop the run if the id column repeats, which would mean a bad load.

    Args:
        name: Which file is being checked, used in the error message.
        frame: The loaded rows.
        id_column: The id column's name.

    Raises:
        ValueError: If any id appears more than once.
    """
    duplicated = frame[id_column][frame[id_column].duplicated()]
    if not duplicated.empty:
        examples = sorted(duplicated.unique().tolist())[:5]
        raise ValueError(
            f"{name}: {len(duplicated)} rows repeat an id, for example {examples}. "
            "A duplicated id means a partial load or a bad concatenation."
        )


def summarise_for_log(name: str, frame: pd.DataFrame) -> None:
    """Log the shape and names of a loaded frame, never its contents.

    Args:
        name: Which file is being summarised.
        frame: The loaded rows.

    Note:
        .dev/DEBUGGING.md section 1: log shape and names. A whole dataframe in a
        log file is unreadable and can leak customer data.
    """
    missing = {
        column: int(count) for column, count in frame.isna().sum().items() if count
    }
    log_step(
        name,
        rows_in=len(frame),
        rows_out=len(frame),
        cols=len(frame.columns),
        missing=missing if missing else "none",
    )


def run_data_ingestion(config: DataIngestionConfig) -> DataIngestionArtifact:
    """Load the raw data, check it, and save an unchanged copy for stage 2.

    Args:
        config: Where the raw files are and where to save the output.

    Returns:
        The paths of the saved files, the row count, and the column names.

    Raises:
        FileNotFoundError: If either raw file is missing.
        ValueError: If a file is empty, a column is missing or out of range, an
            id repeats, or a column is mostly blank. Every one of those means the
            data is not what we think it is, and training on it would produce
            confident nonsense.
    """
    for path in (config.data_path, config.competition_test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {path}. The files are downloaded once from "
                "the competition. See docs/data_inventory.md for their hashes."
            )

    log_step("ingestion", source=config.data_path, reading=True)
    train_frame = pd.read_csv(config.data_path)
    log_step("ingestion", source=config.competition_test_path, reading=True)
    competition_frame = pd.read_csv(config.competition_test_path)

    for name, frame in (("train.csv", train_frame), ("test.csv", competition_frame)):
        if frame.empty:
            raise ValueError(
                f"{name} is empty. An empty file always means the download "
                "failed, so the pipeline must stop here."
            )
        target = "satisfaction"
        check_required_columns(name, frame, set(config.required_columns), target)
        check_service_ratings_in_range(name, frame)
        check_missing_values_are_reasonable(name, frame)
        check_ids_are_unique(name, frame, "id")
        summarise_for_log(name, frame)

    leakage = config.banned_features & set(train_frame.columns)
    log_step("ingestion", banned_features_present_as_columns=sorted(leakage))

    output_dir = config.artifacts_dir / "data_ingestion"
    train_output = save_dataframe(train_frame, output_dir / "raw_train.csv")
    test_output = save_dataframe(competition_frame, output_dir / "raw_test.csv")
    log_step("ingestion", rows_out=len(train_frame), wrote=str(train_output))

    return DataIngestionArtifact(
        raw_data_path=train_output,
        competition_test_path=test_output,
        row_count=len(train_frame),
        column_names=list(train_frame.columns),
    )
