"""
Measures the raw data and prints the facts needed to fill in docs/.

This answers Steps 1.2, 1.3 and 1.6 of .lead/01-DATA.md: how many rows and
columns exist, what each column contains, how many values are missing, and
whether the data passes the quality checks that file lists.

Input:  data/train.csv, data/test.csv, data/sample_submission.csv
Output: printed to stdout only. Nothing is written, so re-running is safe.

Run with: python research/01_data_inventory.py

Note: this is research code, so it uses the csv module from the standard
library. The production pipeline uses pandas (see .dev/PROJECT-STRUCTURE.md).
The standard library is enough here because we only need counts and ranges,
and it avoids adding a dependency for a one-off measurement.
"""

import csv
import logging
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data")

# Columns that are a rating on a fixed scale. Their expected range is checked
# against EXPECTED_RATING_RANGE below.
RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

# The original dataset describes these as 1 (worst) to 5 (best). A 0 would
# mean either a new value or a broken load, so it is reported rather than
# assumed away.
EXPECTED_RATING_RANGE = (1, 5)

# Columns that should never be negative.
NON_NEGATIVE_COLUMNS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def read_header(file_path: Path) -> list[str]:
    """Return the column names of a CSV file, without reading the whole file."""
    with file_path.open(newline="", encoding="utf-8") as file:
        return next(csv.reader(file))


def count_rows_and_missing(file_path: Path) -> tuple[int, Counter]:
    """Count the rows and the missing values per column.

    Args:
        file_path: The CSV file to read.

    Returns:
        The number of data rows, and a counter of how many rows have a blank
        value in each column. Blank means the cell was empty in the file.
    """
    missing_counts: Counter = Counter()
    row_count = 0

    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_count += 1
            for column_name, value in row.items():
                if value is None or value.strip() == "":
                    missing_counts[column_name] += 1

    return row_count, missing_counts


def summarise_column(file_path: Path, column_name: str) -> dict:
    """Return the facts about one column: type, range, and common values.

    Args:
        file_path: The CSV file to read.
        column_name: The column to summarise.

    Returns:
        A dict with the number of values, the number of distinct values, the
        smallest and largest value when they are numbers, and the five most
        common values.
    """
    value_counts: Counter = Counter()
    numbers: list[float] = []

    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw_value = row[column_name]
            value_counts[raw_value] += 1
            try:
                numbers.append(float(raw_value))
            except ValueError:
                # Not a number, so this is a category rather than a quantity.
                pass

    summary = {
        "value_count": sum(value_counts.values()),
        "distinct_count": len(value_counts),
        "most_common": value_counts.most_common(5),
    }
    if numbers:
        summary["min"] = min(numbers)
        summary["max"] = max(numbers)
    return summary


def find_duplicate_rows(file_path: Path, columns_to_ignore: list[str]) -> int:
    """Count rows that repeat an earlier row, ignoring the given columns.

    Args:
        file_path: The CSV file to read.
        columns_to_ignore: Columns that are not compared, such as the row id.

    Returns:
        How many rows are exact repeats of an earlier row.
    """
    seen_rows: set[tuple] = set()
    duplicate_count = 0

    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_key = tuple(
                value for name, value in row.items() if name not in columns_to_ignore
            )
            if row_key in seen_rows:
                duplicate_count += 1
            else:
                seen_rows.add(row_key)

    return duplicate_count


def check_ids_are_unique_and_increasing(file_path: Path) -> dict:
    """Check whether the id column is unique and counts upwards.

    Args:
        file_path: The CSV file to read.

    Returns:
        Whether every id is different, and the smallest and largest id.
    """
    seen_ids: set[str] = set()
    first_id = 0
    last_id = 0
    is_increasing = True
    duplicate_ids = 0

    with file_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            row_id = row["id"]
            if row_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(row_id)

            numeric_id = int(row_id)
            if row_index == 0:
                first_id = numeric_id
            elif numeric_id <= last_id:
                is_increasing = False
            last_id = numeric_id

    return {
        "distinct_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "is_increasing": is_increasing,
        "first_id": first_id,
        "last_id": last_id,
    }


def find_overlapping_ids(train_path: Path, test_path: Path) -> int:
    """Count ids that appear in both the training and the test file.

    Args:
        train_path: The training CSV file.
        test_path: The test CSV file.

    Returns:
        How many ids are shared between the two files.
    """
    train_ids: set[str] = set()
    with train_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            train_ids.add(row["id"])

    overlapping = 0
    with test_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["id"] in train_ids:
                overlapping += 1

    return overlapping


def log_table_heading(title: str, column_names: list[str]) -> None:
    """Log a heading and the column names, so each block is readable alone."""
    logger.info("")
    logger.info("=== %s ===", title)
    logger.info("columns (%d): %s", len(column_names), column_names)


logger = logging.getLogger("data_inventory")


def main() -> None:
    """Measure the raw data and log everything needed for the docs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    train_path = DATA_PATH / "train.csv"
    test_path = DATA_PATH / "test.csv"
    submission_path = DATA_PATH / "sample_submission.csv"

    log_table_heading("FILES", [train_path.name, test_path.name, submission_path.name])

    train_columns = read_header(train_path)
    test_columns = read_header(test_path)
    submission_columns = read_header(submission_path)

    logger.info("train columns  : %s", train_columns)
    logger.info("test columns   : %s", test_columns)
    logger.info("submission cols: %s", submission_columns)
    logger.info(
        "columns only in train: %s", sorted(set(train_columns) - set(test_columns))
    )

    train_rows, train_missing = count_rows_and_missing(train_path)
    test_rows, test_missing = count_rows_and_missing(test_path)
    logger.info("train rows: %d", train_rows)
    logger.info("test rows: %d", test_rows)
    logger.info("train missing values: %s", dict(train_missing) or "none")
    logger.info("test missing values: %s", dict(test_missing) or "none")

    log_table_heading("ID COLUMN", ["train"])
    logger.info("train id: %s", check_ids_are_unique_and_increasing(train_path))
    logger.info("test id: %s", check_ids_are_unique_and_increasing(test_path))
    logger.info("ids in both files: %d", find_overlapping_ids(train_path, test_path))

    log_table_heading("DUPLICATE ROWS (ignoring id)", ["train", "test"])
    logger.info("train duplicate rows: %d", find_duplicate_rows(train_path, ["id"]))
    logger.info("test duplicate rows: %d", find_duplicate_rows(test_path, ["id"]))

    log_table_heading("TARGET COLUMN: satisfaction", ["train"])
    logger.info("train: %s", summarise_column(train_path, "satisfaction"))

    log_table_heading("RATING COLUMNS (expected 1-5)", RATING_COLUMNS)
    for column_name in RATING_COLUMNS:
        summary = summarise_column(train_path, column_name)
        is_outside_range = (
            summary["min"] < EXPECTED_RATING_RANGE[0]
            or summary["max"] > EXPECTED_RATING_RANGE[1]
        )
        logger.info(
            "%-38s min=%s max=%s distinct=%d outside_1_to_5=%s most_common=%s",
            column_name,
            summary["min"],
            summary["max"],
            summary["distinct_count"],
            is_outside_range,
            summary["most_common"],
        )

    log_table_heading("NUMBER COLUMNS", NON_NEGATIVE_COLUMNS)
    for column_name in NON_NEGATIVE_COLUMNS:
        summary = summarise_column(train_path, column_name)
        logger.info(
            "%-30s min=%s max=%s distinct=%d",
            column_name,
            summary["min"],
            summary["max"],
            summary["distinct_count"],
        )

    log_table_heading(
        "TEXT COLUMNS", ["Gender", "Customer Type", "Type of Travel", "Class"]
    )
    for column_name in ["Gender", "Customer Type", "Type of Travel", "Class"]:
        summary = summarise_column(train_path, column_name)
        logger.info(
            "%-18s distinct=%d most_common=%s",
            column_name,
            summary["distinct_count"],
            summary["most_common"],
        )

    log_table_heading("EXAMPLE ROWS", train_columns)
    with train_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            if row_index >= 3:
                break
            logger.info("row %d: %s", row_index, row)

    log_table_heading("SAMPLE SUBMISSION", submission_columns)
    submission_summary = summarise_column(submission_path, "satisfaction")
    logger.info("satisfaction column: %s", submission_summary)
    with submission_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            if row_index >= 3:
                break
            logger.info("row %d: %s", row_index, row)


if __name__ == "__main__":
    main()
