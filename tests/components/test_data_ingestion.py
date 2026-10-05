"""
Tests for src/components/data_ingestion.py

Run with: pytest tests/components/test_data_ingestion.py -v

The tests use small readable tables written inline, not the real data. See
.dev/RULES.md rule 10: "Test data is small and readable."
"""

import pandas as pd
import pytest

from src.components.data_ingestion import (
    MAX_MISSING_SHARE,
    check_ids_are_unique,
    check_missing_values_are_reasonable,
    check_required_columns,
    check_service_ratings_in_range,
)
from src.constants import SERVICE_RATING_COLUMNS


def make_small_frame(rows: int = 5) -> pd.DataFrame:
    """Return a small valid frame with every column the pipeline expects.

    Args:
        rows: How many rows to build.

    Returns:
        A frame with all service ratings at 3, and both delay columns at 0.
    """
    data: dict[str, object] = {"id": list(range(rows)), "satisfaction": [True] * rows}
    for column in SERVICE_RATING_COLUMNS:
        data[column] = [3] * rows
    data["Age"] = [40] * rows
    data["Flight Distance"] = [1000] * rows
    data["Departure Delay in Minutes"] = [0] * rows
    data["Arrival Delay in Minutes"] = [0.0] * rows
    data["Gender"] = ["Male"] * rows
    data["Customer Type"] = ["Loyal Customer"] * rows
    data["Type of Travel"] = ["Personal Travel"] * rows
    data["Class"] = ["Eco"] * rows
    return pd.DataFrame(data)


def test_valid_frame_passes_every_check() -> None:
    """A well-formed frame must not raise."""
    frame = make_small_frame()
    required = set(frame.columns)

    check_required_columns("train.csv", frame, required, "satisfaction")
    check_service_ratings_in_range("train.csv", frame)
    check_missing_values_are_reasonable("train.csv", frame)
    check_ids_are_unique("train.csv", frame, "id")


def test_raises_when_a_required_column_is_missing() -> None:
    """A renamed column must stop the run with a message naming it."""
    frame = make_small_frame().drop(columns=["Online boarding"])
    required = set(make_small_frame().columns)

    with pytest.raises(ValueError, match="Online boarding"):
        check_required_columns("train.csv", frame, required, "satisfaction")


def test_competition_test_may_omit_the_label() -> None:
    """The competition test file has no label, and that must be allowed."""
    frame = make_small_frame().drop(columns=["satisfaction"])
    required = set(make_small_frame().columns)

    check_required_columns("test.csv", frame, required, "satisfaction")


def test_raises_when_a_rating_is_out_of_range() -> None:
    """A rating of 7 means a corrupted load and must stop the run."""
    frame = make_small_frame()
    frame.loc[0, "Seat comfort"] = 7

    with pytest.raises(ValueError, match="Seat comfort"):
        check_service_ratings_in_range("train.csv", frame)


def test_raises_when_ids_repeat() -> None:
    """A repeated id means a bad load and must stop the run."""
    frame = make_small_frame()
    frame.loc[1, "id"] = frame.loc[0, "id"]

    with pytest.raises(ValueError, match="repeat an id"):
        check_ids_are_unique("train.csv", frame, "id")


def test_raises_when_a_column_is_mostly_blank() -> None:
    """A mostly-blank column is the signature of a failed load."""
    frame = make_small_frame(rows=10)
    frame.loc[: int(len(frame) * MAX_MISSING_SHARE), "Age"] = None

    with pytest.raises(ValueError, match="failed load"):
        check_missing_values_are_reasonable("train.csv", frame)


def test_a_few_blanks_are_allowed() -> None:
    """0.04% blanks is an ordinary missing value, not a broken load.

    This is the real case: 204 of 489,743 rows have no recorded arrival delay.
    The check must not fire on it.
    """
    frame = make_small_frame(rows=1000)
    frame.loc[:0, "Arrival Delay in Minutes"] = None

    check_missing_values_are_reasonable("train.csv", frame)
