"""
Tests for src/components/model_evaluation.py

Run with: pytest tests/components/test_model_evaluation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.components.model_evaluation import (
    calculate_metrics,
    read_errors,
    write_submission,
)
from src.utils.common import check_columns


def test_confusion_matrix_counts_add_up_to_the_row_count() -> None:
    """Every row must land in exactly one of the four cells.

    This is the check that catches a transposed confusion matrix, which is the
    most common way an evaluation report quietly lies.
    """
    target = pd.Series([True, True, False, False, True, False])
    probabilities = np.array([0.9, 0.2, 0.8, 0.1, 0.6, 0.4])
    predictions = (probabilities >= 0.5).astype(np.int64)

    metrics = calculate_metrics(target, probabilities, predictions)

    total = (
        metrics.confusion_true_positive
        + metrics.confusion_false_positive
        + metrics.confusion_true_negative
        + metrics.confusion_false_negative
    )
    assert total == len(target)
    assert metrics.confusion_true_positive == 2
    assert metrics.confusion_false_positive == 1
    assert metrics.confusion_false_negative == 1
    assert metrics.confusion_true_negative == 2


def test_perfect_predictions_score_one() -> None:
    """A perfect model must score 1.0 on every metric."""
    target = pd.Series([True, False, True, False])
    probabilities = np.array([1.0, 0.0, 1.0, 0.0])
    predictions = (probabilities >= 0.5).astype(np.int64)

    metrics = calculate_metrics(target, probabilities, predictions)

    assert metrics.roc_auc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0


def test_roc_auc_uses_the_probabilities_not_the_threshold() -> None:
    """AUC measures ranking, so the threshold must not change it.

    Two models with identical hard labels but different confidence must get
    different AUCs. If they do not, the metric is being computed on the wrong
    thing.
    """
    target = pd.Series([True, True, False, False])

    confident = np.array([0.9, 0.8, 0.2, 0.1])
    unconfident = np.array([0.51, 0.49, 0.49, 0.51])

    confident_metrics = calculate_metrics(
        target, confident, (confident >= 0.5).astype(np.int64)
    )
    unconfident_metrics = calculate_metrics(
        target, unconfident, (unconfident >= 0.5).astype(np.int64)
    )

    assert confident_metrics.roc_auc > unconfident_metrics.roc_auc


def test_read_errors_returns_only_mistakes() -> None:
    """The error table must contain wrong rows, and say how confident they were."""
    frame = pd.DataFrame({"id": [1, 2, 3, 4]})
    target = pd.Series([True, True, False, False])
    probabilities = np.array([0.95, 0.05, 0.90, 0.10])
    # id 1 -> 0.95 satisfied (right), id 2 -> 0.05 (wrong), id 3 -> 0.90 (wrong),
    # id 4 -> 0.10 (right).

    errors = read_errors(frame, target, probabilities)

    # A 0.5 threshold on [0.95, 0.05, 0.90, 0.10] calls ids 1 and 3 satisfied.
    # id 1 was satisfied, so it is right. id 3 was not, so it is wrong. id 2 was
    # satisfied but was called not. So the two mistakes are ids 2 and 3.
    assert set(errors["id"]) == {2, 3}
    assert len(errors) == 2


def test_submission_row_count_must_match_the_template(tmp_path) -> None:
    """A submission with the wrong number of rows scores nothing, so it must stop."""
    template_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2, 3], "satisfaction": [0.5] * 3}).to_csv(
        template_path, index=False
    )
    competition = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError, match="wrong number of rows"):
        write_submission(
            competition,
            np.array([0.1, 0.9]),
            template_path,
            tmp_path / "submission.csv",
        )


def test_submission_columns_must_match_the_template(tmp_path) -> None:
    """Kaggle needs exactly `id` and `satisfaction`, in that order."""
    template_path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2], "satisfaction": [0.5, 0.5]}).to_csv(
        template_path, index=False
    )
    competition = pd.DataFrame({"id": [1, 2]})

    rows = write_submission(
        competition, np.array([0.1, 0.9]), template_path, tmp_path / "submission.csv"
    )

    assert rows == 2
    written = pd.read_csv(tmp_path / "submission.csv")
    assert list(written.columns) == ["id", "satisfaction"]
    check_columns("submission", set(written.columns), {"id", "satisfaction"})
