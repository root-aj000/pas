"""
Tests for src/components/model_training.py

Run with: pytest tests/components/test_model_training.py -v

This file holds Test 2 from .dev/RULES.md rule 10: the overfitting test. It is the
one test with no ordinary equivalent, and it catches the worst class of bug in
machine learning - a model that cannot learn at all.
"""

from typing import Any

import pandas as pd
import pytest

from src.components.model_training import (
    MODEL_REGISTRY,
    add_encoding_parameter,
    build_model,
    run_overfitting_check,
    score_model,
)


def make_tiny_dataset(rows: int = 16) -> tuple[pd.DataFrame, pd.Series]:
    """Return a tiny dataset with a learnable pattern.

    Args:
        rows: How many rows to build. Must be even for the labels to balance.

    Returns:
        The features and the labels.
    """
    features = pd.DataFrame(
        {
            "feature_a": [float(index % 4) for index in range(rows)],
            "feature_b": [float(index % 3) for index in range(rows)],
        }
    )
    labels = pd.Series([bool(index % 2) for index in range(rows)])
    return features, labels


def test_overfitting_check_passes_for_hist_gradient_boosting() -> None:
    """The model must be able to memorise 16 rows.

    If it cannot, training is broken - the learning rate is wrong, or the labels
    and features are mismatched. No amount of real data will hide it.
    """
    assert run_overfitting_check("hist_gradient_boosting", {}, seed=42) is True


def test_overfitting_check_passes_for_random_forest() -> None:
    """The same check for the other tree candidate."""
    assert run_overfitting_check("random_forest", {"n_estimators": 10}, seed=42) is True


def test_overfitting_check_passes_for_logistic_regression() -> None:
    """Logistic regression can fit 16 rows given enough iterations.

    It needs the iterations raised, because the default stops before convergence
    on this little data. That is a property of the test, not a bug.
    """
    assert (
        run_overfitting_check("logistic_regression", {"max_iter": 5000}, seed=42)
        is True
    )


def test_overfitting_check_fails_when_no_split_is_possible() -> None:
    """The check must fail when the settings make 16 rows unfittable.

    This is the test proving the test works. A check that cannot fail is not a
    check.

    min_samples_leaf=20 on a 16-row dataset makes every split illegal, so the
    model can only predict the average. scikit-learn accepts the setting, so this
    is a genuine failure of the model rather than a rejected argument.
    """
    with pytest.raises(AssertionError, match="Overfitting check failed"):
        run_overfitting_check(
            "hist_gradient_boosting", {"min_samples_leaf": 20}, seed=42
        )


def test_a_rejected_parameter_surfaces_as_scikit_learns_own_error() -> None:
    """scikit-learn rejects learning_rate=0.0 before training starts.

    Recorded so nobody later mistakes scikit-learn's input validation for our
    overfitting check working. It is a different failure arriving earlier.
    """
    with pytest.raises(Exception, match="learning_rate"):
        run_overfitting_check("hist_gradient_boosting", {"learning_rate": 0.0}, seed=42)


def test_every_registered_model_can_be_built_with_a_seed() -> None:
    """Swapping model must be possible, or MODEL_REGISTRY is decoration."""
    for name in MODEL_REGISTRY:
        model = build_model(name, {}, seed=42)
        assert model is not None


def test_raises_with_the_valid_names_for_an_unknown_model() -> None:
    """A typo in config.yaml must stop the run and list what is valid."""
    with pytest.raises(KeyError, match="hist_gradient_boosting"):
        build_model("gradient_booster", {}, seed=42)


def test_decision_tree_accepts_no_random_state() -> None:
    """DecisionTreeClassifier has no random_state, so build_model must not pass one.

    Passing it anyway would raise a TypeError that says nothing useful.
    """
    model: Any = build_model("decision_tree", {"max_depth": 3}, seed=42)
    assert model.max_depth == 3


def test_score_model_returns_all_five_metrics() -> None:
    """Every reported metric must be present, so runs are measured the same way."""
    features, labels = make_tiny_dataset(rows=200)
    model = build_model("hist_gradient_boosting", {}, seed=42)
    model.fit(features, labels)

    scores = score_model(model, features, labels)

    assert set(scores) == {"roc_auc", "accuracy", "f1", "precision", "recall"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_native_encoding_is_only_valid_for_models_that_read_categories() -> None:
    """A scikit-learn model cannot read `category` dtype, and must say so clearly.

    The message matters: without it the failure surfaces as a dtype error inside
    fit(), which points at the model rather than at the config setting.
    """
    with pytest.raises(ValueError, match="Only .* can read"):
        build_model("logistic_regression", {}, seed=42, categorical_encoding="native")


def test_native_encoding_is_valid_for_realmlp() -> None:
    """RealMLP embeds categories internally, so native encoding is fine for it."""
    model = build_model(
        "realmlp", {"n_ens": 1, "n_epochs": 1}, seed=42, categorical_encoding="native"
    )
    assert type(model).__name__ == "RealMLP_TD_Classifier"


def test_native_encoding_adds_enable_categorical_for_xgboost() -> None:
    """The flag must be derived from the encoding, not set twice in two places."""
    model: Any = build_model(
        "xgboost", {"n_estimators": 5}, seed=42, categorical_encoding="native"
    )
    assert model.get_params()["enable_categorical"] is True


def test_one_hot_does_not_add_the_enable_categorical_flag() -> None:
    """One-hot data has no category columns, so the helper adds nothing.

    Tested on the helper rather than on get_params() because xgboost 3.x already
    defaults enable_categorical to True, so reading it back off a built model
    cannot tell who set it.
    """
    params = add_encoding_parameter("xgboost", {"n_estimators": 5}, "one_hot")
    assert "enable_categorical" not in params


def test_the_helper_rejects_an_unknown_encoding() -> None:
    """A typo in config.yaml must stop the run with the valid options listed."""
    with pytest.raises(ValueError, match="Supported: 'native', 'one_hot'"):
        add_encoding_parameter("xgboost", {}, "nativ")


def test_one_hot_leaves_a_sklearn_model_untouched() -> None:
    """Only xgboost ever gets the flag, whatever the encoding."""
    assert add_encoding_parameter(
        "hist_gradient_boosting", {"max_iter": 5}, "one_hot"
    ) == {"max_iter": 5}
