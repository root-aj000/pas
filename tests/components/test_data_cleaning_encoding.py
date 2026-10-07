"""
Tests for src/components/data_cleaning_encoding.py

Run with: pytest tests/components/test_data_cleaning_encoding.py -v
"""

from typing import Any

import pandas as pd
import pytest

from src import constants as src_constants
from src.components.data_cleaning_encoding import (
    add_arrival_delay_status,
    add_derived_features,
    encode_categorical_columns,
    optionally_clip_outliers,
    split_the_data,
)
from src.constants import SERVICE_RATING_COLUMNS
from src.entity.config_entity import DataCleaningConfig


def make_config(tmp_path, **overrides) -> DataCleaningConfig:
    """Return a small settings object for tests.

    Args:
        tmp_path: pytest's temporary folder, used as the artifacts directory.
        **overrides: Any setting to change from the default.

    Returns:
        A DataCleaningConfig pointing at a temporary folder.
    """
    defaults: dict[str, Any] = {
        "target_column": "satisfaction",
        "features": ["Online boarding"],
        "candidate_features_pending_question": [],
        "derived_feature_names": ["mean_service_rating"],
        "banned_features": frozenset({"id", "satisfaction"}),
        "service_rating_columns": list(SERVICE_RATING_COLUMNS),
        "categorical_columns": ["Gender", "Class"],
        "continuous_columns": ["Age", "Flight Distance"],
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_seed": 42,
        "clip_outliers": False,
        "clip_lower_percentile": 0.1,
        "clip_upper_percentile": 99.9,
        "arrival_delay_median": 0.0,
        "categorical_encoding": "one_hot",
        "route_features_enabled": False,
        "route_smoothing": 20.0,
        "aux_features_enabled": False,
        "categorical_twins_enabled": False,
        "artifacts_dir": tmp_path,
    }
    defaults.update(overrides)
    return DataCleaningConfig(**defaults)


def make_small_frame(rows: int = 100) -> pd.DataFrame:
    """Return a small labelled frame with every column stage 2 reads.

    Args:
        rows: How many rows to build.

    Returns:
        A frame with ids, the label, the ratings and the categoricals.
    """
    data: dict[str, Any] = {
        "id": list(range(rows)),
        # Alternating labels so stratification has something to work with.
        "satisfaction": [bool(index % 2) for index in range(rows)],
        "Age": [20 + (index % 40) for index in range(rows)],
        "Flight Distance": [500 + (index % 2000) for index in range(rows)],
        "Departure Delay in Minutes": [0] * rows,
        "Arrival Delay in Minutes": [float(index % 7) for index in range(rows)],
        "Gender": ["Male" if index % 2 else "Female" for index in range(rows)],
        "Class": ["Business" if index % 3 else "Eco" for index in range(rows)],
    }
    for column in SERVICE_RATING_COLUMNS:
        data[column] = [index % 6 for index in range(rows)]
    return pd.DataFrame(data)


def test_split_produces_three_disjoint_sets(tmp_path) -> None:
    """The split must not overlap, and every row must land somewhere."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=300)

    train, validation, test = split_the_data(frame, config)

    assert len(train) + len(validation) + len(test) == len(frame)
    assert not (set(train["id"]) & set(test["id"]))
    assert not (set(train["id"]) & set(validation["id"]))


def test_split_preserves_the_base_rate(tmp_path) -> None:
    """All three splits must share a base rate, or scores are not comparable."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=300)

    train, validation, test = split_the_data(frame, config)

    rates = [float(part["satisfaction"].mean()) for part in (train, validation, test)]
    assert max(rates) - min(rates) < 0.01


def test_split_is_reproducible_with_the_same_seed(tmp_path) -> None:
    """The same seed must give the same split, every time."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=200)

    first, _, _ = split_the_data(frame, config)
    second, _, _ = split_the_data(frame, config)

    assert list(first["id"]) == list(second["id"])


def test_derived_mean_rating_is_the_average_of_the_thirteen(tmp_path) -> None:
    """mean_service_rating must equal the mean of the 13 ratings, to 6 places."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=10)

    result = add_derived_features(frame, config)

    expected = frame[SERVICE_RATING_COLUMNS].mean(axis=1)
    pd.testing.assert_series_equal(
        result["mean_service_rating"], expected, check_names=False, rtol=1e-6
    )


def test_unknown_arrival_delay_is_its_own_group(tmp_path) -> None:
    """A blank arrival delay must not be called "on time"."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=3)
    frame["Arrival Delay in Minutes"] = [None, 30.0, 0.0]

    result = add_arrival_delay_status(frame, config)

    assert list(result["arrival_delay_status"]) == ["unknown", "delayed", "on_time"]


def test_clipping_is_off_by_default(tmp_path) -> None:
    """Outliers are kept unless config.yaml explicitly asks otherwise."""
    config = make_config(tmp_path, clip_outliers=False)
    frame = make_small_frame(rows=100)

    result = optionally_clip_outliers(frame, config)

    pd.testing.assert_frame_equal(result, frame)


def test_clipping_limits_the_top_and_bottom_percentiles(tmp_path) -> None:
    """When switched on, clipping must actually bound the column."""
    config = make_config(
        tmp_path,
        clip_outliers=True,
        clip_lower_percentile=1.0,
        clip_upper_percentile=99.0,
    )
    frame = make_small_frame(rows=1000)

    result = optionally_clip_outliers(frame, config)

    assert result["Flight Distance"].max() <= frame["Flight Distance"].quantile(0.99)
    assert result["Age"].min() >= frame["Age"].quantile(0.01)


def test_encoding_gives_identical_columns_for_both_frames(tmp_path) -> None:
    """Train and competition must be encoded the same way, or nothing works."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=20)
    categories = {
        "Gender": ["Female", "Male"],
        "Class": ["Business", "Eco"],
    }

    train = encode_categorical_columns(frame, config, categories)
    competition = encode_categorical_columns(frame.head(5), config, categories)

    assert list(train.columns) == list(competition.columns)
    assert "Class_Business" in train.columns


def test_encoding_fills_categories_a_split_has_never_seen(tmp_path) -> None:
    """A category missing from one split must still get a column, filled with 0."""
    config = make_config(tmp_path)
    frame = make_small_frame(rows=10)
    only_every_other = frame.copy()
    only_every_other["Class"] = "Eco"
    categories = {"Gender": ["Female", "Male"], "Class": ["Business", "Eco"]}

    result = encode_categorical_columns(only_every_other, config, categories)

    assert "Class_Business" in result.columns
    assert result["Class_Business"].sum() == 0


def test_raises_when_derived_feature_has_no_calculation(tmp_path) -> None:
    """A derived name with no formula is a config error and must stop the run."""
    config = make_config(tmp_path, derived_feature_names=["not_a_real_column"])
    frame = make_small_frame(rows=10)

    with pytest.raises(ValueError, match="No calculation is defined"):
        add_derived_features(frame, config)


def test_native_encoding_keeps_categoricals_as_category_dtype(tmp_path) -> None:
    """native encoding must keep the four columns, as category dtype."""
    config = make_config(tmp_path, categorical_encoding="native")
    frame = make_small_frame(rows=20)
    categories = {"Gender": ["Female", "Male"], "Class": ["Business", "Eco"]}

    result = encode_categorical_columns(frame, config, categories)

    assert "Class" in result.columns
    assert "Class_Eco" not in result.columns
    assert str(result["Class"].dtype) == "category"


def test_native_encoding_pins_the_same_categories_for_every_split(tmp_path) -> None:
    """Train and test must agree on the category set, or the encoding differs.

    A split that happens to contain only "Eco" must still get a Class_Business
    category, with zero rows in it. Inferring categories per frame is how
    training-serving skew sneaks in through a dtype.
    """
    config = make_config(tmp_path, categorical_encoding="native")
    only_every_other = make_small_frame(rows=10)
    only_every_other["Class"] = "Eco"
    categories = {"Gender": ["Female", "Male"], "Class": ["Business", "Eco"]}

    result = encode_categorical_columns(only_every_other, config, categories)

    assert list(result["Class"].cat.categories) == ["Business", "Eco"]
    assert int((result["Class"] == "Business").sum()) == 0


def test_raises_when_a_category_appears_that_was_never_seen(tmp_path) -> None:
    """An unseen category must stop the run, not silently add a column."""
    config = make_config(tmp_path, categorical_encoding="native")
    frame = make_small_frame(rows=10)
    categories = {"Gender": ["Female", "Male"], "Class": ["Eco"]}

    with pytest.raises(ValueError, match="not seen anywhere"):
        encode_categorical_columns(frame, config, categories)


def test_raises_for_an_unknown_encoding(tmp_path) -> None:
    """A typo in config.yaml must stop the run with a message naming the options."""
    config = make_config(tmp_path, categorical_encoding="nativ")
    frame = make_small_frame(rows=5)
    categories = {"Gender": ["Female", "Male"], "Class": ["Business", "Eco"]}

    with pytest.raises(ValueError, match="Unknown categorical_encoding"):
        encode_categorical_columns(frame, config, categories)



def make_routable_frame(rows: int = 200) -> "pd.DataFrame":
    """Return rows sharing a handful of Flight Distance values.

    Args:
        rows: How many rows to build.

    Returns:
        A frame where five routes repeat, so per-route statistics mean something.
    """
    import pandas as pd

    from src.constants import SERVICE_RATING_COLUMNS

    distances = [500, 500, 1000, 1000, 2000]
    data: dict[str, Any] = {
        "id": list(range(rows)),
        "satisfaction": [bool(index % 2) for index in range(rows)],
        "Age": [30 + (index % 20) for index in range(rows)],
        "Flight Distance": [distances[index % 5] for index in range(rows)],
        "Departure Delay in Minutes": [0] * rows,
        "Arrival Delay in Minutes": [float(index % 5) for index in range(rows)],
        "Gender": ["Male"] * rows,
        "Class": ["Eco"] * rows,
    }
    for column in SERVICE_RATING_COLUMNS:
        data[column] = [3 + (index % 3) for index in range(rows)]
    return pd.DataFrame(data)


def test_route_table_counts_rows_per_route(tmp_path) -> None:
    """Each route's size must equal its row count."""
    from src.components.data_cleaning_encoding import build_route_table

    frame = make_routable_frame(rows=200)
    table, _, _ = build_route_table(
        frame, "Flight Distance", ["Age"], "satisfaction", smoothing=20.0
    )

    assert table.loc[500, "route_size"] == 80
    assert table.loc[1000, "route_size"] == 80
    assert table.loc[2000, "route_size"] == 40


def test_route_target_encoding_blends_toward_the_global_mean(tmp_path) -> None:
    """A tiny route must stay near the global rate, not define its own."""
    import pandas as pd

    from src.components.data_cleaning_encoding import build_route_table

    frame = make_routable_frame(rows=200)
    # One huge route that is all satisfied, one tiny route that is all satisfied.
    big = pd.DataFrame(
        {
            "Flight Distance": [9999] * 1000,
            "Age": [40] * 1000,
            "satisfaction": [True] * 1000,
        }
    )
    tiny = pd.DataFrame(
        {"Flight Distance": [1111] * 2, "Age": [40] * 2, "satisfaction": [True] * 2}
    )
    frame = pd.concat([frame, big, tiny], ignore_index=True)
    _, _, global_rate = build_route_table(
        frame, "Flight Distance", ["Age"], "satisfaction", smoothing=20.0
    )
    table, _, _ = build_route_table(
        frame, "Flight Distance", ["Age"], "satisfaction", smoothing=20.0
    )

    # The 1000-row route is essentially its own rate. The 2-row route is pulled
    # most of the way to the global mean.
    assert table.loc[9999, "route_target_encoded"] > 0.95
    assert abs(table.loc[1111, "route_target_encoded"] - global_rate) < 0.15


def test_oof_target_never_sees_its_own_label(tmp_path) -> None:
    """A row's out-of-fold value must come from other rows only."""
    import pandas as pd

    from src.components.data_cleaning_encoding import add_oof_route_target

    # One route, one row satisfied and the rest not. That row's OOF value must be
    # built without itself, so it must be LOW, not 1.0.
    frame = pd.DataFrame(
        {
            "Flight Distance": [500] * 20,
            "satisfaction": [True] + [False] * 19,
        }
    )
    values = add_oof_route_target(
        frame, "Flight Distance", "satisfaction", smoothing=1.0, seed=42
    )

    assert len(values) == 20
    assert values.iloc[0] < 0.5


def test_unseen_routes_get_globals_not_guesses(tmp_path) -> None:
    """A route never seen in train must get the global mean, never NaN."""
    from src.components.data_cleaning_encoding import (
        apply_route_features,
        build_route_table,
    )

    train_frame = make_routable_frame(rows=200)
    table, global_means, global_rate = build_route_table(
        train_frame, "Flight Distance", ["Age"], "satisfaction", smoothing=20.0
    )
    new_frame = make_routable_frame(rows=10)
    new_frame["Flight Distance"] = 777777

    result = apply_route_features(
        new_frame, table, global_means, global_rate, ["Age"], use_oof_target=None
    )

    assert (result["route_size"] == 0).all()
    assert (result["route_target_encoded"] == global_rate).all()
    assert result["route_mean_Age"].notna().all()


def test_route_feature_names_are_deterministic(tmp_path) -> None:
    """The generated names must be stable, since config.yaml lists them."""
    from src.components.data_cleaning_encoding import route_feature_names

    first = route_feature_names(["Age", "X"])
    second = route_feature_names(["Age", "X"])

    assert first == second
    assert first[:2] == ["route_size", "route_target_encoded"]
    assert "route_mean_Age" in first


def test_twin_names_are_deterministic(tmp_path) -> None:
    """Generated twin names must be stable, since config.yaml lists them."""
    from src.components.data_cleaning_encoding import twin_feature_names

    assert twin_feature_names(["Age", "X"]) == ["Age_cat_", "X_cat_"]
    assert twin_feature_names(["Age", "X"]) == twin_feature_names(["Age", "X"])


def test_twins_hold_the_value_as_a_category(tmp_path) -> None:
    """A twin must carry the integer value as a category, not a number."""
    import pandas as pd

    from src.components.data_cleaning_encoding import add_categorical_twins

    frame = pd.DataFrame({"Age": [25, 30, 25], "Flight Distance": [500, 1000, 500]})
    result = add_categorical_twins(frame, ["Age", "Flight Distance"])

    assert str(result["Age_cat_"].dtype) == "category"
    assert set(result["Age_cat_"].astype(str).unique()) == {"25", "30"}
    assert set(result["Flight Distance_cat_"].astype(str).unique()) == {"500", "1000"}
    # Originals untouched.
    assert str(result["Age"].dtype) == "int64"


def test_twins_do_not_touch_non_numeric_columns(tmp_path) -> None:
    """Only numeric columns get twins. Strings and categories are left alone."""
    import pandas as pd

    from src.components.data_cleaning_encoding import add_categorical_twins

    frame = pd.DataFrame({"Age": [25, 30], "Class": ["Eco", "Business"]})
    result = add_categorical_twins(frame, ["Age"])

    assert "Age_cat_" in result.columns
    assert "Class_cat_" not in result.columns


def _aux_frame(rows: int = 900, seed: int = 0) -> pd.DataFrame:
    """Return a synthetic frame holding the 21 raw columns the aux models read.

    Args:
        rows: How many rows to make.
        seed: Random seed.

    Returns:
        A frame with the 21 CANDIDATE_FEATURE_COLUMNS, so the auxiliary-task
        code can run on it without the competition data.
    """
    import numpy as np

    from src.constants import CANDIDATE_FEATURE_COLUMNS, CATEGORICAL_COLUMNS

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.integers(0, 6, rows) for c in SERVICE_RATING_COLUMNS})
    frame["Age"] = rng.integers(18, 70, rows)
    frame["Flight Distance"] = rng.integers(100, 5000, rows)
    frame["Departure Delay in Minutes"] = rng.integers(0, 60, rows)
    frame["Arrival Delay in Minutes"] = rng.integers(0, 60, rows)
    frame["Gender"] = rng.choice(["Male", "Female"], rows)
    frame["Customer Type"] = rng.choice(["Loyal Customer", "Disloyal Customer"], rows)
    frame["Type of Travel"] = rng.choice(
        ["Business travel", "Personal Travel", "Travel type unclear"], rows
    )
    frame["Class"] = rng.choice(["Business", "Eco", "Premium Economy"], rows)
    # The label follows the ratings, so the predictors have something to learn.
    frame["satisfaction"] = (
        frame[SERVICE_RATING_COLUMNS].mean(axis=1) > rng.uniform(2.5, 3.5)
    ).astype(int)
    assert list(frame.columns) == CANDIDATE_FEATURE_COLUMNS + ["satisfaction"]
    return frame


def test_auxiliary_block_produces_thirty_columns(tmp_path) -> None:
    """The aux block must give 16 aux_p_, 13 aux_ev_ and aux_sum_logp."""
    from src.components.data_cleaning_encoding import (
        add_oof_auxiliary_predictions,
        apply_auxiliary_features,
        aux_feature_names,
        fit_auxiliary_models,
    )

    frame = _aux_frame()
    categories = {
        c: sorted(frame[c].astype(str).unique().tolist())
        for c in src_constants.CATEGORICAL_COLUMNS
    }
    models = fit_auxiliary_models(frame, list(SERVICE_RATING_COLUMNS), categories, 42)
    assert len(models) == 16, "13 ratings plus 3 categoricals"

    result = apply_auxiliary_features(
        frame, models, list(SERVICE_RATING_COLUMNS), categories, None
    )
    names = aux_feature_names(list(SERVICE_RATING_COLUMNS))
    assert len(names) == 30
    assert all(name in result.columns for name in names)
    # A probability, so inside [0, 1]; an expected rating, so inside [0, 5].
    assert result["aux_p_Class"].between(0, 1).all()
    assert result["aux_ev_Food and drink"].between(0, 5).all()
    # A sum of log probabilities, so never positive.
    assert (result["aux_sum_logp"] <= 1e-6).all()

    oof = add_oof_auxiliary_predictions(
        frame, list(SERVICE_RATING_COLUMNS), categories, 42
    )
    assert list(oof.columns) == names


def test_auxiliary_train_values_are_out_of_fold(tmp_path) -> None:
    """Train rows must not be scored by a model that saw them.

    This is the leakage guard. If the out-of-fold path silently fell back to
    in-sample predictions, the auxiliary columns would be far too confident and
    the real model would learn to trust a signal that does not exist on the test
    rows. So the two must disagree noticeably.
    """
    from src.components.data_cleaning_encoding import (
        add_oof_auxiliary_predictions,
        apply_auxiliary_features,
        fit_auxiliary_models,
    )

    frame = _aux_frame(rows=1500, seed=1)
    categories = {
        c: sorted(frame[c].astype(str).unique().tolist())
        for c in src_constants.CATEGORICAL_COLUMNS
    }
    models = fit_auxiliary_models(frame, list(SERVICE_RATING_COLUMNS), categories, 42)
    in_sample = apply_auxiliary_features(
        frame, models, list(SERVICE_RATING_COLUMNS), categories, None
    )
    oof = add_oof_auxiliary_predictions(
        frame, list(SERVICE_RATING_COLUMNS), categories, 42
    )

    for column in ("aux_p_Food and drink", "aux_ev_Food and drink"):
        gap = (in_sample[column] - oof[column]).abs().mean()
        assert gap > 1e-3, f"{column} looks in-sample, not out-of-fold (gap {gap})"
    assert oof["aux_sum_logp"].mean() < in_sample["aux_sum_logp"].mean(), (
        "out-of-fold rows must be less self-consistent than in-sample rows"
    )


def test_auxiliary_columns_arrive_in_the_same_order_everywhere(tmp_path) -> None:
    """All four frames must carry the aux columns in one identical order.

    This is the check that would have caught the 80-minute failure: the
    out-of-fold path built the training columns grouped (all aux_p_, then all
    aux_ev_) while the direct path interleaved them, so the frames agreed on
    which columns existed but not on their order. Stage 2's column-order check
    then rejected the build after every auxiliary model had been fitted.
    """
    from src.components.data_cleaning_encoding import (
        add_oof_auxiliary_predictions,
        apply_auxiliary_features,
        aux_feature_names,
        fit_auxiliary_models,
    )

    frame = _aux_frame(rows=600, seed=2)
    ratings = list(SERVICE_RATING_COLUMNS)
    categories = {
        c: sorted(frame[c].astype(str).unique().tolist())
        for c in src_constants.CATEGORICAL_COLUMNS
    }
    models = fit_auxiliary_models(frame, ratings, categories, 42)
    oof = add_oof_auxiliary_predictions(frame, ratings, categories, 42)

    train_side = apply_auxiliary_features(frame, models, ratings, categories, oof)
    other_side = apply_auxiliary_features(frame, models, ratings, categories, None)

    assert list(train_side.columns) == list(other_side.columns)
    assert list(train_side.columns)[-30:] == aux_feature_names(ratings)
