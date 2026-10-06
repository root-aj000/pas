"""
Splits the data, builds the derived features, and encodes the categories.

Input:  artifacts/data_ingestion/raw_train.csv, artifacts/data_ingestion/raw_test.csv
Output: artifacts/data_cleaning_encoding/train.csv, validation.csv, test.csv,
        competition_test.csv, features.json, split_report.csv

Run with: python -m src.pipeline.stage_02_data_cleaning_encoding

Three jobs, in this order, and the order matters:

1. Split first, before looking at anything. Once you have seen the test data you
   cannot un-look it. See .lead/01-DATA.md Step 1.7.
2. Build the derived features. The strongest derived column, mean_service_rating,
   has the second-best effect size in the whole dataset.
3. One-hot encode the short categoricals.

The split is a stratified random one because this dataset has no date column and
no passenger identifier, so there is no time axis to cut on and no entity to hold
out. The reasoning is written down in docs/split_plan.md.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    CANDIDATE_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    SERVICE_RATING_COLUMNS,
)
from src.entity.config_entity import DataCleaningArtifact, DataCleaningConfig
from src.utils.common import check_columns, log_step, save_dataframe, save_json

# The auxiliary-task targets, verbatim from the reference: the 13 ratings plus the
# three short categoricals that carry real signal. Gender is excluded - it is
# balanced and uninformative, and the reference left it out too.
AUX_TARGETS: list[str] = SERVICE_RATING_COLUMNS + [
    "Class",
    "Type of Travel",
    "Customer Type",
]


def add_derived_features(
    frame: pd.DataFrame,
    config: DataCleaningConfig,
) -> pd.DataFrame:
    """Add the derived columns this project computes rather than reads.

    Args:
        frame: Rows to add columns to. Not modified.
        config: Names the rating columns and which derived columns to build.

    Returns:
        A new frame with the derived columns added.

    Note:
        Each derived column's meaning is in config.yaml, next to its name, so the
        documentation and the calculation cannot drift apart.
    """
    enriched = frame.copy()
    ratings = config.service_rating_columns

    for name in config.derived_feature_names:
        if name == "mean_service_rating":
            enriched[name] = enriched[ratings].mean(axis=1)
        elif name == "worst_service_rating":
            enriched[name] = enriched[ratings].min(axis=1)
        elif name == "count_of_ratings_at_or_below_2":
            enriched[name] = (enriched[ratings] <= 2).sum(axis=1)
        else:
            raise ValueError(
                f"No calculation is defined for derived feature '{name}'. "
                f"Known names: mean_service_rating, worst_service_rating, "
                "count_of_ratings_at_or_below_2. Add the calculation here and the "
                "description in config.yaml."
            )
    log_step(
        "derive",
        rows_in=len(frame),
        rows_out=len(enriched),
        added=[name for name in config.derived_feature_names],
    )
    return enriched


def add_arrival_delay_status(
    frame: pd.DataFrame, config: DataCleaningConfig
) -> pd.DataFrame:
    """Add the three-way arrival delay group, keeping unknown as its own value.

    Args:
        frame: Rows to add a column to. Not modified.
        config: Names the arrival delay column.

    Returns:
        A new frame with `arrival_delay_status` added.

    Note:
        204 rows in the train split have no recorded arrival delay, and we do not
        know what a blank means: an on-time flight, a value never captured, or a
        cancellation. Calling a blank flight "on time" would be a decision
        presented as a fact, so unknown is kept as its own group.
        See docs/column_dictionary.md Note 1.
    """
    enriched = frame.copy()
    delay_column = "Arrival Delay in Minutes"
    enriched["arrival_delay_status"] = np.select(
        [
            enriched[delay_column].isna(),
            enriched[delay_column] > 0,
        ],
        ["unknown", "delayed"],
        default="on_time",
    )
    shares = (
        enriched["arrival_delay_status"].value_counts(normalize=True).round(4).to_dict()
    )
    log_step("arrival_delay_status", rows_in=len(frame), shares=shares)
    return enriched


def route_feature_names(numeric_columns: list[str]) -> list[str]:
    """Return the deterministic names of every generated route column.

    Args:
        numeric_columns: The numeric columns a route profile is built from.

    Returns:
        `route_size`, `route_target_encoded`, then one `route_mean_<column>` per
        numeric column, in that order.

    Note:
    This is the single source of truth for generated names. The builder and the
    checker both read it, so they cannot disagree about what exists.
    """
    return (
        ["route_size", "route_target_encoded"]
        + [f"route_mean_{column}" for column in numeric_columns]
        + [f"routediff_{column}" for column in numeric_columns]
    )


DIGIT_COLUMNS: list[str] = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def digit_feature_names(columns: list[str] | None = None) -> list[str]:
    """Return the deterministic names of the digit features.

    Args:
        columns: Which numeric columns to decompose. Defaults to the four that
            carry a magnitude.

    Returns:
        One `<column>_d-4` .. `<column>_d3` per column, eight each.

    Note:
    (v // 10**k) % 10, for k from -4 to 3. A synthetic generator often writes
    values with structure in their digits that a model reading the number as a
    single quantity cannot see - and the raw value is one column while its digits
    are eight. Reference measured +0.000211 for this block on its own.
    """
    used = DIGIT_COLUMNS if columns is None else columns
    return [f"{column}_d{k}" for column in used for k in range(-4, 4)]


def add_digit_features(
    frame: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Split each numeric column into its decimal digits, one column per place.

    Args:
        frame: Rows to enrich. Not modified.
        columns: Which numeric columns to decompose.

    Returns:
        A new frame with eight int8 columns per input column.

    Note:
    Deterministic arithmetic: no fitting, no label, nothing to leak. For
    k >= 0 this is the k-th digit from the right; for k < 0, the k-th from the
    left of the fractional part. Negative powers are exact in float64, so the
    fractional digits are not rounded away.
    """
    enriched = frame.copy()
    for column in DIGIT_COLUMNS if columns is None else columns:
        value = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
        for k in range(-4, 4):
            enriched[f"{column}_d{k}"] = (value // (10.0**k) % 10).astype("int8")
    return enriched


def frequency_feature_names(columns: list[str]) -> list[str]:
    """Return the deterministic names of the frequency features.

    Args:
        columns: Which columns get a frequency and a rarity.

    Returns:
        One `freq_<column>` and one `rarity_<column>` per column.
    """
    names: list[str] = []
    for column in columns:
        names.append(f"freq_{column}")
        names.append(f"rarity_{column}")
    return names


def add_frequency_features(
    frame: pd.DataFrame, columns: list[str], counts: dict[str, pd.Series]
) -> pd.DataFrame:
    """Add how common, and how rare, each value is.

    Args:
        frame: Rows to enrich. Not modified.
        columns: Which columns get a frequency and a rarity.
        counts: Column name to its value_counts, fitted on the train split.

    Returns:
        A new frame with two float32 columns per input column.

    Note:
    A rate column that most rows have never seen is telling you something the
    rate itself cannot. Rarity is -log(frequency), so a value seen once sits far
    from a value seen ten thousand times, where a plain frequency would squash
    both near zero. Reference measured +0.000128 for the block. Label-free: the
    counts come from the train split's own column.
    """
    enriched = frame.copy()
    total = max(len(frame), 1)
    for column in columns:
        share = frame[column].map(counts[column]).fillna(0.0) / total
        enriched[f"freq_{column}"] = share.astype("float32")
        enriched[f"rarity_{column}"] = (-np.log(share.clip(lower=1.0 / total))).astype(
            "float32"
        )
    return enriched


def count_feature_names(numeric_columns: list[str]) -> list[str]:
    """Return the deterministic names of the value-count features.

    Args:
        numeric_columns: Every numeric column getting a count.

    Returns:
        One `cnt_<column>` per numeric column, in that order.
    """
    return [f"cnt_{column}" for column in numeric_columns]


def add_value_counts(
    frame: pd.DataFrame,
    counts: dict[str, pd.Series],
) -> pd.DataFrame:
    """Add a column per numeric, holding how often that value occurs.

    Args:
        frame: Rows to enrich. Not modified.
        counts: Column name to its value_counts, fitted on the train split.

    Returns:
        A new frame with one `cnt_<column>` float column per numeric.

    Note:
    No fitting and no label: a value's frequency is a property of the data, not
    of the target, so the train-split table is safe for every frame. Flight
    Distance alone has 3,474 distinct values over about a million rows, so
    "this route is common" is real information the network cannot build for
    itself from a normalised distance. The reference measured +0.00007 for the
    value counts on top of the route profile and the target encodings.
    """
    enriched = frame.copy()
    for column, table in counts.items():
        enriched[f"cnt_{column}"] = frame[column].map(table).fillna(0).astype("float32")
    return enriched


def build_route_table(
    train_frame: pd.DataFrame,
    route_key: str,
    numeric_columns: list[str],
    target_column: str,
    smoothing: float,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    """Build per-route statistics from the training rows only.

    Args:
        train_frame: Training rows, with the route key and numeric columns.
        route_key: The column identifying the route. Flight Distance here.
        numeric_columns: Columns to average per route.
        target_column: The label, used only for the smoothed rate.
        smoothing: Blend weight toward the global mean. A route with fewer rows
            than this is pulled most of the way to the overall rate.

    Returns:
        The route table (one row per route), the global column means, and the
        global satisfaction rate. Unseen routes get the globals, never a guess.

    Note:
    The profile and the counts touch no label, so they are safe to compute on the
    full train set. The target rate DOES use the label, so train rows must NOT
    take their values from this table - they get out-of-fold values from
    add_oof_route_target instead. That separation is the whole leakage story.
    """
    grouped = train_frame.groupby(route_key, observed=True)
    table = pd.DataFrame({"route_size": grouped.size()})
    for column in numeric_columns:
        table[f"route_mean_{column}"] = grouped[column].mean()

    global_rate = float(train_frame[target_column].mean())
    route_sums = grouped[target_column].sum()
    route_counts = grouped.size()
    table["route_target_encoded"] = (route_sums + smoothing * global_rate) / (
        route_counts + smoothing
    )

    global_means = {
        column: float(train_frame[column].mean()) for column in numeric_columns
    }
    return table, global_means, global_rate


def apply_route_features(
    frame: pd.DataFrame,
    route_table: pd.DataFrame,
    global_means: dict[str, float],
    global_rate: float,
    numeric_columns: list[str],
    use_oof_target: pd.Series | None,
) -> pd.DataFrame:
    """Add route columns to any frame from the train-built table.

    Args:
        frame: Rows to enrich. Must carry the route key.
        route_table: Per-route statistics from build_route_table.
        global_means: Global column means, for routes never seen in train.
        global_rate: Global satisfaction rate, same purpose.
        numeric_columns: Columns the profile covers.
        use_oof_target: Out-of-fold target values for train rows, or None to use
            the full-train mapping. Train rows must pass their OOF values; every
            other frame passes None.

    Returns:
        A new frame with route_size, route_target_encoded and the profile means.
    """
    enriched = frame.copy()
    route_key = "Flight Distance"

    # Left join on the route. Rows whose route never appeared in train get NaN,
    # which is filled with the globals below - explicitly, not by accident.
    joined = enriched.merge(
        route_table,
        left_on=route_key,
        right_index=True,
        how="left",
        suffixes=("", "_route"),
    )

    joined["route_size"] = joined["route_size"].fillna(0).astype(int)
    for column in numeric_columns:
        profile_column = f"route_mean_{column}"
        joined[profile_column] = joined[profile_column].fillna(global_means[column])
        # The residual: how unusual this passenger is for the route they flew.
        # The route mean says what a typical passenger on this route answers; the
        # residual says this one differs. Reference measured +0.000128 for the
        # profile block and this is the part of it we were missing.
        #
        # `fillna(0.0)` is load-bearing, not tidiness: `Arrival Delay in Minutes`
        # is absent on 204 training rows, and the route mean is a groupby mean so
        # it is still defined there - the subtraction was NaN and took the run
        # down at `check_feature_list`. Zero means "no deviation recorded", which
        # is the honest reading of a missing observation: the model is not told
        # the row is average for its route, it is told nothing about it. 0.04% of
        # rows, so the choice between this and any other neutral value is not
        # measurable; what matters is that it is not NaN.
        joined[f"routediff_{column}"] = (
            joined[column] - joined[profile_column]
        ).fillna(0.0).astype("float32")

    if use_oof_target is not None:
        # Train rows take their out-of-fold values. No row ever sees its own label.
        joined["route_target_encoded"] = use_oof_target.to_numpy()
    else:
        joined["route_target_encoded"] = joined["route_target_encoded"].fillna(
            global_rate
        )

    unseen = int((joined["route_size"] == 0).sum())
    if unseen:
        import logging as _logging

        _logging.getLogger("pipeline").info(
            "[route] %d rows whose route never appeared in train, given globals", unseen
        )
    return joined


def add_oof_route_target(
    train_frame: pd.DataFrame,
    route_key: str,
    target_column: str,
    smoothing: float,
    seed: int,
    folds: int = 5,
) -> pd.Series:
    """Compute out-of-fold target encodings for the training rows.

    Args:
        train_frame: Training rows.
        route_key: The route identifier.
        target_column: The label.
        smoothing: Blend weight toward the fold's global mean.
        seed: Random seed for the fold split.
        folds: How many folds. Five, matching the rest of the project.

    Returns:
        One smoothed rate per training row, from a model that never saw that row.

    Note:
    Each row's value comes from the other four fifths. A row that saw its own
    label in its route mean would leak - the model would learn "rows like me are
    satisfied" from itself. Five folds is the standard cost for preventing that.
    """
    from sklearn.model_selection import StratifiedKFold

    values = pd.Series(index=train_frame.index, dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = train_frame[target_column]
    for fitting_index, scoring_index in splitter.split(train_frame, labels):
        fitting = train_frame.iloc[fitting_index]
        scoring = train_frame.iloc[scoring_index]
        fold_global = float(fitting[target_column].mean())
        grouped = fitting.groupby(route_key, observed=True)[target_column].agg(
            ["sum", "size"]
        )
        mapping = (grouped["sum"] + smoothing * fold_global) / (
            grouped["size"] + smoothing
        )
        values.iloc[scoring_index] = (
            scoring[route_key].map(mapping).fillna(fold_global).to_numpy()
        )
    return values


def optionally_clip_outliers(
    frame: pd.DataFrame, config: DataCleaningConfig
) -> pd.DataFrame:
    """Clip the continuous columns, if config.yaml asks for it.

    Args:
        frame: Rows to clip. Not modified.
        config: Says whether to clip, at which percentiles, and which columns.

    Returns:
        A new frame, clipped or unchanged.

    Note:
        Clipping is OFF by default and that is a measured decision, not an
        oversight. It changes validation ROC-AUC by +0.000076, which is noise, and
        it would invent a ceiling the world does not have. The setting stays so
        the experiment can be re-run. See docs/outlier_findings.md.
    """
    if not config.clip_outliers:
        log_step(
            "clip_outliers",
            applied=False,
            reason="measured unnecessary, see docs/outlier_findings.md",
        )
        return frame

    clipped = frame.copy()
    for column in config.continuous_columns:
        lower = float(frame[column].quantile(config.clip_lower_percentile / 100))
        upper = float(frame[column].quantile(config.clip_upper_percentile / 100))
        clipped[column] = clipped[column].clip(lower=lower, upper=upper)
    log_step(
        "clip_outliers",
        applied=True,
        columns=len(config.continuous_columns),
        lower_percentile=config.clip_lower_percentile,
        upper_percentile=config.clip_upper_percentile,
    )
    return clipped


def split_the_data(
    frame: pd.DataFrame, config: DataCleaningConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, validation and test, stratified on the label.

    Args:
        frame: All labelled rows.
        config: The target column, the two split sizes, and the seed.

    Returns:
        The train, validation and test frames, in that order.

    Raises:
        ValueError: If the two sizes do not leave any rows for training, or the
            target column is missing.
    """
    check_columns("split", set(frame.columns), {config.target_column})
    remaining = 1 - config.test_size
    validation_share_of_remaining = config.validation_size / remaining

    holdout_and_train, test_frame = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=frame[config.target_column],
    )
    train_frame, validation_frame = train_test_split(
        holdout_and_train,
        test_size=validation_share_of_remaining,
        random_state=config.random_seed,
        stratify=holdout_and_train[config.target_column],
    )
    if train_frame.empty:
        raise ValueError(
            f"Split sizes leave 0 training rows: test_size={config.test_size}, "
            f"validation_size={config.validation_size}."
        )

    # The base rate must be the same in all three, or scores wobble for a reason
    # that has nothing to do with the model.
    rates = {
        "train": float(train_frame[config.target_column].mean()),
        "validation": float(validation_frame[config.target_column].mean()),
        "test": float(test_frame[config.target_column].mean()),
    }
    spread = max(rates.values()) - min(rates.values())
    if spread > 0.01:
        raise ValueError(
            f"The three splits do not share a base rate: {rates}. Spread {spread:.4f} "
            "exceeds 1%, so scores would not be comparable between splits."
        )

    overlap = set(train_frame["id"]) & set(test_frame["id"])
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both train and test")

    log_step(
        "split",
        rows_in=len(frame),
        rows_out=len(train_frame),
        train_rows=len(train_frame),
        validation_rows=len(validation_frame),
        test_rows=len(test_frame),
        base_rate_spread=round(spread, 5),
        seed=config.random_seed,
    )
    return train_frame, validation_frame, test_frame


TE_COLUMNS: list[tuple[str, ...]] = [
    ("Flight Distance",),
    ("Age",),
    ("Flight Distance", "Age"),
    ("Flight Distance", "Class"),
    ("Flight Distance", "Type of Travel"),
    ("Age", "Class", "Type of Travel", "Customer Type"),
]


def _composite(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    """Join several columns into one string key, so groupby works on the tuple.

    Args:
        frame: Rows holding the key columns.
        keys: Which columns to join.

    Returns:
        One composite string per row. NaNs become the string "__nan__", so they
        form their own group rather than being dropped.
    """
    joined = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    for key in keys:
        joined = joined + frame[key].astype(object).astype(str) + "|"
    return joined


def target_encoding_column_name(keys: tuple[str, ...]) -> str:
    """Return the feature column name for a key set.

    Args:
        keys: The key columns, e.g. ("Flight Distance", "Age").

    Returns:
        `te_Flight Distance|Age` style, matching the reference codebook.
    """
    # Match the reference column-naming exactly (FD is their alias for Flight
    # Distance), so our numbers compare without a rename pass through the results.
    alias = {"Flight Distance": "FD"}
    prefixes = [alias.get(k, k[:6]) for k in keys]
    return "te_" + "|".join(prefixes)


def add_target_encodings(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    keys: tuple[str, ...],
    target_column: str,
    smoothing: float,
    use_oof: bool,
    seed: int,
    folds: int = 5,
) -> pd.Series:
    """Append one target-encoded column for a key set.

    Args:
        frame: Rows to enrich.
        train_frame: The rows the encoding is fitted on. Must be train.csv rows,
            because train rows must be encoded out-of-fold.
        keys: The grouping columns.
        target_column: The label.
        smoothing: Blend weight toward the prior, matching the reference (20.0).
        use_oof: If True, each row's value comes from the other folds only. Use
            True for train_frame, False for every other frame.
        seed: Random seed for the inner fold.
        folds: Five, matching the rest of the project.

    Returns:
        The encoded values as a float64 Series, in row order.
    """
    prior = float(train_frame[target_column].mean())

    def _map_one(fit_frame: pd.DataFrame, apply_keys: pd.Series) -> pd.Series:
        grouped = pd.DataFrame(
            {"k": _composite(fit_frame, keys), "y": fit_frame[target_column]}
        )
        agg = grouped.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + prior * smoothing) / (agg["count"] + smoothing)
        return apply_keys.map(enc).fillna(prior)

    if use_oof:
        from sklearn.model_selection import StratifiedKFold

        out = pd.Series(index=frame.index, dtype=float)
        labels = train_frame[target_column]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fitting_index, scoring_index in splitter.split(train_frame, labels):
            fitting = train_frame.iloc[fitting_index]
            scoring = frame.iloc[scoring_index]
            out.iloc[scoring_index] = _map_one(
                fitting, _composite(scoring, keys)
            ).to_numpy()
        return out

    return _map_one(train_frame, _composite(frame, keys)).astype(float)


def add_target_encodings_to_all(
    enriched_frames: dict[str, pd.DataFrame], config: DataCleaningConfig
) -> dict[str, pd.DataFrame]:
    """Add the six route/target-encoding columns to every frame.

    Args:
        enriched_frames: The four frames, keyed by split. Train must carry the label.
        config: Gives the seed and the smoothing default.

    Returns:
        The four frames with `te_...` columns appended.

    Note:
    Train gets out-of-fold values; the other three take the full-train mapping.
    Six keys, six columns. The FD-only key already exists as
    route_target_encoded from the route block; we keep both because the fold-safe
    validation shows FD-only being redundant with the route stat, but the
    multi-column keys add information one column cannot express.
    """
    smoothing = config.route_smoothing
    train_frame = enriched_frames["train"]
    for keys in TE_COLUMNS:
        column_name = target_encoding_column_name(keys)
        enriched_frames["train"][column_name] = add_target_encodings(
            train_frame,
            train_frame,
            keys,
            config.target_column,
            smoothing,
            True,
            config.random_seed,
        )
        for name in ("validation", "test", "competition"):
            enriched_frames[name][column_name] = add_target_encodings(
                enriched_frames[name],
                train_frame,
                keys,
                config.target_column,
                smoothing,
                False,
                config.random_seed,
            )
        log_step("target_encoding", column=column_name, smoothing=smoothing)
    return enriched_frames


def add_route_features_to_all(
    enriched_frames: dict[str, pd.DataFrame], config: DataCleaningConfig
) -> dict[str, pd.DataFrame]:
    """Add route-ID columns to every frame from a train-built table.

    Args:
        enriched_frames: The four frames after deriving and clipping, keyed by
            "train", "validation", "test" and "competition".
        config: Says whether route features are on, and the smoothing weight.

    Returns:
        The four frames with route_size, route_target_encoded and the per-route
        profile means added.

    Note:
    The profile and the counts touch no label, so the full-train table is safe
    for every frame. The target rate uses the label, so train rows get
    out-of-fold values and every other frame gets the full-train mapping. That
    separation - not the smoothing, not the fold count - is what prevents leakage.
    """
    numeric_columns = (
        list(config.service_rating_columns)
        + ["Age"]
        + [c for c in config.continuous_columns if c != "Flight Distance"]
    )
    # Flight Distance IS the route key, so it profiles everything except itself.
    numeric_columns = [c for c in numeric_columns if c != "Flight Distance"]

    count_columns = list(config.continuous_columns)
    counts = {c: enriched_frames["train"][c].value_counts() for c in count_columns}
    enriched_frames = {
        name: add_value_counts(frame, counts) for name, frame in enriched_frames.items()
    }
    log_step("value_counts", columns=len(count_columns))

    # Digits, then frequency and rarity of every raw column. Both are arithmetic
    # on columns already present, fitted on the train split's own values, so no
    # label is involved and nothing can leak.
    enriched_frames = {
        name: add_digit_features(frame) for name, frame in enriched_frames.items()
    }
    log_step("digit_features", columns=len(digit_feature_names()))

    frequency_columns = [
        c for c in CANDIDATE_FEATURE_COLUMNS if c in enriched_frames["train"].columns
    ]
    frequency_counts = {
        c: enriched_frames["train"][c].astype(str).value_counts()
        for c in frequency_columns
    }
    enriched_frames = {
        name: add_frequency_features(frame, frequency_columns, frequency_counts)
        for name, frame in enriched_frames.items()
    }
    log_step(
        "frequency_features", columns=len(frequency_feature_names(frequency_columns))
    )

    train_frame = enriched_frames["train"]
    route_table, global_means, global_rate = build_route_table(
        train_frame,
        "Flight Distance",
        numeric_columns,
        config.target_column,
        config.route_smoothing,
    )
    log_step(
        "route_table",
        routes=len(route_table),
        smoothing=config.route_smoothing,
        profile_columns=len(numeric_columns),
    )

    oof_target = add_oof_route_target(
        train_frame,
        "Flight Distance",
        config.target_column,
        config.route_smoothing,
        config.random_seed,
    )

    out: dict[str, pd.DataFrame] = {}
    for name, frame in enriched_frames.items():
        out[name] = apply_route_features(
            frame,
            route_table,
            global_means,
            global_rate,
            numeric_columns,
            use_oof_target=oof_target if name == "train" else None,
        )
    log_step(
        "route_features",
        added=len(route_feature_names(numeric_columns)),
        train_rows=len(out["train"]),
    )
    return out


def aux_feature_names(rating_columns: list[str]) -> list[str]:
    """Return the deterministic names of the 30 auxiliary-task features.

    Args:
        rating_columns: The 13 service ratings.

    Returns:
        16 `aux_p_<target>` (the model's probability of the row's own value),
        13 `aux_ev_<rating>` (the model's expected value for that rating), and
        `aux_sum_logp`. Thirty columns, in that order.
    """
    names = [f"aux_p_{column}" for column in AUX_TARGETS]
    names += [f"aux_ev_{column}" for column in rating_columns]
    return names + ["aux_sum_logp"]


def _aux_codes(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Integer-code the raw columns the auxiliary models read.

    Args:
        frame: Rows to code.
        categories: Column name to its allowed values, fixed from the train split
            so every frame codes identically.

    Returns:
        A frame holding only CANDIDATE_FEATURE_COLUMNS, with the four
        categoricals as integer codes and everything else as-is.
    """
    coded = pd.DataFrame(index=frame.index)
    for column in CANDIDATE_FEATURE_COLUMNS:
        if column in categories:
            coded[column] = pd.Categorical(
                frame[column], categories=categories[column]
            ).codes
        else:
            coded[column] = frame[column]
    return coded


def fit_auxiliary_models(
    train_frame: pd.DataFrame,
    rating_columns: list[str],
    categories: dict[str, list[str]],
    seed: int,
) -> dict[str, Any]:
    """Fit one small model per auxiliary target, predicting it from the rest.

    Args:
        train_frame: Training rows.
        rating_columns: The 13 ratings, which also get an expected-value column.
        categories: Column name to allowed values, for integer-coding.
        seed: Random seed.

    Returns:
        Target name to (model, inputs).

    Note:
    The predictors read only CANDIDATE_FEATURE_COLUMNS - the 21 raw columns, minus
    the one being predicted. Not the engineered features. The reference measured
    twelve variants of this and the rule was consistent: an expected value helps
    exactly when it comes from the other 20 raw columns and nothing else. Feeding
    it the route profile or the target encodings makes it worse than not having it,
    because the extra information overlaps what the label model already knows.
    """
    from xgboost import XGBClassifier

    coded = _aux_codes(train_frame, categories)
    models: dict[str, Any] = {}
    for column in AUX_TARGETS:
        inputs = [c for c in CANDIDATE_FEATURE_COLUMNS if c != column]
        codes, _ = _aux_codes_and_values(coded[column])
        objective = (
            "binary:logistic" if len(np.unique(codes)) <= 2 else "multi:softprob"
        )
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            objective=objective,
            random_state=seed,
        )
        model.fit(coded[inputs], codes)
        models[column] = (model, inputs)
    return models


def apply_auxiliary_features(
    frame: pd.DataFrame,
    aux_models: dict[str, Any],
    rating_columns: list[str],
    categories: dict[str, list[str]],
    precomputed: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add the 30 auxiliary-task columns to any frame.

    Args:
        frame: Rows to enrich.
        aux_models: Target name to (model, inputs), from fit_auxiliary_models.
        rating_columns: The 13 ratings that also get an expected-value column.
        categories: Column name to allowed values, for integer-coding.
        precomputed: For the training frame, the out-of-fold values, already
            assembled with the same column names. When given, nothing is fitted
            here and these values are used verbatim.

    Returns:
        A new frame with the 30 auxiliary-task columns.
    """
    import numpy as np

    order = aux_feature_names(rating_columns)
    if precomputed is not None:
        return pd.concat(
            [
                frame.reset_index(drop=True),
                precomputed.reset_index(drop=True)[order],
            ],
            axis=1,
        )

    coded = _aux_codes(frame, categories)
    own_values: list[np.ndarray] = []
    out = frame.copy()
    for column, (model, inputs) in aux_models.items():
        probabilities = np.asarray(model.predict_proba(coded[inputs]))
        codes, values = _aux_codes_and_values(coded[column])
        own = probabilities[np.arange(len(coded)), codes]
        out[f"aux_p_{column}"] = own
        own_values.append(own)
        if column in rating_columns:
            out[f"aux_ev_{column}"] = probabilities @ values
    out["aux_sum_logp"] = np.log(np.clip(np.column_stack(own_values), 1e-6, 1)).sum(1)
    # The out-of-fold path emits aux_p_* then aux_ev_*; the loop above interleaves
    # them. Reindex so all four frames come out in the same order, or stage 2's
    # column-order check rejects the build after an hour of fitting.
    return pd.concat([frame, out[order]], axis=1)


def _aux_codes_and_values(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the target as contiguous 0..k-1 codes and its sorted values.

    Args:
        series: The target column for these rows.

    Returns:
        The codes (what the model is fitted on, because xgboost's sklearn API
        rejects labels that do not start at 0) and the sorted distinct values
        (what the expected value is computed against).
    """
    values = np.sort(series.dropna().unique().astype(float))
    codes = np.searchsorted(values, series.to_numpy(dtype=float))
    return codes, values


def add_oof_auxiliary_predictions(
    train_frame: pd.DataFrame,
    rating_columns: list[str],
    categories: dict[str, list[str]],
    seed: int,
    folds: int = 5,
) -> pd.DataFrame:
    """Compute the 30 auxiliary-task columns out-of-fold for the training rows.

    Args:
        train_frame: Training rows.
        rating_columns: The 13 ratings.
        categories: Column name to allowed values.
        seed: Random seed.
        folds: Five, matching the rest of the project.

    Returns:
        A frame with the 30 columns, each row scored by models that never saw it.
    """
    from sklearn.model_selection import KFold

    coded = _aux_codes(train_frame, categories)
    blocks = {
        f"aux_p_{column}": np.zeros(len(train_frame), dtype=np.float32)
        for column in AUX_TARGETS
    }
    blocks.update(
        {
            f"aux_ev_{column}": np.zeros(len(train_frame), dtype=np.float32)
            for column in rating_columns
        }
    )
    blocks["aux_sum_logp"] = np.zeros(len(train_frame), dtype=np.float32)

    own_all: list[np.ndarray] = []
    for column in AUX_TARGETS:
        inputs = [c for c in CANDIDATE_FEATURE_COLUMNS if c != column]
        codes, values = _aux_codes_and_values(coded[column])
        objective = (
            "binary:logistic" if len(np.unique(codes)) <= 2 else "multi:softprob"
        )
        own_column = np.zeros(len(train_frame), dtype=np.float32)
        for fitting_index, scoring_index in KFold(
            folds, shuffle=True, random_state=0
        ).split(coded):
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.7,
                tree_method="hist",
                objective=objective,
                random_state=seed,
            )
            model.fit(coded.iloc[fitting_index][inputs], codes[fitting_index])
            probabilities = np.asarray(
                model.predict_proba(coded.iloc[scoring_index][inputs])
            )
            own = probabilities[np.arange(len(scoring_index)), codes[scoring_index]]
            own_column[scoring_index] = own
            if column in rating_columns:
                blocks[f"aux_ev_{column}"][scoring_index] = probabilities @ values
        blocks[f"aux_p_{column}"] = own_column
        own_all.append(own_column)
    blocks["aux_sum_logp"] = np.log(np.clip(np.column_stack(own_all), 1e-6, 1)).sum(1)
    return pd.DataFrame(blocks)


def twin_feature_names(numeric_columns: list[str]) -> list[str]:
    """Return the deterministic names of the categorical twins.

    Args:
        numeric_columns: Every numeric column getting a twin.

    Returns:
        One `<column>_cat_` per numeric column, in the same order.
    """
    return [f"{column}_cat_" for column in numeric_columns]


def add_categorical_twins(
    frame: pd.DataFrame, numeric_columns: list[str]
) -> pd.DataFrame:
    """Add a categorical twin of every numeric column.

    Args:
        frame: Rows to enrich. Not modified.
        numeric_columns: Columns to twin. Must all exist and be numeric.

    Returns:
        A new frame with one `<column>_cat_` category column per numeric column.

    Note:
    The twin is the value as an integer, then as a string, as a category. The
    net learns an embedding per distinct value - per route, per age, per rating.
    Deterministic and labelless: no fitting, no globals, nothing to leak. A
    missing value would become the string "nan" and form its own category, which
    is honest handling rather than imputation.
    """
    enriched = frame.copy()
    for column in numeric_columns:
        enriched[f"{column}_cat_"] = (
            enriched[column].fillna(0).astype(int).astype(str).astype("category")
        )
    return enriched


def add_auxiliary_features_to_all(
    enriched_frames: dict[str, pd.DataFrame], config: DataCleaningConfig
) -> dict[str, pd.DataFrame]:
    """Add the 30 auxiliary-task columns to every frame.

    Args:
        enriched_frames: The four frames, keyed by split. Train must carry the
            label; the others need not.
        config: Gives the ratings list, the seed, and whether aux is on.

    Returns:
        The four frames with 16 `aux_p_`, 13 `aux_ev_` and `aux_sum_logp`.

    Note:
    Three column families, and they say different things. `aux_ev_<rating>` is the
    model's expected value for that rating. `aux_p_<target>` is the probability it
    assigns to the value the row actually has, so it measures how self-consistent
    the row is. `aux_sum_logp` adds up those probabilities in log space across all
    16 targets, which is one number for "how surprising is this row".

    Train rows get out-of-fold values, so no training row is ever scored by a
    model that saw it. Every other frame is scored by models fitted on the full
    train split. That separation is the whole of the leakage protection.
    """
    rating_columns = list(config.service_rating_columns)

    # Fix each categorical's allowed values from the train split, so the train,
    # validation, test and competition frames all encode a category identically.
    categories = {
        column: sorted(
            enriched_frames["train"][column].dropna().astype(str).unique().tolist()
        )
        for column in CATEGORICAL_COLUMNS
    }

    train_frame = enriched_frames["train"]
    aux_models = fit_auxiliary_models(
        train_frame, rating_columns, categories, config.random_seed
    )
    oof_values = add_oof_auxiliary_predictions(
        train_frame, rating_columns, categories, config.random_seed
    )
    log_step("aux_models", models=len(aux_models), inputs_per_model=20)

    out: dict[str, pd.DataFrame] = {}
    for name, frame in enriched_frames.items():
        out[name] = apply_auxiliary_features(
            frame,
            aux_models,
            rating_columns,
            categories,
            oof_values if name == "train" else None,
        )
    log_step("aux_features", added=len(aux_feature_names(rating_columns)))
    return out


def encode_categorical_columns(
    frame: pd.DataFrame,
    config: DataCleaningConfig,
    all_categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Encode the short categorical columns, one of two ways.

    Args:
        frame: Rows to encode.
        config: Names the categorical columns and says which encoding to use.
        all_categories: Column name to every value seen across all splits, so
            train and test get identical columns in identical order.

    Returns:
        A new frame with the categorical columns either replaced by indicator
        columns, or kept as pandas `category` dtype.

    Note:
    native is worth +0.0029 ROC-AUC over one_hot here - more than all
    hyperparameter tuning combined. XGBoost partitions the category values itself,
    which one-hot discards. It is an XGBoost-only feature though: scikit-learn
    estimators cannot read `category` dtype, so config.yaml can still ask for
    one_hot when the model is one of those.

    Raises:
        ValueError: If a value appears that was not seen anywhere, or the
            configured encoding is not one we support. Either means the categories
            differ between files, which must stop the run rather than silently
            produce a different set of columns.
    """
    unexpected = {
        column: sorted(set(frame[column].dropna().unique()) - set(values))
        for column, values in all_categories.items()
    }
    unexpected = {key: value for key, value in unexpected.items() if value}
    if unexpected:
        raise ValueError(
            f"Categories appeared that were not seen anywhere: {unexpected}. "
            "The category values differ between files, which must stop the run."
        )

    if config.categorical_encoding == "native":
        joined = frame.copy()
        for column, values in all_categories.items():
            # Pin the categories rather than letting pandas infer them. Train and
            # test must agree on the category set, or the encoding silently differs
            # between them and the model sees a column it was never trained on.
            joined[column] = pd.Categorical(joined[column], categories=values)
        log_step(
            "encode",
            rows_in=len(frame),
            rows_out=len(joined),
            encoding="native",
            categorical_columns=len(all_categories),
            categories={c: len(v) for c, v in all_categories.items()},
        )
        return joined

    if config.categorical_encoding == "one_hot":
        encoded = pd.concat(
            [
                pd.get_dummies(frame[column], columns=[column], dtype=int).reindex(
                    columns=[f"{column}_{value}" for value in values], fill_value=0
                )
                for column, values in all_categories.items()
            ],
            axis=1,
        )
        joined = pd.concat(
            [frame.drop(columns=list(config.categorical_columns)), encoded], axis=1
        )
        log_step(
            "encode",
            rows_in=len(frame),
            rows_out=len(joined),
            encoding="one_hot",
            encoded_columns=len(encoded.columns),
            categories={c: len(v) for c, v in all_categories.items()},
        )
        return joined

    raise ValueError(
        f"Unknown categorical_encoding '{config.categorical_encoding}'. "
        "Supported: 'native', 'one_hot'."
    )


def build_everything(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    competition_frame: pd.DataFrame,
    config: DataCleaningConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive, group and encode every frame the same way.

    Args:
        train_frame: Labelled training rows.
        validation_frame: Labelled validation rows.
        test_frame: Labelled test rows, held back for one final measurement.
        competition_frame: The competition's unlabelled test rows.
        config: The stage's settings.

    Returns:
        The four encoded frames, in the same order as the arguments.

    Note:
        The competition frame goes through exactly the same steps as the labelled
        ones. Anything else is training-serving skew, and skew is silent.
    """
    all_categories = {
        column: sorted(
            pd.concat(
                [
                    train_frame[column],
                    validation_frame[column],
                    test_frame[column],
                    competition_frame[column],
                ]
            )
            .unique()
            .tolist()
        )
        for column in config.categorical_columns
    }

    # Enrich every frame the same way first. Route statistics come after, because
    # they are built from the train frame and applied to all four.
    enriched_frames: dict[str, pd.DataFrame] = {}
    for name, frame in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("test", test_frame),
        ("competition", competition_frame),
    ):
        enriched = add_derived_features(frame, config)
        enriched = add_arrival_delay_status(enriched, config)
        enriched_frames[name] = optionally_clip_outliers(enriched, config)

    if config.route_features_enabled:
        enriched_frames = add_route_features_to_all(enriched_frames, config)
        enriched_frames = add_target_encodings_to_all(enriched_frames, config)

    if config.aux_features_enabled:
        enriched_frames = add_auxiliary_features_to_all(enriched_frames, config)

    if config.categorical_twins_enabled:
        # Deterministic per-frame transform: no fitting, no globals, nothing shared
        # between frames, so each frame is twinned independently. Must run BEFORE
        # encoding, because the twins themselves are categorical columns.
        for name, frame in enriched_frames.items():
            numeric_columns = [
                c
                for c in frame.columns
                if c
                not in list(config.categorical_columns)
                + [config.target_column, "id", "arrival_delay_status"]
                and str(frame[c].dtype) in ("int64", "float64")
            ]
            enriched_frames[name] = add_categorical_twins(frame, numeric_columns)
        log_step("categorical_twins", added=len(numeric_columns))

    prepared: dict[str, pd.DataFrame] = {}
    for name, frame in enriched_frames.items():
        prepared[name] = encode_categorical_columns(frame, config, all_categories)

    return (
        prepared["train"],
        prepared["validation"],
        prepared["test"],
        prepared["competition"],
    )


def check_feature_list(
    encoded_train: pd.DataFrame,
    encoded_competition: pd.DataFrame,
    config: DataCleaningConfig,
) -> None:
    """Stop the run unless every configured feature exists and matches.

    Args:
        encoded_train: The encoded training rows.
        encoded_competition: The encoded competition rows.
        config: The feature list from config.yaml.

    Raises:
        ValueError: If a configured feature is missing, if a banned column is in
            the list, or if the two frames do not have identical columns.

    Note:
        .dev/DEBUGGING.md section 4: save the feature list with the model and check
        it at prediction time. If training code changes a feature and nobody
        notices, prediction stops with a message instead of quietly producing
        garbage.
    """
    banned = set(config.banned_features)
    banned_in_use = banned & set(config.features)
    if banned_in_use:
        raise ValueError(
            f"config.yaml lists banned columns as features: {sorted(banned_in_use)}. "
            "`satisfaction` is the answer and `id` encodes the organisers' split."
        )

    check_columns("features.train", set(encoded_train.columns), set(config.features))
    check_columns(
        "features.competition", set(encoded_competition.columns), set(config.features)
    )

    # Compare the feature columns only. The training frame carries the label and
    # the competition frame correctly does not, so comparing whole frames would
    # always fail. What must match is the set of columns a prediction is made
    # from, and their order.
    train_features_in_order = [
        column for column in encoded_train.columns if column in set(config.features)
    ]
    competition_features_in_order = [
        column
        for column in encoded_competition.columns
        if column in set(config.features)
    ]
    if train_features_in_order != competition_features_in_order:
        raise ValueError(
            "The training and competition frames offer different features in a "
            f"different order.\n  train:      {train_features_in_order}\n"
            f"  competition: {competition_features_in_order}\n"
            "The competition frame must be encoded exactly like the training one."
        )

    missing_values = int(encoded_train[config.features].isna().sum().sum())
    if missing_values:
        offenders = {
            column: int(count)
            for column, count in encoded_train[config.features].isna().sum().items()
            if count
        }
        raise ValueError(
            f"The configured features still contain {missing_values} missing "
            f"values: {offenders}. The feature list and the preparation steps "
            "disagree, which means the pipeline would train on blanks."
        )
    log_step("feature_check", features=len(config.features), missing_values=0)


def run_data_cleaning_encoding(
    raw_train_path, raw_competition_path, config: DataCleaningConfig
) -> DataCleaningArtifact:
    """Split, derive, encode and save everything the model needs.

    Args:
        raw_train_path: Path to the saved training CSV from stage 1.
        raw_competition_path: Path to the saved competition test CSV from stage 1.
        config: The stage's settings.

    Returns:
        The paths of the saved files, the row counts, and the feature list.

    Raises:
        FileNotFoundError: If stage 1 has not run.
        ValueError: If the split is degenerate, the base rates diverge, the
            feature list does not match the data, or a banned column is used.
    """
    for path in (raw_train_path, raw_competition_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run stage 1 first: "
                "python -m src.pipeline.stage_01_data_ingestion"
            )

    full_data = pd.read_csv(raw_train_path)
    competition_data = pd.read_csv(raw_competition_path)
    log_step(
        "cleaning",
        rows_in=len(full_data),
        competition_rows=len(competition_data),
        target=config.target_column,
    )

    train_frame, validation_frame, test_frame = split_the_data(full_data, config)
    encoded_train, encoded_validation, encoded_test, encoded_competition = (
        build_everything(
            train_frame, validation_frame, test_frame, competition_data, config
        )
    )
    check_feature_list(encoded_train, encoded_competition, config)

    output_dir = config.artifacts_dir
    train_path = save_dataframe(encoded_train, output_dir / "train.csv")
    validation_path = save_dataframe(encoded_validation, output_dir / "validation.csv")
    test_path = save_dataframe(encoded_test, output_dir / "test.csv")
    # The competition frame is prepared here and saved here. Stage 4 must read the
    # prepared file, not the raw one: predicting from raw data would skip the
    # derivation and encoding steps, which is exactly the training-serving skew
    # .dev/TOOLS.md section 2 warns about.
    competition_path = save_dataframe(
        encoded_competition, output_dir / "competition_test.csv"
    )
    save_json(
        {
            "features": config.features,
            "categorical_encoding": config.categorical_encoding,
            "categorical_columns": list(config.categorical_columns),
            "derived_features": config.derived_feature_names,
            "dropped_features": "see config.yaml dropped_features",
            "target_column": config.target_column,
            "random_seed": config.random_seed,
        },
        output_dir / "features.json",
    )

    split_report = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "satisfied_share": round(float(frame[config.target_column].mean()), 4),
            }
            for name, frame in (
                ("train", encoded_train),
                ("validation", encoded_validation),
                ("test", encoded_test),
            )
        ]
    )
    report_path = save_dataframe(split_report, output_dir / "split_report.csv")
    print(split_report.to_string(index=False))

    return DataCleaningArtifact(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        competition_test_path=competition_path,
        categorical_encoding=config.categorical_encoding,
        features_path=output_dir / "features.json",
        split_report_path=report_path,
        train_rows=len(encoded_train),
        validation_rows=len(encoded_validation),
        test_rows=len(encoded_test),
        features=list(config.features),
    )
