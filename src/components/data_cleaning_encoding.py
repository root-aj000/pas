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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataCleaningArtifact, DataCleaningConfig
from src.utils.common import check_columns, log_step, save_dataframe, save_json


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
    return ["route_size", "route_target_encoded"] + [
        f"route_mean_{column}" for column in numeric_columns
    ]


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
    """Return the deterministic names of the auxiliary-task features.

    Args:
        rating_columns: The 13 service ratings.

    Returns:
        One `expected_<rating>` per rating, in the same order.
    """
    return [f"expected_{column}" for column in rating_columns]


def fit_auxiliary_models(
    train_frame: pd.DataFrame,
    rating_columns: list[str],
    other_columns: list[str],
    seed: int,
) -> dict[str, object]:
    """Fit one small model per rating, predicting it from the other columns.

    Args:
        train_frame: Training rows.
        rating_columns: The 13 ratings to predict, one model each.
        other_columns: The inputs. Every column except the one being predicted
            and the label, so no rating ever predicts itself.
        seed: Random seed.

    Returns:
        Rating name to fitted model. Small HistGradientBoosting, 100 trees -
        these are features, not the final model, so they are kept fast.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    models: dict[str, object] = {}
    for column in rating_columns:
        inputs = [c for c in other_columns if c != column]
        model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        # Ratings are 0-5 integers. Treat as classes so predict_proba gives an
        # expected value rather than a point prediction.
        model.fit(train_frame[inputs], train_frame[column].astype(int))
        models[column] = (model, inputs)
    return models


def apply_auxiliary_features(
    frame: pd.DataFrame,
    aux_models: dict[str, object],
    use_oof_frame: pd.DataFrame | None,
    oof_predictions: dict[str, pd.Series] | None,
) -> pd.DataFrame:
    """Add the expected value of each rating to any frame.

    Args:
        frame: Rows to enrich.
        aux_models: Rating name to (model, inputs), from fit_auxiliary_models.
        use_oof_frame: If this frame IS the training frame, pass it here along
            with oof_predictions so train rows get out-of-fold values.
        oof_predictions: Rating name to OOF expected values for train rows, or
            None to predict directly (for validation, test, competition).

    Returns:
        A new frame with one `expected_<rating>` column per rating.
    """

    enriched = frame.copy()
    for column, (model, inputs) in aux_models.items():
        probabilities = model.predict_proba(frame[inputs])
        classes = model.classes_
        enriched[f"expected_{column}"] = (probabilities * classes).sum(axis=1)
    if use_oof_frame is not None and oof_predictions is not None:
        for column, values in oof_predictions.items():
            enriched[f"expected_{column}"] = values.to_numpy()
    return enriched


def add_oof_auxiliary_predictions(
    train_frame: pd.DataFrame,
    rating_columns: list[str],
    other_columns: list[str],
    seed: int,
    folds: int = 5,
) -> dict[str, pd.Series]:
    """Compute out-of-fold expected ratings for the training rows.

    Args:
        train_frame: Training rows.
        rating_columns: The 13 ratings to predict.
        other_columns: The inputs, excluding the label.
        seed: Random seed.
        folds: Five, matching the rest of the project.

    Returns:
        Rating name to one expected value per training row, from models that
        never saw that row.
    """
    from sklearn.model_selection import StratifiedKFold

    out: dict[str, pd.Series] = {
        c: pd.Series(index=train_frame.index, dtype=float) for c in rating_columns
    }
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = train_frame["satisfaction"]
    for fitting_index, scoring_index in splitter.split(train_frame, labels):
        fitting = train_frame.iloc[fitting_index]
        scoring = train_frame.iloc[scoring_index]
        fold_models = fit_auxiliary_models(fitting, rating_columns, other_columns, seed)
        for column, (model, inputs) in fold_models.items():
            probabilities = model.predict_proba(scoring[inputs])
            out[column].iloc[scoring_index] = (probabilities * model.classes_).sum(
                axis=1
            )
    return out


def add_auxiliary_features_to_all(
    enriched_frames: dict[str, pd.DataFrame], config: DataCleaningConfig
) -> dict[str, pd.DataFrame]:
    """Add expected-rating columns to every frame from train-fitted models.

    Args:
        enriched_frames: The four frames, keyed by split. Train must carry the
            label; the others need not.
        config: Says whether aux features are on, and gives the seed.

    Returns:
        The four frames with one `expected_<rating>` column per rating.

    Note:
    Each rating is predicted from every column except itself and the label, so no
    rating ever predicts itself and the label never predicts anything. Train rows
    get out-of-fold values. Every other frame gets predictions from models fitted
    on the full train split. That separation is what prevents leakage.
    """
    rating_columns = list(config.service_rating_columns)
    # Numeric columns only. The categoricals are still raw strings at this point -
    # encoding happens after this step - and HistGradientBoosting cannot read them.
    # Ratings are predicted almost entirely by other ratings in any case, so nothing
    # meaningful is lost. Documented rather than worked around.
    numeric_dtypes = {"int64", "float64", "bool"}
    other_columns = [
        c
        for c in enriched_frames["train"].columns
        if c not in rating_columns + [config.target_column, "id"]
        and str(enriched_frames["train"][c].dtype) in numeric_dtypes
    ]

    train_frame = enriched_frames["train"]
    aux_models = fit_auxiliary_models(
        train_frame, rating_columns, other_columns, config.random_seed
    )
    oof_predictions = add_oof_auxiliary_predictions(
        train_frame, rating_columns, other_columns, config.random_seed
    )
    log_step("aux_models", models=len(aux_models), inputs_per_model=len(other_columns))

    out: dict[str, pd.DataFrame] = {}
    for name, frame in enriched_frames.items():
        if name == "train":
            out[name] = apply_auxiliary_features(
                frame, aux_models, frame, oof_predictions
            )
        else:
            out[name] = apply_auxiliary_features(frame, aux_models, None, None)
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

    if config.aux_features_enabled:
        enriched_frames = add_auxiliary_features_to_all(enriched_frames, config)

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
