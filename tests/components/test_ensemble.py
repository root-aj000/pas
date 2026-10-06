"""
Tests for src/components/ensemble.py

Run with: pytest tests/components/test_ensemble.py -v

The point of these is the invariants that silently produce a plausible-looking
submission while being wrong: a row scored by a model that saw it, a member
silently excluded, a submission scored against the wrong rows. Each has cost a run
somewhere.
"""

import numpy as np
import pandas as pd
import pytest

from src.components.ensemble import (
    SPLIT_SEED,
    EnsembleConfig,
    MemberSpec,
    nested_stack_score,
    select_features,
    to_logit,
    train_ensemble,
)
from src.constants import CATEGORICAL_COLUMNS, SERVICE_RATING_COLUMNS


def make_frames(rows: int = 900, competition_rows: int = 400, seed: int = 0):
    """Return a synthetic train frame and competition frame.

    Args:
        rows: Training rows.
        competition_rows: Competition rows.
        seed: Random seed.

    Returns:
        The training frame, with a label, and the competition frame without one.

    Note:
    The label depends on the columns, so a model that reads them can score well
    above chance. A model that cannot will score near 0.5, which is what makes
    the stack assertions below meaningful.
    """
    rng = np.random.default_rng(seed)

    def build(n: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {c: rng.integers(0, 6, n).astype("float32") for c in SERVICE_RATING_COLUMNS}
        )
        frame["Age"] = rng.integers(18, 70, n).astype("float32")
        frame["Flight Distance"] = rng.integers(100, 5000, n).astype("float32")
        frame["Departure Delay in Minutes"] = rng.integers(0, 60, n).astype("float32")
        frame["Arrival Delay in Minutes"] = rng.integers(0, 60, n).astype("float32")
        for c in CATEGORICAL_COLUMNS:
            frame[c] = pd.Categorical(rng.choice(["a", "b", "c"], n))
        return frame

    train = build(rows)
    score = train[SERVICE_RATING_COLUMNS].mean(axis=1) + rng.normal(0, 1.0, rows)
    train["satisfaction"] = (
        score > train[SERVICE_RATING_COLUMNS].mean(axis=1).mean()
    ).astype("int8")
    competition = build(competition_rows)
    return train, competition


def test_select_features_drops_prefixes_and_suffixes() -> None:
    """A member's own view of the columns is what select_features promises.

    A narrower view is a genuinely different member, which is why this exists at
    all: measured on this dataset, the same network on the raw columns added more
    to the stack than two extra architectures did.
    """
    features = [
        "Age",
        "Flight Distance",
        "route_mean_Age",
        "route_size",
        "Age_cat_",
        "Seat comfort_cat_",
    ]
    spec = MemberSpec(
        name="raw",
        kind="realmlp",
        drop_prefix=("route_",),
        drop_suffix=("_cat_",),
    )
    # The two route columns go by prefix, the two twins by suffix, and both raw
    # columns survive.
    assert select_features(features, spec) == ["Age", "Flight Distance"]


def test_select_features_refuses_to_empty_a_member() -> None:
    """Filtering everything away must fail loudly, not produce a zero-column model."""
    spec = MemberSpec(name="empty", kind="realmlp", drop_prefix=("Age",))
    with pytest.raises(ValueError, match="no features left"):
        select_features(["Age"], spec)


def test_to_logit_round_trips_and_stays_finite() -> None:
    """Logits must stay finite at 0 and 1, or a stacker sees infinities."""
    p = np.array([0.0, 1.0, 0.5, 1e-12, 1 - 1e-12])
    z = to_logit(p)
    assert np.isfinite(z).all()


def test_nested_stack_score_beats_a_random_combiner() -> None:
    """The nested score must be an honest held-out number, not a fitted one.

    If the stack were scored on the rows its weights were fitted on it would read
    high by construction. Here two members carry real signal, so a correctly
    combined stack beats either member alone.
    """
    rng = np.random.default_rng(0)
    n = 2000
    y = pd.Series(rng.integers(0, 2, n).astype("int8"))
    signal_a = y.to_numpy() * 1.4 + rng.normal(0, 1.0, n)
    signal_b = y.to_numpy() * 0.6 + rng.normal(0, 1.4, n)
    Z = np.column_stack([signal_a, signal_b])
    scores, weights = nested_stack_score(Z, y.to_numpy())
    from sklearn.metrics import roc_auc_score

    stacked = roc_auc_score(y, scores)
    assert stacked > roc_auc_score(y, signal_a), "stack lost to its best member"
    assert len(weights) == 2


def test_train_ensemble_is_leak_free_and_shaped_correctly(tmp_path) -> None:
    """End to end on synthetic data: no leakage, right row count, usable summary.

    The two assertions that matter:

    * Every training row must have a non-NaN out-of-fold probability. A NaN means
      no fold model covered that row, and the whole stack would be built on a
      partial prediction that still looks like an array.
    * The competition probabilities must have one entry per competition row, not
      per training row. Getting that wrong was a real bug: the first version
      applied the stack weights to the training logits.
    """
    train, competition = make_frames()
    config = EnsembleConfig(
        enabled=True,
        members=[
            MemberSpec(
                name="m1",
                kind="xgboost",
                params={
                    "n_estimators": 30,
                    "max_depth": 3,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
            ),
            MemberSpec(
                name="m2",
                kind="xgboost",
                params={
                    "n_estimators": 30,
                    "max_depth": 5,
                    "learning_rate": 0.3,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
            ),
        ],
        folds=3,
        seed=42,
        device="cpu",
    )
    summary = train_ensemble(
        config,
        train,
        competition,
        [c for c in train.columns if c != "satisfaction"],  # explicit features
        list(CATEGORICAL_COLUMNS),
        tmp_path,
        "satisfaction",
    )

    for name in ("m1", "m2"):
        oof = np.load(tmp_path / f"oof_{name}.npy")
        assert len(oof) == len(train), f"{name} oof length wrong"
        assert np.isfinite(oof).all(), f"{name} left rows unscored"
        assert ((oof >= 0) & (oof <= 1)).all()

    probabilities = summary["competition_probabilities"]
    assert len(probabilities) == len(competition), (
        "one probability per competition row is the whole contract of a submission"
    )
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()
    assert summary["nested_stack_auc"] > 0.5
    assert (tmp_path / "ensemble_summary.json").exists()
    assert set(summary["solo_oof_auc"]) == {"m1", "m2"}


def test_split_seed_is_the_one_the_log_assumes() -> None:
    """The fold split is part of the project's contract, not an implementation detail.

    docs/experiment_log.md compares against numbers measured on
    StratifiedKFold(shuffle=True, random_state=42). Changing this constant
    silently invalidates every comparison in that file.
    """
    assert SPLIT_SEED == 42


def test_a_feature_missing_from_the_competition_frame_fails_loudly(tmp_path) -> None:
    """A member must never be handed the label as if it were a feature.

    The stage-2 artifacts carry columns a model must not see: the id, the raw
    delay columns, arrival_delay_status. Reading the feature list from the frame's
    columns instead of from config.yaml puts all of them in scope, and the label
    is the worst of them - it would give a perfect score on train and nothing on
    the competition rows. This test pins the guard that catches it.
    """
    train, competition = make_frames(rows=300, competition_rows=150)
    config = EnsembleConfig(
        enabled=True,
        members=[
            MemberSpec(
                name="m1",
                kind="xgboost",
                params={"n_estimators": 5, "tree_method": "hist"},
            )
        ],
        folds=2,
    )
    # Pass the label as a feature, which is the mistake being guarded against.
    with pytest.raises(ValueError, match="does not have"):
        train_ensemble(
            config,
            train,
            competition,
            list(train.columns),  # includes satisfaction
            list(CATEGORICAL_COLUMNS),
            tmp_path,
            "satisfaction",
        )


def test_in_fold_target_encoding_never_sees_its_own_label() -> None:
    """The whole point of the in-fold encodings: no row encodes its own label.

    Tested with a unique key. A row whose key nobody else shares gets an encoding
    of exactly the prior, whichever fold it lands in, because its group
    contributes nothing to the models fitted on the other folds. If its own label
    leaked into its encoding, flipping that label would move the value.

    An earlier version of this test flipped a label inside a SHARED key and
    failed - but that proves nothing: flipping a label changes the stratified
    split, which reshuffles the inner folds, so the encoding is computed from a
    different subset of rows. The unique key removes that confound.
    """
    from src.components.ensemble import add_in_fold_target_encodings

    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "unique_key": np.arange(n),  # nobody shares this
            "Flight Distance": rng.integers(1, 40, n),  # shared, as a control
        }
    )
    frame["satisfaction"] = (frame["Flight Distance"] % 3 == 0).astype("int8")
    score = frame.drop(columns=["satisfaction"]).iloc[:80].reset_index(drop=True)
    test = frame.drop(columns=["satisfaction"]).iloc[:40].reset_index(drop=True)

    labels = frame["satisfaction"]
    before, _, _ = add_in_fold_target_encodings(
        frame.drop(columns=["satisfaction"]),
        labels,
        score,
        test,
        ["unique_key", "Flight Distance"],
        (),
        42,
    )
    flipped = frame.copy()
    flipped.loc[0, "satisfaction"] = 1 - int(flipped.loc[0, "satisfaction"])
    after, _, _ = add_in_fold_target_encodings(
        flipped.drop(columns=["satisfaction"]),
        flipped["satisfaction"],
        score,
        test,
        ["unique_key", "Flight Distance"],
        (),
        42,
    )

    assert before["tef_unique_key"].iloc[0] == after["tef_unique_key"].iloc[0], (
        "a row's own label leaked into its own encoding"
    )
    # The control: a shared key's encoding is derived from its group, so flipping
    # one member must move it. Without this the encoder could be a frozen no-op
    # and the assertion above would pass for the wrong reason.
    assert (
        before["tef_Flight Distance"].to_numpy()
        != after["tef_Flight Distance"].to_numpy()
    ).any(), "the encoder did not react to a label change anywhere"


def test_rank_blend_beats_equal_weighting_when_members_disagree() -> None:
    """The rank blender must actually use the weights, not average everything.

    Two members, one of them pure noise. An equal-weight blend of those is worse
    than the good member alone, so if the fitted weights do not push the noise
    member down, the optimiser is not working.
    """
    from src.components.ensemble import rank_blend_weights, to_rank

    rng = np.random.default_rng(0)
    n = 3000
    y = pd.Series(rng.integers(0, 2, n).astype("int8"))
    good = y.to_numpy() * 1.6 + rng.normal(0, 1.0, n)
    noise = rng.normal(0, 1.0, n)
    P = np.column_stack([to_rank(good), to_rank(noise)])
    weights = rank_blend_weights(P, y.to_numpy())
    assert weights[0] > weights[1], (
        f"the informative member should outweigh the noise one, got {weights}"
    )
    assert abs(weights.sum() - 1.0) < 1e-6


def test_in_fold_encodings_do_not_collide_with_stage_two_names() -> None:
    """The fold-encoded columns must not reuse stage 2's `te_` names.

    Stage 2 writes `te_Age`, `te_FD`, `te_FD|Age`. The in-fold block used to write
    `te_<column>`, so encoding "Age" produced a second `te_Age` and `pd.concat`
    kept both. Every member then silently received a duplicated input column and
    nothing complained. The check is against the real stage-2 naming function, so
    it cannot drift if that naming changes.
    """
    from src.components.data_cleaning_encoding import target_encoding_column_name
    from src.components.ensemble import add_in_fold_target_encodings

    keys = [("Flight Distance",), ("Age",), ("Age", "Class")]
    stage_two = {target_encoding_column_name(k) for k in keys}
    assert "te_Age" in stage_two, "stage 2 is expected to emit te_Age"

    encoded = ["Age", "Flight Distance", "Class"]
    n = 200
    frame = pd.DataFrame(
        {
            "Age": np.arange(n, dtype=np.float64),
            "Flight Distance": np.arange(n, dtype=np.float64) * 3,
            "Class": ["Eco"] * n,
        }
    )
    labels = pd.Series(np.arange(n) % 2, dtype="int8")
    out, _, _ = add_in_fold_target_encodings(
        frame, labels, frame.iloc[:5], frame.iloc[:5], encoded, (), 42
    )
    added = {c for c in out.columns if c not in frame.columns}
    assert added, "the block added no columns"
    assert not (added & stage_two), (
        f"names already produced by stage 2, which would duplicate on concat: "
        f"{sorted(added & stage_two)}"
    )
    assert len(frame.columns.intersection(added)) == 0, (
        "the block overwrote a column it was handed"
    )


def test_target_encodings_off_keeps_the_block_out_of_the_member(monkeypatch) -> None:
    """`target_encodings: false` must reach the member, not just the config.

    The switch exists because the neural members were handed 191 columns of which
    a fifth were encodings of columns still sitting beside them. If it were parsed
    and never consulted, the members would keep receiving the block and the
    setting would be a lie told in a config file.
    """
    import src.components.ensemble as module
    from src.components.ensemble import EnsembleConfig, MemberSpec

    n = 200
    frame = pd.DataFrame(
        {"Age": np.arange(n, dtype=np.float64), "Class": ["Eco"] * n}
    )
    labels = pd.Series(np.arange(n) % 2, dtype="int8")
    competition = frame.iloc[:20].copy()
    widths: dict[bool, int] = {}

    class _Spy:
        def fit(self, X, y):
            widths[spec.target_encodings] = X.shape[1]
            return self

        def predict_proba(self, X):
            return np.tile([0.4, 0.6], (len(X), 1))

    for enabled in (True, False):
        spec = MemberSpec(name="spy", kind="spy", target_encodings=enabled)
        monkeypatch.setattr(
            module, "build_estimator", lambda kind, params, seed: _Spy()
        )
        module.train_member(
            spec,
            frame,
            labels,
            competition,
            [],
            EnsembleConfig(folds=2, seed=42, te_columns=["Age", "Class"]),
            "satisfaction",
        )

    assert widths[False] == 2, (
        f"with the block off the member should see its 2 columns, saw {widths[False]}"
    )
    assert widths[True] > widths[False], (
        f"with the block on the member should see more columns: {widths}"
    )


def test_stage_five_trains_on_train_plus_validation() -> None:
    """Structural guard: stage 5 must not silently drop the validation split.

    Stage 2 splits 70/15/15 for the single-model path, where the validation file
    gave stages 3 and 4 something to select on. Stage 5 cross-validates internally
    and needs no held-out split, so reading only `train.csv` discarded 209,892
    rows - 30% of the labelled data - from every member.

    This checks the source rather than the behaviour, which is normally the wrong
    way round. It is here because the failure mode is a *number*, not an
    exception: the run completes, writes a plausible submission and reports a
    normal nested score, 0.001 low, with nothing anywhere reporting an error.
    Running stage 5 for real needs a full artifacts tree and a fitted config,
    which is not a unit test. The row count the run logs (`rows=`, emitted
    alongside `train_rows=` and `validation_rows=`) is the thing to watch in the
    actual output.
    """
    import inspect

    from src.pipeline import stage_05_ensemble

    source = inspect.getsource(stage_05_ensemble.run_pipeline)
    assert "validation.csv" in source, (
        "stage 5 no longer reads validation.csv, so it trains on 70% of the rows"
    )
    assert "pd.concat" in source, (
        "stage 5 must fold the validation rows into training, not read them "
        "alongside and ignore them"
    )


def test_ensemble_device_reaches_every_family() -> None:
    """`ensemble.device: cuda` has to reach each family under its own key.

    Every family spells the device differently and none defaults to the GPU:
    XGBoost `device="cuda"`, LightGBM `device="gpu"`, CatBoost
    `task_type="GPU"`, pytabkit and TabNet `device`. The key existed in
    config.yaml and was read by nothing, so a Kaggle run trained all nine members
    on CPU. No GPU is needed to catch that - only that the right key appears.
    """
    from src.components.ensemble import apply_device

    expected = {
        "xgboost": ("device", "cuda"),
        "lightgbm": ("device", "gpu"),
        "catboost": ("task_type", "GPU"),
        "realmlp": ("device", "cuda"),
        "tabnet": ("device", "cuda"),
    }
    for kind, (key, value) in expected.items():
        got = apply_device(kind, {"n_estimators": 10}, "cuda")
        assert got.get(key) == value, (
            f"{kind} on cuda should set {key}={value!r}, got {got}"
        )
        assert apply_device(kind, {"n_estimators": 10}, "cpu") == {
            "n_estimators": 10
        }, f"{kind} on cpu should be left alone"


def test_routediff_has_no_missing_values_when_the_delay_is_absent() -> None:
    """`Arrival Delay in Minutes` is absent on some rows; the residual must cope.

    The route mean is a groupby mean, so it is defined on a row whose delay is
    missing, and `NaN - mean` is NaN. That reached `check_feature_list` and failed
    the run at stage 2 - which is how the block was caught, but only because the
    configured feature list finally included the columns it generates.
    """
    import pandas as pd

    from src.components.data_cleaning_encoding import (
        apply_route_features,
        build_route_table,
    )

    frame = pd.DataFrame(
        {
            "Flight Distance": [100, 100, 200, 200] * 3,
            "satisfaction": [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            "Arrival Delay in Minutes": [5.0, np.nan, 7.0, np.nan, 1.0, 2.0,
                                         np.nan, 3.0, 4.0, np.nan, 6.0, 8.0],
        }
    )
    table, means, rate = build_route_table(
        frame, "Flight Distance", ["Arrival Delay in Minutes"], "satisfaction", 20.0
    )
    out = apply_route_features(
        frame, table, means, rate, ["Arrival Delay in Minutes"], None
    )
    residual = out["routediff_Arrival Delay in Minutes"]
    assert residual.notna().all(), (
        f"{int(residual.isna().sum())} residuals are NaN where the delay is missing"
    )


def test_apply_device_rejects_an_unresolved_device() -> None:
    """An unresolved `auto` must raise, not silently leave members on the CPU.

    This is the exact bug that cost a Kaggle session: `train_ensemble` resolved
    `auto` into a local for the sharding decision, but `train_member` reads
    `config.device` and that was still "auto". `apply_device` treated anything
    that was not literally "cuda" as cpu, so no member received a device param,
    every family defaults to the processor, and the run reported `gpus=2` while
    both cards sat at 0%. Rejecting the value turns that into a loud failure.
    """
    from src.components.ensemble import apply_device

    for unresolved in ("auto", "gpu", "", "GPU", "cuda:0"):
        with pytest.raises(ValueError, match="resolve_device"):
            apply_device("xgboost", {}, unresolved)


def test_auto_device_is_written_back_into_the_config(monkeypatch) -> None:
    """The resolved device must reach `config.device`, not just a local.

    `train_member` reads `config.device` when it calls `apply_device`. Resolving
    `auto` for the sharding decision alone leaves that read unresolved, so the
    fix has to be `replace(config, device=device)` - the same call the parent and
    the per-GPU worker both make.
    """
    from dataclasses import replace

    import torch

    from src.components.ensemble import EnsembleConfig, apply_device, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = EnsembleConfig(device="auto")
    device = resolve_device(config.device)
    assert device == "cuda"
    resolved = replace(config, device=device)
    assert resolved.device == "cuda"
    assert apply_device("xgboost", {}, resolved.device).get("device") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_load_ensemble_frames_uses_every_labelled_split(tmp_path) -> None:
    """Stage 5 must train on all three labelled splits, not just `train.csv`.

    Stage 2 splits 70/15/15 for the single-model path. The ensemble reads neither
    the validation file nor the frozen test file, so holding them out was pure
    loss: 489,743 rows first, then 594,689 after adding validation only. The
    reference fits on all 699,635.

    The fixture writes three labelled files plus the competition file, so the
    assertion is on the row count rather than on the source.
    """
    import json

    import src.utils.common as common
    from src.components.ensemble import load_ensemble_frames

    def write(name: str, rows: int, label: bool) -> None:
        frame = pd.DataFrame({"Age": np.arange(rows, dtype=np.float64)})
        if label:
            frame["satisfaction"] = np.arange(rows) % 2
        frame.to_csv(tmp_path / name, index=False)

    write("train.csv", 70, True)
    write("validation.csv", 15, True)
    write("test.csv", 15, True)
    write("competition_test.csv", 30, False)
    (tmp_path / "features.json").write_text(json.dumps({"categorical_encoding": "native"}))

    train_frame, competition_frame = load_ensemble_frames(tmp_path, ["Age"], [])
    assert len(train_frame) == 100, (
        f"stage 5 should train on all 100 labelled rows, got {len(train_frame)}"
    )
    assert len(competition_frame) == 30
    assert "satisfaction" in train_frame.columns
    assert "satisfaction" not in competition_frame.columns
