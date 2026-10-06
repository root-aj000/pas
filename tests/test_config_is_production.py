"""
Tests that config.yaml is a production configuration, not a test one.

Run with: pytest tests/test_config_is_production.py -v

This file exists because of a specific mistake. Smoke runs were done by editing
config.yaml in place, shrinking folds to 2 and trees to 30, and restoring from a
backup afterwards. On one occasion the restore silently dropped a setting that had
been added after the backup was taken, and the pipeline was one command away from
training nine 30-tree models on two folds and calling it the submitted ensemble.

Smoke tests now override settings in memory, so config.yaml is never mutated. This
file is the second line of defence: if a smoke value ever does get committed, the
suite fails instead of the mistake reaching a GPU.
"""

import re
from pathlib import Path

import pytest
import yaml

from src.config.configuration import PipelineConfigReader

CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

# The smallest value a real run would plausibly use. A smoke run goes below these.
MIN_FOLDS = 5
MIN_TREES = 300
MIN_ENSEMBLE_MEMBERS = 3


@pytest.fixture(scope="module")
def config() -> dict:
    """Return the parsed config.yaml.

    Returns:
        The whole configuration as a dictionary.
    """
    return yaml.safe_load(CONFIG.read_text())


def test_config_has_no_smoke_folds(config: dict) -> None:
    """Cross-validation must have enough folds to be worth running."""
    folds = config["ensemble"]["folds"]
    assert folds >= MIN_FOLDS, (
        f"ensemble.folds is {folds}, which is a smoke value. Production runs use "
        f"{MIN_FOLDS} or more; every OOF number in docs/experiment_log.md was "
        f"measured at 10."
    )


def test_every_tree_member_has_a_real_budget(config: dict) -> None:
    """A tree member with a handful of trees is a test fixture, not a model.

    This is the check that would have caught the committed smoke config: four of
    the nine members had n_estimators=30, which trains in seconds and scores
    about 0.948.
    """
    small: list[str] = []
    for member in config["ensemble"]["members"]:
        params = member.get("params") or {}
        trees = params.get("n_estimators", params.get("iterations"))
        if trees is None:
            continue  # neural members count differently
        if trees < MIN_TREES:
            small.append(f"{member['name']}={trees}")
    assert not small, (
        f"tree members below {MIN_TREES} trees: {', '.join(small)}. That is a "
        f"smoke configuration; override in memory instead of editing config.yaml."
    )


def test_neural_members_have_a_real_budget(config: dict) -> None:
    """A neural member at one epoch and one ensemble member learns nothing useful."""
    small: list[str] = []
    for member in config["ensemble"]["members"]:
        if member["kind"] != "realmlp":
            continue
        params = member.get("params") or {}
        if params.get("n_ens", 1) < 2 or params.get("n_epochs", 1) < 2:
            small.append(
                f"{member['name']}=n_ens {params.get('n_ens')}, "
                f"n_epochs {params.get('n_epochs')}"
            )
    assert not small, f"neural members at a smoke budget: {'; '.join(small)}"


def test_ensemble_is_actually_configured(config: dict) -> None:
    """The shipped path is the ensemble, so it must be switched on and populated."""
    ensemble = config["ensemble"]
    assert ensemble["enabled"] is True, (
        "ensemble.enabled is false, so run_pipeline.py trains the single model "
        "from stage 3 and ignores the stack that produced the 0.96050 submission"
    )
    members = ensemble["members"]
    assert len(members) >= MIN_ENSEMBLE_MEMBERS, f"only {len(members)} members"
    names = [m["name"] for m in members]
    assert len(names) == len(set(names)), f"duplicate member names: {names}"


def test_kaggle_dataset_path_is_set_for_the_new_mount(config: dict) -> None:
    """The Kaggle dataset holding train.csv is named, not guessed.

    Two attached datasets both contain a train.csv. Without this key the resolver
    cannot tell them apart and raises, or worse, used to pick by sort order.
    """
    assert config.get("kaggle_dataset_path"), (
        "kaggle_dataset_path is missing; on Kaggle the resolver would have to "
        "choose between the competition mount and the user dataset by sort order"
    )


def test_every_generated_feature_is_in_the_config_list() -> None:
    """Stage 2 generates names that config.yaml must actually list.

    `check_feature_list` only checks that every *configured* feature exists in the
    encoded frames. It does not check the other direction, so a generated column
    that nobody added to `features` is dropped without a word: the artifact gets
    99 columns, the model is handed 70, and the run reports success. The digit,
    frequency and route-residual blocks were all built and shipped that way, so
    this asserts the reverse direction too.
    """
    from src.components.data_cleaning_encoding import (
        CANDIDATE_FEATURE_COLUMNS,
        digit_feature_names,
        frequency_feature_names,
        route_feature_names,
    )

    config = yaml.safe_load(CONFIG.read_text())
    features = set(config["features"])
    reader = PipelineConfigReader()
    cleaning = reader.create_cleaning_config()

    numeric = [
        column
        for column in cleaning.service_rating_columns
        if column != "Flight Distance"
    ]
    numeric += ["Age"] + [
        column
        for column in cleaning.continuous_columns
        if column not in ("Flight Distance", "Age")
    ]

    generated = (
        route_feature_names(numeric)
        + digit_feature_names()
        + frequency_feature_names(list(CANDIDATE_FEATURE_COLUMNS))
    )
    unlisted = sorted({name for name in generated} - features)
    assert not unlisted, (
        f"stage 2 generates {len(unlisted)} columns that config.yaml does not "
        f"list, so they would be dropped: {unlisted[:5]}..."
    )


def test_smoke_tests_do_not_write_config_yaml() -> None:
    """No script in the repository may modify config.yaml.

    Smoke coverage has to override settings in memory. A script that edits the
    config on disk will eventually leave a smoke value behind, and the failure
    will surface as a bad leaderboard score rather than as an error.
    """
    # Only a write whose TARGET is config.yaml counts. Several files read the
    # config and also write something else - reports, model bundles, predictions -
    # and flagging those would be noise that trains everyone to ignore the test.
    write = r"(write_text\(|open\([^)]*[\"'][wa]|yaml\.(safe_)?dump\()"
    offenders: list[str] = []
    for path in list(Path("research").rglob("*.py")) + list(Path("src").rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "config.yaml" in line and re.search(write, line):
                offenders.append(f"{path}:{number}")
    assert not offenders, (
        f"these files read config.yaml and also write: {offenders}. Pass overrides "
        f"as arguments instead."
    )
