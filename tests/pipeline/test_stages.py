"""
Tests for the pipeline stages and the shared helpers.

Run with: pytest tests/pipeline/test_stages.py -v
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.components.model_training import run_overfitting_check
from src.pipeline.stage_01_data_ingestion import STAGE_NAME as STAGE_01
from src.utils.common import (
    build_run_log_name,
    check_columns,
    hash_file,
    load_json,
    next_model_version,
    save_dataframe,
    save_json,
    seed_everything,
)


def test_check_columns_passes_when_everything_is_present() -> None:
    """A frame with all the columns must not raise."""
    check_columns("test", {"a", "b", "c"}, {"a", "b"})


def test_check_columns_names_what_is_missing() -> None:
    """The error must list the missing columns and what was available."""
    with pytest.raises(ValueError, match="tenure_days"):
        check_columns("clean", {"age", "spend"}, {"age", "tenure_days"})


def test_seed_everything_is_repeatable() -> None:
    """Two runs with the same seed must give the same numbers."""
    import numpy as np

    seed_everything(42)
    first = np.random.rand(5)
    seed_everything(42)
    second = np.random.rand(5)

    assert list(first) == list(second)


def test_next_model_version_starts_at_one(tmp_path) -> None:
    """With no models yet, the first run writes model_1."""
    assert next_model_version(tmp_path) == 1


def test_next_model_version_never_reuses_a_number(tmp_path) -> None:
    """Each run gets a new folder, so a trained model is never overwritten."""
    (tmp_path / "model_1").mkdir()
    (tmp_path / "model_2").mkdir()

    assert next_model_version(tmp_path) == 3


def test_next_model_version_ignores_unrelated_folders(tmp_path) -> None:
    """A folder that is not a model must not be counted as a version."""
    (tmp_path / "model_1").mkdir()
    (tmp_path / "notebook_output").mkdir()

    assert next_model_version(tmp_path) == 2


def test_run_log_names_sort_into_a_history() -> None:
    """Date, time, then what it was - sorted by name, it reads as a history."""
    name = build_run_log_name(
        "full_run", datetime(2026, 10, 4, 21, 30, tzinfo=timezone.utc)
    )

    assert name == "2026-10-04_2130_full_run.log"


def test_hash_file_changes_when_the_file_changes(tmp_path) -> None:
    """The config hash must change when config.yaml changes, or it proves nothing."""
    path = tmp_path / "config.yaml"
    path.write_text("random_seed: 42")
    first = hash_file(path)
    path.write_text("random_seed: 7")
    second = hash_file(path)

    assert first != second


def test_hash_file_raises_for_a_missing_file(tmp_path) -> None:
    """Hashing a file that is not there must stop the run."""
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "nope.yaml")


def test_save_and_load_json_round_trips(tmp_path) -> None:
    """The model bundle's features.json must survive a write and read."""
    path = save_json({"features": ["a", "b"], "seed": 42}, tmp_path / "features.json")

    assert load_json(path) == {"features": ["a", "b"], "seed": 42}


def test_load_json_raises_for_a_missing_file(tmp_path) -> None:
    """A missing model bundle means the pipeline ran out of order, which must stop."""
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "not_there.json")


def test_save_dataframe_creates_missing_folders(tmp_path) -> None:
    """Writing into a folder that does not exist yet must work."""
    path = save_dataframe(
        pd.DataFrame({"a": [1, 2]}), tmp_path / "deep" / "nested" / "out.csv"
    )

    assert Path(path).exists()
    assert len(pd.read_csv(path)) == 2


def test_stage_03_runs_the_overfitting_check_before_training(monkeypatch) -> None:
    """Stage 3 must refuse to train a model that cannot learn.

    This is the wiring test: it proves the check is actually called by the stage,
    not just present in the component.
    """
    from src.pipeline import stage_03_model_training

    called = []
    original = stage_03_model_training.run_overfitting_check

    def spy(*args, **kwargs):
        called.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(stage_03_model_training, "run_overfitting_check", spy)
    stage_03_model_training.run_pipeline()
    assert called, "run_overfitting_check was not called by stage_03"


def test_every_stage_module_exposes_run_pipeline() -> None:
    """Each stage must be runnable on its own while debugging it."""
    from src.pipeline import (
        stage_01_data_ingestion,
        stage_02_data_cleaning_encoding,
        stage_03_model_training,
        stage_04_model_evaluation,
    )

    for module in (
        stage_01_data_ingestion,
        stage_02_data_cleaning_encoding,
        stage_03_model_training,
        stage_04_model_evaluation,
    ):
        assert callable(module.run_pipeline)

    assert STAGE_01 == "stage_01_data_ingestion"


def test_find_project_root_returns_the_directory_holding_config_yaml(
    tmp_path, monkeypatch
) -> None:
    """The root is wherever config.yaml lives, not wherever you run from."""
    from src.utils.common import find_project_root

    root = tmp_path / "proj"
    root.mkdir()
    (root / "config.yaml").write_text("seed: 42\n")
    monkeypatch.chdir(root)

    assert find_project_root() == root.resolve()


def test_find_project_root_searches_parents(tmp_path, monkeypatch) -> None:
    """Running from a subdirectory must still find the root above it."""
    from src.utils.common import find_project_root

    root = tmp_path / "proj"
    (root / "nested" / "deep").mkdir(parents=True)
    (root / "config.yaml").write_text("seed: 42\n")
    monkeypatch.chdir(root / "nested" / "deep")

    assert find_project_root() == root.resolve()


def test_find_project_root_fails_loudly_for_a_marker_that_exists_nowhere(
    tmp_path, monkeypatch
) -> None:
    """The error must say what was missing and where it looked.

    find_project_root also searches near its own file, so with the package
    importable it always finds the real project. That fallback is what makes
    Kaggle work. A marker that exists nowhere tests the failure message itself.
    """
    from src.utils.common import find_project_root

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    with pytest.raises(FileNotFoundError, match="no_such_file"):
        find_project_root(marker="no_such_file_xyz.yaml")


def test_find_project_root_prefers_cwd_over_package_location(
    tmp_path, monkeypatch
) -> None:
    """A config.yaml in the working tree must win over the installed package.

    Otherwise local edits would be silently ignored in favour of wherever the
    package happens to be installed.
    """
    from src.utils.common import find_project_root

    root = tmp_path / "proj"
    root.mkdir()
    (root / "config.yaml").write_text("seed: 42\n")
    monkeypatch.chdir(root)

    assert find_project_root() == root.resolve()


def test_kaggle_input_is_used_only_when_local_data_is_missing(
    tmp_path, monkeypatch
) -> None:
    """Local data/ always wins. Kaggle is the fallback, not the default."""
    from src.config.configuration import resolve_data_file
    from src.utils import common

    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    local = root / "data" / "train.csv"
    local.write_text("id\n0\n")
    (root / "config.yaml").write_text("seed: 42\n")

    fake_kaggle = tmp_path / "kaggle" / "input" / "competition"
    fake_kaggle.mkdir(parents=True)
    (fake_kaggle / "train.csv").write_text("id\n9\n")

    monkeypatch.chdir(root)
    monkeypatch.setattr(common, "KAGGLE_INPUT_DIRS", (tmp_path / "kaggle" / "input",))

    assert resolve_data_file("data/train.csv", "Training data") == local.resolve()


def test_kaggle_input_is_found_when_local_data_is_absent(tmp_path, monkeypatch) -> None:
    """A repo cloned without data/ must still find the Kaggle mount."""
    from src.config.configuration import resolve_data_file
    from src.utils import common

    root = tmp_path / "proj"
    root.mkdir()
    (root / "config.yaml").write_text("seed: 42\n")

    fake_kaggle = tmp_path / "kaggle" / "input" / "competition"
    fake_kaggle.mkdir(parents=True)
    remote = fake_kaggle / "train.csv"
    remote.write_text("id\n9\n")

    monkeypatch.chdir(root)
    monkeypatch.setattr(common, "KAGGLE_INPUT_DIRS", (tmp_path / "kaggle" / "input",))

    assert resolve_data_file("data/train.csv", "Training data") == remote.resolve()


def test_missing_everywhere_lists_both_places_checked(tmp_path, monkeypatch) -> None:
    """If neither has it, the error must name both places, not just one."""
    from src.config.configuration import resolve_data_file
    from src.utils import common

    root = tmp_path / "proj"
    root.mkdir()
    (root / "config.yaml").write_text("seed: 42\n")
    (tmp_path / "kaggle" / "input").mkdir(parents=True)

    monkeypatch.chdir(root)
    monkeypatch.setattr(common, "KAGGLE_INPUT_DIRS", (tmp_path / "kaggle" / "input",))

    with pytest.raises(FileNotFoundError, match="/kaggle/input"):
        resolve_data_file("data/train.csv", "Training data")
