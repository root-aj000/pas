"""
Tests for the config reader and the settings contract.

Run with: pytest tests/test_config_entity.py -v

These are Test 3 from .dev/RULES.md rule 10: is the incoming configuration shaped
right? A typo in config.yaml must stop the run with a message naming the setting,
not produce a model trained on the wrong features.
"""

from pathlib import Path

import pytest

from src.config.configuration import PipelineConfigReader, load_yaml_config, require_key

MINIMAL_CONFIG = """
data_path: "data/train.csv"
competition_test_path: "data/test.csv"
sample_submission_path: "data/sample_submission.csv"
artifacts_path: "artifacts"
models_path: "models"
report_path: "reports"
log_path: "logs"
random_seed: 42
target_column: "satisfaction"
banned_features:
  - "id"
  - "satisfaction"
test_size: 0.15
validation_size: 0.15
categorical_encoding: "one_hot"
features:
  - "Online boarding"
candidate_features_pending_question: []
derived_features:
  - name: "mean_service_rating"
    description: "Mean of the 13 ratings."
model_name: "hist_gradient_boosting"
model_params:
  learning_rate: 0.06
metrics:
  - "roc_auc"
decision_threshold: 0.5
max_training_seconds: 3600
"""


def write_config(tmp_path: Path, body: str) -> Path:
    """Write a config file into a temporary folder.

    Args:
        tmp_path: pytest's temporary folder.
        body: The yaml to write.

    Returns:
        The path written.
    """
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


def test_loads_a_valid_config(tmp_path) -> None:
    """A complete config must load and expose every setting."""
    config = load_yaml_config(write_config(tmp_path, MINIMAL_CONFIG))

    assert config["random_seed"] == 42
    assert config["model_name"] == "hist_gradient_boosting"
    assert config["features"] == ["Online boarding"]


def test_raises_when_the_config_file_is_missing(tmp_path) -> None:
    """A missing config must stop the run, not fall back to defaults."""
    with pytest.raises(FileNotFoundError, match="config.yaml not found"):
        load_yaml_config(tmp_path / "does_not_exist.yaml")


def test_raises_when_the_config_is_not_a_mapping(tmp_path) -> None:
    """A yaml list instead of a mapping is a type error, and must be named as one."""
    path = write_config(tmp_path, "- one\n- two\n")

    with pytest.raises(TypeError, match="must contain a mapping"):
        load_yaml_config(path)


def test_require_key_names_the_missing_setting(tmp_path) -> None:
    """The error must say which setting is missing, and not default it silently."""
    with pytest.raises(KeyError, match="learning_rate"):
        require_key({"random_seed": 42}, "learning_rate")


def test_reader_builds_every_stage_config(tmp_path) -> None:
    """All four settings objects must build from one complete config."""
    reader = PipelineConfigReader(write_config(tmp_path, MINIMAL_CONFIG))

    ingestion = reader.create_ingestion_config()
    cleaning = reader.create_cleaning_config()
    training = reader.create_training_config(model_version=3)
    evaluation = reader.create_evaluation_config(model_version=3)

    # Paths resolve against the project root, so the pipeline runs from any
    # directory on any machine. Relative in config.yaml, absolute everywhere else.
    assert ingestion.data_path == Path("data/train.csv").resolve()
    assert ingestion.data_path.is_absolute()
    assert "satisfaction" in ingestion.required_columns
    assert cleaning.test_size == 0.15
    assert training.model_dir == Path("models/model_3").resolve()
    assert evaluation.model_path == Path("models/model_3/model.pkl").resolve()


def test_model_versions_do_not_overwrite_each_other(tmp_path) -> None:
    """Each version must point at its own folder, so no model is overwritten."""
    reader = PipelineConfigReader(write_config(tmp_path, MINIMAL_CONFIG))

    first = reader.create_training_config(model_version=1).model_dir
    second = reader.create_training_config(model_version=2).model_dir

    assert first != second


def test_reader_passes_derived_names_through_without_judging_them(tmp_path) -> None:
    """The reader passes derived names on; the component checks them.

    An earlier version compared derived names against the input column list and
    rejected every derived feature. Derived features are computed *from* those
    columns, so the check was simply wrong. Whether a name has a calculation is
    now checked in one place only - the component that does the calculating.
    See tests/components/test_data_cleaning_encoding.py.
    """
    config = MINIMAL_CONFIG.replace(
        '  - name: "mean_service_rating"', '  - name: "not_a_column"'
    )
    reader = PipelineConfigReader(write_config(tmp_path, config))

    cleaning = reader.create_cleaning_config()

    assert cleaning.derived_feature_names == ["not_a_column"]
