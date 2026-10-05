"""
Reads config.yaml and hands each stage the settings it needs.

This is the only file in the project that reads config.yaml. Components never
touch yaml - they receive a settings object, which is what makes them testable
with no files and no configuration involved.

See .lead/02-A-ARCHITECTURE.md section 3.
"""

from pathlib import Path
from typing import Any

import yaml

from src.constants import (
    CANDIDATE_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    SERVICE_RATING_COLUMNS,
    TARGET_COLUMN,
)
from src.entity.config_entity import (
    DataCleaningConfig,
    DataIngestionConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PipelineConfig,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_yaml_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Return the whole configuration as a dictionary.

    Args:
        config_path: Path to config.yaml.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If config.yaml is missing. Every run needs it, so the
            program stops rather than guessing defaults.
        ValueError: If the file does not parse as a mapping.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. Every setting lives in that "
            "file, so the run cannot continue without it."
        )
    with config_path.open() as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"{config_path} must contain a mapping of settings, got {type(loaded).__name__}"
        )
    return loaded


def require_key(config: dict[str, Any], key: str) -> Any:
    """Return a required setting, or stop with a message naming what is missing.

    Args:
        config: The parsed configuration.
        key: The setting's name.

    Returns:
        The setting's value.

    Raises:
        KeyError: If the setting is absent.
    """
    if key not in config:
        raise KeyError(
            f"config.yaml is missing '{key}'. Found keys: {sorted(config)}. "
            "Add it to config.yaml rather than defaulting it in code - a default "
            "hidden here is a setting nobody can see or change."
        )
    return config[key]


class PipelineConfigReader:
    """Builds the settings object each stage needs, from config.yaml.

    Args:
        config_path: Path to config.yaml.

    Raises:
        FileNotFoundError: If config.yaml is missing.
        ValueError: If a required setting is absent or the wrong shape.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = load_yaml_config(config_path)

    def _read_categorical_encoding(self) -> str:
        """Return the categorical encoding, checked against the two we support.

        Returns:
            Either "native" or "one_hot".

        Raises:
            ValueError: If config.yaml asks for anything else. A typo here would
                otherwise produce a frame of the wrong shape and fail much later,
                in the model, with a message about dtype rather than about config.
        """
        encoding = str(require_key(self.config, "categorical_encoding"))
        allowed = ("native", "one_hot")
        if encoding not in allowed:
            raise ValueError(
                f"config.yaml categorical_encoding is '{encoding}'. "
                f"Supported: {list(allowed)}."
            )
        return encoding

    def create_ingestion_config(self) -> DataIngestionConfig:
        """Return the settings stage 1 needs.

        Returns:
            The stage 1 settings.

        Raises:
            ValueError: If a required column is listed in neither the known
                feature list nor the banned list.
        """
        banned = frozenset(require_key(self.config, "banned_features"))
        required = frozenset(CANDIDATE_FEATURE_COLUMNS) | frozenset({TARGET_COLUMN})
        return DataIngestionConfig(
            data_path=Path(require_key(self.config, "data_path")),
            competition_test_path=Path(
                require_key(self.config, "competition_test_path")
            ),
            artifacts_dir=Path(require_key(self.config, "artifacts_path")),
            required_columns=required,
            banned_features=banned,
        )

    def create_cleaning_config(self, model_version: int = 1) -> DataCleaningConfig:
        """Return the settings stage 2 needs.

        Args:
            model_version: Unused here, kept so every reader takes the same
                argument shape and stages can be called uniformly.

        Returns:
            The stage 2 settings.
        """
        # Only the names are read here. Whether a name has a calculation is
        # checked in the component, which is the only place that knows how to
        # compute one - duplicating the list here would let the two drift apart.
        derived_names = [
            entry["name"] for entry in require_key(self.config, "derived_features")
        ]
        return DataCleaningConfig(
            target_column=str(require_key(self.config, "target_column")),
            features=list(require_key(self.config, "features")),
            candidate_features_pending_question=list(
                require_key(self.config, "candidate_features_pending_question")
            ),
            derived_feature_names=derived_names,
            banned_features=frozenset(require_key(self.config, "banned_features")),
            service_rating_columns=list(SERVICE_RATING_COLUMNS),
            categorical_columns=list(CATEGORICAL_COLUMNS),
            continuous_columns=list(CONTINUOUS_COLUMNS),
            test_size=float(require_key(self.config, "test_size")),
            validation_size=float(require_key(self.config, "validation_size")),
            random_seed=int(require_key(self.config, "random_seed")),
            clip_outliers=bool(self.config.get("clip_outliers", False)),
            clip_lower_percentile=float(self.config.get("clip_lower_percentile", 0.1)),
            clip_upper_percentile=float(self.config.get("clip_upper_percentile", 99.9)),
            arrival_delay_median=float(
                self.config.get("arrival_delay_median_for_model", 0.0)
            ),
            categorical_encoding=self._read_categorical_encoding(),
            route_features_enabled=bool(
                self.config.get("route_features", {}).get("enabled", False)
            ),
            route_smoothing=float(
                self.config.get("route_features", {}).get("smoothing", 20.0)
            ),
            aux_features_enabled=bool(
                self.config.get("auxiliary_features", {}).get("enabled", False)
            ),
            artifacts_dir=Path(require_key(self.config, "artifacts_path"))
            / "data_cleaning_encoding",
        )

    def create_training_config(self, model_version: int) -> ModelTrainerConfig:
        """Return the settings stage 3 needs.

        Args:
            model_version: Which model folder to write into, so a trained model
                is never overwritten.

        Returns:
            The stage 3 settings.
        """
        return ModelTrainerConfig(
            train_data_path=Path(require_key(self.config, "artifacts_path"))
            / "data_cleaning_encoding"
            / "train.csv",
            model_dir=Path(require_key(self.config, "models_path"))
            / f"model_{model_version}",
            model_name=str(require_key(self.config, "model_name")),
            model_params=dict(require_key(self.config, "model_params")),
            features=list(require_key(self.config, "features")),
            target_column=str(require_key(self.config, "target_column")),
            random_seed=int(require_key(self.config, "random_seed")),
            max_training_seconds=int(require_key(self.config, "max_training_seconds")),
            categorical_encoding=self._read_categorical_encoding(),
        )

    def create_evaluation_config(self, model_version: int) -> ModelEvaluationConfig:
        """Return the settings stage 4 needs.

        Args:
            model_version: Which trained model to score.

        Returns:
            The stage 4 settings.
        """
        artifacts_root = Path(require_key(self.config, "artifacts_path"))
        model_dir = (
            Path(require_key(self.config, "models_path")) / f"model_{model_version}"
        )
        return ModelEvaluationConfig(
            validation_data_path=artifacts_root
            / "data_cleaning_encoding"
            / "validation.csv",
            test_data_path=artifacts_root / "data_cleaning_encoding" / "test.csv",
            # The prepared file, not data/test.csv. Stage 4 must never derive or
            # encode anything itself: whatever the model was trained on is what it
            # must be predicted from, or the two drift apart silently.
            competition_test_path=artifacts_root
            / "data_cleaning_encoding"
            / "competition_test.csv",
            sample_submission_path=Path(
                require_key(self.config, "sample_submission_path")
            ),
            model_path=model_dir / "model.pkl",
            features_path=model_dir / "features.json",
            target_column=str(require_key(self.config, "target_column")),
            decision_threshold=float(require_key(self.config, "decision_threshold")),
            metrics=list(require_key(self.config, "metrics")),
            report_dir=Path(require_key(self.config, "report_path"))
            / f"model_{model_version}",
            submission_dir=Path(require_key(self.config, "report_path"))
            / "submissions",
        )

    def create_pipeline_config(self, model_version: int = 1) -> PipelineConfig:
        """Return every stage's settings in one object.

        Args:
            model_version: Which model version this run will produce.

        Returns:
            The full settings object, as run_pipeline.py needs it.
        """
        return PipelineConfig(
            ingestion=self.create_ingestion_config(),
            cleaning=self.create_cleaning_config(),
            training=self.create_training_config(model_version),
            evaluation=self.create_evaluation_config(model_version),
            models_root=Path(require_key(self.config, "models_path")),
            report_root=Path(require_key(self.config, "report_path")),
            artifacts_root=Path(require_key(self.config, "artifacts_path")),
        )
