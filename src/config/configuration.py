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

from src.components.ensemble import EnsembleConfig, MemberSpec, resolve_device
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
from src.utils.common import (
    describe_kaggle_input,
    find_kaggle_file,
    find_project_root,
    get_logger,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")


def resolve_project_path(raw: str | Path) -> Path:
    """Return an absolute path for a setting from config.yaml.

    Args:
        raw: The value as written, usually relative like "data/train.csv".

    Returns:
        The value resolved against the project root, unless it was already
        absolute. An absolute value in config.yaml always wins, untouched.
    """
    path = Path(raw)
    if path.is_absolute():
        return path
    return find_project_root() / path


def resolve_data_file(
    raw: str | Path, description: str, preferred_root: str | Path | None = None
) -> Path:
    """Return the data file to read, from the project tree or Kaggle.

    Args:
        raw: The value from config.yaml.
        description: What the file is, used in the error message.
        preferred_root: A Kaggle dataset directory to search before the general
            mounts. Without it, two mounted datasets both holding train.csv make
            the result depend on alphabetical order.

    Returns:
        The file to read.

    Raises:
        FileNotFoundError: If it is in neither place, listing everywhere checked.

    Note:
    Local files win. Only when the project tree lacks the file are the Kaggle
    input mounts scanned for the same filename. A found file is logged loudly -
    silently reading data from elsewhere is how results stop being reproducible.
    """
    path = resolve_project_path(raw)
    if path.exists():
        return path
    found = find_kaggle_file(path.name, preferred_root)
    if found is not None:
        get_logger().warning(
            "%s not found at %s. Using Kaggle input %s instead.",
            description,
            path,
            found,
        )
        return found
    raise FileNotFoundError(
        f"{description} not found at {path}, and no file named {path.name} "
        "exists anywhere under /kaggle/input/.\n"
        "Attach the competition data with the notebook's Add data button, or "
        "place the files in data/.\n"
        f"What is actually mounted:\n{describe_kaggle_input()}"
    )


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
        resolved = (
            config_path
            if config_path.is_absolute() or config_path.exists()
            else resolve_project_path(config_path)
        )
        self.config_path = resolved
        self.config = load_yaml_config(resolved)

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

    @property
    def kaggle_dataset_path(self) -> str | None:
        """The Kaggle dataset directory to prefer, from config.yaml.

        Returns:
            The configured path, or None when the local tree already holds the
            files or the key is absent.
        """
        raw = self.config.get("kaggle_dataset_path")
        return str(raw) if raw else None

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
            data_path=resolve_data_file(
                require_key(self.config, "data_path"),
                "Training data",
                self.kaggle_dataset_path,
            ),
            competition_test_path=resolve_data_file(
                require_key(self.config, "competition_test_path"),
                "Competition test data",
                self.kaggle_dataset_path,
            ),
            artifacts_dir=resolve_project_path(
                require_key(self.config, "artifacts_path")
            ),
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
            categorical_twins_enabled=bool(
                self.config.get("categorical_twins", {}).get("enabled", False)
            ),
            artifacts_dir=resolve_project_path(
                require_key(self.config, "artifacts_path")
            )
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
        model_params = dict(require_key(self.config, "model_params"))
        if "device" in model_params:
            model_params["device"] = resolve_device(str(model_params["device"]))
        return ModelTrainerConfig(
            train_data_path=resolve_project_path(
                require_key(self.config, "artifacts_path")
            )
            / "data_cleaning_encoding"
            / "train.csv",
            model_dir=resolve_project_path(require_key(self.config, "models_path"))
            / f"model_{model_version}",
            model_name=str(require_key(self.config, "model_name")),
            model_params=model_params,
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
        artifacts_root = resolve_project_path(
            require_key(self.config, "artifacts_path")
        )
        model_dir = (
            resolve_project_path(require_key(self.config, "models_path"))
            / f"model_{model_version}"
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
            sample_submission_path=resolve_data_file(
                require_key(self.config, "sample_submission_path"),
                "Submission template",
                self.kaggle_dataset_path,
            ),
            model_path=model_dir / "model.pkl",
            features_path=model_dir / "features.json",
            target_column=str(require_key(self.config, "target_column")),
            decision_threshold=float(require_key(self.config, "decision_threshold")),
            metrics=list(require_key(self.config, "metrics")),
            report_dir=resolve_project_path(require_key(self.config, "report_path"))
            / f"model_{model_version}",
            submission_dir=resolve_project_path(require_key(self.config, "report_path"))
            / "submissions",
        )

    def create_ensemble_config(self) -> EnsembleConfig:
        """Return the ensemble settings from config.yaml.

        Returns:
            An EnsembleConfig, disabled when the block is absent or off.

        Note:
        Defaults to disabled rather than raising. A config without an ensemble
        block is a config that wants the single-model path, which is the
        behaviour that predates this feature and is covered by the other tests.
        """
        block = self.config.get("ensemble") or {}
        if not isinstance(block, dict):
            block = {}
        members = [
            MemberSpec(
                name=str(entry["name"]),
                kind=str(entry["kind"]),
                params=dict(entry.get("params") or {}),
                drop_prefix=tuple(entry.get("drop_prefix") or ()),
                drop_suffix=tuple(entry.get("drop_suffix") or ()),
                target_encodings=bool(entry.get("target_encodings", True)),
            )
            for entry in (block.get("members") or [])
        ]
        return EnsembleConfig(
            enabled=bool(block.get("enabled", False)),
            members=members,
            folds=int(block.get("folds", 10)),
            seed=int(block.get("seed", 42)),
            device=resolve_device(str(require_key(self.config, "device"))),
            stack_C=float(block.get("stack_C", 1.0)),
            te_columns=[str(c) for c in (block.get("te_columns") or [])],
            combiner=str(block.get("combiner", "logistic")),
            pseudo_label_enabled=bool(block.get("pseudo_label_enabled", False)),
            pseudo_label_high=float(block.get("pseudo_label_high", 0.95)),
            pseudo_label_low=float(block.get("pseudo_label_low", 0.05)),
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
            models_root=resolve_project_path(require_key(self.config, "models_path")),
            report_root=resolve_project_path(require_key(self.config, "report_path")),
            artifacts_root=resolve_project_path(
                require_key(self.config, "artifacts_path")
            ),
        )
