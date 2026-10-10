"""
Typed descriptions of what each stage's settings and outputs must contain.

This file defines the agreement between stages. If stage 1 produces something
that does not match, Python stops the program with a clear message instead of
letting stage 3 fail with a confusing error three steps later.

See .lead/02-A-ARCHITECTURE.md section 2.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataIngestionConfig:
    """Settings needed to load the raw data."""

    data_path: Path
    competition_test_path: Path
    artifacts_dir: Path
    required_columns: frozenset[str]
    banned_features: frozenset[str]


@dataclass(frozen=True)
class DataIngestionArtifact:
    """What stage 1 produced, and where it was saved."""

    raw_data_path: Path
    competition_test_path: Path
    row_count: int
    column_names: list[str]


@dataclass(frozen=True)
class DataCleaningConfig:
    """Settings needed to split, derive features and encode categories.

    Note:
    `dropped_features` maps each column the pipeline decided not to use to the
    measured reason. It is carried into the model card, where a reader needs the
    number, not the name of the file it came from.
    """

    target_column: str
    features: list[str]
    candidate_features_pending_question: list[str]
    derived_feature_names: list[str]
    banned_features: frozenset[str]
    dropped_features: dict[str, str]
    service_rating_columns: list[str]
    categorical_columns: list[str]
    continuous_columns: list[str]
    test_size: float
    validation_size: float
    random_seed: int
    clip_outliers: bool
    clip_lower_percentile: float
    clip_upper_percentile: float
    arrival_delay_median: float
    """The value that WOULD be substituted for a blank delay.

    Blanks are kept as the `arrival_delay_status: unknown` group instead, so this
    is recorded every run and deliberately not applied. It used to be read and used
    by nothing at all.
    """

    arrival_delay_column: str
    """Which column `arrival_delay_status` groups. Previously hard-coded."""

    categorical_encoding: str
    route_features_enabled: bool
    route_smoothing: float
    aux_features_enabled: bool
    categorical_twins_enabled: bool
    artifacts_dir: Path


@dataclass(frozen=True)
class DataCleaningArtifact:
    """What stage 2 produced: the three splits and the feature list."""

    train_path: Path
    validation_path: Path
    test_path: Path
    competition_test_path: Path
    features_path: Path
    split_report_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    features: list[str]
    categorical_encoding: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Settings needed to train one model."""

    train_data_path: Path
    model_dir: Path
    model_name: str
    model_params: dict[str, Any]
    features: list[str]
    target_column: str
    random_seed: int
    max_training_seconds: int
    categorical_encoding: str
    decision_threshold: float
    """The cut used for hard labels here and in stage 4.

    Stage 3 reports accuracy, F1, precision and recall on validation, and stage 4
    reports the same four on validation and test. Without this field the two cut
    at different places - `model.predict`'s baked-in 0.5 here, `config
    .decision_threshold` there - so the same rows scored two ways in one model
    card.
    """


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """What stage 3 produced: the model bundle and what it scored."""

    model_path: Path
    features_path: Path
    config_path: Path
    model_card_path: Path
    model_version: int
    validation_metrics: dict[str, float]
    training_seconds: float
    overfitting_test_passed: bool


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Settings needed to score a trained model and write a submission."""

    validation_data_path: Path
    test_data_path: Path
    competition_test_path: Path
    sample_submission_path: Path
    model_path: Path
    features_path: Path
    target_column: str
    decision_threshold: float
    metrics: list[str]
    report_dir: Path
    submission_dir: Path


@dataclass(frozen=True)
class EvaluationMetrics:
    """The scores we report, so every run is measured the same way."""

    roc_auc: float
    accuracy: float
    f1: float
    precision: float
    recall: float
    confusion_true_positive: int
    confusion_false_positive: int
    confusion_true_negative: int
    confusion_false_negative: int


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """What stage 4 produced: the report and the submission file."""

    report_path: Path
    submission_path: Path
    test_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    submission_rows: int


@dataclass(frozen=True)
class ViewConfig:
    """Which feature views stage 6 should try, and the bar to clear.

    Note:
    A view is a narrower column set the same estimator is trained on, not a
    different architecture. The point is decorrelation: stage 5's four members all
    see the same columns and score below their own best member when stacked, while
    the same model scores 0.9524 on the ratings alone and 0.9591 on everything -
    a +0.0067 spread that is what a stack can combine.

    `min_gain` is the acceptance threshold on the NESTED stack AUC, in AUC units.
    The nested score moves by roughly +/-0.0003 between fold seeds on this
    dataset, so 0.0004 sits just above the measured noise: a real effect clears it
    and a lucky seed does not. Lowering it to 0.0001 would accept views that are
    indistinguishable from re-drawing the folds, which is how a stack grows four
    members and gains nothing.
    """

    enabled: bool
    views: list[dict[str, Any]]
    min_gain: float


@dataclass(frozen=True)
class PipelineConfig:
    """Everything run_pipeline.py needs, gathered from config.yaml."""

    ingestion: DataIngestionConfig
    cleaning: DataCleaningConfig
    training: ModelTrainerConfig
    evaluation: ModelEvaluationConfig
    models_root: Path
    report_root: Path
    artifacts_root: Path

