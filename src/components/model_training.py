"""
Fits the model named in config.yaml and saves it as a self-contained bundle.

Input:  artifacts/data_cleaning_encoding/train.csv
Output: models/model_<N>/model.pkl, features.json, config.json, model_card.md

Run with: python -m src.pipeline.stage_03_model_training

This is the only place a library estimator is created. The name in config.yaml
becomes a class through one dictionary, MODEL_REGISTRY, so trying another method
is a one-line edit in the config and nothing else changes.

All five files in models/model_<N>/ travel together. A model without its feature
list is unmaintainable, because nobody can say what it was trained on.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cloudpickle
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.entity.config_entity import ModelTrainerArtifact, ModelTrainerConfig
from src.utils.common import (
    apply_categorical_encoding,
    check_columns,
    get_git_commit,
    load_json,
    log_step,
    save_json,
    seed_torch,
)


def add_encoding_parameter(
    model_name: str, params: dict[str, Any], categorical_encoding: str
) -> dict[str, Any]:
    """Return the parameters with the right categorical handling added.

    Args:
        model_name: Which family is being built.
        params: The hyperparameters from config.yaml.
        categorical_encoding: "native" or "one_hot", as recorded by stage 2.

    Returns:
        The parameters to build the model with.

    Note:
    `enable_categorical` is an XGBoost-only argument, and it is derived from the
    encoding stage 2 actually used rather than set separately in config.yaml. Two
    switches controlling one thing is two places to forget, and forgetting one
    produces a dtype error deep inside a fit.
    """
    if categorical_encoding not in ("native", "one_hot"):
        raise ValueError(
            f"Unknown categorical_encoding '{categorical_encoding}'. "
            "Supported: 'native', 'one_hot'."
        )
    if model_name == "xgboost" and categorical_encoding == "native":
        return {"enable_categorical": True, **params}
    return dict(params)


def build_realmlp_classifier(**params: Any):
    """Create a RealMLP_TD_Classifier, importing pytabkit only when asked for.

    Args:
        **params: Whatever build_model passes, which is the config.yaml
            hyperparameters plus `random_state`. Matches how a scikit-learn
            class is called, so the registry stays uniform.

    Returns:
        An unfitted RealMLP_TD_Classifier.

    Note:
    RealMLP is a tabular-specific neural network (categorical embeddings, PLR
    numerical embeddings, tuned schedule) - not a basic MLP. Measured 0.959051
    on our 22 features against 0.958868 for XGBoost native. Same pattern as
    xgboost: lazy import so torch stays an optional dependency for runs that
    never use it.
    """
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ImportError as error:
        raise ImportError(
            "config.yaml asks for model_name: realmlp but pytabkit is not "
            "installed. Install it with `uv pip install pytabkit`, or change "
            "model_name to one of the other options."
        ) from error
    return RealMLP_TD_Classifier(**params)


def build_xgboost_classifier(**params: Any):
    """Create an XGBClassifier, importing xgboost only when it is asked for.

    Args:
        **params: Whatever build_model passes, which is the config.yaml
            hyperparameters plus `random_state`. The signature matches how a
            scikit-learn class is called, so the registry stays uniform.

    Returns:
        An unfitted XGBClassifier.

    Note:
    xgboost is not a scikit-learn estimator, so it cannot sit in the registry as a
    plain class - it needs importing, and importing it at module level would make
    it a hard dependency for a model most runs never use. The wrapper gives every
    registry entry the same signature, which is what keeps the dispatch in
    build_model honest.

    An earlier version of this took `(params, seed)`, which build_model does not
    call that way. The test that builds every registered model caught it.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "config.yaml asks for model_name: xgboost but xgboost is not "
            "installed. Install it with `uv pip install xgboost`, or change "
            "model_name to one of the scikit-learn options."
        ) from error
    return XGBClassifier(**params)


# Key: the name used in config.yaml. Value: a class, or a builder with the same
# signature. This is the single dispatch point for the whole project.
#
# Is a registry a "factory", which .dev/RULES.md rule 7 forbids? Yes, technically
# - and it is allowed here for one specific proven reason: we compare several
# methods before choosing one. See .lead/02-C-CHOOSING-THE-METHOD.md Step 7.
# If the project ever uses only one model, delete this and instantiate it directly.
class Estimator(Protocol):
    """What every model in MODEL_REGISTRY must be able to do.

    The registry holds six unrelated classes - four scikit-learn estimators,
    XGBoost, and RealMLP. They share no base class, so this names the three
    methods the pipeline actually calls on them. Declaring it means build_model
    can promise a return type, and callers can use .fit() and .predict_proba()
    without a type checker objecting that the attribute does not exist.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...

    def predict_proba(self, X: Any) -> Any: ...


MODEL_REGISTRY: dict[str, Callable[..., "Estimator"]] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "xgboost": build_xgboost_classifier,
    "realmlp": build_realmlp_classifier,
}

# Models whose constructor takes no `random_state`. Passing one would raise
# TypeError, and per .dev/RULES.md rule 8 the run must stop with a clear message
# rather than fail obscurely.
#
# Empty, and populated by what it is rather than by what someone remembered to
# type: `build_model` inspects the constructor's signature and names any model
# that cannot take the seed. A hand-maintained list is a list that goes stale -
# this one has been an empty frozenset since it was added, so the branch below it
# could never fire and the "must stop with a clear message" it documents had no
# path to produce one.
MODELS_WITHOUT_RANDOM_STATE: frozenset[str] = frozenset()

# Overrides the overfitting check applies so it can do its job.
#
# The check asks "can this model family learn at all?", on 16 rows. Tree models
# are tuned for datasets far larger than that: HistGradientBoosting defaults to
# min_samples_leaf=20, which is more rows than the dataset has, so no split is
# ever legal and the model can only ever predict the average. That is a property
# of the default, not a fault in the model, so the check overrides it.
#
# Anything passed in params wins over these. That is deliberate, and the test
# suite relies on it to prove the check can still fail.
#
# Note what is NOT here: the production hyperparameters from config.yaml. The
# check asks "can this model family learn at all?", which is a question about the
# family. Production settings are deliberately not passed in, because they are
# tuned for 490,000 rows and cannot fit 16 by design - min_samples_leaf=40 is a
# sensible production value and a fatal one on a 16-row dataset. Feeding it in
# made this check fail on every run, which is the same as having no check.
TINY_DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {"min_samples_leaf": 1, "max_iter": 300},
    "random_forest": {"min_samples_leaf": 1, "n_estimators": 20},
    "logistic_regression": {"max_iter": 5000},
    "decision_tree": {"min_samples_leaf": 1},
    # min_child_weight defaults to 1, which is already fine, but 16 rows needs
    # enough trees and a high enough learning rate to converge in one pass.
    "xgboost": {"n_estimators": 200, "learning_rate": 0.5},
    "realmlp": {"n_ens": 1, "n_epochs": 2},
}


def build_model(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    categorical_encoding: str = "one_hot",
) -> "Estimator":
    """Create the model named in the configuration.

    Args:
        model_name: Key from MODEL_REGISTRY, for example "random_forest".
        params: Hyperparameters from config.yaml.
        seed: Random seed, so the model is reproducible.
        categorical_encoding: Which categorical encoding stage 2 used. Only
            xgboost reads `category` dtype; the scikit-learn estimators cannot,
            which is why config.yaml can still ask for one_hot.

    Returns:
        An unfitted model, ready for fit().

    Raises:
        KeyError: If the name is not in the registry, listing the names that are.
        ValueError: If native encoding is paired with a model that cannot read it.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}. "
            "Add a line to MODEL_REGISTRY in src/components/model_training.py if "
            "the method is genuinely needed."
        )
    # Models that read pandas `category` dtype natively. Everything else needs
    # one_hot, because scikit-learn estimators cannot read categories at all.
    NATIVE_CAPABLE_MODELS = frozenset({"xgboost", "realmlp"})
    if categorical_encoding == "native" and model_name not in NATIVE_CAPABLE_MODELS:
        raise ValueError(
            f"categorical_encoding is 'native' but the model is '{model_name}'. "
            f"Only {sorted(NATIVE_CAPABLE_MODELS)} can read pandas `category` dtype. "
            "Set categorical_encoding: one_hot in config.yaml for any other model."
        )
    params = add_encoding_parameter(model_name, params, categorical_encoding)
    model_class = MODEL_REGISTRY[model_name]
    if not _accepts_random_state(model_class):
        raise ValueError(
            f"model '{model_name}' ({model_class.__name__}) has no random_state "
            f"parameter, so this run cannot be reproduced. Add a seed to it, or "
            f"remove it from MODEL_REGISTRY - per .dev/RULES.md rule 8 the run "
            f"stops here rather than training an unseedable model."
        )
    # The caller's `random_state` wins. `model_params` in config.yaml commonly
    # carries one too, and `model_class(random_state=seed, **params)` with both is
    # `TypeError: got multiple values for keyword argument 'random_state'`. The
    # explicit argument is the pipeline's single seed, set once from
    # `random_seed`; a duplicate inside model_params was always a collision, and
    # failing on it at the constructor tells the reader nothing about which of the
    # two seeds was meant to win.
    params.pop("random_state", None)
    return model_class(random_state=seed, **params)


def _accepts_random_state(model_class) -> bool:
    """Whether a model class can be constructed with `random_state`.

    Args:
        model_class: The estimator class.

    Returns:
        True if `random_state` is in the constructor's signature, or the class
        accepts arbitrary keyword arguments.

    Note:
    Inspected rather than looked up in a list. `MODEL_REGISTRY` already holds
    every model this pipeline can build, so asking the class is both shorter and
    impossible to forget when one is added - which is what happened to the
    hand-written list this replaces.
    """
    import inspect

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(model_class.__init__).parameters.values()
    ):
        return True
    return "random_state" in inspect.signature(model_class.__init__).parameters


def score_model(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    """Score a fitted model on held-out rows.

    Args:
        model: A fitted estimator with predict_proba.
        features: The rows to score.
        target: The answers for those rows.
        decision_threshold: The cut used for the hard labels. Must be the same
            value stage 4 uses, or the two reports disagree.

    Returns:
        ROC-AUC, accuracy, F1, precision and recall.

    Note:
        ROC-AUC is reported as well as accuracy because open question 6 asks
        whether the competition scores one or the other. Reporting both means the
        answer is already here. The two agree on the ranking.

        The hard labels are cut at `decision_threshold` applied to the
        probabilities, not taken from `model.predict`. `predict` uses whatever
        0.5 the estimator had baked in, so with any other configured threshold
        stage 3 reported accuracy, F1, precision and recall for one model and
        stage 4 reported the same four for another - on the same rows, in the same
        model card, as two different numbers. `ModelTrainerConfig` had no threshold
        field at all, so the two stages could not agree unless the config happened
        to say 0.5.
    """
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions)),
        "precision": float(precision_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
    }


def run_overfitting_check(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    categorical_encoding: str = "one_hot",
) -> bool:
    """Check that the model family can memorise 16 rows.

    A model that cannot drive the loss to near zero on 16 rows is broken - the
    learning rate is wrong, or the labels and features are mismatched. No amount of
    real data will hide it, and the failure is otherwise found days later.

    This is Test 2 from .dev/RULES.md rule 10, and Step 3.3 of
    .lead/03-TRAIN-AND-TUNE.md. The same check runs as a unit test in
    tests/components/test_model_training.py; running it here as well means a
    training run stops rather than producing a model that cannot learn.

    Args:
        model_name: Which family to check.
        params: Overrides on top of TINY_DATASET_OVERRIDES. The production
            settings from config.yaml are deliberately not passed in - see the
            note on TINY_DATASET_OVERRIDES. The tests use this argument to force a
            failure and prove the check is not vacuous.
        seed: Random seed.

    Returns:
        True if the model memorised the rows.

    Raises:
        AssertionError: If the model cannot fit 16 rows, naming the model. The
            message says to fix the code, not to look for a data problem.
    """
    # The pattern must be learnable by every family in the registry, including a
    # linear one. An earlier version used `index % 2` as the label, which is not
    # separable by a straight line, so logistic regression could never pass no
    # matter how it was configured. `index % 4 >= 2` is a clean threshold on
    # feature_a.
    features = pd.DataFrame(
        {
            "feature_a": [float(index % 4) for index in range(16)],
            "feature_b": [float(index % 3) for index in range(16)],
        }
    )
    target = pd.Series([bool(index % 4 >= 2) for index in range(16)])

    check_params = {**TINY_DATASET_OVERRIDES.get(model_name, {}), **params}
    # The tiny dataset has no categorical columns, so it is always built as one_hot
    # regardless of what the real data uses. native here would ask for a category
    # column that does not exist in a 16-row frame of numbers.
    model = build_model(model_name, check_params, seed, categorical_encoding="one_hot")
    model.fit(features, target)
    memorised = bool((model.predict(features) == target).all())
    if not memorised:
        raise AssertionError(
            f"Overfitting check failed for '{model_name}'. The model cannot "
            "memorise 16 rows, so training itself is broken - check the "
            "hyperparameters and the label, not the data. See "
            ".lead/03-TRAIN-AND-TUNE.md Step 3.3."
        )
    log_step("overfitting_check", model=model_name, rows=16, memorised=True)
    return True


def write_model_card(
    model_path: Path,
    model_name: str,
    params: dict[str, Any],
    validation_metrics: dict[str, float],
    features: list[str],
    seed: int,
    dropped_features: dict[str, str] | None = None,
) -> Path:
    """Write the document that lets someone who was not here understand the model.

    Args:
        model_path: Where the model file is, for the record.
        model_name: Which method was used.
        params: The hyperparameters used.
        validation_metrics: What it scored on the validation split.
        features: The exact feature list, in order.
        seed: The seed used.
        dropped_features: Column name to the measured reason it is not a feature.
            Written into the card.

    Returns:
        The path of the written card.

    Note:
        Written in plain English, because the reader is a person who has never
        seen this project. Template from .lead/09-TEMPLATES.md section 4.
    """
    git_commit = get_git_commit()
    feature_lines = "\n".join(f"- {name}" for name in features)
    # The reasons themselves, not a pointer to config.yaml. The reader of a model
    # card is someone who was not here and will not go looking; "see config.yaml"
    # is the one thing that cannot be acted on from the document in front of them.
    dropped = dropped_features or {}
    # A list of names is accepted as well as a name -> reason mapping, because
    # that is the shape an older features.json carries: stage 2 wrote the literal
    # string "see config.yaml dropped_features" under this key, and a reader
    # upgrading the artifacts would hand us a string or a list. Normalised here so
    # the caller does not have to know which it has.
    if isinstance(dropped, str):
        entries = [(dropped, "recorded before the reason was stored")]
    elif isinstance(dropped, dict):
        entries = list(dropped.items())
    else:
        entries = [(str(name), "no reason recorded") for name in dropped]
    dropped_lines = (
        "\n".join(f"- `{column}` - {reason}" for column, reason in entries)
        or "- (none recorded)"
    )
    card = f"""# Model Card: {model_name}

**Written automatically by stage 3. Do not edit by hand - re-run the pipeline.**

## What it predicts
Whether a passenger was satisfied with their flight. The output is a probability
from 0 to 1.

**Who it is for.** A Kaggle Playground Series S6E10 submission file. Nobody
consumes this operationally.

**Who it is NOT for.** This is not a system that can warn an airline that a flight
is about to go badly. Every feature is the passenger's own answer on the same
survey form as the label, so the model infers one survey answer from the others.
It cannot predict a reaction, because nothing is known until after the flight.

## Data it needs
{len(features)} features, listed in `features.json` in this folder, in this order.
Prediction refuses to run if the input does not have exactly these columns.

## How it was trained
- Method: `{model_name}`
- Hyperparameters: `{params}`
- Random seed: {seed}
- Git commit: {git_commit}
- Trained on: `artifacts/data_cleaning_encoding/train.csv`
- Scored on: `artifacts/data_cleaning_encoding/validation.csv`, which the model
  never saw during fitting

## How well it does
Measured on the validation split:

| Metric | Value |
|---|---|
| ROC-AUC | {validation_metrics["roc_auc"]:.6f} |
| Accuracy | {validation_metrics["accuracy"]:.6f} |
| F1 | {validation_metrics["f1"]:.6f} |
| Precision | {validation_metrics["precision"]:.6f} |
| Recall | {validation_metrics["recall"]:.6f} |

**The bar it had to beat.** A one-line rule on cabin class,
`Class == Business`, scores 0.7770 accuracy and 0.7786 ROC-AUC. See
`docs/eda_findings.md` section 4.

## Known weaknesses
- The training data is **synthetic**, generated by the competition organisers from
  a model fitted to a real survey. A good score here is not a finding about real
  airline passengers.
- These columns were measured and dropped. If any of them matter in the real
  world, this model misses it:

{dropped_lines}
- Whether `0` in twelve of the thirteen service ratings means "worst possible" or
  "not applicable" is unanswered. If it means "not applicable", the model is
  treating a missing value as a number. See `docs/column_dictionary.md` Note 3.

## The exact feature list
{feature_lines}

## What to do if it fails
Fall back to the `Class == Business` rule. It is 0.7770 accurate and needs no
model file. See `docs/runbook_competition.md`.
"""
    card_path = model_path.parent / "model_card.md"
    card_path.write_text(card)
    return card_path


def run_model_training(
    config: ModelTrainerConfig, validation_data_path: Path | None = None
) -> ModelTrainerArtifact:
    """Train the model named in config.yaml and save the bundle.

    Args:
        config: Which model, which hyperparameters, which features, where to save.
        validation_data_path: Where the held-out rows are. Read for scoring only,
            never for fitting.

    Returns:
        The paths in the saved bundle, the validation scores, and how long it took.

    Raises:
        FileNotFoundError: If the prepared training data is missing, meaning
            stage 2 has not run.
        KeyError: If model_name is not in MODEL_REGISTRY.
        ValueError: If a configured feature is missing from the data.
        AssertionError: If the model cannot memorise 16 rows.
    """
    if not config.train_data_path.exists():
        raise FileNotFoundError(
            f"{config.train_data_path} not found. Run stage 2 first: "
            "python -m src.pipeline.stage_02_data_cleaning_encoding"
        )

    log_step(
        "train",
        model=config.model_name,
        params=config.model_params,
        seed=config.random_seed,
        source=config.train_data_path,
    )

    # Called with no params on purpose. Production settings are tuned for 490,000
    # rows and cannot memorise 16, so passing them here would fail every run.
    # What this proves is that the estimator can learn at all - if it cannot, no
    # amount of data will help and the run stops now rather than in three days.
    run_overfitting_check(
        config.model_name,
        {},
        config.random_seed,
        categorical_encoding=config.categorical_encoding,
    )

    train_data = pd.read_csv(config.train_data_path)
    # CSV does not carry dtypes, so a `category` column written by stage 2 comes
    # back as `str`. The contract for how the data was encoded lives beside the
    # features themselves, in features.json, which stage 2 wrote - so it is read
    # from there rather than assumed here.
    contract = load_json(config.train_data_path.parent / "features.json")
    categorical_columns = list(contract.get("categorical_columns", []))
    category_maps = contract.get("category_maps") or {}
    encoding = str(contract.get("categorical_encoding", "one_hot"))
    train_data = apply_categorical_encoding(
        train_data,
        categorical_columns,
        encoding,
        "train",
        category_maps,
    )
    log_step(
        "train",
        encoding=encoding,
        categorical_columns=len(categorical_columns),
        pinned_levels=len(category_maps),
    )
    check_columns(
        "train", set(train_data.columns), set(config.features) | {config.target_column}
    )

    if config.model_name == "realmlp":
        # RealMLP trains on torch, so torch and CUDA RNGs need seeding too - the
        # random_state passed to the estimator is not enough. GPU training is still
        # not bit-reproducible; compare at 0.0005 tolerance, never exactly.
        seed_torch(config.random_seed)
    model = build_model(
        config.model_name,
        config.model_params,
        config.random_seed,
        categorical_encoding=config.categorical_encoding,
    )

    started = time.monotonic()
    model.fit(train_data[config.features], train_data[config.target_column])
    training_seconds = time.monotonic() - started

    if training_seconds > config.max_training_seconds:
        raise ValueError(
            f"Training took {training_seconds:.0f}s, over the budget of "
            f"{config.max_training_seconds}s set in config.yaml. A training run "
            "that overruns its budget means the settings changed without being "
            "committed. Stop and check."
        )
    log_step("train", rows_in=len(train_data), fit_seconds=round(training_seconds, 1))

    validation_metrics: dict[str, float] = {}
    if validation_data_path is not None and Path(validation_data_path).exists():
        validation_data = apply_categorical_encoding(
            pd.read_csv(validation_data_path),
            categorical_columns,
            encoding,
            "validate",
            category_maps,
        )
        validation_metrics = score_model(
            model,
            validation_data[config.features],
            validation_data[config.target_column],
            config.decision_threshold,
        )
        log_step(
            "validate",
            rows_in=len(validation_data),
            **{name: round(value, 6) for name, value in validation_metrics.items()},
        )
    else:
        log_step("validate", skipped=True, reason="no validation file found")

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / "model.pkl"
    # cloudpickle, not joblib: the tuned RealMLP recipe stores wd_sched,
    # p_drop_sched and ls_eps_sched as compiled lambdas, and plain pickle
    # cannot serialise those. cloudpickle serialises them by value. joblib.load
    # still reads the file back, because cloudpickle output is valid pickle.
    # cloudpickle.dump wants a file object or a str, not a Path, so open it.
    with model_path.open("wb") as handle:
        cloudpickle.dump(model, handle)

    # The model bundle carries its own encoding contract, not just a feature list.
    # .lead/03-TRAIN-AND-TUNE.md Step 3.6: all five files travel together, and a
    # model that cannot be fed correctly is unmaintainable. Stage 4 reads this
    # file, so the categorical columns and encoding have to be in it.
    features_path = save_json(
        {
            "features": config.features,
            "target_column": config.target_column,
            "categorical_encoding": str(
                contract.get("categorical_encoding", "one_hot")
            ),
            "categorical_columns": list(contract.get("categorical_columns", [])),
            "derived_features": "see config.yaml",
        },
        config.model_dir / "features.json",
    )
    config_path = save_json(
        {
            "model_name": config.model_name,
            "model_params": config.model_params,
            "random_seed": config.random_seed,
            "git_commit": get_git_commit(),
            "validation_metrics": validation_metrics,
            "training_seconds": round(training_seconds, 2),
            "train_rows": len(train_data),
            "features": config.features,
        },
        config.model_dir / "config.json",
    )
    card_path = write_model_card(
        model_path,
        config.model_name,
        config.model_params,
        validation_metrics,
        config.features,
        config.random_seed,
        # Read from stage 2's contract, which reads it from config.yaml, so the
        # card cannot quote a different set of dropped columns than the run used.
        (load_json(config.train_data_path.parent / "features.json").get(
            "dropped_features"
        ) or {}),
    )
    log_step("train", wrote=str(model_path), version=config.model_dir.name)

    return ModelTrainerArtifact(
        model_path=model_path,
        features_path=features_path,
        config_path=config_path,
        model_card_path=card_path,
        model_version=int(config.model_dir.name.split("_")[-1]),
        validation_metrics=validation_metrics,
        training_seconds=training_seconds,
        overfitting_test_passed=True,
    )
