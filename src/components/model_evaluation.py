"""
Scores the trained model on the frozen test split and writes the submission.

Input:  models/model_<N>/model.pkl, features.json, and the three prepared splits
Output: reports/model_<N>/evaluation.md, reports/submissions/submission_model_<N>.csv

Run with: python -m src.pipeline.stage_04_model_evaluation

Two jobs that belong together because they are the same job: measure the model,
then act on the measurement. The evaluation report and the submission file are
both outputs of "what does this model do on data it has never seen".

Note on the architecture: .lead/PROJECT-STRUCTURE.md warns against adding a fifth
pipeline stage. Writing a submission file is where the test set gets opened, so it
belongs here rather than in a stage of its own.
"""

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from src.constants import BASELINE_ACCURACY, BASELINE_ROC_AUC
from src.entity.config_entity import (
    EvaluationMetrics,
    ModelEvaluationArtifact,
    ModelEvaluationConfig,
)
from src.utils.common import (
    apply_categorical_encoding,
    check_columns,
    load_json,
    log_step,
    save_dataframe,
)

# How many of the worst-scoring wrong rows to keep for the report. Separate from
# `read_errors`' own default because the count of mistakes is reported separately
# and must not be read off the length of a truncated table.
ERROR_PREVIEW_LIMIT = 20

# The metrics stage 4 can compute, and how the report names them. `config.yaml`'s
# `metrics` list selects from these keys; see `write_report`.
METRIC_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}
SUPPORTED_METRICS = frozenset(METRIC_LABELS)


def calculate_metrics(
    target: pd.Series,
    probabilities: NDArray[np.float64],
    predictions: NDArray[np.int64],
) -> EvaluationMetrics:
    """Turn raw predictions into the reported scores.

    Args:
        target: The answers.
        probabilities: The model's probability of satisfaction for each row.
        predictions: The hard labels, after the threshold is applied.

    Returns:
        Every reported metric, plus the confusion matrix counts.

    Note:
        ROC-AUC is computed on the probabilities, not the hard labels. Ranking is
        what AUC measures, and a threshold would throw that away.

        `labels=[0, 1]` is passed so the matrix is always 2x2. Without it sklearn
        drops the labels absent from the data, so a threshold above every predicted
        probability - or a degenerate split - yields a 1x1 matrix and `.ravel()`
        unpacks one value into four names: `ValueError: not enough values to
        unpack (expected 4, got 1)`. That is a confusing way to learn the threshold
        is too high.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, predictions, labels=[0, 1]
    ).ravel()
    return EvaluationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        f1=float(f1_score(target, predictions)),
        precision=float(precision_score(target, predictions)),
        recall=float(recall_score(target, predictions)),
        confusion_true_positive=int(true_positive),
        confusion_false_positive=int(false_positive),
        confusion_true_negative=int(true_negative),
        confusion_false_negative=int(false_negative),
    )


def read_errors(
    frame: pd.DataFrame,
    target: pd.Series,
    probabilities: NDArray[np.float64],
    limit: int = 20,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return the rows the model got wrong, worst-confidence first.

    Args:
        frame: The scored rows, used only for their ids.
        target: The answers.
        probabilities: The model's probability of satisfaction.
        limit: How many rows to return.
        decision_threshold: The cut used for the hard labels. This must be the
            same value the reported metrics used; at the default 0.5 a
            non-default threshold produces an error table describing a different
            model's mistakes than the one the report scores.

    Returns:
        A frame of the model's worst mistakes, at most `limit` rows.

    Note:
        .lead/02-C Step 9 calls this the highest-yield hour of the whole step. The
        second row of such a table is usually the next feature idea.

        The return value is truncated, so `len(result)` is the preview size rather
        than the number of mistakes. Callers that want the real count have to count
        it themselves - logging this length as "N wrong rows" reported 20 whatever
        the true figure was.
    """
    scored = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "actual_satisfied": target.to_numpy(),
            "predicted_probability": probabilities,
        }
    )
    scored["was_wrong"] = (scored["predicted_probability"] >= decision_threshold) != scored[
        "actual_satisfied"
    ]
    wrong = scored[scored["was_wrong"]].copy()
    wrong["confidence_in_the_wrong_answer"] = (
        1 - wrong["predicted_probability"]
    ).where(wrong["actual_satisfied"], wrong["predicted_probability"])
    return wrong.sort_values("confidence_in_the_wrong_answer", ascending=False).head(
        limit
    )


def write_evaluation_report(
    report_path,
    model_name: str,
    version: int,
    validation: EvaluationMetrics,
    test: EvaluationMetrics,
    errors: pd.DataFrame,
    submission_rows: int,
    decision_threshold: float,
    metrics: list[str],
) -> None:
    """Write the evaluation report in plain English.

    Args:
        report_path: Where to write the markdown file.
        model_name: Which method was scored.
        version: Which model version.
        validation: Scores on the validation split.
        test: Scores on the frozen test split, measured once.
        errors: The rows the model got wrong.
        submission_rows: How many rows the submission file holds.
        decision_threshold: The cut used for the hard labels.
        metrics: Which metrics to put in the report's Scores table. Must be names
            this stage can produce.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(metrics) - SUPPORTED_METRICS)
    if unknown:
        raise ValueError(
            f"config.yaml asks for {unknown}, which stage 4 cannot produce. "
            f"Supported: {sorted(SUPPORTED_METRICS)}."
        )
    # Built from `metrics`, so the list in config.yaml decides the table. It used to
    # be read into `ModelEvaluationConfig` and then never consulted: the table above
    # was hard-coded, so setting `metrics: [roc_auc]` did not remove f1, precision and
    # recall, and - because the key is `require_key`d - deleting it crashed the run
    # while changing it did nothing. That is the worst of both: a setting that looks
    # live and is not.
    #
    # ROC-AUC is always kept. It is the headline number, it is the one the
    # competition scores, and removing it from the table while leaving it in the
    # comparison below would produce a report nobody could read.
    shown = [name for name in SUPPORTED_METRICS if name in metrics]
    if "roc_auc" not in shown:
        shown.insert(0, "roc_auc")
    rows = "\n".join(
        f"| {label} | {getattr(validation, name):.6f} | {getattr(test, name):.6f} |"
        for name, label in METRIC_LABELS.items()
        if name in shown
    )
    # A fenced block, not to_markdown(). pandas needs the tabulate package for
    # to_markdown(), and adding a dependency to render ten rows of a report is
    # exactly what .dev/RULES.md rule 9 warns against.
    error_preview = "```\n" + errors.head(10).to_string(index=False) + "\n```"
    report = f"""# Evaluation: {model_name} (model_{version})

**Written automatically by stage 4. Do not edit by hand - re-run the pipeline.**

## Scores

| Metric | Validation | Test (measured once) |
|---|---|---|
{rows}

## Against the bar

| | Accuracy | ROC-AUC |
|---|---|---|
| do nothing | 0.5564 | 0.5000 |
| one-column rule, `Class == Business` | {BASELINE_ACCURACY:.4f} | {BASELINE_ROC_AUC:.4f} |
| **this model, on the test split** | **{test.accuracy:.4f}** | **{test.roc_auc:.4f}** |

## The confusion matrix, in plain words

Using a threshold of {decision_threshold} on the test split:

| | Actually satisfied | Actually not satisfied |
|---|---|---|
| **Predicted satisfied** | {test.confusion_true_positive:,} caught | {test.confusion_false_positive:,} false alarms |
| **Predicted not satisfied** | {test.confusion_false_negative:,} missed | {test.confusion_true_negative:,} correct |

Of the people who were satisfied, the model catches {test.recall:.1%} and misses
{test.confusion_false_negative:,}. Of the rows it flags, {test.precision:.1%} are
genuinely satisfied.

## The rows it got wrong

The model's most confident mistakes. `.lead/02-C` Step 9: if these all look alike,
that is the next feature idea. If they look random, it is a data problem.

{error_preview}

## Submission

{submission_rows:,} rows, one per row of `data/test.csv`.

## What is not yet decided

- **Whether ROC-AUC or accuracy is the competition metric.** Open question 6.
  Both are reported above so the answer is already here.
- **Whether this model is good enough to submit.** That is `.lead/04`'s go/no-go
  decision, and it needs a target number agreed in advance. Open question 7.
- **Whether the metric is worth tuning further.** Tuning is `.lead/03` Step 11,
  and it comes after the method is chosen, not before.
"""
    report_path.write_text(report)


def write_submission(
    competition_frame: pd.DataFrame,
    probabilities: NDArray[np.float64],
    sample_submission_path,
    output_path,
) -> int:
    """Write the competition submission file.

    Args:
        competition_frame: The competition's test rows.
        probabilities: The model's probability of satisfaction for each row, as
            returned by predict_proba.
        sample_submission_path: The template Kaggle provided.
        output_path: Where to write our submission.

    Returns:
        The number of rows written.

    Raises:
        FileNotFoundError: If the template is missing.
        ValueError: If the row count does not match the template. A submission with
            the wrong number of rows scores nothing, and Kaggle's error message
            does not say which row is missing.
    """
    if not sample_submission_path.exists():
        raise FileNotFoundError(
            f"{sample_submission_path} not found. It is the template Kaggle "
            "provided, and the submission must match its columns exactly."
        )

    template = pd.read_csv(sample_submission_path)
    if len(template) != len(competition_frame):
        raise ValueError(
            f"The template has {len(template)} rows but data/test.csv has "
            f"{len(competition_frame)}. A submission with the wrong number of "
            "rows scores nothing."
        )

    submission = pd.DataFrame(
        {
            "id": competition_frame["id"].to_numpy(),
            "satisfaction": probabilities,
        }
    )
    # Reindexed onto the template's column order, then compared as an ordered list.
    # `check_columns` computes `expected - actual`, so it only ever caught template
    # columns the submission lacked - and the submission's two columns are hard-coded
    # just above, so it could catch exactly one thing. It never checked order, never
    # checked for extra columns, and so could not detect the failure it is standing
    # in for. A competition that reads the template's second column by position
    # scores `id` instead of the prediction.
    submission = submission.reindex(columns=list(template.columns))
    check_columns("submission", set(submission.columns), set(template.columns))
    if list(submission.columns) != list(template.columns):
        raise ValueError(
            f"submission columns {list(submission.columns)} do not match the "
            f"template's {list(template.columns)}. A submission has to match the "
            f"template's columns, in its order."
        )
    save_dataframe(submission, output_path)
    return len(submission)


def run_model_evaluation(config: ModelEvaluationConfig) -> ModelEvaluationArtifact:
    """Score the trained model on the frozen test split and write the submission.

    Args:
        config: Where the model, the data, the template and the outputs are.

    Returns:
        The paths written, both sets of metrics, and the submission row count.

    Raises:
        FileNotFoundError: If the model or the prepared data is missing, meaning
            an earlier stage has not run.
        ValueError: If the saved feature list does not match the prepared data.
    """
    for path in (
        config.model_path,
        config.features_path,
        config.validation_data_path,
        config.test_data_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Stages run in order: "
                "python -m src.pipeline.stage_03_model_training"
            )

    model = joblib.load(config.model_path)
    # Checked before it is dereferenced. The `check_columns` call came second, so a
    # contract missing `features` produced a bare KeyError from the line above
    # rather than the intended message naming features.json - and the file is read
    # twice below, once for each of the two purposes it serves.
    saved = load_json(config.features_path)
    check_columns("evaluate", set(saved.keys()), {"features"})
    features: list[str] = saved["features"]

    # The encoding contract comes from features.json, written by stage 2 beside the
    # features themselves. CSV does not carry dtypes, so without this the four
    # categorical columns arrive as `str` and xgboost rejects them.
    categorical_columns = list(saved.get("categorical_columns", []))
    encoding = str(saved.get("categorical_encoding", "one_hot"))
    category_maps = saved.get("category_maps") or {}

    validation_data = apply_categorical_encoding(
        pd.read_csv(config.validation_data_path),
        categorical_columns,
        encoding,
        "evaluate.validation",
        category_maps,
    )
    test_data = apply_categorical_encoding(
        pd.read_csv(config.test_data_path),
        categorical_columns,
        encoding,
        "evaluate.test",
        category_maps,
    )
    check_columns(
        "evaluate.validation",
        set(validation_data.columns),
        set(features) | {config.target_column},
    )
    check_columns(
        "evaluate.test", set(test_data.columns), set(features) | {config.target_column}
    )

    # Predicted once. This was called twice on validation and once on test, and on
    # 105k rows by 178 features with a RealMLP it is the most expensive operation
    # in the stage - three times over, for one number.
    validation_probabilities = model.predict_proba(validation_data[features])[:, 1]
    validation_metrics = calculate_metrics(
        validation_data[config.target_column],
        validation_probabilities,
        (validation_probabilities >= config.decision_threshold).astype(int),
    )
    test_probabilities = model.predict_proba(test_data[features])[:, 1]
    test_metrics = calculate_metrics(
        test_data[config.target_column],
        test_probabilities,
        (test_probabilities >= config.decision_threshold).astype(np.int64),
    )
    log_step(
        "evaluate",
        validation_rows=len(validation_data),
        test_rows=len(test_data),
        test_roc_auc=round(test_metrics.roc_auc, 6),
        test_accuracy=round(test_metrics.accuracy, 6),
    )

    # The configured threshold, not `read_errors`' own 0.5. The metrics above cut
    # at `decision_threshold` and the error table cut at 0.5, so with any other
    # setting the rows a reviewer reads as "wrong" were the rows a different model
    # would have got wrong - and the confusion matrix in the report, which is
    # built from these labels, disagreed with the metrics printed beside it.
    errors = read_errors(
        test_data,
        test_data[config.target_column],
        test_probabilities,
        decision_threshold=config.decision_threshold,
        limit=ERROR_PREVIEW_LIMIT,
    )
    # `read_errors` returns a truncated table, so `len(errors)` is the preview size
    # and not the number of mistakes. Logging that as "N wrong rows" reported 20
    # when the real figure was in the thousands.
    wrong_total = int(
        ((test_probabilities >= config.decision_threshold).astype(np.int64)
         != test_data[config.target_column].to_numpy()).sum()
    )
    log_step("errors", wrong_rows=wrong_total, preview=len(errors))

    competition_frame = apply_categorical_encoding(
        pd.read_csv(config.competition_test_path),
        categorical_columns,
        encoding,
        "evaluate.competition",
        category_maps,
    )
    check_columns("evaluate.competition", set(competition_frame.columns), set(features))
    competition_probabilities = model.predict_proba(competition_frame[features])[:, 1]

    submission_path = (
        config.submission_dir
        / f"submission_model_{config.model_path.parent.name.split('_')[-1]}.csv"
    )
    submission_rows = write_submission(
        competition_frame,
        competition_probabilities,
        config.sample_submission_path,
        submission_path,
    )
    log_step("submission", rows_out=submission_rows, wrote=str(submission_path))

    report_path = config.report_dir / "evaluation.md"
    write_evaluation_report(
        report_path,
        model_name=str(
            load_json(config.model_path.parent / "config.json")["model_name"]
        ),
        version=int(config.model_path.parent.name.split("_")[-1]),
        validation=validation_metrics,
        test=test_metrics,
        errors=errors,
        submission_rows=submission_rows,
        decision_threshold=config.decision_threshold,
        metrics=list(config.metrics),
    )

    return ModelEvaluationArtifact(
        report_path=report_path,
        submission_path=submission_path,
        test_metrics=test_metrics,
        validation_metrics=validation_metrics,
        submission_rows=submission_rows,
    )
