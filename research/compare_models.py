"""
Compares every candidate method on the same split, features, seed and metric.

Run with: python research/compare_models.py

This decides which single model goes forward. It is a one-off decision tool and is
not part of the production pipeline - see .lead/02-C-CHOOSING-THE-METHOD.md Step 7.

The rules that make the comparison honest, all from .lead/02-C Step 7:

| Rule | Why |
|---|---|
| Same features for every candidate | Otherwise we are comparing data, not methods |
| Same split | Otherwise we are comparing luck |
| Same metric | Otherwise the numbers mean nothing |
| Same seed, from config.yaml | Removes one source of randomness |
| Default settings, no tuning | Tuning five methods properly is weeks of work, and Step 11 is last, not first |
| Validation only, never the test split | .lead/01-DATA.md Step 1.7 rule 3 |

Both metrics are reported. Open question 6 asks whether the competition scores
ROC-AUC or accuracy, and it is not settled. Reporting both means the answer is
already here.
"""

import sys
import time
from pathlib import Path

# Running this file as `python research/compare_models.py` puts research/ on the
# import path, not the project root, so `import src...` fails. Adding the root
# explicitly keeps the documented command working without an installed package or
# a setup.py. Nothing else needs this: the pipeline is always run from the root as
# `python run_pipeline.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import pandas as pd

from src.components.model_training import build_model
from src.config.configuration import PipelineConfigReader
from src.utils.common import (
    check_columns,
    get_logger,
    log_step,
    save_dataframe,
    setup_logging,
)

# The shortlist from docs/method_plan.md Step 5, with each one's reason. Written
# down before anything was run, so nobody can quietly drop the ones that do badly.
#
# The params are DEFAULT settings on purpose. .lead/02-C Step 7 requires every
# candidate to be run once with defaults first.
CANDIDATES: dict[str, dict[str, object]] = {
    "logistic_regression": {},
    "decision_tree": {},
    "random_forest": {},
    "hist_gradient_boosting": {},
}

# xgboost is in MODEL_REGISTRY only if it is installed. It is not a scikit-learn
# estimator, so it is registered separately below rather than pretending otherwise.
XGBOOST_REASON = (
    "Added because the owner requires easy hyperparameter optimisation. Both "
    "XGBoost and LightGBM have first-class Optuna support and mature "
    "regularisation knobs. Its advantage is tuning ergonomics, not accuracy."
)

BASELINES: list[tuple[str, float, float]] = [
    ("do nothing", 0.5564, 0.5000),
    ("one-column rule: Class == Business", 0.7770, 0.7786),
]


def build_xgboost(params: dict[str, object], seed: int):
    """Create an XGBoost classifier, if xgboost is installed.

    Args:
        params: Hyperparameters from config.yaml.
        seed: Random seed.

    Returns:
        An unfitted XGBClassifier.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError(
            "xgboost is not installed. Install it with "
            "`uv pip install xgboost`, or remove it from CANDIDATES."
        ) from error
    return XGBClassifier(random_state=seed, **params)


def score_candidate(
    name: str,
    model,
    features: list[str],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    seed: int,
) -> dict[str, object]:
    """Fit one candidate and score it on the validation split.

    Args:
        name: The candidate's name, for the output table.
        model: An unfitted estimator.
        features: The exact feature list, identical for every candidate.
        train_data: The training rows.
        validation_data: The held-out rows.
        seed: The random seed.

    Returns:
        One row of results.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    started = time.monotonic()
    model.fit(train_data[features], train_data["satisfaction"])
    fit_seconds = time.monotonic() - started

    probabilities = model.predict_proba(validation_data[features])[:, 1]
    predictions = model.predict(validation_data[features])

    return {
        "model": name,
        "roc_auc": round(
            float(roc_auc_score(validation_data["satisfaction"], probabilities)), 6
        ),
        "accuracy": round(
            float(accuracy_score(validation_data["satisfaction"], predictions)), 6
        ),
        "fit_seconds": round(fit_seconds, 1),
    }


def main() -> int:
    """Run the comparison and print one table.

    Returns:
        0 if the comparison ran, 1 if no candidate could be fitted.

    Raises:
        FileNotFoundError: If the pipeline has not prepared the data. Run
            `python run_pipeline.py` first.
    """
    setup_logging()
    logger = get_logger()

    reader = PipelineConfigReader()
    config = reader.create_cleaning_config()
    artifacts = config.artifacts_dir

    train_path = artifacts / "train.csv"
    validation_path = artifacts / "validation.csv"
    for path in (train_path, validation_path):
        if not path.exists():
            print(
                f"{path} not found. Run `python run_pipeline.py` first to prepare "
                "the splits."
            )
            return 1

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(validation_path)
    features: list[str] = config.features
    check_columns("compare", set(train_data.columns), set(features) | {"satisfaction"})
    check_columns(
        "compare", set(validation_data.columns), set(features) | {"satisfaction"}
    )
    seed = config.random_seed

    log_step(
        "compare",
        train_rows=len(train_data),
        validation_rows=len(validation_data),
        features=len(features),
        candidates=len(CANDIDATES),
        seed=seed,
    )
    logger.info(
        "Every candidate gets the same features, the same split, the same seed and "
        "default settings. Tuning comes later, and only for the winner."
    )
    logger.info("")

    results: list[dict[str, object]] = []
    for name, params in CANDIDATES.items():
        model = build_model(name, params, seed)
        row = score_candidate(name, model, features, train_data, validation_data, seed)
        results.append(row)
        logger.info(
            "  %-28s roc_auc=%.6f  accuracy=%.6f  fit=%ss",
            name,
            row["roc_auc"],
            row["accuracy"],
            row["fit_seconds"],
        )

    # XGBoost is optional, so a missing install is reported rather than fatal.
    try:
        xgboost_model = build_xgboost({}, seed)
        row = score_candidate(
            "xgboost", xgboost_model, features, train_data, validation_data, seed
        )
        results.append(row)
        logger.info(
            "  %-28s roc_auc=%.6f  accuracy=%.6f  fit=%ss",
            "xgboost",
            row["roc_auc"],
            row["accuracy"],
            row["fit_seconds"],
        )
    except ImportError as error:
        logger.warning("xgboost skipped: %s", error)

    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    baseline_rows = pd.DataFrame(
        [
            {"model": name, "roc_auc": auc, "accuracy": accuracy}
            for name, accuracy, auc in BASELINES
        ]
    )
    combined = pd.concat([table, baseline_rows], ignore_index=True)

    logger.info("")
    logger.info("=" * 78)
    logger.info("VALIDATION COMPARISON - all from one run, seed %d", seed)
    logger.info("=" * 78)
    print(combined.to_string(index=False))

    output = Path(reader.config["report_path"]) / "model_comparison.csv"
    save_dataframe(combined, output)
    logger.info("")
    logger.info("wrote %s", output)

    best = table.iloc[0]
    bar = BASELINES[1][1]
    logger.info("")
    logger.info("Best candidate: %s at roc_auc %.6f", best["model"], best["roc_auc"])
    logger.info("One-column rule to beat: %.4f", bar)
    logger.info(
        "Margin over the rule: %+.6f",
        float(best["roc_auc"]) - bar,
    )
    logger.info("")
    logger.info(
        "Next: .lead/02-C Step 9, read the errors of the winner. Not tuning until "
        "Step 11."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
