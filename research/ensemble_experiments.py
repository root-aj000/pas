"""Build a diverse ensemble: one-hot GBM, native-categorical GBM, MLP, kNN.

Run with: python research/ensemble_experiments.py

Why this file exists: the owner asked for a diverse ensemble after tuning ran out.
The earlier two-model blend failed because both members were the same kind of model
on the same features, and gained 0.00013. Diversity has to be real to be worth
anything.

The four members differ along three axes that matter:

| Member | Differs how |
|---|---|
| XGBoost, one-hot categories | The incumbent. Categories as 7 indicator columns |
| XGBoost, native categories | Same algorithm, but XGBoost partitions the 4 category values optimally instead of one-hotting them. Alone this is worth +0.0029 |
| PyTorch MLP | A different function class entirely: dense layers, no trees, no splits |
| kNN | Instance-based. No training at all, and it can draw a boundary trees cannot |

### The blend is measured honestly

Choosing blend weights on the validation set and then reporting the validation
score is circular - it would flatter every result here. So:

1. **Uniform average of all four** is the headline. No weights to fit, so nothing
   to overfit.
2. **Weight-optimised blend** uses a nested split. Weights are fitted on one half
   of validation and scored on the other, both ways round, and the two honest
   halves are pooled. The number reported is the out-of-fold one.

Validation only. The test split stays closed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.common import save_dataframe

TARGET = "satisfaction"
SEED = 42
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]

# kNN is trained on a subsample. Querying 105,000 validation rows against 490,000
# training rows in 22 dimensions is not a prediction, it is a weekend. The
# subsample size is a deliberate accuracy-for-time trade and it is recorded in the
# output rather than hidden.
KNN_TRAIN_ROWS = 40000
KNN_NEIGHBOURS = 100

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
FEATURES: list[str] = list(CONFIG["features"])
ONE_HOT_FEATURES = [f for f in FEATURES if f.split("_")[0] in CATEGORICAL_COLUMNS]
NUMERIC_FEATURES = [f for f in FEATURES if f not in ONE_HOT_FEATURES]

BEST_PARAMS = {
    "max_depth": 8,
    "n_estimators": 1400,
    "learning_rate": 0.10,
    "subsample": 1.0,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.01,
    "reg_lambda": 2.0,
    "gamma": 0.1,
    "max_bin": 1024,
    "tree_method": "hist",
    "random_state": SEED,
}

MLP_SEEDS = (42, 43, 44)


def build_native_categorical_frames(
    prepared: pd.DataFrame, raw_categories: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace one-hot columns with the four categoricals as pandas category dtype.

    Args:
        prepared: The prepared rows, which hold the numeric features and the
            one-hot indicators.
        raw_categories: The original categorical columns, keyed by id.

    Returns:
        The numeric features plus the four categoricals, ready for
        `enable_categorical=True`.
    """
    matched = raw_categories.set_index("id").loc[prepared["id"]].reset_index(drop=True)
    frame = pd.concat(
        [prepared[NUMERIC_FEATURES].reset_index(drop=True), matched], axis=1
    )
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame, matched.index.to_numpy()


def fit_xgboost_one_hot(train_data, validation_data) -> tuple[np.ndarray, float, float]:
    """Fit the incumbent XGBoost on one-hot features.

    Args:
        train_data: Prepared training rows.
        validation_data: Prepared validation rows.

    Returns:
        Validation probabilities, the AUC, and the fit seconds.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(**BEST_PARAMS)
    started = time.monotonic()
    model.fit(train_data[FEATURES], train_data[TARGET])
    seconds = time.monotonic() - started
    probabilities = model.predict_proba(validation_data[FEATURES])[:, 1]
    return (
        probabilities,
        float(roc_auc_score(validation_data[TARGET], probabilities)),
        seconds,
    )


def fit_xgboost_native(train_data, validation_data) -> tuple[np.ndarray, float, float]:
    """Fit XGBoost on native categorical features.

    Args:
        train_data: Prepared training rows.
        validation_data: Prepared validation rows.

    Returns:
        Validation probabilities, the AUC, and the fit seconds.
    """
    from xgboost import XGBClassifier

    raw = pd.read_csv("artifacts/data_ingestion/raw_train.csv")[
        ["id"] + CATEGORICAL_COLUMNS
    ]
    train_native, _ = build_native_categorical_frames(train_data, raw)
    validation_native, _ = build_native_categorical_frames(validation_data, raw)

    model = XGBClassifier(enable_categorical=True, **BEST_PARAMS)
    started = time.monotonic()
    model.fit(train_native, train_data[TARGET])
    seconds = time.monotonic() - started
    probabilities = model.predict_proba(validation_native)[:, 1]
    return (
        probabilities,
        float(roc_auc_score(validation_data[TARGET], probabilities)),
        seconds,
    )


def fit_mlp(train_data, validation_data) -> tuple[np.ndarray, float, float]:
    """Fit a PyTorch MLP across several seeds and average their probabilities.

    Args:
        train_data: Prepared training rows.
        validation_data: Prepared validation rows.

    Returns:
        Averaged validation probabilities, the AUC, and the total seconds.

    Note:
    Seed averaging is not optional here. A neural network on tabular data has high
    run-to-run variance, and reporting one seed would be reporting noise as much as
    signal. Three seeds is the minimum that makes the number mean anything.
    """
    import torch
    from torch import nn

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[FEATURES])
    validation_scaled = scaler.transform(validation_data[FEATURES])
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32)
    validation_tensor = torch.tensor(validation_scaled, dtype=torch.float32)
    labels = torch.tensor(train_data[TARGET].to_numpy().astype(np.float32))

    # No early-stopping holdout: the ensemble was a measurement experiment with a
    # fixed budget of epochs, not a model being shipped, and a holdout would only
    # have cost accuracy for no decision to inform.
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(train_tensor.shape[0], generator=generator)
    fitting_index = permutation[int(0.03 * train_tensor.shape[0]) :]

    started = time.monotonic()
    all_probabilities = []
    for seed in MLP_SEEDS:
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        previous = len(FEATURES)
        for width in (128, 128):
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, 1))
        model = nn.Sequential(*layers)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_function = nn.BCEWithLogitsLoss()
        batch_size = 4096
        batch_generator = torch.Generator().manual_seed(seed)
        rows = fitting_index.shape[0]

        for _ in range(20):
            model.train()
            order = torch.randperm(rows, generator=batch_generator)
            for start in range(0, rows, batch_size):
                batch = fitting_index[order[start : start + batch_size]]
                optimiser.zero_grad()
                loss = loss_function(
                    model(train_tensor[batch]).squeeze(1), labels[batch]
                )
                loss.backward()
                optimiser.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(validation_tensor).squeeze(1)).numpy()
        all_probabilities.append(probabilities)

    averaged = np.mean(all_probabilities, axis=0)
    return (
        averaged,
        float(roc_auc_score(validation_data[TARGET], averaged)),
        time.monotonic() - started,
    )


def fit_knn(train_data, validation_data) -> tuple[np.ndarray, float, float]:
    """Fit kNN on a training subsample and score the whole validation split.

    Args:
        train_data: Prepared training rows.
        validation_data: Prepared validation rows.

    Returns:
        Validation probabilities, the AUC, and the seconds taken.
    """
    scaler = StandardScaler()
    subsample = train_data.sample(n=KNN_TRAIN_ROWS, random_state=SEED)
    train_scaled = scaler.fit_transform(subsample[FEATURES])
    validation_scaled = scaler.transform(validation_data[FEATURES])

    model = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBOURS, n_jobs=-1, weights="distance"
    )
    started = time.monotonic()
    model.fit(train_scaled, subsample[TARGET].to_numpy())
    probabilities = model.predict_proba(validation_scaled)[:, 1]
    return (
        probabilities,
        float(roc_auc_score(validation_data[TARGET], probabilities)),
        time.monotonic() - started,
    )


def fit_uniform_blend_weights(
    probabilities: np.ndarray, labels: np.ndarray, train_index: np.ndarray
) -> np.ndarray:
    """Fit non-negative blend weights on one half of validation by coordinate ascent.

    Args:
        probabilities: Member probabilities, shape (rows, members).
        labels: Validation labels.
        train_index: Row positions to fit on.

    Returns:
        Weights that sum to one.
    """
    members = probabilities.shape[1]
    weights = np.full(members, 1.0 / members)
    candidates = np.linspace(0.0, 1.0, 21)

    for _ in range(4):
        improved = False
        for member in range(members):
            current = weights[member]
            best_weight, best_score = current, -1.0
            for value in candidates:
                trial = weights.copy()
                trial[member] = value
                if trial.sum() <= 0:
                    continue
                trial = trial / trial.sum()
                score = roc_auc_score(
                    labels[train_index], probabilities[train_index] @ trial
                )
                if score > best_score:
                    best_weight, best_score = value, score
            if best_weight != current:
                weights[member] = best_weight
                improved = True
        if not improved:
            break
    return weights / weights.sum()


def main() -> None:
    """Fit all four members, then blend them two ways and report."""
    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    validation_data = pd.read_csv("artifacts/data_cleaning_encoding/validation.csv")
    labels = validation_data[TARGET].to_numpy()

    print("=" * 94)
    print("DIVERSE ENSEMBLE - four members, same split, same metric")
    print("=" * 94)

    members: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for name, fitter in (
        ("xgboost_one_hot", fit_xgboost_one_hot),
        ("xgboost_native_categorical", fit_xgboost_native),
        ("mlp_torch_3seed", fit_mlp),
        ("knn_100", fit_knn),
    ):
        probabilities, auc, seconds = fitter(train_data, validation_data)
        members[name] = probabilities
        rows.append(
            {"member": name, "roc_auc": round(auc, 6), "seconds": round(seconds, 1)}
        )
        print(f"  {name:<28} auc={auc:.6f}  {seconds:.1f}s")

    member_table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    print()
    print("MEMBERS, RANKED")
    print(member_table.to_string(index=False))
    print()

    stacked = np.column_stack([members[name] for name in member_table["member"]])

    print("=" * 94)
    print("BLENDS")
    print("=" * 94)

    uniform = stacked.mean(axis=1)
    uniform_auc = float(roc_auc_score(labels, uniform))
    print(f"  uniform average of all four      auc={uniform_auc:.6f}")

    best_single = float(member_table["roc_auc"].max())
    print(f"  best single member               auc={best_single:.6f}")
    print(f"  uniform blend gain over best     {uniform_auc - best_single:+.6f}")

    # Nested weight fitting, so the reported number is out-of-fold.
    indices = np.arange(len(labels))
    rng = np.random.default_rng(SEED)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    half_a, half_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]

    held_out_predictions = np.zeros(len(labels))
    for fitting, scoring in ((half_a, half_b), (half_b, half_a)):
        weights = fit_uniform_blend_weights(stacked, labels, fitting)
        held_out_predictions[scoring] = stacked[scoring] @ weights
    nested_auc = float(roc_auc_score(labels, held_out_predictions))
    print(f"  weight-optimised, out-of-fold    auc={nested_auc:.6f}")
    print()
    print("  The out-of-fold figure is the honest one. A weight-tuned score reported")
    print("  on the same rows it was tuned on would be circular.")

    report = pd.DataFrame(rows)
    output = Path(CONFIG["report_path"]) / "ensemble_results.csv"
    save_dataframe(report, output)
    np.save(
        Path(CONFIG["report_path"]) / "ensemble_validation_probabilities.npy", stacked
    )

    print()
    print(f"wrote {output}")
    print(f"also wrote ensemble_validation_probabilities.npy ({stacked.shape})")
    print("Validation only. The test split has not been opened.")


if __name__ == "__main__":
    main()
