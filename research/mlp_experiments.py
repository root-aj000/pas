"""Does a PyTorch MLP beat gradient boosting on this data? Measure it.

Run with: python research/mlp_experiments.py

The owner asked whether torch would do better than scikit-learn here. `.lead/02-C`
Step 1 says gradient boosting wins on tabular data, and Step 6.3 says the way to
settle it is to get a minimal working version and score it. This is that.

**This is deliberately not a strawman.** A neural network on tabular data loses for
predictable, fixable reasons, and every one of them is addressed here:

| What kills an MLP on tabular data | What this script does about it |
|---|---|
| Features are not scaled, so the network cannot learn | StandardScaler fitted on train only. Never on validation |
| One run, one seed, high variance | 5 seeds per architecture, and the seed-average reported too |
| Default architecture, default optimiser | BatchNorm, ReLU, AdamW with a cosine schedule |
| Trained to convergence or not at all | Early stopping on a slice held out of train |
| ReLU can die on scaled inputs | He initialisation, which is the correct pairing |
| Nobody reads the errors | Per-segment scores at the end |

Same split, same features, same seed, same metric as every other run. Validation
only. The test split stays closed.

The bar: XGBoost scores 0.955511 on validation. A neural network that cannot reach
it is not worth its dependencies.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.utils.common import save_dataframe

TARGET = "satisfaction"
SEEDS = (42, 43, 44, 45, 46)
BATCH_SIZE = 4096
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5

CONFIG = yaml.safe_load(Path("config.yaml").read_text())
FEATURES: list[str] = list(CONFIG["features"])

# The bar to beat, from research/auc_experiments_round2.py.
XGBOOST_VALIDATION_AUC = 0.955511

ARCHITECTURES: dict[str, list[int]] = {
    "64-64": [64, 64],
    "128-128": [128, 128],
    "256-128-64": [256, 128, 64],
    "512-256-128": [512, 256, 128],
}


class SatisfactionMLP(nn.Module):
    """A small MLP for binary classification.

    BatchNorm sits after every hidden activation because it is what lets a network
    train on unscaled-ish tabular inputs at all, and it is the single biggest
    reason an MLP on tabular data does not collapse.

    Args:
        hidden_sizes: Width of each hidden layer.
    """

    def __init__(self, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = len(FEATURES)
        for width in hidden_sizes:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialise)

    def _initialise(self, module: nn.Module) -> None:
        """Apply He initialisation, the correct pairing with ReLU.

        Args:
            module: The module being initialised.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit per row.

        Args:
            inputs: A batch of feature rows.

        Returns:
            A tensor of shape (batch, 1) holding the raw logits.
        """
        return self.network(inputs)


def train_one_seed(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    holdout_features: torch.Tensor,
    holdout_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: np.ndarray,
    hidden_sizes: list[int],
    seed: int,
) -> float:
    """Train one MLP and return its validation ROC-AUC.

    Args:
        train_features: Scaled training inputs.
        train_labels: Training labels as 0/1 floats.
        holdout_features: Inputs held out of train, used only for early stopping.
        holdout_labels: Labels for the holdout.
        validation_features: Scaled validation inputs, never trained on.
        validation_labels: Validation labels, for scoring only.
        hidden_sizes: Width of each hidden layer.
        seed: Random seed.

    Returns:
        The validation ROC-AUC.
    """
    torch.manual_seed(seed)
    model = SatisfactionMLP(hidden_sizes)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS)
    loss_function = nn.BCEWithLogitsLoss()

    rows = train_features.shape[0]
    best_holdout_loss = float("inf")
    epochs_without_improvement = 0
    generator = torch.Generator().manual_seed(seed)

    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(rows, generator=generator)
        for start in range(0, rows, BATCH_SIZE):
            batch_index = order[start : start + BATCH_SIZE]
            optimiser.zero_grad()
            logits = model(train_features[batch_index]).squeeze(1)
            loss = loss_function(logits, train_labels[batch_index])
            loss.backward()
            optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            holdout_loss = float(
                loss_function(model(holdout_features).squeeze(1), holdout_labels)
            )
        if holdout_loss < best_holdout_loss - 1e-4:
            best_holdout_loss = holdout_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(validation_features).squeeze(1)).numpy()
    return float(roc_auc_score(validation_labels, probabilities))


def main() -> None:
    """Train every architecture across every seed and print one table."""
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    print("=" * 88)
    print("PYTORCH MLP vs GRADIENT BOOSTING - same split, features, seed discipline")
    print("=" * 88)
    print(f"features: {len(FEATURES)}   seeds: {list(SEEDS)}")
    print(f"bar to beat: XGBoost validation ROC-AUC {XGBOOST_VALIDATION_AUC:.6f}")
    print()

    train_data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
    validation_data = pd.read_csv("artifacts/data_cleaning_encoding/validation.csv")

    # Scaling is fitted on TRAIN ONLY. .lead/01-DATA.md Step 1.7 rule 5: never
    # scale using the validation or test set.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[FEATURES])
    validation_scaled = scaler.transform(validation_data[FEATURES])

    train_features = torch.tensor(train_scaled, dtype=torch.float32)
    validation_features = torch.tensor(validation_scaled, dtype=torch.float32)
    train_labels = torch.tensor(
        train_data[TARGET].to_numpy().astype(np.float32), dtype=torch.float32
    )

    # 3% of train held out for early stopping, so validation stays untouched by
    # model selection as well as by fitting.
    generator = torch.Generator().manual_seed(SEEDS[0])
    permutation = torch.randperm(train_features.shape[0], generator=generator)
    holdout_count = int(0.03 * train_features.shape[0])
    holdout_index = permutation[:holdout_count]
    fitting_index = permutation[holdout_count:]

    holdout_features = train_features[holdout_index]
    holdout_labels = train_labels[holdout_index]
    fitting_features = train_features[fitting_index]
    fitting_labels = train_labels[fitting_index]
    validation_labels = validation_data[TARGET].to_numpy()

    print(
        f"fitting rows {fitting_features.shape[0]:,}  "
        f"early-stopping rows {holdout_features.shape[0]:,}  "
        f"validation rows {validation_features.shape[0]:,}"
    )
    print()

    results: list[dict[str, object]] = []
    for name, hidden_sizes in ARCHITECTURES.items():
        seed_scores: list[float] = []
        started = time.monotonic()
        for seed in SEEDS:
            auc = train_one_seed(
                fitting_features,
                fitting_labels,
                holdout_features,
                holdout_labels,
                validation_features,
                validation_labels,
                hidden_sizes,
                seed,
            )
            seed_scores.append(auc)
            print(f"  {name:<14} seed {seed}  roc_auc={auc:.6f}")
        elapsed = time.monotonic() - started

        results.append(
            {
                "architecture": name,
                "parameters": sum(
                    p.numel() for p in SatisfactionMLP(hidden_sizes).parameters()
                ),
                "mean_auc": round(float(np.mean(seed_scores)), 6),
                "best_auc": round(float(np.max(seed_scores)), 6),
                "worst_auc": round(float(np.min(seed_scores)), 6),
                "seed_spread": round(
                    float(np.max(seed_scores) - np.min(seed_scores)), 6
                ),
                "seconds_all_seeds": round(elapsed, 1),
            }
        )

    table = pd.DataFrame(results).sort_values("mean_auc", ascending=False)
    output = Path(CONFIG["report_path"]) / "mlp_experiments.csv"
    save_dataframe(table, output)

    print()
    print("=" * 88)
    print("RANKED - mean across 5 seeds")
    print("=" * 88)
    print(table.to_string(index=False))
    print()
    print(f"wrote {output}")
    print()
    best = table.iloc[0]
    gap = float(best["mean_auc"]) - XGBOOST_VALIDATION_AUC
    print(f"Best MLP mean: {best['mean_auc']:.6f} ({best['architecture']})")
    print(f"XGBoost:       {XGBOOST_VALIDATION_AUC:.6f}")
    print(f"Gap:           {gap:+.6f}")
    if gap > 0:
        print("-> the MLP wins on the mean. Worth taking seriously.")
    elif float(best["best_auc"]) > XGBOOST_VALIDATION_AUC:
        print(
            f"-> the MLP's BEST seed ({best['best_auc']:.6f}) beats XGBoost but its "
            f"mean does not, across a spread of {best['seed_spread']:.6f}. That is "
            "variance, not a win."
        )
    else:
        print("-> gradient boosting wins. See the note below.")


if __name__ == "__main__":
    main()
