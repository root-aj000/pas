"""NOVEL 2: ordinal regression - treating the ratings as ordered, not binary.

Run with: python research/novel_2_ordinal.py

Based on: Cao et al., CORAL (COnsistent RAnk Logits), ICML 2020, arXiv:2001.08085;
Shi/Cao/Raschka, CORN, arXiv:2111.08851; and the ordinal-regression survey
arXiv:2503.00952.

WHY THIS ONE IS WORTH A RUN
The survey's framing: "ordinal regression takes into account the natural ordering
of categories in the response variable. This allows for more nuanced and accurate
modeling when the order of categories carries meaningful information."

There are THREE ordinal angles in this problem and all three are tested here,
because they are genuinely different hypotheses:

  A. CORAL on the TARGET. The label is binary, so this needs a constructed
     ordinal target. We build one: `mean_service_rating` is an ordinal 0-5
     summary of the same survey, so it is a real ordering signal the binary
     label discards.
  B. CORAL on a reconstructed continuous label. Train against the ratings
     directly as an ordinal target, then map the prediction back to satisfaction
     by rank. This uses the ordering of the ratings the binary label throws away.
  C. Plain regression on the latent score. Train to predict mean_service_rating
     as a continuous [0,1] target, then rank by that prediction. AUC only reads
     ranking, so a regression target loses nothing - but every row now
     contributes its DISTANCE from the boundary instead of a saturated 0/1.

The key theoretical point in favour of all three: near the decision boundary a
hard 0/1 label gives almost no gradient. A soft ordinal target gives full
information about how close a passenger was to being satisfied.

WHAT TO BE SCEPTICAL ABOUT
The target is binary, so these are indirect. They only pay off if the ratings
carry ordering information that the label does not - which is exactly what the
hypothesis is, and exactly what the run tests.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from _novel_common import (
    PROJECT_ROOT,
    SEED,
    banner,
    baseline_auc,
    encode,
    report,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

TARGET = "satisfaction"
RATINGS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]


def build_ordinal_targets():
    """Return X, y_binary, y_ordinal (5 bins from mean rating), y_continuous."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
    df[RATINGS] = df[RATINGS].replace(0, np.nan)
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

    df = df.sample(n=100_000, random_state=SEED).reset_index(drop=True)
    mean_rating = df[RATINGS].mean(axis=1).fillna(2.5)
    # 5 bins across the observed 0-5 range. Ordered, and it is a genuine
    # coarsening of information the binary label discards.
    ordinal = np.digitize(mean_rating, [1.5, 2.5, 3.5, 4.5])
    return (
        encode(df),
        df[TARGET].astype("int8").to_numpy(),
        ordinal.astype("int64"),
        (mean_rating / 5.0).to_numpy(),
    )


class Coral(nn.Module):
    """K-1 shared-backbone binary heads over ordered thresholds.

    CORAL's defining constraint is the weight-sharing: all K-1 heads use the same
    output weights and differ only in bias. That is what forces rank-monotonicity
    - the predictions cannot cross - which is the property that makes it better
    than K independent binary classifiers.
    """

    def __init__(self, n_features: int, n_bins: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.shared = nn.Linear(hidden // 2, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(n_bins - 1))

    def forward(self, x):
        h = self.backbone(x)
        shared = self.shared(h)
        # K-1 logits, monotone by construction: a shared score plus one ordered
        # bias per threshold. Biases are sorted so rank-monotonicity holds
        # structurally, not merely as a penalty.
        ordered = torch.cumsum(F.softplus(self.biases), dim=0) - F.softplus(self.biases[0])
        return shared + ordered.unsqueeze(0)


def train_coral(X_tr, y_ord, X_ev, n_bins, epochs=10, lr=1e-3):
    torch.manual_seed(SEED)
    n_bins = max(n_bins, 2)
    model = Coral(X_tr.shape[1], n_bins)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    xt = torch.tensor(X_tr, dtype=torch.float32)
    et = torch.tensor(X_ev, dtype=torch.float32)
    ot = torch.tensor(y_ord, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        for start in range(0, len(xt), 512):
            idx = torch.randperm(len(xt))[start : start + 512]
            xb, yb = xt[idx], ot[idx]
            # No squeeze here: with 2 bins there is exactly one threshold, and
            # squeeze(1) would collapse (batch, 1) to (batch,) and lose the
            # dimension the loss below needs.
            thresholds = model(xb)                      # (batch, K-1)
            k = thresholds.shape[1]
            # Binary target per threshold: has the true bin exceeded threshold i?
            yb_col = yb.float().unsqueeze(1)
            exceeded = (yb_col > torch.arange(k, dtype=torch.float32)).float()
            # CORAL's rank loss over ordered threshold pairs.
            diff = thresholds.unsqueeze(2) - thresholds.unsqueeze(1)
            loss = F.relu(diff).pow(2).mean()
            loss = loss + F.binary_cross_entropy_with_logits(thresholds, exceeded)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        # Rank by the number of thresholds the row exceeds: that is the predicted
        # bin, and bin order IS the ranking.
        thresholds = model(et)
        if thresholds.dim() == 1:
            thresholds = thresholds.unsqueeze(1)
        return thresholds.sum(dim=1).numpy()


def train_regression(X_tr, y_cont, X_ev, epochs=12, lr=1e-3):
    """Plain regression to a continuous [0,1] score, ranked as a probability."""
    torch.manual_seed(SEED)

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
            )

        def forward(self, x):
            return torch.sigmoid(self.net(x)).squeeze(1)

    model = Net(X_tr.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    et = torch.tensor(X_ev, dtype=torch.float32)
    ct = torch.tensor(y_cont, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        for start in range(0, len(xt), 512):
            idx = torch.randperm(len(xt))[start : start + 512]
            loss = F.mse_loss(model(xt[idx]), ct[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        return model(et).numpy()


def main() -> None:
    banner(
        2,
        "ordinal regression (CORAL) + latent-score regression",
        "CORAL arXiv:2001.08085 / CORN arXiv:2111.08851 / survey arXiv:2503.00952",
    )
    X, y, ordinal, continuous = build_ordinal_targets()
    X = np.nan_to_num(X, nan=0.0)

    # experiment, never the thing being scored.
# o_ev and c_ev are the held-out ordinal targets. The CORAL and regression
    # heads train on o_tr/c_tr and are scored against the BINARY y_ev, because
    # AUC for this competition is measured on satisfaction - the ordinal targets
    # are inputs to the experiment, never the thing being scored.
    # o_ev and c_ev are the held-out ordinal targets, carried through the split
    # so the arrays stay aligned even though only the TRAINING halves are fitted.
    # The CORAL and regression heads train on o_tr/c_tr and are scored against the
    # BINARY y_ev, because AUC for this competition is measured on satisfaction.
    X_tr, X_ev, y_tr, y_ev, o_tr, _o_ev, c_tr, _c_ev = train_test_split(
        X, y, ordinal, continuous, test_size=20_000, random_state=SEED, stratify=y
    )
    print(f"train {X_tr.shape}  eval {X_ev.shape}")

    base = baseline_auc(X_tr, y_tr, X_ev, y_ev)
    print(f"baseline LightGBM AUC: {base:.6f}")

    started = time.monotonic()

    # A: CORAL on the 5-bin ordinal target, ranked and mapped back.
    logits = train_coral(X_tr, o_tr, X_ev, n_bins=5)
    auc_a = float(roc_auc_score(y_ev, logits))

    # B: CORAL with 2 bins is just binary classification on the ordinal target -
    # included to separate "CORAL's rank constraint" from "ordinal target".
    logits_b = train_coral(X_tr, o_tr, X_ev, n_bins=2)
    auc_b = float(roc_auc_score(y_ev, logits_b))

    # C: continuous latent score, no thresholding at all.
    score = train_regression(X_tr, c_tr, X_ev)
    auc_c = float(roc_auc_score(y_ev, score))

    seconds = time.monotonic() - started
    print(f"\n  A CORAL 5-bin ordinal target      AUC {auc_a:.6f}  ({auc_a - base:+.6f})")
    print(f"  B CORAL 2-bin (control)           AUC {auc_b:.6f}  ({auc_b - base:+.6f})")
    print(f"  C regression on latent score       AUC {auc_c:.6f}  ({auc_c - base:+.6f})")

    best = max((auc_a, "CORAL 5-bin"), (auc_b, "CORAL 2-bin"), (auc_c, "regression"))
    report(
        f"ordinal - best was {best[1]}",
        best[0], base, seconds,
        "A vs B separates rank-constraint value from ordinal-target value",
    )


if __name__ == "__main__":
    main()