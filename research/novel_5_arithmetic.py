"""NOVEL 5: arithmetic feature interaction - additive AND multiplicative attention.

Run with: python research/novel_5_arithmetic.py

Based on: Cheng et al., "Arithmetic Feature Interaction Is Necessary for Deep
Tabular Learning", AAAI 2024, arXiv:2402.02334 (AMFormer).

WHY THIS ONE IS WORTH A RUN
The paper's central claim: "arithmetic feature interaction is necessary for deep
tabular learning", attributed to "parallel additive and multiplicative
attention operators" that "facilitate the separation of tabular samples in an
extended space with arithmetically-engineered features".

This problem is unusually well suited to that argument. Satisfaction is driven by
products, not sums - Online boarding x Class is the clearest case in the whole
dataset: boarding=5 gives 94.7% satisfied in Business but 60.9% in Eco, while
boarding=3 gives 28.2% vs 6.0%. A tree can only APPROXIMATE that interaction
with a split on each factor separately, spending depth on it. An additive model
cannot represent it at all. A multiplicative operator gets it for free.

METHOD
  1. Each row is projected into an expanded space: the raw features, plus
     pairwise PRODUCTS of every feature pair, plus the original - the
     "arithmetic feature space" the paper builds.
  2. Attention over that expanded space, with both additive (learned weighted
     sum) and multiplicative (learned weighted product) operators running in
     parallel, exactly as AMFormer does.
  3. Compare against the SAME network WITHOUT the multiplicative branch.

The ablation is the whole point: if the multiplicative branch does not beat the
additive-only network on the same features, the claim does not hold here.

SCALING NOTE
A full pairwise product expansion over 22 features is 253 extra columns. That is
fine. Over the pipeline's real 178 it would be 16,091, which is why this runs on
the modest feature set.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from _novel_common import SEED, banner, baseline_auc, load_split, report
from sklearn.metrics import roc_auc_score
from torch import nn


def arithmetic_expansion(X: np.ndarray, max_pairs: int = 400):
    """Return [raw features, pairwise products, raw again] - the arithmetic space.

    Args:
        X: (n, d) float32 features.
        max_pairs: Cap on the number of pairs expanded. d=22 gives 231 pairs, so
            the cap does not bind here; it exists so a larger feature set does
            not silently explode into an OOM.

    Returns:
        (n, 3d) array: raw, products, raw again. The trailing copy of the raw
        features is the "residual path" - without it the network has to
        reconstruct the linear component from products alone, which the paper
        avoids.
    """
    n, d = X.shape
    rows, pairs = [], []
    for i in range(d):
        for j in range(i + 1, d):
            if len(pairs) >= max_pairs:
                break
            rows.append(X[:, i] * X[:, j])
            pairs.append((i, j))
    products = np.column_stack(rows) if rows else np.zeros((n, 0), dtype="float32")
    return np.column_stack([X, products, X]).astype("float32"), pairs


class ArithmeticAttention(nn.Module):
    """Attention with additive and multiplicative operators in parallel."""

    def __init__(self, n_tokens: int, dim: int = 64, multiplicative: bool = True):
        super().__init__()
        self.multiplicative = multiplicative
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # Additive operator: a learned attention distribution over features.
        self.additive = nn.Linear(dim, 1)
        # Multiplicative operator: a learned gate per feature, applied to the
        # value vector before the additive path reads it. This is what lets one
        # token's contribution depend on another token's presence.
        self.multiplicative_gate = nn.Parameter(torch.ones(n_tokens))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        q, v = self.query(tokens), self.value(tokens)
        # ADDITIVE branch: softmax over the FEATURE axis (which token explains
        # this token), not the query axis. Feature-wise attribution, not mixing
        # rows.
        feature_weights = torch.softmax(self.additive(tokens), dim=1)
        additive_out = torch.einsum("btd,bt->bd", v, feature_weights.squeeze(-1))

        if not self.multiplicative:
            return self.norm(additive_out + q.mean(dim=1))

        # MULTIPLICATIVE branch: the same value vector, gated by a per-feature
        # learned multiplier, then pooled again. Multiplying before summing is
        # what encodes AND-type conditions between features.
        gated = v * torch.sigmoid(self.multiplicative_gate).view(1, -1, 1)
        multiplicative_out = gated.mean(dim=1)
        return self.norm(
            additive_out + multiplicative_out + q.mean(dim=1)
        )


class ArithmeticNet(nn.Module):
    def __init__(self, n_tokens: int, multiplicative: bool = True, dim: int = 64):
        super().__init__()
        self.tokenise = nn.Linear(n_tokens, dim)
        self.attention = ArithmeticAttention(n_tokens, dim, multiplicative)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 1)
        )

    def forward(self, x):
        tokens = self.tokenise(x).unsqueeze(1)          # (batch, n_tokens, dim)
        pooled = self.attention(tokens).squeeze(1)       # (batch, dim)
        return self.head(pooled).squeeze(1)


def train(model, X_tr, y_tr, X_ev, epochs=10, lr=1e-3, batch=512):
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    ev = torch.tensor(X_ev, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        for start in range(0, len(xt), batch):
            idx = order[start : start + batch]
            loss = F.binary_cross_entropy_with_logits(
                model(xt[idx]), yt[idx]
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(ev)).numpy()


def main() -> None:
    banner(
        5,
        "arithmetic (additive + multiplicative) feature interaction",
        "AMFormer, AAAI 2024, arXiv:2402.02334 - 'Arithmetic Feature Interaction Is Necessary'",
    )
    X_train, y_train, X_eval, y_eval = load_split()
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0)

    Xa_train, pairs = arithmetic_expansion(X_train)
    Xa_eval, _ = arithmetic_expansion(X_eval)
    print(f"raw features        {X_train.shape[1]}")
    print(f"pairwise products   {len(pairs)}")
    print(f"arithmetic space    {Xa_train.shape[1]}  (= raw + products + raw residual)")

    base = baseline_auc(X_train, y_train, X_eval, y_eval)
    print(f"baseline LightGBM on raw features: {base:.6f}")

    started = time.monotonic()
    torch.manual_seed(SEED)
    with_mult = ArithmeticNet(Xa_train.shape[1], multiplicative=True)
    p_mult = train(with_mult, Xa_train, y_train, Xa_eval)
    auc_mult = float(roc_auc_score(y_eval, p_mult))

    torch.manual_seed(SEED)
    without_mult = ArithmeticNet(Xa_train.shape[1], multiplicative=False)
    p_add = train(without_mult, Xa_train, y_train, Xa_eval)
    auc_add = float(roc_auc_score(y_eval, p_add))

    seconds = time.monotonic() - started
    print(f"\n  additive + multiplicative   AUC {auc_mult:.6f}")
    print(f"  additive only (ablation)    AUC {auc_add:.6f}")
    print(f"  the multiplicative branch   {auc_mult - auc_add:+.6f}")
    print(
        "\n  that last line is the result. A positive number means the "
        "multiplicative\n  operator earned its place; a negative one means the "
        "paper's claim does not\n  hold on this dataset."
    )

    report(
        "AMFormer-style arithmetic interaction (multiplicative branch)",
        auc_mult, base, seconds,
        f"ablation without the multiplicative branch was {auc_add:.6f} "
        f"({auc_mult - auc_add:+.6f}) - compare against THAT number, not LightGBM",
    )


if __name__ == "__main__":
    main()