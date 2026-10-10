"""NOVEL 1: ModernNCA - a retrieval model instead of a parametric classifier.

Run with: python research/novel_1_modernnca.py

Based on: Ye et al., "Revisiting Nearest Neighbor for Tabular Data: A Deep
Tabular Baseline Two Decades Later", arXiv:2407.03257 (ICLR 2025).

WHY THIS ONE IS WORTH A RUN
Every model in this repo is parametric: it learns a function x -> p(y), and
competes with another parametric model on how well that function fits. ModernNCA
is non-parametric. It learns an EMBEDDING SPACE, then predicts a point as a
softmax-weighted vote of its neighbours. The paper reports it ranked first in
classification across 300 datasets, beating XGBoost in most cases and CatBoost
on 114 with 81 ties.

That matters for an ensemble, not just for a leaderboard. The thing that bought
this project's last stack gain was DECORRELATION - the two RealMLP members held
0.4131 of 1.0 in stack weight because they disagreed with the boosted trees. A
retrieval model is a third, independent kind of disagreement. It cannot be
replicated by adding another LightGBM.

It is also the only method here that treats `Flight Distance` as real geometry
rather than as a number to be split on.

METHOD
  1. Encoder phi(x) = MLP(x), BatchNorm + ReLU.
  2. Within a minibatch, compute squared distances between every pair of
     embeddings, softmax them with temperature tau, and predict each row as the
     weighted vote of its neighbours' one-hot labels.
  3. Cross-entropy on that soft prediction.
  4. Stochastic Neighborhood Sampling: sample a subset of training rows in each
     batch as the neighbour candidates, use ALL of them at inference. This is
     what makes it tractable.

IMPLEMENTATION NOTE
This is a faithful reimplementation from the paper's equations, not their code.
The one deviation that matters: at inference we vote over the full training set
rather than a subsample, which is what the paper's Eq. 4 strategy does, and is
the expensive part.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from _novel_common import SEED, banner, baseline_auc, load_split, report
from torch import nn


class Encoder(nn.Module):
    """The learned metric. Deliberately small - the paper finds a linear head on
    top of this is competitive, so depth is not where the value is."""

    def __init__(self, n_features: int, hidden: int = 128, embedding: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, embedding),
        )

    def forward(self, x):
        return self.net(x)


def nca_predict(embeddings, train_embeddings, train_onehot, tau: float, chunk: int = 4096):
    """Softmax-weighted neighbour vote, with gradients, chunked over query rows.

    Args:
        embeddings: (n_query, d) query embeddings, requires grad.
        train_embeddings: (n_ref, d) reference embeddings, detached.
        train_onehot: (n_ref, n_classes) reference labels as one-hot.
        tau: Softmax temperature on the negative squared distance.
        chunk: Query rows per chunk, so a large eval set does not OOM.

    Returns:
        (n_query, n_classes) probability matrix.
    """
    chunks = []
    for start in range(0, len(embeddings), chunk):
        q = embeddings[start : start + chunk]
        # ||a-b||^2 = |a|^2 + |b|^2 - 2ab, which avoids a big broadcast and is
        # what makes this affordable.
        dist2 = (
            (q * q).sum(1, keepdim=True)
            + (train_embeddings * train_embeddings).sum(1).unsqueeze(0)
            - 2.0 * (q @ train_embeddings.T)
        ).clamp_min(0.0)
        weights = torch.softmax(-dist2 / tau, dim=1)
        chunks.append(weights @ train_onehot)
    return torch.cat(chunks, dim=0)


def main() -> None:
    banner(
        1,
        "ModernNCA (retrieval / non-parametric)",
        "arXiv:2407.03257, ICLR 2025 - ranked #1 in classification over 300 datasets",
    )
    X_train, y_train, X_eval, y_eval = load_split()
    print(f"train {X_train.shape}  eval {X_eval.shape}  features {X_train.shape[1]}")

    base = baseline_auc(X_train, y_train, X_eval, y_eval)
    print(f"\nbaseline LightGBM AUC: {base:.6f}")

    device = torch.device("cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # mean_service_rating is NaN where every rating was the 0 missing code. Fill
    # before standardising: a NaN column poisons every distance below, because
    # ||a-b||^2 expands to |a|^2 + |b|^2 - 2ab and |a|^2 is NaN.
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0)

    mean, std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((X_train - mean) / std, dtype=torch.float32)
    Xev = torch.tensor((X_eval - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)

    encoder = Encoder(X_train.shape[1]).to(device)
    optimiser = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    tau = 4.0
    candidates = 2048
    epochs = 12
    batch = 512

    started = time.monotonic()
    for epoch in range(epochs):
        encoder.train()
        permutation = torch.randperm(len(Xtr))
        total = 0.0
        for start in range(0, len(Xtr), batch):
            idx = permutation[start : start + batch]
            xb, yb = Xtr[idx], ytr[idx]

            # SNS: the minibatch is the query set, a random SUBSET of the training
            # set is the neighbour candidate set. Using the whole 80k as candidates
            # for every batch is what makes the naive version unaffordable.
            cand = permutation[
                torch.randint(0, len(Xtr), (min(candidates, len(Xtr)),))
            ]
            ref = Xtr[cand]
            ref_onehot = F.one_hot(ytr[cand], num_classes=2).float()

            emb_all = encoder(torch.cat([xb, ref], dim=0))
            q_emb, ref_emb = emb_all[: len(xb)], emb_all[len(xb) :]

            probabilities = nca_predict(q_emb, ref_emb, ref_onehot, tau)
            loss = F.nll_loss(torch.log(probabilities + 1e-8), yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)

        # At inference the paper uses the WHOLE reference set, not a subsample.
        encoder.eval()
        with torch.no_grad():
            train_emb = encoder(Xtr)
            eval_emb = encoder(Xev)
            train_onehot = F.one_hot(ytr, num_classes=2).float()
            probabilities = nca_predict(eval_emb, train_emb, train_onehot, tau).numpy()
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_eval, probabilities[:, 1]))
        print(f"  epoch {epoch + 1:>2}/{epochs}  loss {total / len(Xtr):.4f}  AUC {auc:.6f}")

    seconds = time.monotonic() - started
    report(
        "ModernNCA",
        auc,
        base,
        seconds,
        "if this is near the baseline, the retrieval framing adds nothing here",
    )


if __name__ == "__main__":
    main()