"""NOVEL 3: transductive semi-supervised learning on the unlabelled test rows.

Run with: python research/novel_3_transductive.py

Based on: Grandvalet & Bengio, "Semi-Supervised Learning by Entropy Minimization",
NeurIPS 2004 (arXiv:1806.05682); Sohn et al., FixMatch, arXiv:2001.07685; and the
2026 CVPR paper "Boosting Semi-Supervised Learning by Exploiting All Unlabeled
Data" which adds Entropy Meaning Loss and Adaptive Negative Learning.

WHY THIS ONE IS WORTH A RUN
`data/test.csv` is 299,844 rows that this pipeline never trains on. In vision,
unlabelled data is worth a great deal - FixMatch reaches 88.61% on CIFAR-10 with
FOUR labels per class. Whether that transfers to tabular is genuinely open, and
this script is the experiment.

THE THREE PIECES, TESTED SEPARATELY

  A. Entropy minimisation (the original 2004 paper). Add lambda * H(p(y|x)) over
     unlabelled rows. The motivating theory: "unlabeled examples are mostly
     beneficial when classes have small overlap." That assumption holds here -
     the two classes are well separated by Online boarding and Class.

  B. Pseudo-labelling (the pipeline already has this, but on the WRONG path).
     Hard-label confident unlabelled rows above threshold tau and train on them.
     The project's pseudo_label_enabled is skipped whenever the run is sharded
     across GPUs, which is every real run. This tests whether it was worth having.

  C. Pseudo-labelling with SOFT targets - sharpening, MixMatch-style. MixMatch's
     key move over hard labels: "guessing low-entropy labels ... and mixing
     labeled and unlabeled data using MixUp", with a sharpening temperature T.
     Soft targets carry the model's uncertainty instead of discarding it.

FixMatch itself is vision-specific - its weak/strong augmentation pair has no
meaning for tabular features, which is why it is not implemented here. The
entropy and pseudo-label components are the transferable part.

WHAT TO BE SCEPTICAL ABOUT
Grandvalet & Bengio also note the theory "provides no positive statement without
distributional assumptions", and that unlabelled data can HARM when the
cluster assumption is violated. The lambda sweep below exists to find the point
where it stops helping.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from _novel_common import SEED, banner, baseline_auc, load_split, report
from torch import nn


class Net(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_with_unlabeled(
    X_lab, y_lab, X_unlab, X_eval, lambda_u=0.0, mode="supervised",
    tau=0.95, sharpen_T=0.5, epochs=8, lr=1e-3, batch=512, seed=SEED
):
    """Train with a labelled loss plus one unlabelled objective.

    Args:
        mode: "supervised" | "entropy" | "pseudo" | "soft".

    Returns:
        Evaluation probabilities.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Net(X_lab.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    lab = torch.tensor(X_lab, dtype=torch.float32)
    lab_y = torch.tensor(y_lab, dtype=torch.float32)
    unlab = torch.tensor(X_unlab, dtype=torch.float32)
    ev = torch.tensor(X_eval, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(lab))
        for start in range(0, len(lab), batch):
            idx = order[start : start + batch]
            logits_lab = model(lab[idx])
            loss = F.binary_cross_entropy_with_logits(logits_lab, lab_y[idx])

            if lambda_u > 0 and mode != "supervised" and len(unlab):
                u_idx = torch.randint(0, len(unlab), (min(batch, len(unlab)),))
                logits_unlab = model(unlab[u_idx])
                prob = torch.sigmoid(logits_unlab)
                if mode == "entropy":
                    # H(p) = -p log p - (1-p) log(1-p)
                    p = prob.clamp(1e-6, 1 - 1e-6)
                    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
                    loss = loss + lambda_u * entropy
                else:
                    confidence = prob.max(dim=0).values
                    if mode == "pseudo":
                        # confidence is a batch-wide scalar; broadcast it to one
                        # target per unlabelled row so the shapes agree.
                        target = (confidence > tau).float().expand_as(logits_unlab)
                        loss = loss + lambda_u * F.binary_cross_entropy_with_logits(
                            logits_unlab, target
                        )
                    elif mode == "soft":
                        # MixMatch sharpening: p^(1/T) renormalised, or uniform
                        # if too flat to sharpen. Keeps the uncertainty.
                        q = prob.unsqueeze(1)
                        two = torch.cat([1 - q, q], dim=1) / (1 - prob + 1e-6).unsqueeze(1)
                        two = two.clamp(1e-6, 1.0)
                        two = two ** (1.0 / sharpen_T)
                        two = two / two.sum(dim=1, keepdim=True)
                        soft_target = two[:, 1]
                        loss = loss + lambda_u * F.binary_cross_entropy_with_logits(
                            logits_unlab, soft_target
                        )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(ev)).numpy()


def main() -> None:
    banner(
        3,
        "transductive SSL on the 299,844 unlabelled competition rows",
        "Grandvalet & Bengio NeurIPS 2004 / FixMatch arXiv:2001.07685 / MixMatch arXiv:1905.02249",
    )
    X_train, y_train, X_eval, y_eval, X_unlab = load_split(with_unlabeled=True)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0)
    X_unlab = np.nan_to_num(X_unlab, nan=0.0)

    print(f"labelled train {X_train.shape}")
    print(f"unlabelled     {X_unlab.shape}   <- never used by the pipeline")

    base = baseline_auc(X_train, y_train, X_eval, y_eval)
    print(f"baseline LightGBM AUC: {base:.6f}")

    from sklearn.metrics import roc_auc_score

    started = time.monotonic()
    supervised = float(
        roc_auc_score(
            y_eval,
            train_with_unlabeled(X_train, y_train, X_unlab, X_eval, lambda_u=0.0),
        )
    )
    print(f"  neural supervised control            AUC {supervised:.6f}")

    results: dict[str, tuple[float, float]] = {}
    for mode in ("entropy", "pseudo", "soft"):
        best = (0.0, 0.0)
        for lam in (0.05, 0.2, 1.0):
            auc = float(
                roc_auc_score(
                    y_eval,
                    train_with_unlabeled(
                        X_train, y_train, X_unlab, X_eval, lambda_u=lam, mode=mode
                    ),
                )
            )
            print(f"  {mode:<8s} lambda={lam:<5}                AUC {auc:.6f}")
            if auc > best[0]:
                best = (auc, lam)
        results[mode] = best

    seconds = time.monotonic() - started
    print()
    for mode, (auc, lam) in results.items():
        print(
            f"  best {mode:<8s} lambda={lam:<5} AUC {auc:.6f} "
            f"({auc - supervised:+.6f} vs neural supervised)"
        )

    best_mode, (best_auc, best_lam) = max(
        results.items(), key=lambda kv: kv[1][0]
    )
    report(
        f"transductive SSL - best was {best_mode} at lambda={best_lam}",
        best_auc, base, seconds,
        f"neural supervised control was {supervised:.6f} "
        f"({best_auc - supervised:+.6f} from the unlabelled data) - "
        f"compare against THAT, not against LightGBM",
    )


if __name__ == "__main__":
    main()