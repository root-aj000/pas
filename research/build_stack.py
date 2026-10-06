"""
Combine member predictions with a logistic regression on logits, and score it honestly.

Why logits: ROC-AUC depends only on the ranking of the scores, so members are
combined in logit space rather than as raw probabilities. This is what the
reference notebook does and it is worth the trouble - it is a linear model with
one weight per member, so it cannot overfit 490k rows even with 38 members.

Why the score is nested: the stacker has weights, so scoring it on rows its own
weights were fitted on would flatter it. Every number printed here comes from
StratifiedKFold(5, shuffle=True, random_state=7), where each row's score comes
from weights fitted on the other rows. The submission at the end is then fitted
on all rows, which is the one place the weights see everything - that is correct,
because the test rows are still unlabelled.

Run with: python -m research.build_stack --out reports/submissions/stack.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_MEMBERS = Path("research/members")
DEFAULT_ARTIFACTS = Path("artifacts/data_cleaning_encoding")
SPLIT_SEED = 7


def logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits, clipped away from 0 and 1.

    Args:
        p: Probabilities.

    Returns:
        The log-odds.
    """
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def load_members(
    members_dir: Path, names: list[str] | None
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read each member's out-of-fold and test predictions from disk.

    Args:
        members_dir: Directory holding the oof_*.npy and test_*.npy files.
        names: Member names to load, or None for every member present.

    Returns:
        The member names, an (rows x members) out-of-fold matrix in logit space,
        and the matching test matrix.
    """
    if names is None:
        # Exclude .partial on purpose. run_member checkpoints every fold, so a
        # partial file exists for every in-flight member, and its unscored rows
        # are still zeros. Globbing "oof_*.npy" picks it up as a member named
        # "<name>.partial" and stacks a half-empty array into the result - a
        # silent corruption with no error anywhere.
        names = sorted(
            p.stem[4:]
            for p in members_dir.glob("oof_*.npy")
            if not p.stem.endswith(".partial")
        )
    missing = [
        n
        for n in names
        if not (members_dir / f"oof_{n}.npy").exists()
        or not (members_dir / f"test_{n}.npy").exists()
    ]
    if missing:
        raise SystemExit(f"missing predictions for: {missing}")
    oof = np.column_stack([logit(np.load(members_dir / f"oof_{n}.npy")) for n in names])
    test = np.column_stack(
        [logit(np.load(members_dir / f"test_{n}.npy")) for n in names]
    )
    return names, oof, test


def nested_stack_score(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score a stack with weights fitted out-of-fold.

    Args:
        Z: Member logits, rows by members.
        y: Labels.

    Returns:
        One held-out score per row.
    """
    out = np.zeros(len(y))
    for fit_index, score_index in StratifiedKFold(
        5, shuffle=True, random_state=SPLIT_SEED
    ).split(Z, y):
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(Z[fit_index], y[fit_index])
        out[score_index] = model.decision_function(Z[score_index])
    return out


def greedy_select(
    names: list[str], Z: np.ndarray, y: np.ndarray, max_members: int
) -> list[str]:
    """Add members one at a time, keeping one only if it improves the stack.

    Args:
        names: Member names.
        Z: Member logits.
        y: Labels.
        max_members: Stop after this many picks.

    Returns:
        The chosen member names, in the order they were picked.

    Note:
    Each candidate is judged by the same nested score as the full stack, so a
    member has to earn its place on held-out rows. The reference noted this
    greedy path is still chosen on the rows it is scored on and did not submit
    it; keeping the nested score as the criterion makes it honest enough to
    report, but the full stack is the primary answer.
    """
    chosen: list[int] = []
    best = -np.inf
    while len(chosen) < max_members:
        candidate_best, candidate_index = best, None
        for index in range(Z.shape[1]):
            if index in chosen:
                continue
            trial = chosen + [index]
            score = roc_auc_score(y, nested_stack_score(Z[:, trial], y))
            if score > candidate_best + 1e-7:
                candidate_best, candidate_index = score, index
        if candidate_index is None:
            break
        chosen.append(candidate_index)
        best = candidate_best
        print(f"  + {names[candidate_index]:20s} -> {best:.6f}", flush=True)
    return [names[i] for i in chosen]


def main() -> None:
    """Score every member and the stack, then write the submission."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", default=None, help="comma-separated, default all")
    parser.add_argument("--out", default="reports/submissions/submission_stack.csv")
    parser.add_argument("--max-members", type=int, default=12)
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--members-dir", default=str(DEFAULT_MEMBERS))
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    args = parser.parse_args()

    members_dir = Path(args.members_dir)
    artifacts = Path(args.artifacts)
    names = args.members.split(",") if args.members else None
    names, Z, test_Z = load_members(members_dir, names)

    y = pd.read_csv(artifacts / "train.csv", usecols=["satisfaction"])[
        "satisfaction"
    ].to_numpy()

    roster_path = members_dir / "roster.csv"
    if roster_path.exists():
        roster = pd.read_csv(roster_path)
        counts = {
            str(r["member"]): r["folds"]
            for _, r in roster.iterrows()
            if "folds" in roster
        }
        seen = {n: counts[n] for n in names if n in counts}
        if len(set(seen.values())) > 1:
            print("WARNING: members cross-validated with different fold counts:")
            for name, folds in sorted(seen.items()):
                print(f"  {name:24s} folds={folds}")
            print("  Mixing is arithmetically legal - every row is still out of")
            print("  fold - but the weaker-folded members carry less signal.")
            print()

    print("=== members, scored out-of-fold on the training rows ===")
    solo = []
    for index, name in enumerate(names):
        auc = roc_auc_score(y, Z[:, index])
        solo.append(auc)
        print(f"  {name:22s} {auc:.6f}")
    print("\nrank correlation between members (Spearman, off-diagonal):")
    ranks = np.column_stack(
        [pd.Series(Z[:, i]).rank().to_numpy() for i in range(len(names))]
    )
    corr = pd.DataFrame(ranks).corr(method="spearman").to_numpy()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if corr[i, j] < 0.995:
                print(f"  {names[i]:18s} {names[j]:18s} {corr[i, j]:.4f}")

    print("\n=== combinations ===")
    print(f"  best single member      {max(solo):.6f}  ({names[int(np.argmax(solo))]})")
    print(f"  mean of logits          {roc_auc_score(y, Z.mean(1)):.6f}")
    stack_score = nested_stack_score(Z, y)
    print(f"  logistic stack, nested  {roc_auc_score(y, stack_score):.6f}")

    print("\n=== greedy forward selection (nested) ===")
    greedy_select(names, Z, y, args.max_members)

    if args.no_submission:
        return

    final = LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)
    prediction = 1 / (1 + np.exp(-final.decision_function(test_Z)))
    competition = pd.read_csv(artifacts / "competition_test.csv", usecols=["id"])
    submission = pd.DataFrame(
        {"id": competition["id"].values, "satisfaction": prediction}
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    print("\n=== stack weights ===")
    for name, weight in sorted(
        zip(names, final.coef_[0]), key=lambda pair: -abs(pair[1])
    ):
        print(f"  {name:22s} {weight:+.4f}")
    print(f"\nwrote {out_path}  rows={len(submission)}")
    (members_dir / "stack_meta.json").write_text(
        json.dumps(
            {
                "members": names,
                "solo_oof_auc": {n: round(a, 6) for n, a in zip(names, solo)},
                "nested_stack_auc": round(roc_auc_score(y, stack_score), 6),
                "mean_logit_auc": round(roc_auc_score(y, Z.mean(1)), 6),
                "split_seed": SPLIT_SEED,
                "n_train_rows": len(y),
                "y": yaml.safe_load(Path("config.yaml").read_text())["target_column"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
