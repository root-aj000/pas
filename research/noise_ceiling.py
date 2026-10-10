"""Is the label noise irreducible, and is the signal exhausted?

Run with: python research/noise_ceiling.py

This answers the question every further modelling decision depends on: when four
model families all land on the same 0.96xxx, is that a ceiling imposed by the
labels, or is it four models failing in the same way?

FOUR MEASUREMENTS, each answering a separate question.

1. Are there contradictory duplicates? Group rows by their exact feature vector
   and look for any group carrying both labels. If one exists, the label is not a
   function of the features and no model can do better than the group majority.

2. Are the labels Bernoulli draws? Group rows by a coarse but informative
   signature - mean rating and worst rating - so groups are large. Inside a group,
   compare the observed label variance to p(1-p) for that group's rate. A ratio of
   exactly 1.000 means the labels behave like coin flips at that rate, and the
   residual variance is sampling noise that no model can recover. A ratio ABOVE 1
   means the signature is missing information other columns could supply.

3. Where is the ceiling? Take the model's own out-of-fold probabilities as an
   estimate of the true p(x), draw a fresh Bernoulli label per row from that p, and
   score it. If the achieved AUC equals that, the model has recovered p as well as
   the data allows and the remaining gap to 1.0 is not model error.

4. Is p exhausted by the ratings alone? Train on the 13 raw ratings, then on the
   full configured feature set, with the same model, folds and seed. If both land
   on the same number the features are exhausted and nothing more is available. If
   the larger set wins, the features carry signal the ratings do not.

The answer decides what is worth doing next. Measurement 2 says whether label
noise can be attacked at all. Measurement 4 says whether feature work still pays.
Together they bound every remaining hour of effort.

Nothing here changes the shipped model or any artifact. This reads the prepared
train split and writes to reports/.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.common import ensure_project_root

ensure_project_root()

import json

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# config.yaml is YAML, not JSON - it carries anchors and comments. The JSON in this
# file is only ever written out, never read back from config.
CONFIG = yaml.safe_load(Path("config.yaml").read_text())
TARGET = str(CONFIG["target_column"])
SEED = 42
N_SPLITS = 5
ARTIFACTS = Path("artifacts/data_cleaning_encoding")
REPORTS = Path(CONFIG["report_path"])

RATING_COLUMNS = [
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

# The six non-rating columns that survive into config.yaml's feature list. The two
# delay columns are deliberately excluded: config.yaml lists both under
# `dropped_features`, so "the raw file" here means the 19 columns the pipeline
# actually considers, not the 21 in data/train.csv.
DEMOGRAPHIC_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
    "Age",
    "Flight Distance",
]


def load_features() -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, list[str]]]:
    """Read the prepared train split and ordinal-code its string columns.

    Returns:
        The frame, the labels, the configured feature list, and the pinned
        category levels.

    Note:
    `features.json` is the contract stage 2 wrote, and its `category_maps` are the
    pinned level lists. Coding with those rather than `factorize` keeps the same
    column meaning as the shipped pipeline. Without this the four string columns
    arrive as `str` and the estimator rejects them.
    """
    contract = json.loads((ARTIFACTS / "features.json").read_text())
    features = list(contract["features"])
    category_maps = contract.get("category_maps") or {}
    frame = pd.read_csv(ARTIFACTS / "train.csv", usecols=features + [TARGET])
    labels = frame.pop(TARGET).astype(int).to_numpy()
    for column in [c for c in features if str(frame[c].dtype) in ("object", "str")]:
        levels = category_maps.get(column)
        frame[column] = (
            pd.Categorical(frame[column], categories=levels).codes
            if levels
            else pd.factorize(frame[column])[0]
        )
    return frame, labels, features, category_maps


def cross_validated_auc(
    frame: pd.DataFrame, labels: np.ndarray, columns: list[str], splits: list
) -> tuple[float, np.ndarray]:
    """Score one column set with one model, out of fold.

    Args:
        frame: The prepared rows.
        labels: The labels.
        columns: Which columns to fit on.
        splits: The fold list, shared across every configuration so the only
            thing that differs between them is the feature set.

    Returns:
        The pooled out-of-fold AUC, and the out-of-fold probabilities.

    Note:
    HistGradientBoosting rather than XGBoost, because it accepts an ordinal-coded
    column without a categorical dtype and because it is fast enough to run seven
    configurations in a few minutes. Every configuration uses the same model, so
    the comparison between them is a comparison of feature sets and not of models.
    """
    matrix = frame[columns].to_numpy(dtype=np.float32)
    out_of_fold = np.zeros(len(labels))
    for fitting, scoring in splits:
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, random_state=SEED
        )
        model.fit(matrix[fitting], labels[fitting])
        out_of_fold[scoring] = model.predict_proba(matrix[scoring])[:, 1]
    return float(roc_auc_score(labels, out_of_fold)), out_of_fold


def measurement_1_duplicate_labels(features: list[str], labels: np.ndarray) -> dict:
    """Look for feature vectors that carry both labels.

    Args:
        features: Every configured feature.
        labels: The labels.

    Returns:
        A dict of counts, and a printed report.

    Note:
    This is the strongest form of the claim "the noise is irreducible". A group of
    rows with identical inputs and different outputs is a proof of it: no function
    of the features can separate them, so any model's error on that group is
    unavoidable. Finding none does NOT prove the noise is reducible, only that it
    is not visible in exact duplicates.
    """
    print("=" * 78)
    print("MEASUREMENT 1  contradictory duplicates")
    print("=" * 78)
    frame = pd.read_csv(
        ARTIFACTS / "train.csv", usecols=[c for c in features if not c.endswith("_cat_")] + [TARGET]
    )
    groups = frame.groupby(
        [c for c in frame.columns if c != TARGET], dropna=False
    )[TARGET].agg(["size", "nunique"])
    repeated = groups[groups["size"] > 1]
    conflicting = repeated[repeated["nunique"] > 1]
    print(f"  feature vectors seen more than once : {len(repeated):,}")
    print(f"  ...carrying both labels             : {len(conflicting):,}")
    if len(conflicting):
        rows = int(conflicting["size"].sum())
        print(f"  rows in conflicting groups          : {rows:,}")
        print("  => the label is NOT a function of these features. Provable ceiling.")
    else:
        print("  => none. The noise is not visible in exact duplicates.")
        print("     Measurement 2 tests whether it exists at all.")
    print()
    return {
        "repeated_vectors": len(repeated),
        "conflicting_vectors": len(conflicting),
        "rows_in_conflict": int(conflicting["size"].sum()) if len(conflicting) else 0,
    }


def measurement_2_bernoulli(labels: np.ndarray) -> dict:
    """Test whether labels are Bernoulli draws at their group's rate.

    Args:
        labels: The labels.

    Returns:
        A dict of statistics, and a printed report.

    Note:
    This is the decisive test. Take rows sharing a coarse signature - mean rating
    and worst rating - so each group holds hundreds of rows. If the label were a
    deterministic function of the signature, every row in a group would carry the
    same label and the variance would be 0. If labels were sampled Bernoulli(p) for
    the group's true p, the variance would be p(1-p).

    The observed/Bernoulli ratio is therefore a direct read on how much of the
    within-group variation is sampling noise. A ratio of 1.000 means ALL of it is.
    Groups near 0 or 1 are skipped: p(1-p) approaches 0 there and the ratio becomes
    numerically unstable, which is a property of the estimator rather than of the
    data.
    """
    print("=" * 78)
    print("MEASUREMENT 2  are the labels Bernoulli draws?")
    print("=" * 78)
    frame = pd.read_csv(
        ARTIFACTS / "train.csv", usecols=RATING_COLUMNS + [TARGET]
    )
    ratings = frame[RATING_COLUMNS].replace(0, np.nan)
    signature = (
        (ratings.mean(axis=1).round(1) * 10).fillna(-1).astype(int) * 100
        + ratings.min(axis=1).fillna(-1).round(1).astype(int)
    )
    grouped = (
        pd.DataFrame({"signature": signature, "label": labels})
        .groupby("signature")["label"]
        .agg(["size", "mean"])
    )
    large = grouped[grouped["size"] >= 200]
    print(f"  groups of >=200 rows sharing (mean, worst): {len(large):,}\n")
    print(f"  {'group rate':>11} {'n':>7} {'observed var':>13} {'Bernoulli var':>14} {'ratio':>7}")

    ratios: list[float] = []
    for value, row in large.iterrows():
        rate = float(row["mean"])
        if rate < 0.03 or rate > 0.97:
            continue
        observed = float(labels[signature == value].var(ddof=1))
        expected = rate * (1 - rate)
        ratio = observed / expected
        ratios.append(ratio)
        if len(ratios) <= 10:
            print(
                f"    {rate:9.4f} {int(row['size']):7d} {observed:13.5f} "
                f"{expected:14.5f} {ratio:7.3f}"
            )

    array = np.array(ratios)
    median_ratio = float(np.median(array))
    print(f"\n  median observed/Bernoulli variance ratio : {median_ratio:.3f}  (n={len(array)})")
    print(f"  fraction of groups with ratio > 1.5      : {float((array > 1.5).mean()):.1%}")
    print()
    if 0.95 <= median_ratio <= 1.05:
        print("  VERDICT: ratio ~1.000. Inside a group the labels behave exactly like")
        print("  coin flips at that group's rate. The whole residual variance is")
        print("  sampling noise, and it is IRREDUCIBLE. Every model that reaches the")
        print("  same AUC has recovered p(x) as well as the data permits.")
    elif median_ratio > 1.05:
        print("  VERDICT: ratio above 1. The signature misses information that other")
        print("  columns still supply, so the noise is not yet irreducible.")
    else:
        print("  VERDICT: ratio below 1. Labels within a group are MORE similar than")
        print("  independent draws, which suggests clustering or an unmodelled feature.")
    print()
    return {
        "groups": len(array),
        "median_ratio": round(median_ratio, 4),
        "fraction_above_1_5": round(float((array > 1.5).mean()), 4),
    }


def measurement_3_ceiling(frame: pd.DataFrame, labels: np.ndarray, splits: list) -> dict:
    """Estimate the ceiling implied by the estimated true p(x).

    Args:
        frame: The prepared rows.
        labels: The labels.
        splits: The fold list.

    Returns:
        A dict of AUCs, and a printed report.

    Note:
    Takes the full configured feature set, estimates p(x) out of fold, and then
    asks what AUC that p would score against labels freshly drawn from it. Because
    the fresh draw has no noise beyond the Bernoulli sampling itself, any gap
    between the two numbers is model error. No gap means the model is at the
    ceiling.

    Also reports what the ceiling WOULD be with k independent labels per row. That
    is not achievable here - one passenger fills in one form - but it localises the
    noise: per-observation, not per-person.
    """
    print("=" * 78)
    print("MEASUREMENT 3  where is the ceiling?")
    print("=" * 78)
    _, probabilities = cross_validated_auc(
        frame, labels, list(frame.columns), splits
    )
    achieved = float(roc_auc_score(labels, probabilities))
    generator = np.random.default_rng(SEED)
    draws = [
        float(roc_auc_score((generator.random(len(labels)) < probabilities).astype(int), probabilities))
        for _ in range(20)
    ]
    ceiling = float(np.mean(draws))
    spread = float(np.std(draws))
    print(f"  achieved out-of-fold AUC                  : {achieved:.6f}")
    print(f"  AUC against freshly drawn Bernoulli labels: {ceiling:.6f} +-{spread:.6f}")
    gap = achieved - ceiling
    print(f"  difference                                 : {gap:+.6f}")
    print()
    print("  ceiling if k independent labels existed per row:")
    for k in (2, 3, 5):
        averages = [
            float(
                roc_auc_score(
                    ((generator.random((k, len(labels))) < probabilities).mean(axis=0) >= 0.5).astype(int),
                    probabilities,
                )
            )
            for _ in range(10)
        ]
        print(f"    k={k}: {np.mean(averages):.6f}")
    print()
    if abs(gap) < 3 * spread + 1e-4:
        print("  VERDICT: achieved equals the ceiling within noise. The model has")
        print("  recovered p(x) as well as this data allows.")
    elif gap > 0:
        print("  VERDICT: achieved is ABOVE the ceiling, which means the ceiling")
        print("  estimate is low - the estimated p is sharper than the truth.")
    else:
        print(f"  VERDICT: {abs(gap):.6f} of model error remains. There is headroom.")
    print()
    return {
        "achieved": round(achieved, 6),
        "ceiling": round(ceiling, 6),
        "ceiling_std": round(spread, 6),
        "gap": round(gap, 6),
    }


def measurement_4_exhausted(
    frame: pd.DataFrame, labels: np.ndarray, features: list[str], splits: list
) -> dict:
    """Compare feature sets under one model, one fold structure, one seed.

    Args:
        frame: The prepared rows.
        labels: The labels.
        features: Every configured feature.
        splits: The fold list.

    Returns:
        A dict of AUCs and deltas, and a printed report.

    Note:
    The question is whether p is exhausted by the 13 raw ratings, or whether the
    engineered features still carry signal. Every configuration uses the same
    estimator, the same folds and the same seed, so the only variable is the column
    set and the differences are attributable to it.

    Three reference points rather than two, because "ratings vs everything" hides
    which part does the work: demographics alone, the raw columns, and the full set.
    """
    print("=" * 78)
    print("MEASUREMENT 4  is p exhausted by the ratings alone?")
    print("=" * 78)
    ratings = [c for c in RATING_COLUMNS if c in features]
    demographics = [c for c in DEMOGRAPHIC_COLUMNS if c in features]

    configurations = [
        ("A. the 13 raw ratings ONLY", ratings),
        ("C. demographics only, NO ratings", demographics),
        ("D. ratings + demographics (raw columns)", ratings + demographics),
        ("B. all configured features", features),
    ]
    scores: dict[str, float] = {}
    for label, columns in configurations:
        started = time.monotonic()
        score, _ = cross_validated_auc(frame, labels, columns, splits)
        scores[label] = score
        print(
            f"  {label:44s} AUC={score:.6f}  "
            f"({time.monotonic() - started:4.0f}s, {len(columns):3d} cols)"
        )

    a = scores["A. the 13 raw ratings ONLY"]
    c = scores["C. demographics only, NO ratings"]
    d = scores["D. ratings + demographics (raw columns)"]
    b = scores["B. all configured features"]
    print()
    print(f"  ratings only -> demographics only : {c - a:+.6f}")
    print(f"  ratings only -> raw columns       : {d - a:+.6f}")
    print(f"  raw columns  -> all engineered     : {b - d:+.6f}")
    print(f"  ratings only -> all engineered     : {b - a:+.6f}")
    print()
    verdict = "exhausted" if b - a < 0.001 else "NOT exhausted"
    print(f"  VERDICT: p is {verdict}. Engineered features are worth {b - a:+.6f} AUC")
    print("  over the ratings alone.")
    if b - a < 0.001:
        print("  Further feature work cannot pay, and neither can any label-noise")
        print("  method, because measurement 2 shows the noise is irreducible.")
    else:
        print("  Feature work still pays. The ceiling is not where the ratings put it,")
        print("  so measure 3's number is a floor rather than a ceiling.")
    print()
    return {
        "ratings_only": round(a, 6),
        "demographics_only": round(c, 6),
        "raw_columns": round(d, 6),
        "all_features": round(b, 6),
        "delta_ratings_to_all": round(b - a, 6),
        "verdict": verdict,
    }


def latent_factor_probe(frame: pd.DataFrame, labels: np.ndarray) -> dict:
    """Test whether the 13 ratings share one dominant factor the mean misses.

    Args:
        frame: The prepared rows.
        labels: The labels.

    Returns:
        A dict of statistics, and a printed report.

    Note:
    The 13 ratings are one person's answers on one form, so they are not
    independent: a passenger who rates everything poorly rates the wifi poorly
    too. If that shared structure is strong, one factor should carry most of the
    variance, and the label should load on it.

    Two things are measured. PC1's share of the rating variance, and whether PC1
    predicts the label better than the plain mean. The second is the one that
    matters - a factor that explains variance the label does not load on is
    worthless here, however clean it looks.
    """
    print("=" * 78)
    print("PROBE  latent satisfaction factor in the 13 ratings")
    print("=" * 78)
    ratings = frame[[c for c in RATING_COLUMNS if c in frame.columns]].to_numpy(float)
    standardised = (ratings - ratings.mean(axis=0)) / ratings.std(axis=0)
    correlation = np.corrcoef(standardised.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    share = float(eigenvalues[0] / len(RATING_COLUMNS))
    effective_rank = float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())
    principal = standardised @ eigenvectors[:, -1]
    if np.corrcoef(principal, labels)[0, 1] < 0:
        principal = -principal
    loading = float(np.corrcoef(principal, labels)[0, 1])

    print(f"  PC1 share of rating variance           : {share:.1%}")
    print(f"  top-3 share                            : {eigenvalues[:3].sum() / len(RATING_COLUMNS):.1%}")
    print(f"  participation ratio (effective rank)   : {effective_rank:.2f} of 13")
    print(f"  corr(PC1, label)                       : {loading:+.4f}")
    print(f"  AUC of PC1 alone                       : {roc_auc_score(labels, principal):.6f}")
    print(f"  AUC of the plain mean                  : {roc_auc_score(labels, ratings.mean(axis=1)):.6f}")
    print()
    if share < 0.4:
        print("  VERDICT: no single dominant factor. The ratings are only loosely")
        print("  correlated, so a one-factor summary would discard most of what they")
        print("  carry. A latent-factor representation is unlikely to beat fitting the")
        print("  13 columns directly, which is what the model already does.")
    else:
        print("  VERDICT: one factor dominates. Worth testing whether it predicts the")
        print("  label better than the mean before building anything on it.")
    print()
    return {
        "pc1_variance_share": round(share, 4),
        "effective_rank": round(effective_rank, 3),
        "pc1_label_correlation": round(loading, 4),
        "pc1_auc": round(float(roc_auc_score(labels, principal)), 6),
        "mean_auc": round(float(roc_auc_score(labels, ratings.mean(axis=1))), 6),
    }


def main() -> int:
    """Run all measurements and write the summary.

    Returns:
        0 always. The point is the printed table, not a gate.
    """
    frame, labels, features, _ = load_features()
    print(f"rows={len(labels):,}  configured features={len(features)}")
    print(f"positives={labels.mean():.4f}\n")

    splits = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            frame, labels
        )
    )

    results = {
        "duplicate_labels": measurement_1_duplicate_labels(features, labels),
        "bernoulli": measurement_2_bernoulli(labels),
        "ceiling": measurement_3_ceiling(frame, labels, splits),
        "feature_exhaustion": measurement_4_exhausted(frame, labels, features, splits),
        "latent_factor": latent_factor_probe(frame, labels),
    }

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(json.dumps(results, indent=2))

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "noise_ceiling.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())