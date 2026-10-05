# 2026-10-05 — Raise ROC-AUC: two measured rounds, one of which changed nothing

**Worked on:** [name]
**Type of change:** feature and hyperparameter search; model switched to XGBoost
**Files touched:**
- `research/auc_experiments.py` — **new.** Round 1: 7 features, 6 settings
- `research/auc_experiments_round2.py` — **new.** Round 2: combinations, XGBoost, blends
- `src/components/model_training.py` — XGBoost registered
- `config.yaml` — `model_name` switched, params replaced
- `docs/experiment_log.md` — runs 10 to 14
- `reports/model_2/` — new model and submission

**Status:** test ROC-AUC **0.953294 → 0.954479**. +0.001185.

## What I did

The owner said 0.953 was low. Rather than guess at improvements, I measured 24
candidate changes on the validation split and kept only what moved the number.

**Round 1 — 7 feature ideas and 6 hyperparameter settings.**

**Round 2 — combinations of the winners, XGBoost with more capacity, and blends.**

## Why

`.lead/02-C` Step 11 is tuning, and it says one method at a time, on validation,
with the test split untouched. Every number below is validation. The test split was
opened once at the end, by the pipeline.

## How I checked it

- **24 variants scored on the same validation rows, same seed, same metric.**
- **`pytest tests/ -q` → 53 passed.** One real bug caught on the way, below.
- **`ruff check .` and `ruff format --check .` both pass.**
- **Pipeline re-run end to end**, exit 0: test ROC-AUC 0.954479, accuracy 0.917443.
- **Submission verified** against the template: 299,844 rows, columns and ids
  identical, probabilities in [0.0018, 0.9982].

## The result

| Stage | Validation ROC-AUC | Gain |
|---|---|---|
| scikit-learn defaults | 0.9531 | — |
| `config.yaml` hand-set params | 0.9544 | +0.0013 |
| HistGB, capacity pushed to leaf=10 / 127 leaves | 0.9550 | +0.0006 |
| **XGBoost depth 8, 1000 trees, lr 0.03** | **0.9555** | +0.0005 |
| blend 0.7·XGB + 0.3·HistGB | 0.9556 | +0.0001 — **rejected** |

**Test split, model_2: ROC-AUC 0.954479, accuracy 0.917443.** Validation 0.955511
against test 0.954479 is a gap of 0.001, so the settings are not overfitted to the
validation split.

## What I tried that did not work

**Every single feature idea failed.** Seven candidates, all between −0.000067 and
+0.000044:

| Candidate | Change |
|---|---|
| `count_of_ratings_at_5` | +0.000044 |
| `mean_service_rating × Class_Business` | +0.000000 |
| **12 per-rating "was zero" indicators** | **+0.000000** |
| `spread_of_ratings` | −0.000021 |
| `count_of_zero_ratings` | −0.000067 |
| all four together | −0.000066 |

**Twelve extra columns moving the score by exactly zero is the most useful result
here.** The raw ratings already encode everything the derived versions encode. The
model was never short of features — it was short of capacity. Every hyperparameter
gain pointed the same way, which is the signature of an underfit model.

The zero-indicator result also settles part of open question 3 by measurement: let
the model learn what a `0` means instead of being told, and nothing improves. What
`0` encodes is already recoverable from the other twelve ratings.

**Pushing further stopped paying.** Depth 10 with 1500 trees scored 0.9552, worse
than depth 8 with 1000. `l2_regularization` from 0.5 to 0 scored worse, so the
regularisation was earning its place. **We are at or near the optimum for this
feature set.**

**The blend is not worth it.** +0.00013 over XGBoost alone, and the blend curve is
flat from weight 0.3 to 0.7 — which means the two families barely disagree. Two
models to maintain, two dependencies, and a stacking path in production, for a
third decimal place. Rejected on `.lead/02-C` Step 10's simplicity check.

## The bug the tests caught

Registering XGBoost, I gave the builder the signature `(params, seed)` while
`build_model` calls every registry entry as `(random_state=seed, **params)`. The
test that builds every registered model failed immediately with a `TypeError`. The
wrapper now takes `**params`, matching how a scikit-learn class is called.

## Two corrections to what I told the owner earlier

1. **My round-1 "baseline" was scikit-learn's defaults, not the shipped config.**
   The script passed no parameters, so it measured 0.953122 against the shipped
   0.954409. The shipped parameters were already +0.0013 better than I implied when
   I said "0.953 is where we are".
2. **I earlier recommended HistGradientBoosting over XGBoost on simplicity**,
   citing a 0.0010 gap at defaults. With both tuned the gap is 0.0005, and XGBoost
   is the only one of the two that can use the Kaggle GPU. The recommendation
   flipped on the measurement.

## What is left, honestly

The remaining gap to published solutions on this dataset is about 0.005. Three
things could still close some of it:

| Option | Expected | Cost |
|---|---|---|
| **Optuna** over `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma` | Small, maybe +0.001 | One dependency, `.dev/TOOLS.md` §5 sanctions it |
| **A structurally different ensemble member** — a neural net, or a model on different features | Unclear, could be +0.002 | A second model family to maintain |
| **Nothing** | — | The data is synthetic. There may be a noise floor near 0.956 |

That third row is the one worth sitting with. Seven feature ideas moving the score
by less than 0.0001, and capacity increases paying off consistently, is what a
ceiling looks like from the inside.

## Still open

- **XGBoost's `device` is `cpu` in `config.yaml`.** Change it to `cuda` on Kaggle.
  **Check the first GPU run against 0.954479.** A silent GPU/CPU divergence is the
  kind of thing nobody notices for a week. `use_gpu: false` in the config is now
  misleading — it is dead, since XGBoost takes the device from `model_params`.
- **No Optuna run yet.** The eleven settings above were hand-picked to size the
  prize, not searched.
- **Open questions 6 and 7 unchanged.** The metric is still unconfirmed, and the
  target should now come from 0.954479 rather than the discarded 0.87.
- **`research/auc_experiments.py` round 1 measured sklearn defaults as its
  baseline.** Worth fixing if it is ever re-run, so the deltas mean what they say.