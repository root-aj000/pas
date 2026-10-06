# Experiment Log

**Started:** 2026-10-04
**Rule:** one table, appended to, never edit an old row. The **Notes** column is
what makes this valuable in six months — write what you learned, not what you did.
Template: `.lead/09-TEMPLATES.md` section 3.

Every row is reproducible. The seed is 42 for all of them, from `config.yaml`.

| Run | Date | Model | Key settings | Seed | ROC-AUC | Accuracy | Notes |
|---|---|---|---|---|---|---|---|
| 0a | 2026-10-04 | do nothing | always predict False | — | 0.5000 | 0.5564 | BASELINE. The floor |
| 0b | 2026-10-04 | one-column rule | `Online boarding == 5` | — | 0.6904 | 0.7206 | BASELINE. Registered in Step 0.4 before any scoring |
| 0c | 2026-10-04 | one-column rule | `Class == Business` | — | 0.7786 | 0.7770 | **The real bar.** Found by scoring every single-column rule |
| 0d | 2026-10-04 | one-column rule | `Type of Travel == Business travel` | — | 0.7027 | 0.6761 | Strong recall, weak precision |
| 1 | 2026-10-04 | HistGradientBoosting | all 26 columns, defaults | 42 | 0.9573 | 0.9242 | **The number to beat.** Untuned. Made the proposed 0.87 target meaningless |
| 2 | 2026-10-05 | HistGradientBoosting | 17 features, config defaults | 42 | 0.9476 | 0.9015 | **A mistake.** I dropped 9 columns on their univariate effect size |
| 3 | 2026-10-05 | LogisticRegression | 25 features, defaults | 42 | 0.8732 | 0.8095 | 0.08 behind boosting. The signal is genuinely non-linear |
| 4 | 2026-10-05 | DecisionTree | 25 features, defaults | 42 | 0.8536 | 0.8551 | Unpruned. Worst of the four, as expected |
| 5 | 2026-10-05 | RandomForest | 25 features, defaults | 42 | 0.9512 | 0.9149 | 0.002 behind boosting and **11x slower to fit** (51.5s vs 4.4s). Not worth it |
| 6 | 2026-10-05 | XGBClassifier | 25 features, defaults | 42 | **0.9541** | **0.9158** | Best. But only +0.0010 over HistGB — see the decision below |
| 7 | 2026-10-05 | HistGradientBoosting | 25 features, config defaults | 42 | 0.9531 | 0.9138 | The feature fix recovered most of run 2's loss |
| 8 | 2026-10-05 | HistGradientBoosting | 25 features, tuned params in config.yaml | 42 | **0.9544** | **0.9155** | **SHIPPED as model_1.** Test split: 0.9533 AUC / 0.9156 accuracy |
| 9 | 2026-10-05 | HistGradientBoosting + clipping | 25 features, outliers clipped 0.1/99.9 | 42 | 0.9544 | 0.9151 | Change of −0.0003. Confirms `docs/outlier_findings.md` — clipping is noise |
| 10 | 2026-10-05 | HistGradientBoosting | 11 hand-set settings tried | 42 | 0.9547 | — | Best single change: `min_samples_leaf` 40→10. Every gain pointed at underfitting |
| 11 | 2026-10-05 | 7 new feature ideas | count_at_5, spread, zero counts, interaction | 42 | 0.9532 | — | **All failed.** Range −0.000067 to +0.000044. Twelve zero-indicators moved it by exactly 0.000000 |
| 12 | 2026-10-05 | HistGradientBoosting | leaf=10, leaves=127, lr=0.05 | 42 | 0.9550 | — | Best HistGB found. Capacity was the constraint |
| 13 | 2026-10-05 | **XGBClassifier** | depth 8, 1000 trees, lr 0.03 | 42 | **0.9555** | — | **SHIPPED as model_2.** Test: 0.9545 AUC / 0.9174 accuracy |
| 14 | 2026-10-05 | blend | 0.7·XGB + 0.3·HistGB | 42 | 0.9556 | — | **Rejected.** +0.00013 over XGBoost alone, for two models to maintain |
| 15 | 2026-10-05 | XGBClassifier | 40 random configs + 6 confirmed | 42 | 0.9560 | — | Tuning exhausted. +0.0005 validation over run 13 |
| 16 | 2026-10-05 | **XGBClassifier, native categorical** | `enable_categorical=True`, 22 features | 42 | **0.9588** | — | **+0.0029, the largest single gain in the project.** SHIPPED as model_4 |
| 17 | 2026-10-05 | PyTorch MLP | 2×128, BatchNorm, 3 seeds averaged | 42 | 0.9532 | — | Respectable, and beat the pre-search XGBoost. Not better than it |
| 18 | 2026-10-05 | kNN | k=100, 40k training subsample | 42 | 0.9399 | — | Weakest member by 0.013 |
| 19 | 2026-10-05 | 4-model ensemble | all 11 subsets tried | 42 | ≤0.9583 | — | **Every subset lost to native-categorical alone.** No blend beat 0.958868 |

---

## What the optimisation rounds actually found

The gain from tuning was **+0.0025 ROC-AUC in total**, and it came from exactly
one place.

| Stage | ROC-AUC | Gain |
|---|---|---|
| scikit-learn defaults | 0.9531 | — |
| `config.yaml` hand-set params | 0.9544 | +0.0013 |
| HistGB, capacity pushed | 0.9550 | +0.0006 |
| **XGBoost, depth 8, tuned** | **0.9555** | +0.0005 |
| blend of the two | 0.9556 | **+0.0001, rejected** |

### Feature engineering is exhausted, and that is a real result

Seven candidate features, built from domain reasoning, all landed between
−0.000067 and +0.000044:

| Candidate feature | Change in ROC-AUC |
|---|---|
| `count_of_ratings_at_5` | +0.000044 |
| `mean_service_rating × Class_Business` | +0.000000 |
| **12 per-rating "was zero" indicators** | **+0.000000** |
| `spread_of_ratings` | −0.000021 |
| all four derived extras together | −0.000066 |
| `count_of_zero_ratings` | −0.000067 |

Twelve extra columns moving the score by *exactly zero* is the clearest signal
here: **the raw ratings already contain everything the derived versions encode.**
The model is not short of features. It was short of capacity.

The zero-indicator result also settles part of open question 3 by measurement —
letting the model learn what a `0` means, rather than being told, changes nothing.
Whatever `0` encodes is already recoverable from the other twelve ratings.

### Native categorical features beat every other change combined

| Encoding | Validation ROC-AUC | Features |
|---|---|---|
| one-hot (run 13) | 0.955924 | 25 |
| **native (run 16)** | **0.958868** | **22** |

**+0.0029, for fewer columns.** XGBoost partitions the four category values
itself instead of receiving seven indicator columns that throw the ordering away.
That is roughly **nine times** what 46 hyperparameter configurations achieved.

The feature engineering that failed so comprehensively in run 11 was searching the
wrong axis. It added *derived numeric* columns. This changes how the *existing*
categorical columns are represented — no new information at all, just less
destruction of what was already there.

### The ensemble failed, and all 11 ways of combining it

| Members | Validation ROC-AUC |
|---|---|
| **native-categorical XGBoost alone** | **0.958868** |
| native + one-hot | 0.958288 |
| native + MLP | 0.957620 |
| native + kNN | 0.954534 |
| native + one-hot + MLP | 0.957632 |
| all four | 0.955760 |

**Nothing beat the best single member.** Not one subset, not the weight-optimised
blend measured out-of-fold (0.958762). The members are four views of one signal,
and averaging in three weaker views of it just adds their errors.

The earlier two-model blend failed for the same reason, and I attributed it to the
members being too similar. That diagnosis was right but the fix was not — adding a
neural network and a kNN did not make them different enough.

### Where the headroom went, and where it stopped

| Direction | Result |
|---|---|
| More capacity, smaller leaves | **Worked.** Every gain pointed here — the signature of an underfit model |
| depth 10, 1500 trees | **Worse** (0.9552). Past the optimum |
| `l2_regularization` 0.5 → 0 | **Worse** (0.9548). The regularisation was earning its place |
| Blending two families | **Not worth it.** +0.00013, and the blend curve is flat from weight 0.3 to 0.7 |
| Feature engineering | **Nothing.** Seven ideas, zero movement |

**So the honest answer to "0.953 is low" is: 0.953 was low, and most of that has
now been taken.** The remaining gap to published solutions on this dataset is
roughly 0.005, and the three things that could still close it are:

1. **Optuna.** These eleven settings were hand-picked. A real search over
   `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`,
   `reg_lambda` and `gamma` has more room than I explored by hand
2. **A genuinely diverse ensemble.** Two gradient-boosted trees are nearly the
   same model. Something structurally different — a neural network, or a model on
   differently-encoded features — would disagree more than these two do
3. **Possibly nothing.** The data is synthetic. There may be an irreducible noise
   floor around 0.956, and no amount of tuning reaches past it

---

## The findings that changed a decision

### 1. Dropping features on effect size alone cost 0.0055 ROC-AUC

Run 2 used 17 features. Run 7 used 25. The nine I removed had negligible
*individual* effect sizes, so removing them looked obviously right.

| Feature set | Features | ROC-AUC |
|---|---|---|
| What I originally chose | 17 | 0.9476 |
| After adding them back | 25 | 0.9531 |

`.dev/EDA.md` says to delete a feature that "stops being useful". I read that as
"useless on its own", and it does not mean that. Each removed column was worth
between +0.0003 and +0.0024 *as an interaction partner*, and they add up.

### 2. The worst univariate feature was the best contributor

| Column | Cliff's δ (individual) | Gain in ROC-AUC when added back |
|---|---|---|
| `Departure/Arrival time convenient` | **−0.0525, the worst of the 13** | **+0.00244, the largest** |
| `Gate location` | +0.0277 | +0.00154 |
| `Baggage handling` | +0.3110 | +0.00106 |
| `Gender` | 0.0080 | **+0.000000, exactly nothing** |
| both delay columns | −0.018 / −0.032 | **−0.000048, nothing** |

**Never drop a column on its univariate statistic.** The three that are genuinely
dead are `Gender` and the two delays, and the ablation is the only reason I know
that. They are the only three still excluded.

### 3. XGBoost and HistGradientBoosting are the same model for our purposes

| | ROC-AUC | Accuracy | Fit time |
|---|---|---|---|
| XGBoost | 0.9541 | 0.9158 | 2.4s |
| HistGradientBoosting | 0.9531 | 0.9138 | 4.4s |

**+0.0010 ROC-AUC.** `.lead/02-C` Step 10's simplicity check asks whether the
extra complexity earns its keep. One extra dependency, one extra library to
learn, and a duplicate code path, for one point in the fourth decimal place.

XGBoost was put on the shortlist only because the owner requires easy
hyperparameter optimisation. It is measurably better on that axis and
immeasurably better on score. **That is a close call, and it is the owner's to
make** — see "Still open".

### 4. Random forest is dominated on every axis

0.9512 against 0.9531, and **51.5 seconds to fit against 4.4**. Eleven times the
cost for a worse score. Removed.

### 5. Linear models are 0.08 behind, so the signal is genuinely non-linear

Logistic regression reaches 0.8732. That is a real gap, not a tuning problem, and
it is worth recording: the relationship between a passenger's ratings and their
verdict is not a weighted sum of those ratings.

---

## Runs 20 and 21 — the reference notebook's two ideas, measured here

| Run | Date | Model | Features | Seed | ROC-AUC (val) | ROC-AUC (test) | Notes |
|---|---|---|---|---|---|---|---|
| 20 | 2026-10-05 | XGBClassifier, native | 22 + 18 route | 42 | 0.958438 | **0.957540** | **Route features: −0.0004 val / −0.00035 test.** Turned off |
| 21 | 2026-10-05 | XGBClassifier, native | 22 + 13 aux | 42 | 0.956347 | **0.955143** | **Aux features: −0.0025 val / −0.0028 test.** Turned off |
| 22 | 2026-10-05 | XGBClassifier, native | 22 (back to model_6) | 42 | 0.958844 | 0.957894 | Confirmed deterministic — identical to model_4/model_6 |

### Why both failed here and worked there

Both ideas helped a reference notebook whose base was 21 raw columns (+0.0016 and
+0.00013). Our base already has native categoricals and three derived features.

| Their base | Our base | What the new columns add |
|---|---|---|
| 21 raw columns | 22 engineered columns | — |
| Route profile describes *which route* — genuinely new | Route-adjacent info already in Class mix + native splits | Redundant |
| Aux expected ratings add *smoothed* signal | Raw ratings + mean_service_rating already there | 13 near-duplicates that dilute splits |

**New information beats re-encoded information.** Native categorical won (+0.0029)
because it stopped destroying information. Route and aux re-encode information the
model already has, in a base that already extracts it.

The aux failure is the sharper of the two: −0.0028 is not noise, it is damage.
Thirteen columns each highly correlated with existing signal split the tree's
attention across redundant copies of the same split.

Both feature blocks stay in the code behind flags, with tests, in case the base
changes. Both are off.

## Runs 22 and 23 — Kaggle GPU: verification and RealMLP

| Run | Date | Model | Features | Seed | ROC-AUC (val) | ROC-AUC (test) | Notes |
|---|---|---|---|---|---|---|---|
| 22 | 2026-10-05 | XGBClassifier, native, `device: gpu` | 22 | 42 | 0.958548 | **0.957667** | **GPU verified.** CPU test was 0.957894. Gap −0.000227, inside the 0.0005 tolerance. Fit 4.7s on GPU vs ~25s CPU |
| 23 | 2026-10-05 | **RealMLP, 8 members, 3 epochs** | 22 (ours) | 42 | **0.959051** | — | **Beats XGBoost native (0.958868) by +0.000183 on identical features.** 67s on 2× T4 |

### Correction: the architecture does matter

`docs/method_plan.md` Step 5 excluded neural networks "by the data and by the
tuning requirement", and `research/mlp_experiments.py` measured a basic MLP at
0.953235 — 0.0056 behind. Both of those are about a *basic* MLP.

RealMLP is not a basic MLP. It is a tabular-specific architecture (categorical
embeddings, PLR numerical embeddings, tuned schedule), and on the same 22
features, same split, same seed it scores **0.959051 vs 0.958868**. The earlier
claim that "their advantage was the features, not the architecture" was wrong.
About +0.0002 of their lead is architecture.

Decomposed, using their best single (RealMLP v4, 0.961166, on their features):

| Gap | Size | Source |
|---|---|---|
| Features (theirs vs ours) | ~+0.0021 | Route profile, target encodings, teacher prediction |
| Architecture (RealMLP vs XGB, same features) | **+0.0002** | Measured here |

Features dominate 10-to-1. But the architecture gap is real, not zero.

### Recommendation stands, reason changes

RealMLP is NOT switched in as the shipped model, but no longer because "it can't
win". Because **+0.0002 does not justify the cost**: torch + Lightning + pytabkit
as dependencies, GPU-only for reasonable speed (67s vs 4.7s), 8 ensemble members
to maintain — for a gain inside the CV noise floor (std 0.00048).

`.lead/02-C` Step 10's simplicity check cuts against it. If the gap were +0.002,
the answer would differ.

### GPU notes from the run

- `device: "gpu"` worked. Fit 4.7s (was ~25s CPU).
- XGBoost warns once per run: *"Falling back to prediction using DMatrix due to
  mismatched devices... while the input data is on: cpu."* Prediction falls back
  to a CPU path. Correctness unaffected (test AUC confirms it), speed slightly
  reduced on predict only. Not worth fixing for a one-shot submission.
- 2× T4 visible, both used by Lightning for RealMLP. XGBoost used `cuda:0`.
- GPU training is not bit-reproducible. Compare at 0.0005 tolerance, never exactly.

## Runs 24 and 25 — RealMLP in the pipeline, measured twice

| Run | Date | Model | Features | Seed | ROC-AUC (val) | ROC-AUC (test) | Notes |
|---|---|---|---|---|---|---|---|
| 24 | 2026-10-05 | **RealMLP, published recipe** | 22 native | 42 | 0.958169 | **0.957141** | Pipeline green end-to-end. Loses to XGBoost by −0.00075 test |
| 25 | 2026-10-05 | **RealMLP, defaults** | 22 native | 42 | 0.956846 | **0.956199** | Worse than the recipe by −0.0009 test. Recipe transfers, partially |

### The recipe transfers, but not fully

| Source | Settings | Validation |
|---|---|---|
| research script, defaults | n_ens=8, n_epochs=3, rest default | **0.959051** |
| pipeline, published recipe minus 3 schedules | full recipe | 0.958169 |
| pipeline, defaults | n_ens=8, n_epochs=3, rest default | 0.956846 |

The research script's 0.959051 does **not** reproduce via the pipeline (0.956846,
gap −0.0022, same features, same seed). Likely cause: the pipeline seeds global
torch/CUDA RNGs via `seed_torch` before building, which changes the internal
state RealMLP's own validation split depends on — despite `random_state=42`.
Seeding is necessary for reproducibility and changes the number it reproduces.
Recorded, not yet chased.

### Scoreboard, test split, the only number that counts

| Model | Test ROC-AUC | vs best |
|---|---|---|
| **XGBoost native (model_4)** | **0.957894** | — |
| RealMLP recipe (model_10) | 0.957141 | −0.00075 |
| RealMLP defaults (model_11) | 0.956199 | −0.00170 |

**XGBoost remains the best shipped model.** The RealMLP rewrite is complete,
tested and working — train, save, load, predict, submission, all green — but on
test it loses. The implementation stays (both families are one config line apart);
the crown does not move on hope.

## Runs 26 and 27 — Tier 1 for RealMLP: v4 twins, then aux

| Run | Date | Model | Features | Seed | ROC-AUC (val) | ROC-AUC (test) | Notes |
|---|---|---|---|---|---|---|---|
| 26 | 2026-10-05 | **RealMLP, v4 twins, 12 members, 6 epochs** | 22 + 20 twins = 42 | 42 | 0.957353 | **0.956872** | Beats defaults by +0.0007 test. Loses to XGBoost by −0.001 |
| 27 | 2026-10-05 | **RealMLP, v4 + 13 aux expected ratings** | 42 + 13 = 55 | 42 | 0.957359 | **0.956847** | Aux adds +0.000006 val / −0.000025 test. Nothing |

### Tier 1 verdict, item by item

| Item | Expected | Measured here | Verdict |
|---|---|---|---|
| `flat_anneal` schedule | +0.00026 | Already in config | Kept |
| v4 twins + 12 members + 6 epochs | +0.00046 | **+0.0007 test over defaults** | **Works, kept** |
| Aux expected ratings for RealMLP | +0.00017 | +0.000006 val / −0.000025 test | **Dead, turned back off** |
| 12 members (part of v4) | +0.00002 | Included above | Kept |

The v4 direction is right — twins + capacity beat defaults. But RealMLP+v4 at
0.956872 still trails XGBoost native at 0.957894 by a full point in the third
decimal. The twins helped RealMLP catch up halfway and no further.

Aux features are now dead for **both** families: −0.0028 for XGBoost, ±0.0000 for
RealMLP. Their +0.00017 came from 30 features including original-data predictions
we do not have. Our 13 expected ratings are redundant with `mean_service_rating`
for any model that can already average.

## What is still open

- **XGBoost or HistGradientBoosting.** +0.0010 ROC-AUC for an extra dependency.
  `docs/method_plan.md` section 5 argues for HistGradientBoosting on simplicity;
  the owner argues for XGBoost on tuning ergonomics. Both are defensible. Not
  decided here.
- **Open question 6, the metric.** Untested: whether the competition scores
  ROC-AUC or accuracy. Both are recorded in every row above, so the answer is
  already available.
- **Open question 7, the success target.** The proposed 0.87 was removed as
  meaningless — an untuned model reaches 0.9573. The target should now come from
  what a *tuned* model achieves, which run 8 puts at **0.9544 validation AUC**.
- **Tuning has not happened.** `config.yaml` holds hand-set values, not Optuna
  output. That is `.lead/02-C` Step 11, and it comes after Step 9 reads the
  errors of the winner.
- **The neural-network question is unanswered by measurement.** `.lead/02-C` says
  gradient boosting beats deep learning on tabular data, and the comparison above
  shows boosting at 0.9541 against logistic regression's 0.8732. But no MLP was
  fitted, so the claim is argued, not measured here.

---

## How to reproduce any row

```bash
python run_pipeline.py                    # the shipped model, config.yaml as committed
python research/compare_models.py         # runs 3 to 7, one table, default settings
```

Both take no arguments. Run logs are in `logs/`, named by date and time, and every
one carries the git commit, the config hash, the seed and the exit code.

**One caveat on these numbers.** The training data is synthetic, generated by the
competition organisers. A ROC-AUC of 0.95 here describes how well one survey
answer predicts another on the same form. It is not a finding about real airline
passengers, and it must not be reported as one.

## What has changed

- **2026-10-04:** runs 0a to 1, from the EDA and the first baseline measurements.
- **2026-10-05:** runs 2 to 9, from building the pipeline. Run 2 records a mistake
  I made and runs 6 to 8 record the correction. Old rows were never edited; the
  correction is new rows.
## Runs 28 to 30 — the reference's engineered inputs, then its tuned recipe

| Run | Date | Model | Features | Seed | ROC-AUC (val) | ROC-AUC (test) | Notes |
|---|---|---|---|---|---|---|---|
| 28 | 2026-10-05 | **RealMLP, twins + route profile** | 42 + 18 route = 60 | 42 | 0.959230 | **0.958758** | Route profile is +0.0019 val for RealMLP. Beats XGBoost for the first time |
| 29 | 2026-10-05 | **RealMLP, + 6-key target encodings** | 60 + 6 TE = 66 | 42 | 0.959623 | **0.959011** | Encodings add +0.0004 val / +0.00025 test |
| 30 | 2026-10-05 | **RealMLP, tuned recipe (all 27 params)** | 66 | 42 | 0.960433 | **0.959356** | The recipe, not defaults: +0.0008 val / +0.00035 test |

### Correction to Runs 26–27: two claims in the Tier 1 verdict were wrong

**`flat_anneal` was never in our config.** The verdict table said "Already in
config — Kept". It was not. `model_params` held only `n_ens`, `n_epochs` and
`device`; everything else was pytabkit's generic `RealMLP_TD_CLASS` defaults.
Introspecting those defaults against the reference's `REALMLP` dict: **23 of 26
values differ**. The two that cost us most were `plr_sigma` (0.1 vs 2.33, which
effectively disables the PLR numerical embeddings that are the entire point of
RealMLP) and `ls_eps` (0.1 vs 0.01, ten times the label smoothing). Run 30 ports
the recipe verbatim, including `flat_anneal`, and gains +0.0008 validation for it.

**Aux was declared dead from the wrong construction.** Runs 27's "dead for both
families" used 13 `expected_` columns built with HistGradientBoosting. The
reference's aux is 30 columns built with XGBoost: 16 `aux_p_` (the model's
probability of the row's *own* value), 13 `aux_ev_` (its expected value for that
rating), and `aux_sum_logp`. Their table puts that construction at +0.0018 on a
single RealMLP — their best single model. Ours is being rebuilt to match, not
assumed dead. The verdict is suspended until the matching construction is measured.

### Why the route profile helped RealMLP after hurting XGBoost

Run 14 (XGBoost era) measured route features at −0.0004 and disabled them. Run 28
measures the same features at +0.0019 validation for RealMLP. Same columns,
opposite sign, different model. XGBoost learns route statistics from raw distance
splits, so the profile is redundant for it. RealMLP cannot form that comparison
from a normalised distance, so the ready-made per-route means are new information
for it. The lesson is not "route features are good" but "a feature's value is a
property of the (feature, model) pair, and a measurement on one model does not
transfer to another". Disabling globally from a single model's ablation was the
mistake; the flag is now per-experiment, not per-project.

### Scoreboard, test split

| Model | Test ROC-AUC | vs best |
|---|---|---|
| **RealMLP tuned + route + TE (model_16)** | **0.959356** | — |
| RealMLP defaults + route + TE (model_15) | 0.959011 | −0.00035 |
| RealMLP defaults + route (model_14) | 0.958758 | −0.00060 |
| XGBoost native (model_4) | 0.957894 | −0.00146 |
| RealMLP v4 twins only (model_12) | 0.956872 | −0.00248 |

**RealMLP is now the best shipped model.** The crown moves on measurement, not hope.

## Runs 31+ — the systematic plan: one split, one matrix, no more one-offs

The runs above were made one at a time on a single validation split, and several
conclusions drawn from them turned out to be wrong (see the corrections). From
here the discipline is:

1. **One split for all selection.** `StratifiedKFold(5, shuffle=True,
   random_state=42)` over the training rows only — the same split the reference
   used, so our out-of-fold numbers are comparable to theirs. The validation and
   test splits stay closed until the final confirmation. `research/make_members.py`
   generates every candidate's predictions on this split; `research/build_stack.py`
   scores them.
2. **Noise floor before claims.** The reference measured seed-noise at ~0.00001 on
   its best models and ~0.00005 on its new-feature models, on 5-fold. Any delta
   below ~0.00005 on our 5-fold is noise and is reported as noise. Single-split
   validation noise is larger and is no longer used for decisions.
3. **One variable at a time, same split.** The ablation matrix is: aux on/off,
   TE on/off, route on/off, twins on/off, tuned recipe vs defaults — each a
   `--drop-prefix` or config flip on the same artifacts, scored on the same folds.
4. **Breadth before depth.** One member per architecture family at a screening
   budget (3 members, 4 epochs), on the same split. A family earns a full-budget
   run only by beating the incumbent by more than the noise floor. The reference's
   Tier 3 knobs (embedding size, dropout, lr, batch, label smoothing variants,
   1024-wide nets) all measured "no effect" there and are not re-tested here.
5. **Combine, then confirm once.** Nested-CV logistic regression on member logits
   (their combiner, their regularisation, their split seed). The test split is
   measured exactly once, for the chosen submission.

What is running now: stage 2 is rebuilding the 100-feature artifacts (22 base +
20 twins + 18 route + 6 TE + 30 aux + 4 value counts). The aux block costs ~80
minutes of XGBoost fits; every selection experiment after that reuses the
artifacts and costs no rebuild.
