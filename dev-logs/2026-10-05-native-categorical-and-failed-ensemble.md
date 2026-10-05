# 2026-10-05 — Native categorical features (+0.0029) and a failed ensemble

**Worked on:** [name]
**Type of change:** categorical encoding switched; ensemble tested and rejected
**Files touched:**
- `config.yaml` — `categorical_encoding: native`; features 25 → 22
- `src/components/data_cleaning_encoding.py` — both encodings implemented
- `src/components/model_training.py` — encoding contract, XGBoost registered
- `src/components/model_evaluation.py` — same contract on read
- `src/utils/common.py` — `apply_categorical_encoding`
- `src/entity/config_entity.py` — encoding carried through three dataclasses
- `src/config/configuration.py` — reads and validates the setting
- `research/ensemble_experiments.py` — **new**
- `research/xgb_search.py`, `research/mlp_experiments.py` — **new**
- `tests/` — 53 → 62 tests

**Status:** test ROC-AUC **0.954810 → 0.957894**. +0.003084.

## What I did

Two requests: a diverse ensemble of four models, and native categorical features.
**One produced the project's largest single gain. The other produced nothing.**

## The win — native categorical features

| Encoding | Validation ROC-AUC | Features |
|---|---|---|
| one-hot | 0.955924 | 25 |
| **native** | **0.958868** | **22** |

**+0.0029, for fewer columns.** That is roughly **nine times** what 46
hyperparameter configurations achieved (+0.0005).

XGBoost partitions the four category values itself. One-hot gave it seven
indicator columns and threw away the fact that `Eco Plus` is between `Eco` and
`Business` rather than a fourth unrelated thing.

**The lesson connects to run 11.** Seven feature-engineering ideas had failed
completely, twelve extra columns moving the score by *exactly zero*. Those were all
derived **numeric** columns. This changes the representation of **existing**
categorical columns and adds no new information at all. We were looking for signal
in the wrong place and destroying signal we already had.

## The failure — the ensemble

| Members | Validation ROC-AUC |
|---|---|
| **native-categorical XGBoost alone** | **0.958868** |
| native + one-hot XGBoost | 0.958288 |
| native + MLP | 0.957620 |
| native + kNN | 0.954534 |
| all four | 0.955760 |

Members first: native 0.958868, one-hot 0.955924, MLP 0.953235, kNN 0.939895.

**All 11 subsets lost to the best single member.** So did the weight-optimised
blend, measured out-of-fold so the number was honest (0.958762).

The members are four views of one signal. Averaging three weaker views of a thing
into a stronger one adds their errors and none of their information. kNN at
0.9399 hurt most, which is what you would expect — it is the furthest from the
others in kind.

**Rejected. Not built.** A stacking service, two more models to maintain and a
fourth dependency, for a measured loss.

## How the blend was measured honestly

Choosing blend weights on validation and reporting validation would be circular.
So the weight-optimised blend used a nested split: weights fitted on one half of
validation, scored on the other, both ways round, pooled. That number —
0.958762 — is out-of-fold. Even so it lost.

## Four bugs on the way, all caught by a check

1. **CSV does not carry pandas `category` dtype.** The encoding was written by
   stage 2 and came back as `str`, so XGBoost rejected it with an error about
   `enable_categorical` — pointing at the model when the cause was the file
   format. Fixed by re-applying the dtype from the contract stage 2 writes.
2. **The model bundle did not carry the contract.** Stage 4 reads
   `models/model_4/features.json`, and stage 3 had written only a feature list.
   `.lead/03` Step 3.6 says all five files travel together; this is what that rule
   is for. The bundle now records the encoding and the categorical columns.
3. **`enable_categorical` is XGBoost-only.** `build_model` now refuses
   `categorical_encoding: native` with any other model, and says why, instead of
   letting it fail as a dtype error inside `fit()`.
4. **XGBoost 3.x already defaults `enable_categorical=True`,** so my test that
   read it back off a built model could not tell who set it. Rewritten to test the
   helper directly.

## The PyTorch MLP, for the record

The owner asked twice whether torch would do better. Now measured, not argued:
**0.953235** across three seeds averaged, against 0.958868 for the native
categorical XGBoost. It beat the *pre-search* XGBoost (0.955511 at defaults), so it
is a real model — it just does not win. `.lead/02-C` Step 1's claim held up.

BatchNorm, He initialisation, AdamW, three seeds averaged, and a 3% early-stopping
holdout. Not a strawman.

## What I did not do

- **No ensemble was built.** It lost.
- **No GPU run.** This machine has no GPU. `device: "cpu"` is still in config.yaml.
- **No Optuna.** The 40-config random search was the substitute, and it showed
  tuning is exhausted.
- **I did not touch `Gender`.** It measured at exactly +0.000000 in an earlier
  ablation, but dropping it now would change the feature count from the 22 the
  +0.0029 was measured on. Left alone deliberately.

## Still open

- **`Gender` should go.** It is measured at exactly zero and costs a column. Left
  only because it was in the measured set.
- **The 0.965 target is still 0.007 away.** Native categorical closed a quarter of
  the original gap. What remains: a genuinely independent ensemble member,
  pseudo-labelling on the 299k unlabelled test rows, or an irreducible noise floor.
- **`research/auc_experiments.py` round 1 measured sklearn defaults as its
  baseline**, not the shipped config. The deltas in that file mean less than they
  appear to.
- **Open questions 6 and 7 unchanged.** Metric still unconfirmed; the target
  should now be based on 0.957894, not the discarded 0.87.