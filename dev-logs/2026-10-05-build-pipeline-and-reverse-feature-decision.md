# 2026-10-05 — Build the pipeline, and reverse a feature decision with evidence

**Worked on:** [name]
**Type of change:** new production code, new tests, one reversed decision
**Files touched:**
- `config.yaml` — **new.** Every setting
- `run_pipeline.py` — **new.** The only entry point, no arguments
- `src/` — **new.** constants, entity, config, utils, components, pipeline
- `tests/` — **new.** 53 tests
- `research/compare_models.py` — **new.** `.lead/02-C` Steps 7 and 8
- `requirements.txt` — xgboost added, 128 pinned packages
- `.gitignore` — artifacts, reports, model binaries
- `docs/experiment_log.md` — **new**
- `docs/method_plan.md` — the feature shortlist reversed
- `dev-logs/2026-10-04-*.md` — earlier today

**Status:** the pipeline runs end to end on CPU. One decision reversed on
measurement. Method choice not final — two candidates still in contention.

## What I did

Built the whole pipeline from `.lead/02-A-ARCHITECTURE.md`: four components, four
thin stages, one entry point, a typed settings contract, shared helpers, and 53
tests including the overfitting test. Then ran the candidate comparison from
`.lead/02-C` Steps 7 and 8, and **reversed the feature list I recommended
yesterday.**

## Why the feature decision reversed

Yesterday I dropped nine columns because each had a negligible individual effect
size, citing `.dev/EDA.md`'s "delete a feature that stops being useful".

Then the pipeline ran and I measured what those columns were actually worth:

| Feature set | Features | ROC-AUC |
|---|---|---|
| What I recommended | 17 | 0.9476 |
| With them added back | 25 | **0.9531** |

An ablation of each dropped column one at a time found where the 0.0055 was:

| Column | Cliff's δ alone | Gain when added back |
|---|---|---|
| `Departure/Arrival time convenient` | **−0.0525, worst of the 13** | **+0.00244, the largest** |
| `Gate location` | +0.0277 | +0.00154 |
| `Baggage handling` | +0.3110 | +0.00106 |
| `Gender` | 0.0080 | **+0.000000** |
| both delays | −0.018 / −0.032 | **−0.000048** |

**The column with the worst univariate effect size is the biggest contributor to
the model.** I had read "not useful" as "not useful on its own", and those are
different things. Only `Gender` and the two delays are genuinely dead, and the
ablation is the only reason I know it — my statistics said all nine were.

## How I checked it

- **53 tests pass.** `pytest tests/ -q` → `53 passed in 1.21s`.
- **`ruff check .` and `ruff format --check .` both pass.** Including the 23
  `LOG015` violations that were already in `research/01_data_inventory.py`.
- **The pipeline runs end to end**, exit 0, 24 seconds: `python run_pipeline.py`.
  699,635 rows in, 489,743 / 104,946 / 104,946 out, 25 features, 0 missing.
- **The submission is verified against the template**: 299,844 rows, columns and
  ids identical, all probabilities in [0, 1], and not a constant.
- **Six failures during the build are all kept in `logs/`.** Each one found a real
  bug. Listed below.
- **Both metrics recorded on every run**, so open question 6 cannot block us.

## The bugs the tests and the runs found

Seven, and every one is in this log because a check caught it, not because I
looked hard enough.

1. **The overfitting check could never pass.** `HistGradientBoosting` defaults to
   `min_samples_leaf=20`, and the check dataset has 16 rows, so no split was ever
   legal and the model could only predict the average. It failed on every run.
   Fixed by giving the check tiny-dataset overrides — and by *not* passing it the
   production settings, which are tuned for 490,000 rows and cannot fit 16 by
   design.
2. **The check's dataset was not linearly separable.** The label was `index % 2`,
   which no straight line can separate, so logistic regression could never pass
   however it was configured. Changed to a threshold on one feature.
3. **`configuration.py` rejected every derived feature.** It compared derived
   names against the *input* column list. Derived features are computed *from*
   those columns, so the check was simply wrong. Deleted it — the component that
   does the calculating already guards it, in one place instead of two.
4. **Stage 4 read the raw competition file.** Stage 2 prepared that frame but
   never saved it, so stage 4 fell back to the raw columns and crashed on the
   first missing feature. Stage 2 now saves `competition_test.csv` and stage 4
   reads it. This was training-serving skew waiting to happen.
5. **`read_errors` was typed for a Series but given an ndarray.** `predict_proba`
   returns an array. It crashed on the first real run. Fixed the signature rather
   than wrapping the value.
6. **The feature check compared whole frames**, so it always failed: the training
   frame has `satisfaction` and the competition frame correctly does not. It now
   compares the feature columns and their order.
7. **`to_markdown()` needs `tabulate`.** Adding a dependency to render ten rows of
   a report is what `.dev/RULES.md` rule 9 warns against. Used a fenced
   `to_string()` instead.

## What I got wrong, plainly

- **I wrote a whole docs page about outliers and nothing about the other findings.**
  Ten phases of EDA, one documented topic. Corrected:
  `docs/eda_findings.md` now covers all of it.
- **I recommended dropping features on a statistic that does not measure what I
  used it for.** Cost 0.0055 ROC-AUC, caught by the pipeline rather than by
  reading my own advice again.
- **I edited `config.yaml` with a substring match and corrupted it.** `"features:"`
  matches inside `"banned_features:"`, so my edit overwrote the banned list and
  dropped `test_size` and `validation_size`. `require_key` caught the missing
  settings with a message naming them, which is exactly why it exists.

## What I did not do

- **I did not commit anything.** The repository has no commits at all, so every
  run log currently records `git_commit: unknown` and the forensic record that
  `.dev/DEBUGGING.md` section 8.2 requires is not yet complete. Committing is
  yours to ask for.
- **I did not tune.** `.lead/02-C` Step 11 comes after Step 9 reads the winner's
  errors. The hyperparameters in `config.yaml` are hand-set, not Optuna output.
- **I did not fit a neural network.** `docs/method_plan.md` argues against it on
  the evidence available. That argument is not yet measured here, and the log says
  so.
- **I did not decide between XGBoost and HistGradientBoosting.** XGBoost wins by
  0.0010 ROC-AUC; HistGradientBoosting avoids a dependency. Both defensible, and
  the call is yours.
- **I deleted four model folders** from runs that trained successfully and then
  failed in stage 4. They were byte-identical to the kept one, never evaluated,
  and from debugging runs. Logs for those runs are untouched.

## Still open

- **XGBoost or HistGradientBoosting** — +0.0010 ROC-AUC against one fewer
  dependency.
- **Open question 6, the metric.** Whether the competition scores ROC-AUC or
  accuracy. Both are recorded everywhere, so this cannot block the next step.
- **Open question 7, the target.** The old 0.87 is dead; run 8's 0.9544 is the
  honest reference for setting a new one.
- **The GPU path is untested.** Everything here ran on CPU. `config.yaml` has
  `use_gpu: false` and a note on the XGBoost parameters that need changing. The
  first GPU run should be checked against this CPU run's numbers before it is
  trusted — a silent GPU/CPU divergence is the kind of thing nobody notices for
  a week.
- **Stage 3 writes a model even when stage 4 later fails**, leaving an orphan
  model folder. Harmless here and the run log records it, but on a longer pipeline
  it would need cleaning up. Not worth building machinery for now.