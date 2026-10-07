# Problem Statement

**Written:** 2026-10-04
**Covers:** Steps 0.1 to 0.5 and the Gate of `.lead/00-PROBLEM-AND-BASELINE.md`.
**Source of this statement:** the project owner, in conversation, on 2026-10-04.
**Status:** four boxes are OPEN. See [open_questions.md](open_questions.md).

```text
PROBLEM STATEMENT
Date:        2026-10-04
Owner:       OPEN — not yet named. The person who would act on the output.
Agreed with: OPEN — not yet recorded.

We want to predict whether a passenger was satisfied with their flight, for the
299,844 flights in data/test.csv, so that [PERSON/TEAM] can [SPECIFIC ACTION].

Today they [WHAT THEY DO NOW], which costs [TIME / MONEY / RISK].

We will know this model is useful if [SPECIFIC MEASURABLE TARGET — see the
proposed target below, which needs the owner's sign-off].

Out of scope: Predicting satisfaction for any flight outside data/test.csv.
             Explaining to a passenger why they were unhappy.
             Anything about an individual passenger's identity.
             Any claim that a result from this synthetic dataset is a real
             finding about real airline passengers.
```

The four OPEN items are not stylistic gaps. `.lead/00` Step 0.1 says to fill
this in *with the person who will use the model, not with the data*, and writing
them alone would be building the wrong thing politely. What I need from the
owner is listed as one short list at the bottom of this page.

---

## What is actually being asked for

The dataset is the Kaggle Playground Series S6E10 file set. The task as given
is a prediction task:

| Item | Value |
|---|---|
| Input | `data/train.csv` — 699,635 past flights, with the answer |
| To predict | `data/test.csv` — 299,844 flights, answer withheld |
| Output | One `satisfaction` value per row of `test.csv` |
| Submission file | `data/sample_submission.csv` — has `id` and `satisfaction` columns |

`satisfaction` is `True` or `False`. The submission template contains
continuous probability scores (not a constant), with values ranging from
~0.004 to ~0.997. The mean happens to be close to the base rate (0.4438 vs
0.4436), which may have caused confusion.

Full row counts, primary keys and file hashes are in
[data_inventory.md](data_inventory.md). The label is defined in
[label_definition.md](label_definition.md).

---

## Step 0.2 — The seven questions

| # | Question | Answer |
|---|---|---|
| 1 | Who makes the decision using this model's output? | **ANSWERED 2026-10-04.** Nobody operationally. The only consumer is the competition scoring of `sample_submission.csv`. There is no business decision attached. |
| 2 | What do they do today instead? | **ANSWERED 2026-10-04.** Nothing exists today — no one predicts satisfaction, so there is no current practice to measure. The "current practice" baseline is genuinely not buildable, and that is an honest gap rather than a skipped step. |
| 3 | What does a wrong prediction cost? | **ANSWERED 2026-10-04.** No direct cost. A submission can be replaced at any time, and nothing operational depends on it. |
| 4 | What is the base rate? | **KNOWN.** 310,339 of 699,635 training rows are `True` = **44.36% satisfied, 55.64% not satisfied**. Measured 2026-10-04. |
| 5 | Is the decision reversible? | **YES.** A submission can be replaced at any time, so the cost of a wrong answer is low. |
| 6 | How often does the world change? | **KNOWN: never.** The files are a static snapshot from August 2025. Nothing updates, so there is no retraining schedule and no drift to monitor. See [data_inventory.md](data_inventory.md). |
| 7 | What is the smallest useful version? | **ANSWERED 2026-10-04:** one model that beats the simple rule, plus a valid `sample_submission.csv`. No serving, no API, no monitoring. |

### On question 4 — why it matters more than usual here

The base rate is 44.36%, so this is **not** the severely imbalanced problem
`.lead/00` warns about. "Always answer False" is 55.64% accurate, and no model
can look impressive by beating that. That is a relief, and it is also a trap:
with a base rate this balanced, plain accuracy is defensible and might be used
by default, and accuracy hides the thing that actually matters — how many rows
the model claims are satisfied when they are not.

See the proposed metric below.

---

## Step 0.3 — Is machine learning the right tool?

**Yes. Written down here so it does not get relitigated later.**

`.lead/00` Step 0.3 says to use a simple rule instead when there are fewer than
about 20 rules a person could write down, when very little data exists, or when
the pattern changes every week. None of those hold:

| Test in Step 0.3 | This project |
|---|---|
| Under ~20 rules? | No. 21 inputs, including 13 separate service ratings that interact with each other. Nobody can write this as a rule list. |
| Little data? | No. 699,635 labelled rows. |
| Answer must be exact and checkable? | No. It is a judgement, recorded as a survey answer. |
| Pattern changes weekly? | No. It is a fixed snapshot. |
| Depends on many inputs at once? | **Yes.** This is the ML case. |
| Lots of historical data with the outcome recorded? | **Yes.** |
| Being right most of the time is valuable? | **Yes**, once question 5 is answered. |

**One honest qualification,** carried over from
[label_definition.md](label_definition.md). The features are the passenger's own
ratings on the same survey form as the answer. So the model is not predicting a
reaction — it is inferring one answer from the others on the same form. That is
a legitimate thing to model and it is what this dataset can do, but it is **not**
a system that can warn an airline that a flight is about to go badly, because
nothing is known until after the flight. Whether that limitation is acceptable
is accepted as it stands: open question 2 was answered on 2026-10-04.

---

## Step 0.4 — The baselines

`.lead/00` Step 0.4 calls this the most skipped step in the whole process. These
are the numbers every later model has to beat.

**Measured by:** `python research/02_baselines.py`
**Measured on:** all 699,635 rows of `data/train.csv`, on 2026-10-04
**Evidence:** `logs/dev/2026-10-04_2131_measure_baselines.log`

### Why the whole file and no split

A baseline is a rule with no fitted parameters. "Always answer False" learns
nothing, and the simple rule's threshold was written into the script *before*
the script was run. There is nothing here that can be overfitted, so there is
nothing to hold back.

### Baseline 1 — do nothing

Always predict "not satisfied". This is the floor.

| Measure | Value |
|---|---|
| Accuracy | **0.5564** |
| Satisfied passengers flagged | 0 of 310,339 |
| Satisfied passengers caught | 0% |

### Baseline 2 — one simple rule

**The rule, fixed before the run:** `Inflight entertainment` is rated 5 out of 5
→ predict satisfied; otherwise predict not satisfied.

The reason for that choice, recorded before the run: a passenger's overall
satisfaction is dominated by whether the things they remember were good, and
in-flight entertainment is the service this survey consistently shows as the
strongest single driver. One feature, one threshold, no tuning.

| Measure | Value |
|---|---|
| Accuracy | **0.6455** |
| Precision | 0.6711 |
| Recall | 0.3937 |
| Rows flagged satisfied | 182,068 |

### The finding that matters: the registered rule is not the best one

I printed all thirteen service ratings scored the same way, so that no
single-feature result could be quietly cherry-picked. One is much stronger than
the rule I registered.

| Rule — "rated 5 means satisfied" | Accuracy | Precision | Recall |
|---|---|---|---|
| **Online boarding** | **0.7203** | **0.8751** | 0.4311 |
| Inflight wifi service | 0.6586 | 0.9504 | 0.2430 |
| Seat comfort | 0.6524 | 0.6711 | 0.4245 |
| Inflight entertainment *(registered above)* | 0.6455 | 0.6711 | 0.3937 |
| On-board service | 0.6343 | 0.6566 | 0.3679 |
| Ease of Online booking | 0.6232 | 0.7543 | 0.2231 |
| Baggage handling | 0.6199 | 0.6096 | 0.3981 |
| Cleanliness | 0.6172 | 0.6305 | 0.3311 |
| Leg room service | 0.6164 | 0.6151 | 0.3615 |
| Checkin service | 0.6030 | 0.6103 | 0.2902 |
| Gate location | 0.5894 | 0.6248 | 0.1862 |
| Food and drink | 0.5855 | 0.5655 | 0.2831 |
| Departure/Arrival time convenient | 0.5283 | 0.4373 | 0.2212 |

**What follows from this, and it matters:**

1. The data clearly holds signal. A single column with one threshold gets
   0.7203 accuracy and 0.8751 precision. Step 0.4's question — "does the data
   even hold the signal" — is answered yes.
2. **The honest bar for any model is 0.7203 accuracy, not 0.6455.** I am keeping
   the weaker pre-declared rule registered as the baseline, because swapping the
   goalposts after seeing the results is exactly what Step 0.4 exists to prevent.
   But nobody should use 0.6455 as the number to clear. It would be too easy.
3. A human-readable one-line rule at 0.72 is a serious result. If a model beats
   it by two points, Step 2.7's question applies: is that worth the complexity?
   Sometimes the answer is no, and that is a successful outcome.

### Update, 2026-10-04 — the EDA found a stronger baseline

`research/02_eda.ipynb` scored every single-column rule on the train split, not
just the registered one. One of them is much better.

| Rule | Accuracy | Precision | Recall |
|---|---|---|---|
| **`Class == Business`** | **0.7770** | 0.7257 | 0.7995 |
| `Online boarding == 5` *(the registered baseline)* | 0.7206 | 0.8754 | 0.4315 |
| `Type of Travel == Business travel` | 0.6761 | 0.5841 | 0.9366 |
| `Customer Type == Loyal Customer` | 0.5487 | 0.4953 | 0.9206 |
| do nothing | 0.5564 | 1.0000 | 0.0000 |

**The honest bar is therefore 0.7770, not 0.7203.** A one-line rule on cabin
class beats the registered rating rule on accuracy *and* recall, at a comparable
precision. The number 0.7203 in the sections above was correct when it was
written and is superseded by this one.

Two things this says about the problem itself:

- **Cabin class matters more than any service rating.** Business class is
  satisfied at 4.3 times the rate of Eco (72.6% against 16.8%). Cramer's V for
  `Class` is 0.3931, against 0.6816 Cliff's delta for the best rating.
- **A single column is already a good model.** `.lead/02-B` Step 2.7 asks
  whether a model that beats the simple rule by a small margin is worth the
  complexity. At 0.7770 with one boolean condition, that question is now a
  genuine question rather than a formality.

The registered baseline stays registered. Choosing the goal after seeing the
scores is what Step 0.4 exists to prevent, and the full table is printed in
`logs/dev/` either way.

---

## The metric — PROPOSED, not agreed

`.lead/02-C` Step 3 says the metric must come from the business decision, not
from what is easy to compute. There is no decision yet (question 5), so this is
a proposal with its evidence attached:

**PROPOSED: ROC-AUC as the headline metric, with precision reported at a fixed
recall so the result stays honest and readable.**

| Evidence for it | |
|---|---|
| `sample_submission.csv` ships a **continuous** column, not `True`/`False` | Ranking metrics want scores. Accuracy wants hard labels. The template implies the organisers expect a score. |
| It is filled with the constant 0.44357272006117476 | That is the base rate. A constant submission scores 0.5 AUC, which is the floor for a ranking metric. |
| `Online boarding` alone reaches 0.8751 precision at 0.4311 recall | It ranks well and is conservative. That is an AUC-shaped result, not an accuracy-shaped one. |

**This is strong evidence, not proof.** I could not reach the competition page
to confirm the metric — Kaggle returned an error. Recorded as open question 6.
Do not build the evaluation around an unconfirmed metric.

---

## Step 0.5 — The risks

| Risk | Likelihood | What we will do about it |
|---|---|---|
| **The data is synthetic.** Playground datasets are generated by the organisers. A model can score well on the leaderboard and predict nothing true about real passengers. | High | Stated in the problem statement's out-of-scope, and it must be repeated in the model card. No result from this project may be presented as a finding about real airlines. |
| **The model cannot help before a flight.** 14 of the 21 features are only known after the passenger answers the survey. | High | Recorded in [label_definition.md](label_definition.md). If the goal is warning an airline in advance, this dataset is the wrong dataset and no modelling fixes it. Owner's decision — open question 2. |
| **Nobody owns the output.** A model with no named person behind it gets built and never used. | Medium | Open question 5. The gate stays unticked until this is answered. |
| **The `satisfaction` cut point is unconfirmed.** We do not know what question was asked or where the line between satisfied and dissatisfied sits. The base rate is therefore a number without a definition. | Medium | Open question 3. Affects what 0.4436 means, not whether we can model it. |
| **The metric is unconfirmed**, and accuracy would hide false alarms. At a 44% base rate a mediocre model looks respectable. | High | Open question 6. Fix the metric before any model is built. |
| **`id` leaks the train/test split.** Every training id is lower than every test id. A model that sees `id` scores well and means nothing. | Medium | Already handled — `id` is banned as a feature in [data_inventory.md](data_inventory.md). |
| The team does not trust the output | Low for now | No team exists yet. Revisit if this becomes a real project. |
| The decision it supports is itself cancelled | Low | Nothing has been decided yet. |

---

## Gate status for `.lead/00-PROBLEM-AND-BASELINE.md`

- [x] The problem is written in one paragraph, in plain language
- [ ] The last line carries a measurable number — a target is proposed below and
      the bar it must beat is now measured (0.7770), but the target number itself
      is still open question 7
- [x] The person who will act on the output is named — **there is nobody**. The
      only consumer is the competition scoring of `sample_submission.csv`,
      confirmed 2026-10-04
- [x] The current practice is written down: **there is none.** Recorded, not
      skipped. Nothing predicts satisfaction today
- [x] The base rate is known: **44.36% satisfied** (see
      [label_definition.md](label_definition.md))
- [x] The "do nothing" and "one simple rule" baselines are measured —
      **0.5564** and **0.7770**, see Step 0.4 above and the 2026-10-04 update
- [x] ML has been chosen over rules and simple lookups, and the reason written
      down
- [x] Risks are listed
- [x] The statement has been agreed by the person who asked for the model —
      the owner set the scope on 2026-10-04: build a submission for `test.csv`
      from a competition-supplied dataset

**Eight of ten are met.** The two that remain both need one number from a person:
the success target (open question 7) and, behind it, the metric (question 6).
Everything else the analysis can reach has been reached — the EDA in
`research/02_eda.ipynb` has now been run.

### The proposed measurable target

For the owner to accept, change, or reject:

> We will know this model is useful if it beats the 0.7770-accuracy
> `Class == Business` rule by a margin worth the extra machinery, on a frozen
> test set, and we can produce a valid `sample_submission.csv` from it.
> A specific ROC-AUC figure is deliberately absent — see the note below.

**This target is wrong and should be replaced.** 2026-10-04, after the EDA:
`HistGradientBoostingClassifier` with **default settings and no tuning at all**
reached **0.9573 validation ROC-AUC** — nine points past the proposed 0.87. A gate
that every model passes, including a bad one, is not a gate.

What the target should be instead is open question 7, and the honest input to
that decision is:

| Number | Value | Status |
|---|---|---|
| Do nothing | 0.5564 accuracy / 0.5000 AUC | Measured floor |
| `Class == Business`, one column, one condition | 0.7770 accuracy / 0.7786 AUC | **Measured. This is the bar.** |
| Untuned `HistGradientBoosting`, all features | 0.9242 accuracy / 0.9573 AUC | Measured, no tuning |
| A *good* tuned model | unknown | Not yet measured — that is Step 8 of the method plan |

The target should be set from the last row once it exists, not guessed now. See
[`method_plan.md`](method_plan.md).

---

## What I need from the owner

Four questions block the gate. They are questions 5, 2, 3 and 6 in
[open_questions.md](open_questions.md), which also lists six more.

1. **Who acts on this output, and what will they do with it?** (question 5 — the
   person-acted-on gate box, and the metric cannot be chosen without it)
2. **Is it acceptable that the model can only run after the survey is answered?**
   (question 2 — if the goal is to predict satisfaction *before* a flight flies,
   this dataset cannot do it and we should stop now rather than build it)
3. **What was the survey question, and where does the line between satisfied and
   dissatisfied sit?** (question 3 — or is it acceptable to treat 44.36% as the
   answer without knowing?)
4. **Is ROC-AUC the metric this is scored on?** (question 6 — a link to the
   competition overview, or the word "accuracy", settles it)

Plus two low-priority confirmations: that we may use these Kaggle files for this
project (question 1), and whether the git branch should be `main` rather than
`master` (question 10).

---

## What has changed

- **2026-10-04, after the outlier work:** the proposed ROC-AUC target of 0.87 was
  removed as meaningless — an untuned default model reaches 0.9573. Also added
  the measured ROC-AUC for every baseline (0.7786 for the strongest rule).
- **2026-10-04, after the EDA:** Steps 0.2 rows 1, 2, 3, 5 and 7 answered by the
  owner. Added the 2026-10-04 baseline update after `research/02_eda.ipynb` found
  that `Class == Business` reaches 0.7770, which supersedes 0.7203 as the bar.
- **2026-10-04, later:** this page replaced an earlier stub written the same
  day, which recorded only the template with OPEN markers. Added Steps 0.2, 0.3,
  0.4 and 0.5, the measured baselines, the proposed metric, the proposed target,
  and the list of questions above. The earlier version's content is fully carried
  forward; nothing was removed.