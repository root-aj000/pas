# Open Questions

**Started:** 2026-10-04
**Last updated:** 2026-10-04, after the first EDA run
**Rule:** one list, everyone adds, nobody deletes. Answered questions stay in the
table with their answer, marked closed. This file is the answer to "what next".

Per `.lead/09-TEMPLATES.md`: *"Anything unclear goes here with a name against it,
and the project stops moving on that point until it is answered."*

Questions 1 to 4 were raised by the `.lead/01` documents and are numbered the
way those documents refer to them. Questions 5 onwards were added on
2026-10-04 by `.lead/00`.

| # | Question | Asked of | Asked on | Answer | Closed |
|---|----------|----------|----------|--------|--------|
| 1 | May we use these Kaggle Playground S6E10 files for this project, and store them on this machine outside the competition? | Project owner | 2026-10-04 | **CLOSED.** The dataset is supplied by the competition itself and we are building a submission against `test.csv`. No permission question arises. | 2026-10-04 |
| 2 | Is it acceptable that this model can only score a passenger **after** they answer the survey? | Project owner | 2026-10-04 | **CLOSED.** Accepted. We predict the competition target from the other answers on the same form. The model is not a pre-flight warning system and is not presented as one. | 2026-10-04 |
| 3 | What question were passengers actually asked, and where does the line between satisfied and dissatisfied sit? | Kaggle / dataset authors | 2026-10-04 | **PARTLY ANSWERED.** The competition target is whatever `satisfaction` says it is, so the modelling does not need this. It still matters for describing results honestly, and for the 292 blank arrival delays and the unexplained `0` ratings. Not blocking. | |
| 4 | How long after the flight do passengers answer? If some answers arrive much faster than others, that is a bias we cannot detect — there is no date column in any file. | Kaggle / dataset authors | 2026-10-04 | | |
| 5 | **Who acts on this output, and what will they do with it?** | Project owner | 2026-10-04 | **CLOSED.** Nobody in an operational sense. The only consumer is the competition submission file: `sample_submission.csv`, scored by Kaggle. There is no business decision attached. | 2026-10-04 |
| 6 | **Is ROC-AUC the metric this is scored on?** Evidence points to yes (the submission template wants continuous scores). I could not confirm it — the competition page returned an error. | Project owner | 2026-10-04 | | |
| 7 | **What is the success number?** My original proposal of ROC-AUC ≥ 0.87 was **wrong** — an untuned default model reaches 0.9573, so 0.87 would be a gate everything passes. The bar to beat is measured: `Class == Business` at 0.7770 accuracy / 0.7786 AUC. What a *good tuned* model achieves is not yet measured, and that is the number the target should come from. Set it after `docs/method_plan.md` Step 8. | Project owner | 2026-10-04 | | |
| 8 | What is the cost of a wrong prediction? | Project owner | 2026-10-04 | **CLOSED.** No direct cost. A submission can be replaced at any time, and nothing operational depends on it. Recorded so nobody later mistakes this for a system that harms anyone. | 2026-10-04 |
| 9 | What is the smallest useful version? | Project owner | 2026-10-04 | **CLOSED.** The smallest useful version is one model that beats the 0.7770 baseline and produces a valid `sample_submission.csv`. No serving, no API, no monitoring. | 2026-10-04 |
| 10 | The git branch is `master` with no commits. Should it be `main`? Cosmetic, but it is a one-line fix now and an annoyance later. | Project owner | 2026-10-04 | | |

---

## Which gate each question blocks

| Question | Blocks |
|---|---|
| 5, 7 | `.lead/00` Gate — "the person who will act on the output is named", and the measurable target |
| 6 | `.lead/02-C` Step 3 — the metric must come from the decision, and it must be settled before any model is compared |
| 8 | `.lead/04` Step 4.3 — costing the errors |
| 2 | `.lead/01` Gate — whether this dataset can do the job at all |
| 1 | `.lead/01` Step 1.1 gate — written permission to use the data |
| 3, 4 | Nothing structural. They change how we *describe* results, not whether we can build them. |
| 9, 10 | Nothing. Scope and tidiness. |

**Six questions are now closed.** The one that still blocks real work is
question 6, the metric. Everything else the analysis can reach has been reached.

---

## Not a question, but decided

**Outliers: found, judged real, and deliberately not removed.** See
[`outlier_findings.md`](outlier_findings.md). Zero rows fall outside a plausible
domain range, clipping changes validation ROC-AUC by +0.000076, and the extremes
are among the strongest signals in the data. No clipping, no deletion.

---

## Standing constraint — added 2026-10-04

> **The Kaggle access token must never be used, for any purpose.**

Stated by the project owner. It is a hard rule, not a preference. It means:

- **No Kaggle API calls.** Not to download, not to upload a submission, not to
  read competition metadata. The `kaggle` CLI and `kagglesdk` are installed in
  `.venv` and **must not be used**, even though they are available.
- **No reading or copying `~/.kaggle/kaggle.json`** or any credential file.
- **No token in code, config, notebooks, logs or documentation.**
- **The competition page is read in a browser only**, never through the API.

The data is already downloaded to `data/`, so nothing in this project needs the
API. The one thing this costs us: the evaluation metric (question 6) cannot be
confirmed by API and must be read off the competition page by hand.

`.dev/RULES.md` rule 8 already forbids secrets in code and git. This is the
project-specific instance of it, and it is stricter: the token is never *used*,
not merely never committed.

---

## Closed since the first version

- **`docs/split_plan.md` now exists.** It was a dangling link from
  [data_inventory.md](data_inventory.md). Written on 2026-10-04, because the EDA
  had to split the data before exploring it — `.lead/01-DATA.md` Step 1.7 puts
  the split before the exploration, not after. The metric did not turn out to
  matter for the split, because a stratified random split on `satisfaction` is
  correct whether the metric is accuracy or AUC.

---

## What changed

- **2026-10-04:** questions 1, 2, 5, 8 and 9 closed by the project owner —
  we are building a submission for `test.csv` against a competition-supplied
  dataset. Question 3 downgraded to non-blocking. Question 7 needs the new
  baseline number from the EDA. Standing constraint added: the Kaggle access
  token is never used.
- **2026-10-04, first version:** questions 1 to 4 raised by the `.lead/01`
  documents, questions 5 to 10 by `.lead/00`.

## How to close a question

Reply with the answer and the date. Edit the row, put the answer in the Answer
column and today's date in Closed. Do not delete the row — a question that was
asked and answered is worth more than one that was never asked, and the next
person usually has the same question.

If an answer changes something already written, add a dated **What has changed**
section at the bottom of that document. Do not edit the text above it.