# 09 — Templates

**Who this is for:** everyone who has to write something down.

Copy the template, fill it in, save it in the project. Every one of these exists
because a real project needed it and did not have it.

**The rule for all of these:** write it the day you decide it, not later. A
decision written a month later has already been forgotten, and gets re-litigated.

---

## Where to save what

| Template | Save as |
|---|---|
| Problem statement | `docs/problem_statement.md` |
| Column dictionary | `docs/column_dictionary.md` |
| Experiment log | `docs/experiment_log.md` — one table, appended to |
| Model card | `models/<name>/model_card.md` |
| Go / no-go decision | `docs/decisions/<date>_go_no_go.md` |
| Retraining record | `docs/decisions/<date>_retraining.md` |
| Incident note | `docs/incidents/<date>_<what_broke>.md` |
| Runbook | `docs/runbook_<name>.md` |
| Open questions | `docs/open_questions.md` — kept, never deleted |
| **Dev log entry** | `dev-logs/<date>_<short_name>.md` — see [`../.dev/DEV-LOG.md`](../.dev/DEV-LOG.md) |
| **Run log** | `logs/<date>_<time>_<what>.log` — written by the code, see [`../.dev/DEBUGGING.md`](../.dev/DEBUGGING.md) section 8 |

Create the `docs/`, `docs/decisions/` and `dev-logs/` folders when
you start. An empty folder beats a missing one, because a missing folder means
nobody wrote it down.

---

## 1. Problem statement

One paragraph. No technical words. Fill it in with the person who asked for the
model, and get their agreement.

```text
PROBLEM STATEMENT
Date:        [date]
Owner:       [name, role - the person who acts on the output]
Agreed with: [name, role - confirms this is the right problem]

We want to predict [WHAT] for [WHO], so that [PERSON/TEAM] can [SPECIFIC ACTION].

Today they [WHAT THEY DO NOW], which costs [TIME / MONEY / RISK].

We will know this model is useful if [SPECIFIC MEASURABLE TARGET].

Out of scope:   [what this model will deliberately NOT do]
```

**"Out of scope" matters more than it looks.** Most scope arguments come from
things nobody ever agreed were out of scope.

---

## 2. Column dictionary

The single most useful document in a data project. One row per column.

```text
| Column | Meaning | Type | Units | Example | Known at prediction time? | Trust it? |
|--------|---------|------|-------|---------|--------------------------|-----------|
| tenure_days | Days since first payment | int | days | 412 | Yes | Yes - note: from first payment, not signup page |
| ltv | Lifetime value | float | USD | 1450.75 | Yes | Excludes refunds |
| n_tickets | Support tickets | int | count | 4 | Yes | Only counts tickets with status='resolved' |
```

Rules:

- Add a row the moment you add a column. Not later.
- Never leave the "Known at prediction time?" column blank. `N/A` is allowed;
  blank is not. That column is the leakage check.
- Write down what changed and when. If a definition changed, add a new row with
  the date.

---

## 3. Experiment log

One table. Append to it. Never edit old rows.

```text
| Run | Date | Model | Key settings | Seed | Recall | Precision | Net value | Notes |
|-----|------|-------|--------------|------|--------|-----------|-----------|-------|
| 1 | 2024-06-01 | simple rule | no login 14 days | - | 0.60 | 0.15 | - | BASELINE |
| 2 | 2024-06-02 | logreg | lr=0.01, 12 features | 42 | 0.55 | 0.12 | - | Worse than baseline |
| 3 | 2024-06-03 | rf | depth=8, n=300 | 42 | 0.63 | 0.18 | +28,950/mo | Best so far |
| 4 | 2024-06-04 | rf + 3 features | depth=8, n=300 | 42 | 0.66 | 0.19 | +34,100/mo | +tickets_per_month |
```

The **Notes** column is what makes this valuable in six months. Write what you
learned, not what you did — including "worse than baseline, investigate".

---

## 4. Model card

Lives with the model. This is what a new person reads to understand what they are
holding.

```text
MODEL CARD: churn_v3

WHAT IT PREDICTS
  Which customers will cancel within 30 days.
  Output is a score from 0 to 1. Above 0.62 = predicted to churn.

WHO IT IS FOR
  The retention team, for the weekly customer contact list.

WHO IT IS NOT FOR
  Customers with tenure under 3 months - the model does not detect them at all.
  Do not use for annual plan customers; they churn at half the rate and the
  threshold is not calibrated for them.

DATA IT NEEDS
  Customers, transactions, support_tickets - see features.json for the exact
  list and order.  Must be no older than 24 hours.

HOW IT WAS TRAINED
  Training data:  2023-01-01 to 2024-05-02 (label window of 30 days)
  Data version:   features_2024_05_02
  Seed:           42
  Commit:         3f9a1c2
  Trained:        2024-06-05
  Library version: scikit-learn 1.5.0

HOW WELL IT DOES
  Measured once on the frozen test set (2024-03 to 2024-05, 18,000 customers):
    Catches 63% of the 300 customers who churned
    2,000 contacts per month
    Estimated net value: +28,950 USD per month
    Beats baseline simple rule (60%, 4,000 contacts)
  Known weak spot: tenure under 3 months - recall 0%.

THRESHOLD
  0.62. Chosen with the retention team on 2024-06-06: at 0.50 we would make
  9,000 contacts a month; the team can make 2,000. REVIEW IN 3 MONTHS.

WHAT TO DO IF IT FAILS
  Fallback order: churn_v2 -> simple rule -> do nothing. See runbook.
  Alert: [name]. See docs/runbook_churn_v3.md.

MONITORING
  Weekly drift and quality report. Monthly accuracy against actual outcomes.
  Owner: [name]
```

---

## 5. Go / no-go decision

One page. Signed and dated. This is what makes stopping a real option.

```text
DECISION RECORD

Date:      [date]
Model:     churn_v3, commit 3f9a1c2
Test set:  18,000 rows, 2024-03-01 to 2024-05-02, seed 42 (frozen 2024-06-04)

DECISION:  GO / GO WITH CONDITIONS / NO-GO

RESULTS
  Catches        63% of the 300 customers who churned
  Contacts       2,000 per month
  Net value      +28,950 USD per month estimated
  vs baseline    simple rule 60% / 4,000 contacts
  vs do nothing  0% / 0 contacts

KNOWN WEAKNESSES
  - Does not detect customers with tenure under 3 months (recall 0%)
  - Performance depends on plan mix; untested for annual plans
  - Requires all 9 input tables to be fresh within 24 hours

CONDITIONS
  [e.g. review threshold after 3 months of production data]
  [e.g. do not use for annual plan customers]

DECIDED BY:  [name], [role]
AGREED BY:   [name], [role - the decision owner]
```

---

## 6. Retraining record

```text
RETRAINING RECORD

Date:        [date]
Previous:    churn_v3  Trigger: [scheduled / accuracy drop / drift / business change]
New:         churn_v4

WHY:         Accuracy fell from 63% to 54% in March. Plan mix changed.

DATA USED:   2023-07-01 to 2024-08-02 (label window 30 days)
DATA VERSION: features_2024_08_02
RE-CHECKED:  Label definition - unchanged. Leakage review - passed.
SEED:        42        COMMIT: 7a2e9b1

RESULT ON NEW TEST SET (2024-06 to 2024-08, 20,100 rows, never used before):
  churn_v3 (current)   recall 0.51
  churn_v4 (new)       recall 0.66

DECISION:   Promoted churn_v4.
ROLLOUT:    Shadow from [date] for 1 week. Canary 5% from [date].
            Full only after shadow review.
ROLLBACK:   Keep churn_v3 until [date + 1 month]. Rollback = point serving back.
```

---

## 7. Incident note

Write it the day it happens, while you remember. Ten minutes now beats an hour of
reconstruction later.

```text
INCIDENT: 2024-06-14 - predictions file empty

WHAT HAPPENED
  The weekly prediction file was empty for 2,000 customers instead of 2,000
  rows. Retention team used yesterday's list.

WHEN DETECTED   2024-06-14 09:15, by [name], from the row count check
HOW SEVERE      Medium. One week of contacts missed.
WHO AFFECTED    Retention team; 2,000 customers not contacted

WHY
  The upstream customers load failed at 03:00 (disk full on the database
  server). The pipeline logged a failure but continued, because the load step
  caught the error instead of stopping.

FIXED
  Made the load step stop the pipeline on failure. Re-ran for 2024-06-14.
  Output verified against yesterday: 2,011 rows.

PREVENTED
  - Load step now raises instead of catching (src/components/data_ingestion.py)
  - Added a row count check that stops the run - this alert already existed
    but went to an address nobody reads. Changed the recipient to [name].

FOLLOW-UP DATE   2024-06-21 - confirm the fix held
```

---

## 8. Runbook

One page, for the person having a bad day. See the template in
`.lead/05-PACKAGE-SERVE-DEPLOY.md` Step 5.6.

**Test it by following it yourself, from a clean machine, without asking anyone
anything.**

---

## 9. Open questions

One list. Everyone can add. Nobody deletes — mark them closed with the answer.

```text
| # | Question | Asked of | Asked on | Answer | Closed |
|---|----------|----------|----------|--------|--------|
| 1 | Does `tenure` start at signup page or first payment? | Data team | 2024-06-01 | First payment | 2024-06-02 |
| 2 | Are annual plan refunds included in `ltv`? | Data team | 2024-06-01 | No | 2024-06-03 |
| 3 | Max contacts the team can make per week? | Retention | 2024-06-04 | 500 | 2024-06-04 |
| 4 | ... | | | | |
```

**This file is the answer to "what next".** Anything unclear goes here with a
name against it, and the project stops moving on that point until it is answered.

If this list is empty for weeks, the project is either genuinely clear or people
have stopped asking. Ask someone.

---

## A note on why there are so many documents

Each of these templates exists because a specific thing went missing on a specific
project and cost real time. None of them is for the sake of process.

The test for whether a document is worth keeping:

> **Has it ever answered a question, prevented a mistake, or stopped an
> argument?**

If yes, keep it. If no, delete it. A folder of documents nobody reads is worse
than no folder — it creates the impression that everything is written down.

Review this list every quarter and delete what stopped being useful.