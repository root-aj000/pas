# 06 — Monitor, Maintain, and Retire

**Before this file:** [`05-PACKAGE-SERVE-DEPLOY.md`](05-PACKAGE-SERVE-DEPLOY.md) is
done. The model is running and someone owns it.
**After this file:** the model is looked after for as long as it exists.
**Time it usually takes:** a few hours per month, ongoing.

---

## Why this file exists

**A model does not break loudly. It decays quietly.**

The code keeps running. The predictions keep arriving. Nobody gets an error.
Meanwhile the predictions get steadily worse, because the world changed and
nobody noticed. This is the most common way machine learning projects fail after
launch — not a crash, just slow decay that nobody sees.

If you do nothing else from this folder, do this file.

---

## Step 6.1 — Monitor five things, on a schedule

Set these up before launch, not after. Each needs a number, a threshold, and a
person who acts.

| What to watch | How to measure | Alert when |
|---|---|---|
| **1. Is it still running?** | Success/failure count, last successful run | Any failure, or no run when one was due |
| **2. Is the data still the same?** | Data drift (Evidently) | Drift beyond the agreed limit |
| **3. Is the data still good?** | Data quality checks (Great Expectations) | Any failed check |
| **4. Is the model still right?** | Model decay: score on recent actuals | Score drops below the go/no-go level |
| **5. Is it still worth it?** | The business metric it was built for | Metric stops improving |

Numbers 1 and 3 are easy and must be automatic from day one.
Numbers 2 and 4 need Evidently (`.dev/TOOLS.md` section 4).
Number 5 is a human check — but it must be on the calendar, not optional.

### A weekly routine

Pick a day. Write it in the calendar. It takes 20 minutes.

```text
EVERY WEEK, [day]:
  1. Read last week's runs. Any failures?
  2. Look at the drift report. Anything flagged?
  3. Check the output row count. Roughly the expected number?
  4. Look at the sample of predictions. Do they still look sensible to you?
  5. Note anything odd, even if you cannot explain it.

EVERY MONTH:
  6. Score recent predictions against what actually happened.
  7. Compare to the model card. Is it still inside the promised range?
  8. Check the business metric.

EVERY QUARTER:
  9. Full review: still needed? still worth it? still correct?
```

**An unexplained oddity is a reason to investigate, not a reason to move on.**
Most decay is visible days before the score collapses.

---

## Step 6.2 — Understand the three ways a model degrades

They have different causes and different responses. Know which one you are
looking at.

### 1. Data drift — the inputs changed

The customer's behaviour changed. Or our data collection changed. The model is
still receiving data, but not the data it was trained on.

```text
Symptom:  input distributions move. Average income shifts, a new device type
          appears, a marketing campaign changes traffic.
Cause:    the world changed, not the model.
Response: investigate what changed and whether the model still makes sense.
          Often needs retraining. Sometimes the world changed in a way that
          invalidates the whole approach — that is worth knowing.
```

### 2. Concept drift — the relationship changed

The inputs are the same, but what predicts the outcome has changed.

```text
Symptom:  inputs look normal, but accuracy falls.
Cause:    the world learned something the model does not know.
Response: this is the dangerous one, because monitoring the inputs shows
          nothing wrong. Only checking predictions against actual outcomes
          catches it.
```

**This is why number 4 above — scoring recent predictions against reality — is
not optional.** Drift reports will not catch it.

### 3. Data quality failure — the pipeline broke

```text
Symptom:  sudden change in predictions, or missing values appear.
Cause:    an upstream job failed, a column was renamed, a units bug was
          introduced.
Response: the data quality checks catch this. This is the cheapest of the
          three to fix, and the easiest to miss if you have no checks.
```

---

## Step 6.3 — Retrain on purpose

Decide the trigger **now**, while you are calm, not during an incident.

**Trigger on any of:**

- Scheduled: every 3 months, or every time the business cycle repeats.
- Performance: accuracy drops below the level you promised in the model card.
- Data drift: a monitored feature drifts beyond its limit.
- Business change: a new product, pricing change, or a new competitor.
- A bug in the pipeline that changed the data.

**The retraining procedure — the same every time:**

```text
1. Record why you are retraining.
2. Pull the latest data. Run the quality checks. Stop if any fail.
3. Check the label definition and the leakage question still hold.
   Definitions change. This must be re-confirmed, not assumed.
4. Train with the existing settings as the starting point.
5. Evaluate on a NEW, recent test set. Never reuse the old one.
6. Compare against the current live model, on the same test data.
7. Decide: better, same, or worse.
8. If better by a margin that matters -> promote, through shadow and canary
   again, like the first time. Never straight to full.
9. Log everything in MLflow, with a reason.
10. Update the model card: new date, new scores, new version.
```

**Step 8 matters.** People assume a retrained model is fine because it scored
well. A model evaluated on recent data and then switched to live, with no shadow
period, is how a small regression becomes an outage.

**Keep the old model.** You cannot roll back to something you deleted.

---

## Step 6.4 — Know when to stop using it

Models should be retired. It is not a failure; it is maintenance.

**Retire the model when:**

- The business need disappears, or changes so much the model no longer applies.
- The cost of maintaining it exceeds the benefit.
- The accuracy can no longer be beaten by the simple rule.
- The data it depends on stops being collected, or becomes illegal to use.
- A better approach replaces it.

**The process — deliberately boring:**

```text
1. Tell everyone who uses the output, with a date.
2. Keep the fallback available until the date passes.
3. Archive the model, its config, its card and its logs. Do not delete.
4. Record what was learned, so the next attempt starts further ahead.
5. Remove the code and the scheduled jobs, in the same commit.
```

Retiring a model properly takes an afternoon. Leaving a dead model running takes
months of everyone's attention, and keeps serving bad predictions.

---

## Step 6.5 — Feed what you learn back into the work

Every deployed model teaches you something the notebook never could. Capture it,
or lose it.

**Add these to the project notes each quarter:**

- Which features actually turned out to matter.
- Which segments it fails on, and whether that has changed.
- What the real base rate turned out to be.
- What the business actually did with the predictions.
- What you would do differently.

**And feed back the outcomes.** The most valuable data you will ever have is the
real result of the model's predictions, joined back against what it said. That
closed loop is what makes the next version better, and it only exists if you
build the join deliberately.

---

## Gate — the model is properly maintained once all of these are true

- [ ] All five things monitored, each with a number, a threshold and a named person
- [ ] A weekly and a monthly routine exist, in the calendar
- [ ] Alerts reach a person who will act on them
- [ ] Drift reports are generated and read, not just generated
- [ ] Retraining trigger decided and written down in advance
- [ ] Retraining procedure written, and includes a new test set each time
- [ ] Old models kept, so a rollback is possible
- [ ] Retirement conditions written, and a plan for telling users
- [ ] Archived models keep their card, config and logs
- [ ] Outcomes are joined back to predictions, to improve the next version

---

## Where to go from here

You have reached the end of the pipeline. Go back to
[`README.md`](README.md) for the full picture, or use
[`07-NEW-DEV-ONBOARDING.md`](07-NEW-DEV-ONBOARDING.md) if you are new and
overwhelmed. The next cycle starts again at
[`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md), because a new
quarter or a new problem is a new problem.

## The whole project in one line

> Understand the problem. Build a baseline. Get the data right and define
> everything. Split it honestly. Find signal without cheating. Train
> reproducibly. Evaluate honestly, including who it fails. Decide in writing.
> Deploy gradually, with a fallback. Watch it forever.

## Common mistakes in this phase

- **No monitoring at all**, discovering the model was 20% worse six months later.
- **Alerts going nowhere**, so a daily failure is discovered a month later.
- **Only watching inputs**, and missing concept drift, which only shows up in the
  outcomes.
- **Retraining with no shadow period**, and shipping a regression straight to
  everyone.
- **Reusing the same test set**, so the score keeps looking fine while the model
  quietly worsens.
- **Never retiring anything**, and paying maintenance for a dead model forever.