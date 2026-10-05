# 05 — Package, Work Out How It Will Be Served, and Deploy Gradually

**Before this file:** [`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md)
says GO.
**After this file:** the model is running in production, safely, with a fallback
and an owner.
**Time it usually takes:** 1–2 weeks for a first deployment.

---

## Why this file exists

The model works. The test score is good. Then the actual work starts, and this is
where projects that skipped the process fall over:

- The model cannot be reproduced, so a bug cannot be fixed.
- It is deployed with no idea what happens when it fails at 2am.
- It replaces the old process instantly, so any mistake hits everyone at once.
- Nobody owns it, so when it degrades nobody notices for two months.

None of this is about machine learning. It is about change management, and it is
the part that decides whether real people actually benefit.

---

## Step 5.1 — Package it so it can be repeated

Before it goes anywhere, prove it can be rebuilt from what you saved.

**Everything needed lives in one folder** (see
[`03-TRAIN-AND-TUNE.md`](03-TRAIN-AND-TUNE.md) for the contents):

```text
models/churn_v3/
├── model.pkl
├── features.json      # exact feature list, in order
├── config.json        # every setting, including seed and threshold
├── model_card.md
└── train.log
```

**Write the model card.** This is the document that lets someone who was not
there understand what they are holding. Template in
[`09-TEMPLATES.md`](09-TEMPLATES.md). It must cover:

- What it predicts, in plain language.
- Who it is for, and who it is **not** for (the known weak groups).
- What data it needs, and how fresh that data must be.
- How well it performed, on which data, measured once.
- The threshold, and why it was chosen.
- Training date, data version, git commit, model version.
- Known weaknesses and what to do about them.

**The reproducibility test.** Give the folder to someone who has never seen the
project, on a clean machine, and ask them to produce the same predictions. If
they have to ask a question, something is missing. Fix it now.

---

## Step 5.2 — Decide how it will be served

This is a decision, and it belongs to the team, not to one developer.

### Batch or online?

| | **Batch prediction** | **Online (live) prediction** |
|---|---|---|
| What it does | Scores the whole population on a schedule | Scores one request at a time |
| Use when | The decision is made once a day or a week | The answer is needed in the moment |
| Complexity | Low — a scheduled script | Higher — an API, hosting, monitoring |
| Failure impact | Delayed by one cycle | Immediate, per request |

**Choose batch if you can.** A scheduled script that produces a daily list of
customers to call is dramatically simpler to build, deploy and maintain than a
live API, and it usually matches the real decision anyway.

**Do not build a live API because it sounds more impressive.** A live endpoint
that must be available 99.9% of the time, for a decision that happens once a
month, is effort with no benefit.

### If it must be live

Use the serving design already chosen in `.dev/TOOLS.md` section 2b: MLflow
serves the model, Feast supplies the features at request time, so the served
features are the same definitions used in training.

**Then answer these, in writing:**

| Question | Why it matters |
|---|---|
| How many predictions per second, at peak? | Decides the hardware |
| How long may a prediction take? | Decides the timeout |
| What happens if the model is unavailable? | See Step 5.3 |
| What happens if a feature is missing at request time? | Must not be a 500 error |
| Who is on call when it breaks? | Named person, not a rota that does not exist |

---

## Step 5.3 — Decide what happens when it fails

This must be written down **before** launch. Every production model will fail
eventually.

**The fallback rule:** when the model cannot answer, the system must do something
sensible and predictable — never crash, and never silently produce nothing.

Options, in order of preference:

1. **Serve the previous model version.** The safest. If v3 breaks, v2 takes over.
2. **Fall back to the simple rule.** Often good enough — the baseline from
   [`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md).
3. **Serve the baseline (do nothing).** Predicting nobody churns. Predictable and
   safe.

```text
FALLBACK ORDER:
  1. Current model, if it responds within 500ms and all features are present
  2. Previous model version, if available
  3. The simple rule ("no login for 14 days")
  4. Nobody. Log an alert and carry on with today's list.
```

**And decide who sees the alerts.** An alert nobody reads is not monitoring. One
named person, one way to receive it.

---

## Step 5.4 — Deploy gradually

**Never switch everything over at once.** Three stages, in order.

### Stage 1 — Shadow (run alongside, change nothing)

The new model scores every request, but its output goes to a log file. The old
process still does the work.

Run for one to two weeks. Compare:

- Does the new model agree with what actually happened?
- Are the inputs arriving the way the training data looked?
- Are there errors, timeouts or missing features?

**This stage finds most problems, with zero risk.** Do not skip it to save time.

### Stage 2 — Canary (a small group gets the new answer)

A small slice of real cases — 5%, then 25% — use the model. Everything else keeps
the old process.

Watch, for at least a week at each stage:

- Does the business metric move in the right direction?
- Do the people receiving the output complain, or trust it?
- Do errors or unusual patterns appear?

**Watch the complaints.** A model can be statistically fine and operationally
useless if the people using it do not believe it. That feedback is data, and it
is the leading cause of projects being abandoned.

### Stage 3 — Full

Only when the earlier stages were clean, and the decision owner has said yes in
writing.

**At every stage, keep the old way of working available.** Removing it early means
you have no way back.

---

## Step 5.5 — Hand it over properly

The work is not done when the model is live. It is done when someone else can
maintain it.

**Before you call it finished, hand over these five things in writing:**

1. **What it does, and what it does not do.** Including the known weak groups.
2. **How to run it** — the exact commands, from a clean machine.
3. **How to rebuild it** — retrain from scratch and reproduce the model.
4. **What to do when it breaks** — the fallback order and who to call.
5. **Who owns it** — a named person, and who covers them.

Then watch that person do the retraining once. If they cannot, the handover is
not finished — the knowledge is still only in your head.

---

## Step 5.6 — Write the runbook

A runbook is a short document written for the person having a bad day. Keep it to
one page.

```text
RUNBOOK: churn_v3

WHAT IT DOES:        Lists customers likely to cancel in the next 30 days.
HOW TO RUN IT:      python run_pipeline.py
SHOULD PRODUCE:      reports/churn_v3_predictions.csv, about 2,000 rows

IF IT FAILS:
  - "feature missing"     -> check the upstream load finished; re-run
  - "model not found"     -> wrong path, or venv not active
  - "file is empty"       -> upstream data was empty; DO NOT use today's output
  - nothing at all        -> use the fallback (step 5.3) and alert [name]

HOW TO TELL IT IS WRONG:   row count far from ~2,000, or churn rate far from 0.25%

WHO TO CALL:        [name], [contact]
LAST REVIEWED:      [date]
```

Test it by following it yourself, from a clean machine, without asking anyone a
question. If you get stuck, the runbook is not finished.

---

## Gate — the model is not "live" until all of these are true

- [ ] Model, feature list, config, model card and log in one folder
- [ ] Model card written, including who the model is **not** for
- [ ] Someone else reproduced the predictions from the saved files alone
- [ ] Batch vs online decided and recorded, with the reason
- [ ] If live: latency, throughput, timeout and cost requirements written down
- [ ] Fallback order written, and it does not crash
- [ ] Alerts go to a named person
- [ ] Shadow stage run for at least a week
- [ ] Canary stage run at two sizes, with the business watching the metric
- [ ] Old process still available at every stage
- [ ] Five handover items done, and the owner has retrained it once themselves
- [ ] Runbook written, and followed successfully by someone else

**If any box is unticked: the model is not production-ready**, however good the
test score was.

---

## What to do next

Go to [`06-MONITOR-AND-MAINTAIN.md`](06-MONITOR-AND-MAINTAIN.md): watch it, catch
drift, retrain on purpose, and eventually retire it.

## Common mistakes in this phase

- **Building a live API when a daily batch job would do.**
- **No fallback**, so one bad night stops all work.
- **Switching over fully on day one**, turning a small bug into a large one.
- **Skipping the shadow stage** to save two weeks.
- **Treating it as live once the code is deployed**, and not handing it over.
- **No runbook**, so every incident depends on one person being awake.