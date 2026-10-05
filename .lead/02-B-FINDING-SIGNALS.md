# 02 — Finding Signals and Building Features

**Before this file:** [`01-DATA.md`](01-DATA.md) is done. Data is clean,
understood, and split properly.
**After this file:** you know which features actually carry signal, and you have
a first model measured against the baseline.
**Time it usually takes:** 2–5 days. Do not rush this.

---

## Why this file exists

The internet says "train a model". In real projects, the model is the last 10% of
the work. The 90% before it — finding what genuinely predicts the answer — is what
determines whether the project is worth anything.

Also: you will find some patterns that look excellent and are **completely
useless**. This file is largely about telling those apart from real signal.

---

## Step 2.1 — Set up before you look

Three small things, so the exploration is reproducible.

```text
1. Use the TRAIN set only.     Never explore the validation or test set.
2. Set the random seed and log it.
3. Save every chart and table you generate into reports/ with a clear name.
   reports/eda_01_column_types.md
   reports/eda_02_churn_rate_over_time.png
```

Why rule 1 matters: once you have looked at the test data, you cannot un-look.
Every decision you make afterwards is contaminated, and the final score will be
optimistic without anyone being able to explain why.

The detailed method for the exploration itself is in `.dev/EDA.md`. Follow the
phases there. What follows is the order to do it in, and the traps.

---

## Step 2.2 — Understand one column at a time

For every candidate feature, in this order:

1. **What is it, and what would I expect it to do?** Write the expectation down
   *before* looking at the relationship. If the data surprises you, that is
   information.
2. **How is it distributed?** Skewed? Multiple peaks? Lots of missing?
3. **Are the extreme values real?** A customer spending 10 million is either
   fraud or a bug. Find out which before keeping it.

Keep a running notes file. One line per column: what it is, what it looked like,
whether you are keeping it. You will not remember by the time you build features.

---

## Step 2.3 — Compare everything against the label

This is the single most productive hour in the whole project.

**Split the data into the two groups — the ones that churned and the ones that
did not — and compare every feature across them.**

```text
| Feature          | Churned (n=300)  | Stayed (n=119,700) | Difference |
|------------------|------------------|--------------------|------------|
| Tenure days      | 41               | 380                | Huge       |
| Support tickets  | 4.2              | 0.4                | 10x        |
| Last login days  | 38               | 3                  | Huge       |
| Plan             | 90% monthly      | 55% monthly        | Noticeable |
| Age              | 34               | 35                 | None       |
```

Any column with a big difference is a candidate. Read the table from top to
bottom and you have your feature shortlist, in about ten minutes.

**Then check each one for leakage** — see Step 1.5. For every shortlisted
feature, ask:

> "At the moment of the prediction, did I know this value?"

The `support_tickets` column above is the kind that is often wrong: if tickets
are recorded when a customer asks to cancel, the leak is enormous and the model
will look brilliant and fail completely.

---

## Step 2.4 — Watch out for the misleading patterns

Some strong-looking patterns are worthless. Check each against this list.

| Trap | What it looks like | How to detect it |
|---|---|---|
| **Leakage** | Near-perfect accuracy | Ask: did I know this at prediction time? |
| **A proxy for the wrong thing** | Predicts well, for the wrong reason | What does it mean in the real world? |
| **An artifact of the split** | Great on test, useless in production | Does the pattern hold in every time period? |
| **Only true for one group** | Works overall, fails badly for one segment | Evaluate per segment — see [`04`](04-EVALUATION-AND-GO-NO-GO.md) |
| **Too good to be true** | Better than the current practice by a mile | You have almost certainly leaked |
| **Popularity bias** | Works, but only for big customers | Check it works for small ones too |

**On the last one:** if the base rate is 0.25%, always predicting "no" gives
99.75% accuracy. If your model reports 99.8% accuracy, you have not built a
model. You have built a number.

---

## Step 2.5 — Build features

A **feature** is an input to the model. Raw columns rarely make good features.

Two sources of good features:

**1. Domain knowledge — the strongest source, and the most neglected.**

Combine what you already know about the business:

```text
Birth date                          -> age
First purchase and last purchase    -> customer_lifetime_days
Total spend and number of orders    -> average_order_value
Support tickets and tenure          -> tickets_per_month
```

Ask the person who owns the process: *"What would a good salesperson look at?"*
Then turn their answer into columns. This conversation is the highest-value
ten minutes in the project.

**2. Combinations of raw columns.**

Combine columns that mean something together. Just make sure each new feature
satisfies the leakage question from Step 1.5.

**Rules for features:**

- One clear name that says what it means. `tickets_per_month`, not `f_7`.
- Documented in a docstring: what it is, how it is calculated, units.
- Registered in Feast as a definition — see `.dev/TOOLS.md` section 2 and
  `.dev/PROJECT-STRUCTURE.md`. A feature not defined in Feast causes
  training-serving skew.
- Known at prediction time.
- If a feature stops being useful, delete it. Do not keep it "just in case" —
  every unused feature adds noise and a chance to break something later.

**A note on number of features:** more is not better. Fifty features on a small
dataset will overfit. Start with the ten that survived Step 2.3, and add more
only if the validation score improves.

---

## Step 2.6 — Build the first real model

Start simple, on purpose. You are not trying to win.

```python
# Step 2.6: The first model is deliberately the simplest sensible one.
# Its job is to give us an honest number to compare everything against.
# A complicated model here would tell us nothing, because we would not
# know whether the improvement came from the model or from the data.
```

**Recommended order — do them in this order and stop when validation stops
improving:**

1. **A logistic regression or linear model.** Always run this. It is fast, and it
   shows you which features really carry weight.
2. **A decision tree.** Small, readable. You can print the whole thing.
3. **A random forest.** Handles messy data well without much tuning.
4. **Gradient boosting (XGBoost, LightGBM).** Usually the strongest on tabular
   problems.
5. **A neural network.** Only if the problem genuinely needs one — images, text,
   or very large datasets. Do not add one for fashion on tabular data.

**Why the order matters:**

- Linear first gives you an honest read on the data, and features it cannot use
  are genuinely useless.
- A small decision tree you can print and show to a non-technical person is
  worth more for trust than a model three points better.
- More complex models can only be justified once you can point at the simpler
  model and say why it is not enough.

**Log every run:** settings used, score achieved, how long it took. See
`.dev/CODE-STANDARDS.md`.

---

## Step 2.7 — Compare against the baseline

Now put the numbers side by side. This is the first real checkpoint.

```text
| Model              | Catch rate | Calls per month | Notes                    |
|--------------------|------------|-----------------|--------------------------|
| Do nothing         | 0%         | 0               | The floor                |
| Simple rule        | 60%        | 4,000           | Baseline to beat         |
| Logistic regression| 55%        | 2,100           | Worse than the rule!     |
| Decision tree      | 64%        | 4,200           | Slightly better          |
| Random forest      | 63%        | 4,150           | No gain, more complexity |
```

Read this table honestly:

- **The logistic regression being worse than the simple rule** is worth
  investigating. It often means a feature is badly scaled, or the most useful
  signal is not linear and needs to be expressed as a category.
- **Random forest matching the decision tree** means the complexity is not
  earning its keep. Simplify.
- **Anything worse than the simple rule** means you do not have a model yet.

**Do not proceed past this step with a model that cannot beat the baseline.** Fix
it, or stop and record the finding.

---

## Gate — do not start tuning until all of these are true

- [ ] All exploration done on the train set only
- [ ] Every feature documented with meaning, units and calculation
- [ ] Every shortlisted feature checked for leakage — confirmed known at
      prediction time
- [ ] Suspicious patterns investigated, and documented either way
- [ ] At least three models tried, starting from the simplest
- [ ] All runs logged with settings and scores
- [ ] The best model beats the simple-rule baseline on the validation set
- [ ] You can explain, in plain language, why the model beats the baseline

**If any box is unticked: stop.** Do not tune. Tuning a model that cannot beat a
one-line rule wastes days.

---

## What to do next

Go to [`03-TRAIN-AND-TUNE.md`](03-TRAIN-AND-TUNE.md): set the run up properly,
tune the settings, and make it reproducible.

## Common mistakes in this phase

- **Exploring the test set.** Undoable, and it quietly inflates every number.
- **Trusting a feature without checking what it means.** Leakage lives here.
- **Starting with the most complex model.** You then cannot tell what helped.
- **Keeping every feature "just in case."** Noise, and more to break later.
- **Believing a great score without comparing to the baseline.**
- **Not writing down why a feature was kept.** The next person cannot tell a
  reasoned choice from an accident.