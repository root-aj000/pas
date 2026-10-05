# 04 — Evaluate Properly, and Decide Whether to Ship

**Before this file:** [`03-TRAIN-AND-TUNE.md`](03-TRAIN-AND-TUNE.md) is done. The
model is frozen and saved.
**After this file:** you have an honest evaluation and a written ship-or-stop
decision.
**Time it usually takes:** half a day of analysis, and a conversation.

---

## Why this file exists

This is the phase that most often gets done badly, and the phase where projects
either earn trust or lose it.

A model can have an excellent average score and be **useless**, because:

- It only works for the majority group.
- Its errors are concentrated exactly where the cost is highest.
- It beats an accuracy number but loses the actual business.
- Everyone agreed the threshold on a Tuesday afternoon without writing it down.

An average number is not an evaluation.

---

## Step 4.1 — Run the test score, once

Now you may open the test set. Once.

```text
Before running, confirm you have written down:
  - model version and settings     (done in file 03)
  - which rows, which date, seed   (done in file 03)
  - what score counts as a pass    (done in file 03)
```

Log it to MLflow, exactly as every other run.

If the test score is much worse than the validation score, that is **information,
not a disaster**. It usually means:

| Gap | Usual cause |
|---|---|
| Test much worse | The test period is unusual, or you have leakage that validation did not expose |
| Test much better | Suspicious. The split leaked. Stop and check it |
| Test about the same | Healthy. This is what you want |

Do not retune after seeing the test score. If the gap is large, understand why,
fix the cause, and treat the next test set as the real one. Do not peek twice.

---

## Step 4.2 — Confusion matrix first, before any average number

**Accuracy is almost never the right measure.** With a 0.25% base rate, "no" is
99.75% correct by doing nothing.

Start with the confusion matrix. With the churn example:

```text
                        Actually churned    Actually stayed
Predicted "will churn"        190                  1,810
Predicted "will stay"         110                119,700
```

Read it in plain words:

- **Correctly caught:** 190 of the 300 people who churned (63%)
- **Missed:** 110. These people get no warning.
- **False alarms:** 1,810. Real people, wasting time.

Now the project becomes a conversation about people, not percentages:

> "We would catch 190 of the 300. The retention team would make 2,000 calls a
> month, of which 1,810 are unnecessary. Is that the right trade?"

That question can be answered by a manager. "What is the accuracy?" cannot.

---

## Step 4.3 — Cost the errors

Assign a number to each kind of mistake. Not for precision — for honesty.

| | Actually churned | Actually stayed |
|---|---|---|
| **Predicted churn** | Benefit: we saved them | Cost: wasted call |
| **Predicted stay** | Cost: we lost them, unnoticed | Benefit: no waste |

Worked example:

```text
Missing a churner      costs 200 USD (lost customer)
Wasted call            costs 5 USD (30 minutes of staff time)

Catching 190 leavers  = 190 x 200  = 38,000 USD saved
Wasting 1,810 calls   = 1,810 x 5  =  9,050 USD cost

Net: +28,950 USD per month
```

Now you have something far better than an accuracy figure: a number the business
can compare against the cost of running the project. And it is checkable by
someone with no technical background at all.

**If the net is negative or trivial, stop the project here.** That is a
successful outcome — you have avoided building something expensive and
useless. Record it and move on.

---

## Step 4.4 — Choose the threshold with a person, in writing

A model outputs a score, not a decision. **Where you draw the line is a business
decision, not a technical one.**

- Raise the threshold → fewer false alarms, more missed churners.
- Lower it → the reverse.

Do this together with the person who owns the decision:

1. Pick the cost of each mistake (Step 4.3).
2. Let the model tell you the trade-off at each threshold.
3. Choose one, based on how many calls the team can actually make.
4. **Write down the chosen number and the reason.**

```text
THRESHOLD: 0.62
REASON:    At 0.50 we would make 9,000 calls a month. The team can make
           2,000. At 0.62 we make 2,000 and still catch 63% of leavers.
REVIEWED:  [name], [date]
```

This number goes in `config.yaml` as `decision_threshold`, and it is reviewed on a
schedule — because the cost of a call and the value of a customer both change.

**Never leave the default threshold in place and hope.** A default of 0.50 is
meaningless when 99.75% of cases are negative.

---

## Step 4.5 — Check who it works for

An average can hide a disaster. Always break the results down by segment:

| Segment | Are we calling them? | Do they churn more? | Verdict |
|---|---|---|---|
| Tenure < 3 months | 0% | They churn 4x more | **Model never catches them** |
| Tenure 1 year+ | 45% | Slightly more | Works well |
| Monthly plan | 38% | Yes | Works |
| Annual plan | 0% | They churn half as often | Fine |

Overall the score looks acceptable. In reality the model is **completely blind to
every new customer** — exactly the group with the most value.

**Always check at minimum:** by time period, by customer size, by plan type, by
region or language, and by any group that is known to behave differently.

```python
# Always do this before believing an average.
for segment_name, group in data.groupby("tenure_band"):
    log_step("evaluate", rows_out=len(group), segment=segment_name, recall=score(group))
```

If a group performs badly, you have three honest choices:

1. **Fix it** — add features that help that group.
2. **Exclude it** — do not apply the model to that group, and say so.
3. **Accept it, in writing** — with the business impact stated.

What you must not do is ship a model that fails a group without noticing.

---

## Step 4.6 — Sanity-check the actual errors

Numbers tell you that something is wrong. Reading individual cases tells you
why, and the "why" is always where the next feature comes from.

Look at:

- 20 cases the model got wrong with high confidence — why was it sure?
- 20 cases it got wrong with low confidence — those are the honest unknowns.
- Cases where it was right, to make sure it is not right for a silly reason.

```text
| customer | actual | predicted | notes |
|----------|--------|-----------|-------|
| 10023    | churn  | 0.05      | Business closed. Nothing in the data said so. |
| 10087    | churn  | 0.51      | Two tickets in one day. We only count monthly. |
| 10204    | stay   | 0.94      | Complained loudly, then stayed 3 years. |
```

The second row is gold. It is not bad luck — it is a specific, fixable thing:
ticket count is measured per month, but a burst of tickets predicts churn
strongly. That is your next feature.

**Do this before and after every significant model change.** It takes an hour and
it finds more improvements than any amount of parameter tuning.

---

## Step 4.7 — Write the go / no-go decision

Now the decision. In writing, with numbers, signed and dated.

```text
DECISION:  GO / NO-GO / GO WITH CONDITIONS
DATE:      [date]
MODEL:     churn_v3, commit 3f9a1c2

RESULTS ON THE FROZEN TEST SET (one honest run):
  Catches 63% of the 300 customers who churned
  2,000 calls per month
  Net saving estimated at 28,950 USD per month
  Beats the simple rule (60%, 4,000 calls)

KNOWN WEAKNESSES:
  - Does not catch customers with tenure under 3 months
  - Performance drops if plan mix changes

CONDITIONS: [e.g. review threshold after 3 months of data]

DECIDED BY: [name], [role]
```

"GO WITH CONDITIONS" is a real and often correct answer — for example, use the
model for customers with tenure over 3 months, and review after three months.

**A NO-GO is a good outcome.** Recording "we built this, and it is not worth it,
here is why" is valuable work. It protects the team from building it again next
quarter.

---

## Gate — nothing gets deployed until all of these are true

- [ ] Test score measured once, from the frozen split
- [ ] Test result explained — a large gap from validation is understood
- [ ] Confusion matrix reviewed in plain language
- [ ] Every error type given a cost, and the net benefit calculated
- [ ] Threshold chosen with the decision owner, and written down with a reason
- [ ] Results broken down by segment, with failures named
- [ ] Actual errors read by a human, and findings recorded
- [ ] Every known weakness documented, not just the strengths
- [ ] Go / no-go decision written, dated and signed
- [ ] Decision recorded in MLflow against the exact model version

**If any box is unticked: stop.** Do not deploy. It is always cheaper to delay
than to deploy something and explain it later.

---

## What to do next

If the decision is **GO**: go to
[`05-PACKAGE-SERVE-DEPLOY.md`](05-PACKAGE-SERVE-DEPLOY.md).

If the decision is **NO-GO**: stop the project, write down what was learned, and
tell the people who asked. That is a complete, honest outcome.

## Common mistakes in this phase

- **Reporting accuracy** on an imbalanced problem, and celebrating 99.8%.
- **Leaving the threshold at its default**, and never writing down why it is
  what it is.
- **Only ever looking at the average**, and shipping a model blind to your
  newest and most valuable customers.
- **Never reading individual errors**, and so never finding the next feature.
- **Deciding what counts as success after seeing the result.**
- **Treating NO-GO as failure.** It is the cheapest possible outcome.