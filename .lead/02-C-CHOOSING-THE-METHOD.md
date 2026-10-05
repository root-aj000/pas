# 02-C — Choosing the Method

**Before this file:** [`02-B-FINDING-SIGNALS.md`](02-B-FINDING-SIGNALS.md) is
done. You know which features carry signal, and you know the problem type.
**After this file:** you have a shortlist of methods, a working version of each,
and a scored comparison table.
**Where to look up the exact function:** [`02-D-METHODS-CATALOGUE.md`](02-D-METHODS-CATALOGUE.md).

---

## Why this file exists

"Train a model" is where most tutorials stop, and where a new developer gets
stuck. The real work is a series of decisions:

> What kind of problem is this? What are the constraints? What can I even try?
> How do I look up a method I have never used? How do I compare five of them
> fairly? How do I know when to stop?

None of that is in the tutorials. All of it is here, in order.

---

## The 12 steps

Work through these in order. Do not skip to step 8.

| # | Step | Output |
|---|---|---|
| 1 | Classify the problem | One sentence naming the task type |
| 2 | Write the constraints | Latency, interpretability, data size |
| 3 | Confirm the metric | The number that decides success |
| 4 | Rebuild the trivial baseline | The number to beat |
| 5 | List 3–5 candidate method *families* | A shortlist, with a reason each |
| 6 | Look up each one | A minimal working snippet per candidate |
| 7 | Wire them all the same way | One interface, so swapping is one line |
| 8 | Score them on validation | One table, same metric, same data |
| 9 | Read the errors | What the winner is still getting wrong |
| 10 | Pick the simplest that meets the bar | A decision, with the number |
| 11 | Only now, tune the winner | Better settings for one model |
| 12 | Document the choice | Why, in writing, for the next person |

**Steps 1–5 take an hour and save weeks.** Steps 11 is where people start, and
it is the one that should come last.

---

## Step 1 — Classify the problem

Answer these seven questions. Write the answers down.

| # | Question | If the answer is... |
|---|---|---|
| 1 | **What kind of task?** binary classification / multiclass / regression / ranking / forecasting / clustering / anomaly detection / recommendation | You now know which catalogue section to read |
| 2 | **What kind of data?** tabular / time series / text / images / audio / mixed | Text and images are a completely different world |
| 3 | **What is one prediction about?** one row / one customer / one session / one document | It decides how you split the data |
| 4 | **How many rows, roughly?** <1k / 1k–100k / >100k | This alone decides linear vs trees vs deep learning |
| 5 | **How fast must the answer be?** batch (hours) / seconds / milliseconds | Rules out whole method families |
| 6 | **Must a human be able to explain it?** yes, for a regulator or a customer / no | Rules out black boxes |
| 7 | **What is the target?** a number / a category / a probability / a ranking | A probability needs different handling than a hard label |

### The data size rule that surprises people

```text
Under ~1,000 rows    → linear models and logistic regression. A deep network
                       will memorise the noise.
1,000 to ~100,000    → gradient boosting (XGBoost / LightGBM / CatBoost). This
                       is the sweet spot for tabular data.
Over ~100,000        → still gradient boosting first. Deep learning only pays
                       off with very large datasets, or with images/text/audio.
```

**If your data is tabular, gradient boosting is the answer more often than not.**
Neural networks are the default answer for everything in tutorials, and they are
routinely beaten by gradient boosting on spreadsheet-shaped problems.

---

## Step 2 — Write the constraints down

Constraints eliminate methods faster than anything else. Fill this in:

```text
Latency budget:      __________ (batch / 100ms / 10ms)
Interpretability:    __________ (must explain / internal only)
Data privacy:        __________ (features must be explainable to a customer?)
Retraining cost:     __________ (hourly / weekly / monthly)
Explainable to:      __________ (engineer / manager / regulator / customer)
```

**What each constraint eliminates:**

| Constraint | Eliminates |
|---|---|
| Answer needed in under 10ms | Large ensembles, deep learning, anything calling a database per row |
| Must explain every decision | Neural networks, boosted trees (mostly), k-NN. Leaves linear models, single trees, rules |
| Retraining must be cheap | Deep learning, huge ensembles |
| Under 1,000 rows | Almost everything except linear models and single trees |
| Features include free text or images | Everything in the classic tabular catalogue — a different toolset |

Write the constraints down **before** choosing. Otherwise you will choose the
favourite method and then discover the constraint, and rationalise.

---

## Step 3 — Confirm the metric

A method cannot be compared without a number. Before comparing anything, agree
the number.

| Task | Start with | Then use |
|---|---|---|
| Binary classification, imbalanced | `recall` + `false positives at a given recall` | `average_precision_score` (better than ROC-AUC when rare) |
| Binary classification, balanced | `roc_auc_score` | `f1_score` |
| Regression | `mean_absolute_error` | RMSE, MAPE |
| Ranking | recall at k, or NDCG | — |
| Forecasting | MAE / MASE | Error by horizon |
| Clustering | silhouette score (with care) | Business-meaningful description of the clusters |

**The rule:** the metric must come from the business decision in
[`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md). Not from what is easy
to compute.

Full list of metric functions:
[`02-D-METHODS-CATALOGUE.md`](02-D-METHODS-CATALOGUE.md) section 16.

---

## Step 4 — Confirm the trivial baseline

Already done in [`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md) Step
0.4. Confirm it is still there, and keep it in the comparison table. Every later
number is measured against it.

---

## Step 5 — List 3–5 candidate method families

Not 15 candidates. Three to five, chosen for different *reasons*, not because
they scored well on a leaderboard.

A good shortlist looks like this:

```text
1. Logistic regression    - baseline that always works, shows which features
                            really carry linear signal
2. Random forest          - handles messy data, few settings to tune
3. Gradient boosting      - usually the strongest on tabular data
4. (maybe) k-NN or SVM   - only if there is a reason: e.g. k-NN because the
                            data is very small
```

**Rules for the shortlist:**

- Every candidate needs a **reason**, not a score.
- Include the simplest thing that could work, always.
- If a method is only on the list because it won a Kaggle competition, remove it.
- Write the list down before running anything. Otherwise you will quietly drop
  the ones that did badly, which is how you end up unable to explain your
  choice.

---

## Step 6 — Look up a method you have never used

This is a skill, and it is learnable. The repeatable procedure:

### 6.1 Find the function

Search, in this order:

1. **The cheatsheet** — scikit-learn's
   [algorithm cheat sheet](https://scikit-learn.org/stable/algorithm_selection.html)
   maps problem type to method. For other libraries, the library's own docs index.
2. **The API docs' class list** — scanning the class names tells you what exists.
3. **The library's "user guide" section for your task** — usually a page per task
   type with a recommended default.
4. Ask: "what is the *simplest* thing in this library for my task?" Libraries
   have a default recommended method. Start there.

### 6.2 Check it is usable before you invest in it

For any candidate, confirm:

| Question | Why it matters |
|---|---|
| Is it stable in the installed version? | Some are experimental |
| Does it work inside a `Pipeline`? | If not, it will not work for production |
| Does it handle missing values itself? | Or must you impute first |
| Does it accept `random_state`? | Otherwise it is not reproducible |
| Is it actively maintained? | Check when the last release was |
| Does it need scaling first? | SVM and k-NN do; trees do not |

### 6.3 Get a minimal working version

Smallest code that runs, before any tuning:

```python
"""Minimal working version of a candidate, on 500 rows, no tuning.

The point of this file is to answer one question: does this method run at all
on our data, and what does it score with default settings? Nothing here is
final.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")
features = ["tenure_days", "support_tickets_last_30_days", "monthly_plan"]
target = "churned"

train_data, holdout = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data[target]
)

model = RandomForestClassifier(random_state=42)
model.fit(train_data[features], train_data[target])

score = recall_score(holdout[target], model.predict(holdout[features]))
print(f"recall = {score:.3f}")
```

Run it. If it runs and gives a number, you now know the method is viable. If it
throws an error, you have learned something useful for about ten minutes.

**Do this for every candidate before wiring any of them properly.** Cheap,
throwaway, and it saves hours.

### 6.4 If the method is genuinely new to you

- Read the library's user guide page for the task, not the whole API reference.
- Find one worked example and copy its data shape.
- Check the "see also" list — it usually links to the simpler alternative.
- Time-box it: two hours. If you cannot get it working in two hours, either it
  is the wrong tool or you should use the boring option. Say so in the notes.

---

## Step 7 — Wire them all the same way

So that comparing them is fair, and swapping is one line.

The mechanism is the `MODEL_REGISTRY` in
[`02-A-ARCHITECTURE.md`](02-A-ARCHITECTURE.md) — one dictionary, one dispatch
function.

**The rules that keep the comparison honest:**

| Rule | Why |
|---|---|
| Every candidate gets the **same features** | Otherwise you are comparing data, not methods |
| Every candidate is trained on the **same split** | Otherwise you are comparing luck |
| Every candidate is scored with the **same metric** | Otherwise the numbers mean nothing |
| Every candidate gets the **same seed** | Removes one source of randomness |
| Every candidate is **run once with default settings** first | Tuning five methods properly is weeks of work |

**The honest, unglamorous way to do this** — a comparison script that takes a
list of names and prints a table:

```python
"""Compares candidate models on the same split with the same metric.

Run with: python research/compare_models.py
Writes a table to reports/model_comparison.md. This decides which single
model goes forward. It is not part of the production pipeline.
"""

import pandas as pd
from sklearn.model_selection import cross_validate

from src.components.model_training import build_model

FEATURES = ["tenure_days", "support_tickets_last_30_days", "monthly_plan"]
TARGET = "churned"
SEED = 42

CANDIDATES = {
    "logistic_regression": {},
    "decision_tree": {},
    "random_forest": {},
    "hist_gradient_boosting": {},
}


def compare_candidates() -> pd.DataFrame:
    """Train each candidate on the same data and return scores for each."""
    data = pd.read_csv("artifacts/data_cleaning_encoding/train.csv")

    results = []
    for name, params in CANDIDATES.items():
        model = build_model(name, params, SEED)
        scores = cross_validate(
            model,
            data[FEATURES],
            data[TARGET],
            cv=5,
            scoring="recall",
        )
        results.append(
            {
                "model": name,
                "recall_mean": scores["test_score"].mean(),
                "recall_std": scores["test_score"].std(),
            }
        )
        print(f"{name}: recall {results[-1]['recall_mean']:.3f}")

    return pd.DataFrame(results).sort_values("recall_mean", ascending=False)
```

This script belongs in `research/`, not in the production pipeline. It is a
one-off decision tool. Keep it — the next person will need it when the data
changes.

---

## Step 8 — Score them on validation

One table. Every number from the same run.

```text
| Model                    | Recall | Precision | Avg calls/mo | Params | Notes |
|--------------------------|--------|-----------|--------------|--------|-------|
| Do nothing (baseline)    | 0.00   | -         | 0            | 0      | The floor |
| Simple rule (baseline)   | 0.60   | 0.15      | 4,000        | 0      | Must beat |
| Logistic regression      | 0.55   | 0.12      | 2,100        | 12     | Underperforms the rule |
| Decision tree (depth 4)  | 0.62   | 0.17      | 4,300        | 13     | Readable, could show it |
| Random forest            | 0.63   | 0.18      | 4,150        | 45     | |
| Hist gradient boosting  | 0.66   | 0.19      | 4,020        | 300    | Best |
```

**Read it like this:**

- The two baselines are the floor. Anything that cannot beat the simple rule is
  not a candidate.
- Note the trade-off column, not just recall. More recall here means more calls.
- The most complex model is only worth it if the gain is big enough to matter to
  the decision. 0.63 → 0.66 is real, and it is not automatically worth it.

---

## Step 9 — Read the errors of the winner

Before committing, read what the winning model gets wrong. It is the highest
yield hour in the whole step.

```text
| customer | actual | predicted | notes                                     |
|----------|--------|-----------|-------------------------------------------|
| 10023    | churn  | 0.05      | Business closed. Nothing in the data says. |
| 10087    | churn  | 0.51      | 2 tickets in one day; we only count monthly |
```

The second row is your next feature. That is how features get invented —
by reading errors, not by guessing.

**If all the errors look the same, you have a feature idea. If they look random,
you probably have a data problem.** Check before going further.

---

## Step 10 — Pick the simplest thing that meets the bar

The decision rule, written down:

> We chose **<method>** because it scored <number>, beating the simple rule
> (<number>), and the extra <n>% is worth the added complexity of <what>.
> We rejected <method> because <reason>.
> Simpler option we did not use: <option>, because <reason>.

**The simplicity check.** Before accepting the winner, ask:

- Does the simpler option fail badly, or only slightly worse? *Slightly worse is
  usually not worth the complexity.*
- Could the business use the simpler one and get 90% of the value with none of
  the maintenance? If yes, that is worth raising.
- Can you explain the chosen method to the person who will use it? If not, that
  is a real cost, not an imaginary one.

**It is completely normal to choose the simpler model.** "We used a single
decision tree, it scored 0.62, and anyone can read why it made each call" is a
strong outcome — and far more common in real projects than the tutorials imply.

---

## Step 11 — Only now, tune the winner

One method. Not five.

Tuning is in [`03-TRAIN-AND-TUNE.md`](03-TRAIN-AND-TUNE.md). The short version:
tune the winner, set a time limit, then re-check that the result still beats the
simpler option.

---

## Step 12 — Document the choice

In the experiment log, with the number and the reason. See
[`09-TEMPLATES.md`](09-TEMPLATES.md).

This is what stops the project being re-litigated next quarter by someone who
thinks a different method would be better. They may be right — but they will
see what you tried, and what it scored.

---

## Gate — the method is chosen when all of these are true

- [ ] Problem classified: task type, data type, unit of prediction, size
- [ ] Constraints written down, and methods eliminated by them
- [ ] Metric agreed, and it comes from the business decision
- [ ] Trivial and simple-rule baselines present in the comparison
- [ ] 3–5 candidates listed with a reason each, before scoring
- [ ] Every candidate run once with default settings on the same split
- [ ] Every candidate scored with the same metric, same features, same seed
- [ ] Errors of the top candidate read by a human
- [ ] The simplest option that meets the bar chosen, with the reasoning written
- [ ] The winner wired through `MODEL_REGISTRY`, changeable by one config line
- [ ] Choice recorded in the experiment log with numbers

**If any box is unticked: stop.** You are about to tune a method you have not
chosen.

---

## Common mistakes in this phase

- **Tuning five methods properly.** Weeks of work, and no better answer than
  running each once and picking.
- **Starting from the most complex method** "because it is best". On tabular
  data it usually is not.
- **Comparing candidates on different data or splits.** The comparison means
  nothing.
- **Choosing on the leaderboard.** Those scores come from different preprocessing
  and different data. Useless as a basis for your choice.
- **Optimising the wrong metric**, because it was the default.
- **Never reading the errors**, and so never finding the next feature.
- **Forgetting that a linear model is a real candidate.** It is fast,
  reproducible, explainable, and often within a couple of points.
- **Writing the shortlist after scoring**, so it looks like you had a plan.
- **Building a heavy abstraction to compare three models.** A dictionary and one
  loop, as above.