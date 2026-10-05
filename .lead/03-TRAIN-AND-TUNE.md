# 03 — Train and Tune Properly

**Before this file:** [`02-B-FINDING-SIGNALS.md`](02-B-FINDING-SIGNALS.md) is done.
A simple model already beats the baseline.
**After this file:** you have a model, its best settings, and a run anyone can
repeat exactly.
**Time it usually takes:** 1–3 days, or one overnight search.

---

## Why this file exists

Real projects rarely fail here — this is the part the internet tells you about.
But two things still go wrong:

1. **Tuning on the test set**, so the final score is a lie.
2. **A run that cannot be repeated**, so nobody can tell whether a change helped
   or you just got lucky.

Both are avoidable with a little discipline.

---

## Step 3.1 — Make the run reproducible first

Before any training, set the seed. Set it in `config.yaml`, call it at the top
of the training script, and log the value.

The full function is in `.dev/CODE-STANDARDS.md`. The rule:

> Same data + same settings + same seed = same result.

Without this you cannot compare two runs honestly, and the whole project becomes
superstition.

Also fix these, once, and stop thinking about them:

- **All hyperparameters in `config.yaml`.** Nothing tuned inside the training
  loop — see `.dev/CODE-STANDARDS.md` section 5.
- **Every run logged** with its settings, its scores and its duration.
- **The training script reads a config and prints it** before it starts, so the
  log always says what was run.

---

## Step 3.2 — Run the training pipeline properly

A training run should read like a story in the log. Use the helpers from
`.dev/DEBUGGING.md`.

```text
[train]    starting run churn_v3 seed=42
[train]    config learning_rate=0.001 max_depth=8 n_estimators=300
[train]    loaded train rows=84000 validation rows=18000 features=12
[train]    removed 4 features unused or unavailable at prediction time
[train]    trained in 41s
[train]    validation: recall=0.63 precision=0.61 f1=0.62
[train]    overfitting test passed: model memorised 16 rows
[train]    saved models/churn_v3/  commit=3f9a1c2  logged to MLflow run 118
```

Every line there is load-bearing. When something looks wrong three weeks later,
this log tells you what happened without any guessing.

**The order of operations, always:**

```text
1. Load train set.            Log row count.
2. Drop columns unavailable at prediction time.   Log which, and why.
3. Fit any scaling/encoding on the TRAIN set only.
4. Train.
5. Score on validation.
6. Run the overfitting test.
7. Save the model, the feature list and the config together.
8. Log everything to MLflow.
```

Step 3 is the one people get wrong. If you fit a scaler on all the data, the
test set's own distribution leaks into training. Fit on train, apply to the rest.

---

## Step 3.3 — Run the overfitting test

This belongs in the test suite, and it runs before you trust any training run.

**The idea:** a model must be able to memorise 16 rows perfectly. If it cannot,
something is broken — the gradients are not flowing, the learning rate is wrong,
or the labels and features are mismatched. No amount of real data will hide this.

```python
def test_model_can_overfit_tiny_dataset():
    """The model must be able to memorise 16 rows.

    If it cannot, training is broken. This is the fastest way to find out.
    """
    tiny_features, tiny_labels = make_tiny_dataset(rows=16)
    model = build_model(input_size=tiny_features.shape[1])

    train(model, tiny_features, tiny_labels, epochs=200)

    assert model.predict(tiny_features) == tiny_labels
```

If this test fails, stop and fix it. Do not go looking for data problems — this
one is a code bug, and no amount of data will fix it.

---

## Step 3.4 — Tune the settings

Only now. Not before.

**What is worth tuning** (in order of impact, usually):

| Setting | Typical range to search |
|---|---|
| Learning rate | 0.0001 to 0.1 |
| Depth / complexity | 3 to 12 |
| Number of trees or estimators | 100 to 1000 |
| Dropout (if using a neural network) | 0 to 0.5 |
| Number of features per split | 2 to all |

**What not to tune:** anything that is a fact rather than a choice. Do not tune
your features or your label.

**Using Optuna** (`.dev/TOOLS.md` section 5):

1. Write one small file with a function that trains the model with given settings
   and returns one score.
2. Give Optuna the ranges.
3. Set a time limit — one night is normal.
4. Read the winner, write those values into `config.yaml`, then commit.

**Which score to optimise:** the one that matches the problem you wrote in
[`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md). If catching leavers
matters more than wasted calls, optimise recall — but read Step 4.4 first,
because the score you optimise shapes how the model behaves for every group.

**Stop when it stops improving.** A search that keeps finding 0.1% gains for six
hours is not useful. Set the time limit in advance and honour it.

---

## Step 3.5 — Know when you are done

Stop tuning when:

- The search finished its time budget, or
- Validation score has not improved meaningfully for several attempts, or
- The gains are so small they would not change the decision in the real world.

Then do the final fit and freeze it.

**The final fit:** train on train + validation together, using the settings you
chose. You now have more data, and the settings are already decided, so no
decision is being made on data the model has seen.

**Then** evaluate once on the test set. That is the one honest number. See
[`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md).

Write down, before you see the test score:

- Which model and settings are being tested.
- Which test rows, which date, which seed.
- What score you would consider a pass.

Write it down **first**. Deciding what counts as success after seeing the score
is how people talk themselves into shipping something that does not work.

---

## Step 3.6 — Save everything you will need to repeat this

Save these **together**, in one folder, or the model becomes unreproducible:

```text
models/churn_v3/
├── model.pkl          # the trained model
├── features.json      # exact feature list, in order
├── config.json        # every setting used, including the seed
├── model_card.md      # what it does, how well, what it needs
└── train.log          # full log of the run
```

And in MLflow: the same settings, the same metrics, the git commit hash, and the
model file. See `.dev/TOOLS.md` section 1.

**The test of reproducibility:** give `model.pkl` and `features.json` to someone
who has never seen the project. They should be able to reproduce the predictions
exactly, with no questions to ask. If they cannot, something is missing.

---

## Gate — do not start evaluating until all of these are true

- [ ] Random seed set and logged
- [ ] All hyperparameters in `config.yaml`
- [ ] Every run logged with settings, scores and duration
- [ ] Scaling fitted on the training set only
- [ ] The overfitting test passes
- [ ] Tuning finished inside a pre-set time limit
- [ ] Final model trained on train + validation
- [ ] Test rows, date, seed and pass threshold written down **before** seeing
      the score
- [ ] Model, feature list, config, model card and log saved together
- [ ] MLflow run created, with the git commit hash
- [ ] Someone else can reproduce the predictions from the saved folder alone

**If any box is unticked: stop.** An unreproducible result cannot be trusted, and
cannot be maintained by anyone else.

---

## What to do next

Go to [`04-EVALUATION-AND-GO-NO-GO.md`](04-EVALUATION-AND-GO-NO-GO.md): measure
it properly, look at who it works for and who it fails, and make a real ship-or-stop
decision in writing.

## Common mistakes in this phase

- **Tuning until the test score looks good.** You now have no idea what your
  model actually does.
- **Not running the overfitting test** and spending days on a code bug.
- **Fitting the scaler on all the data.** A subtle, invisible leak.
- **No time limit on the search.** Gains of 0.1% for six hours.
- **Saving the model without the feature list.** It becomes unmaintainable.
- **Deciding what counts as success after seeing the score.**