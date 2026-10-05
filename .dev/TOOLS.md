# ML Tools

The libraries we use, and the job each one does. Use the right one instead of
writing the thing yourself.

| What you need to do | Use | Section |
|---|---|---|
| Track experiments, version models | MLflow | 1 |
| Serve the same features in training and live | Feast | 2 |
| Put a trained model behind an API (later) | MLflow serving | 2b |
| Check that incoming data is valid | Great Expectations | 3 |
| Detect data drift and model decay | Evidently | 4 |
| Search for the best settings | Optuna | 5 |

**One tool per job.** Never use two libraries for the same task.

---

## 1. MLflow — experiment tracking and model registry

MLflow is an open-source platform for managing the machine learning lifecycle.

It does two jobs for us:

- **Experiment tracking.** Log the settings you used, the results you got, and
  the files you produced. So you can answer "why is this model better?" and
  "can we reproduce it?" months later.
- **Model registry.** Store model versions and give them names, so the model
  running in production is a known, named version and not a mystery file.

What we log for every training run:

- Settings: model type, learning rate, number of layers, random seed. Every
  value from `config.yaml` (see [CODE-STANDARDS.md](CODE-STANDARDS.md)).
- Metrics: accuracy, precision, recall, training time.
- The trained model file itself.
- The **exact git commit hash** of the code that produced it.
- The input data version, so a result can be traced back to its data.

Why it matters: without it, every result is a story someone remembers
differently. With it, results are facts.

Rule: **every training run is logged.** A run that is not logged did not happen.

Model files are never left loose on someone's machine. Every trained model goes
to the model registry (or an artifact store such as S3) together with:

- the model file,
- the commit hash it was trained from,
- its `model_card.md` (see [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md)).

That way, any model in production can be traced back to the exact code and data
that produced it.

---

## 2. Feast — feature store

Feast is an open-source feature store. It makes sure a feature is built the same
way during training and during live serving.

It solves a real, silent failure: **training-serving skew**. This is when the
features at training time are slightly different from the features at serving
time. The model still runs, still returns numbers, and quietly gets worse. You
only notice weeks later.

Feast solves it by keeping one definition of each feature and one place where
its values come from.

What we put in Feast:

- Feature definitions: name, type, where the value comes from, how fresh it
  must be.
- Feature views: the tables the features are read from.
- Offline store: the historical data used for training.
- Online store: the live values used when serving.

Where the files live: everything Feast owns sits in `feature_repo/` at the top
level of the project — see [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md). That is
the only place a feature is defined.

### How the two halves work together

- **Training time** — Feast reads historical values from the offline store, so
  the training data is built from the definitions, not from ad-hoc code.
- **Serving time** — Feast serves the same feature values from the online store.
  The model asks for features by name; it never computes them itself.

Both sides read the *same definition*. That is the whole point: it is what makes
training-serving skew impossible rather than merely unlikely.

Two commands cover the daily use:

```bash
cd feature_repo
feast apply      # read the definitions and register them
feast materialize-incremental 2024-01-01 2024-02-01  # copy new values into the online store
```

Rule: **a feature used by a model must be defined in Feast, not in a notebook
or in the training script.**

---

## 2b. Model serving

Serving means putting a trained model behind an HTTP endpoint so other software
can call it.

MLflow can do this. It loads a model version from the registry and serves it.

**We are not building this yet.** There is no service to serve, so there is
nothing to set up. Adding a serving stack now would be building for a problem we
do not have.

Add it when both of these are true:

1. Another system needs predictions over HTTP.
2. The model is trained regularly enough that hand-copying files will not work.

When that day comes, the order is: MLflow serves the model, and it asks Feast for
features at request time so the served features are the same definitions used in
training. That pairing is the reason both tools exist.

If a deployment method other than MLflow serving is chosen later, write it down
here first, so there is one answer rather than three.

---

## 3. Great Expectations — data quality checks

Great Expectations lets you state what "good data" means, then checks incoming
data against those rules automatically.

Rules are written in plain English:

- `amount` must be greater than 0.
- `hour` must be between 0 and 23.
- `customer_id` must never be missing and must be unique.

Each check returns pass or fail. Fail fast, at the start of the pipeline, not
three steps later when the model returns nonsense.

Why it matters: most model surprises are data problems, not model problems. A
check that runs in a second is cheaper than a bad prediction in production.

Rule: **every raw data file is validated before anything uses it.** Failed check
means stop.

---

## 4. Evidently — drift and decay monitoring

Evidently monitors data and model performance over time and tells you when
something has changed underneath you.

It answers two questions:

- **Data drift.** Has the input data changed shape? For example, average income
  was 50k in training and is 80k now. The model still runs, but it was never
  trained on that data.
- **Model decay.** Has accuracy dropped? Same code, same model, worse results.

It compares the current data against the training data and produces a readable
report.

Why it matters: production models fail quietly. This is the alarm bell.

Rule: **run the drift report on a schedule (weekly is a reasonable start) and
look at it.** An alert nobody reads is not monitoring.

---

## 5. Optuna — automatic hyperparameter tuning

Hyperparameters are the knobs you set by hand: learning rate, dropout, number
of layers, number of trees.

Guessing them wastes days. Optuna tries many combinations and reports which one
worked best.

How we use it:

1. Write one small script that trains the model with a given set of knobs and
   returns a score.
2. Give Optuna the ranges to search.
3. Run it overnight.
4. Read the winning settings, paste them into the training script as the
   starting point.

Why it matters: it beats human guessing, and it runs while you sleep.

Rules:

- One study per model. The goal function stays in its own small file.
- Set a time limit. Stop the search when it has run long enough.
- The winning settings get written into `config.yaml` as plain values.
  Do not keep Optuna as a permanent dependency of training.

---

## Quick rules

- One tool per job. No duplicates.
- Log everything: run, settings, results, model file.
- Stop the pipeline when a data check fails.
- Trust no model in production until drift is monitored.