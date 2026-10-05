# Project Structure

Where every file lives. If you are about to create a file and are not sure
where it goes, this page answers it.

**The architecture explained in full — what each folder is for, who calls what,
and where a library method gets plugged in — is
[`.lead/02-A-ARCHITECTURE.md`](../.lead/02-A-ARCHITECTURE.md).** This page is the
quick lookup. Do not follow one without the other; when they disagree, the
architecture file is the source of truth.

---

## Top level

```
.
├── config.yaml         # Every setting: paths, hyper-parameters, seed, threshold
├── run_pipeline.py     # THE ONLY entry point. No arguments. Ever.
├── research/           # Notebooks — exploration, never imported by src/
├── src/                # Production code
├── artifacts/          # Intermediate outputs, one folder per stage
├── data/               # Datasets
├── feature_repo/       # Feast feature definitions
├── models/             # Trained model files, one folder per version
├── dev-logs/           # What we changed and why — one file per piece of work
├── docs/               # Problem statement, decisions, runbooks, incident notes
├── logs/               # What the code did — one file per run
│   └── dev/            # Exploratory runs, kept, never deleted
├── reports/            # Generated reports and charts
├── tests/              # Test files
├── venv/               # Virtual environment
├── .lead/              # What to do, in what order, and what to decide
├── .dev/               # How to write the code
├── requirements.txt    # Python dependencies
└── README.md           # What the project does
```

**The one-line summary of the architecture:**

> Settings live in one file. Components do the work. Stages do the ordering.
> Each stage hands its output to the next one as a saved file.

**And the one-line summary of the logging:**

> One command, no arguments. Every change written up in `dev-logs/`. Every run
> recorded in `logs/`. Nothing deleted.

Only `src/` is allowed to grow by itself. Artifacts, models, reports,
research notebooks and run logs are outputs, not the application.

---

## `data/`

```
data/
├── raw/                 # Original data, never edited
├── interim/             # Partly cleaned data
└── processed/           # Final data used for training
```

- `raw/` is read-only. If data is wrong, fix it with a script and write a new
  file. Never edit it by hand.
- Every file that a script creates must say, in the script, where it writes to.

---

## `feature_repo/` — feature definitions

This folder belongs to Feast. It is the **only** place a feature is defined.

```
feature_repo/
├── feature_store.yaml   # Feast settings: registry, offline store, online store
├── features/            # Feature definitions, one file per entity
└── data/                # Small sample data, so `feast apply` works locally
```

Two commands are used from inside this folder:

```bash
cd feature_repo
feast apply                          # read the definitions and register them
feast materialize-incremental 2024-01-01 2024-02-01   # refresh the online store
```

### `feature_repo/` vs `src/components/` — do not confuse them

These are two different jobs. Putting them in one place is the mistake.

| | `feature_repo/features/` | `src/components/data_cleaning_encoding.py` |
|---|---|---|
| Question it answers | *What is this feature called, and where does its value come from?* | *How do I work out this value from raw data?* |
| Contains | Declarations only: name, type, source table, freshness | Python code: the calculation, and the checks |
| Read by | Feast, at training and serving time | You, when running the pipeline |
| Example | `customer_age` comes from table `customers`, must be under 1 day old | `data_cleaning_encoding.py` computes age from `date_of_birth` |

In one sentence: **Feast says where the value lives, `src/` works out the
value.** A feature definition without the code behind it points at nothing; code
without a definition is not visible to Feast and will cause training-serving
skew.

Rule: if you are **declaring** a feature, it goes in `feature_repo/features/`.
If you are **calculating** one, it goes in
`src/components/data_cleaning_encoding.py`. There is no `src/features/` folder —
that was the old flat layout, and it hid where feature code actually lives.

---

## `src/`

The production code. Full explanation in
[`.lead/02-A-ARCHITECTURE.md`](../.lead/02-A-ARCHITECTURE.md); this is the shape
and the reason for each folder.

```
src/
├── constants/__init__.py    # Values that never change: column names, formats
├── entity/config_entity.py  # The contract: what each stage's settings and
│                            # outputs must contain (typed dataclasses)
├── config/configuration.py  # The only file that reads config.yaml
├── utils/
│   ├── __init__.py
│   └── common.py            # Logging, file IO, yaml, save/load
├── components/              # THE WORK HAPPENS HERE
│   ├── data_ingestion.py           # Load raw data
│   ├── data_cleaning_encoding.py   # Clean and encode
│   ├── model_training.py           # Fit and save — library methods go here
│   └── model_evaluation.py         # Score the model
└── pipeline/                # THE ORDER HAPPENS HERE — no real logic
    ├── stage_01_data_ingestion.py
    ├── stage_02_data_cleaning_encoding.py
    ├── stage_03_model_training.py
    └── stage_04_model_evaluation.py
```

Each stage runs on its own:

```bash
python -m src.pipeline.stage_01_data_ingestion
python -m src.pipeline.stage_02_data_cleaning_encoding
python -m src.pipeline.stage_03_model_training
python -m src.pipeline.stage_04_model_evaluation
```

If Feast is in use, register the feature definitions first, so the pipeline is
building features Feast knows about:

```bash
cd feature_repo && feast apply
```

Rules:

- **Components do the work. Stages do the ordering.** If a stage starts
  containing real logic, that logic belongs in a component.
- **A component does not read `config.yaml`.** It receives a settings object.
  That is what makes it testable without files.
- **Stages hand over saved files, not memory.** So any stage can be re-run alone,
  and you can inspect exactly what the previous stage produced.
- **One file per job.** No `data/`, `features/`, `models/` sub-folders inside
  `src/` — that was the old flat style, and it hid the architecture. The stage
  number already says the order.
- **One `model_training.py`, not `model_training_v2.py`.** Fix the file instead
  of copying it.
- **A library is imported in exactly one component.** If two components need the
  same thing, it belongs in `utils/common.py` or the model registry.
- **No `utils` dumping ground.** A helper stays next to the code that uses it.
  `utils/common.py` is only for things genuinely used everywhere.

---

## `artifacts/`

```
artifacts/
├── data_ingestion/raw_data.csv
└── data_cleaning_encoding/
    ├── train.csv
    ├── test.csv
    └── features.json
```

One folder per stage. This is the hand-off between stages. Everything a stage
produced is here, in plain files you can open with any tool.

**Safe to delete** — every stage can rebuild it. Keep it out of git.

---

## `research/`

```
research/
├── data_ingestion.ipynb
├── data_cleaning_encoding.ipynb
├── model_training.ipynb
└── model_evaluation.ipynb
```

For exploration. Free to be messy.

- **Never imported by `src/`.** If production code imports a notebook, the
  project cannot run without Jupyter.
- **The moment code is needed twice, it moves into `src/components/`.** Give it
  a name and a docstring, and call it from a stage.
- Notebooks are where the method comparison happens — see
  [`02-C-CHOOSING-THE-METHOD.md`](../.lead/02-C-CHOOSING-THE-METHOD.md) Step 7
  for the script that turns that into a table.

---

## `dev-logs/` — what we changed, and why

```
dev-logs/
├── 2024-06-01-project-kickoff.md
├── 2024-06-05-first-model-baseline.md
├── 2024-06-12-fix-merge-duplicating-rows.md
└── archive/
    └── 2024-q1/
```

- One file per **piece of work**, not per day.
- Plain English, written the same day, by the person who did the work.
- **Never delete or edit an old entry.** Add a dated correction.
- Committed to git — this is the project's history.
- Read by the leader, reviewers, and you in three months. See
  [`DEV-LOG.md`](DEV-LOG.md).

**A three-line fix still gets an entry.** There is no size threshold.

---

## `logs/` — what the code did

```
logs/
├── 2024-06-12_0930_full_run.log
├── 2024-06-12_1102_train.log
├── 2024-06-12_1400_pytest.log
└── dev/
    └── 2024-06-12_2210_try_lower_threshold.log
```

- One file per run, named `date_time_what`.
- **Append-only.** Never edited, never deleted. A failed run is evidence.
- `logs/` holds real runs. `logs/dev/` holds experiments and scratch commands —
  those stay too, kept out of the way of the real history.
- Every run records the git commit, a hash of `config.yaml`, the seed, the
  packages, and the exit code. See [`DEBUGGING.md`](DEBUGGING.md) section 8.
- Read by whoever is debugging. See [`DEBUGGING.md`](DEBUGGING.md) section 9 for
  how this differs from `dev-logs/`.

---

## `run_pipeline.py` — the only entry point

```bash
python run_pipeline.py
```

**No arguments. Ever.** Everything that varies lives in `config.yaml`, so
changing a setting is a committed edit and the git history shows which settings
produced which result.

```bash
# WRONG — each one is a different, unlogged run
python run_pipeline.py --model random_forest --seed 7
python run_pipeline.py staging
```

**Arguments are allowed for tests and development**, and those logs go in
`logs/dev/` and are kept:

```bash
pytest tests/ -k test_overfit -v          # fine
python research/compare_candidates.py     # fine
python -m src.pipeline.stage_03_model_training   # fine, a single stage
```

Why the rule exists: one command means one run. It removes any question about
which flags produced last month's number, and it removes "it works if you pass
`--staging`" — which is not true, it is only unreported.

---

## `models/`

```
models/
├── model_1/
│   ├── model.pkl          # The trained model
│   ├── features.json      # Exact feature list, in order
│   ├── config.json        # Every setting used, including the seed
│   ├── model_card.md      # What it does, how well, what it needs
│   └── train.log          # Full log of the run
└── model_2/
```

- Each model gets its own folder. Never overwrite a trained model.
- All five files travel together. A model without its feature list is
  unmaintainable — see [TOOLS.md](TOOLS.md) section 1.
- Record the git commit hash in `config.json` and in the model card.
- Every model is also uploaded to the model registry with MLflow. The local
  folder is a copy, not the source of truth.

---

## `tests/`

One test file per component.

```
tests/
├── components/
│   ├── test_data_ingestion.py
│   ├── test_data_cleaning_encoding.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
├── pipeline/
│   └── test_stages.py
└── test_config_entity.py
```

The overfitting test lives in `tests/components/test_model_training.py`. See rule
10 in [RULES.md](RULES.md).

---

## `reports/`

Generated output: charts, drift reports, evaluation tables, model comparison
tables. Safe to delete, so keep it out of git.

---

## Naming rules

| Thing | Rule | Example |
|---|---|---|
| Files and folders | `snake_case` | `data_cleaning_encoding.py` |
| Functions and variables | `snake_case`, descriptive | `average_spend_per_visit` |
| Classes and dataclasses | `PascalCase` | `DataIngestionConfig` |
| Constants | `SHOUTING_SNAKE_CASE` | `MAX_AGE = 120` |
| Booleans read as a question | prefix `is_` / `has_` | `is_empty`, `has_income` |
| Stages | `stage_<NN>_<what_it_does>` | `stage_03_model_training.py` |
| Models | `model_<N>` | `models/model_3/` |
| Tests | `test_<what_it_checks>` | `test_raises_when_visits_is_zero` |

Avoid single-letter names except for loop counters and maths (`i`, `x`, `y`).

---

## Adding a new file

Ask three questions:

1. Does this belong in `src/components/`, `src/pipeline/`, `src/entity/`,
   `src/config/`, `research/`, `tests/`, `feature_repo/`, or `config.yaml`?
2. Is the job already done by an existing file or library? If yes, extend that
   file instead of creating a new one.
3. Can I do this without a new file at all?

Most of the time the honest answer is option 3.

If you are still unsure:

| The file contains | It goes in |
|---|---|
| A setting that might change between runs | `config.yaml` |
| A value that can never change (a column name, a date format) | `src/constants/` |
| A typed description of what something must contain | `src/entity/config_entity.py` |
| The actual work — loading, cleaning, training, scoring | `src/components/` |
| Ordering and logging between components | `src/pipeline/stage_NN_*.py` |
| Reading or parsing `config.yaml` | `src/config/configuration.py` |
| A helper used by more than one component | `src/utils/common.py` |
| A library method, an estimator, a transform | `src/components/`, or `MODEL_REGISTRY` |
| A feature *declaration* (name, type, source, freshness) | `feature_repo/features/` |
| An exploration that may be deleted | `research/` |
| A written record of a change | `dev-logs/` |

**If you are about to add a new pipeline stage, stop.** A fifth stage means the
pipeline is doing two things at once. Split the component instead, or ask whether
it belongs in an existing stage.

If you are still unsure:

| The file contains | It goes in |
|---|---|
| A feature *declaration* (name, type, source, freshness) | `feature_repo/features/` |
| Code that *computes* a feature value | `src/components/data_cleaning_encoding.py` |
| Settings, hyperparameters, the random seed | `config.yaml` |
| A shared logging helper used everywhere | `src/utils/common.py` |
| A one-off exploration that may be deleted | `research/` |