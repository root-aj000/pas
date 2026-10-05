# 02-A — Architecture: How the Code Is Wired

**Before this file:** [`01-DATA.md`](01-DATA.md) is done. You know the data.
**After this file:** you know where every piece of code lives, who calls what,
and how a method from a library gets plugged in.
**Read this before writing any production code**, and again whenever you feel
lost — it is a map, not a test.

---

## Why this file exists

You can follow a tutorial, copy a notebook, and still not know how to turn that
into a codebase that another person can work on. Tutorials hide the wiring. Real
projects are mostly wiring.

This file is the wiring. It answers four questions that get skipped everywhere:

1. Where does my code go?
2. What calls what?
3. Where does a method from a library actually get plugged in?
4. What does a new developer see first?

**The one-line version:**

> Settings live in one file. Components do the work. Stages do the ordering.
> Each stage hands its output to the next one as a saved file.

Everything below is detail on that sentence.

---

## The whole picture

```text
                       ┌─────────────────────────────────────────┐
                       │  RESEARCH — notebooks                   │
                       │  (free to be messy, never production)   │
                       │                                         │
                       │  research/                              │
                       │    data_ingestion.ipynb                 │
                       │    data_cleaning_encoding.ipynb         │
                       │    model_training.ipynb                 │
                       │    model_evaluation.ipynb               │
                       └───────────────┬─────────────────────────┘
                                       │ anything reusable moves down ↓
┌──────────────┐                       │
│ template.py  │───────────────────────┤  research entry point
└──────────────┘                       │
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────┐
│                              config.yaml                                   │
│         Every setting in the project. Paths, hyper-parameters,              │
│         thresholds, seeds, feature lists. No code, just values.             │
└──────────────────────────────────────┬─────────────────────────────────────┘
                                       │ read by
┌──────────────────────────────────────▼─────────────────────────────────────┐
│  src/                                                                     │
│                                                                             │
│  constants/__init__.py      values that never change (column names,         │
│                             formats, fixed category lists)                 │
│                                                                             │
│  entity/                                                                 │
│    config_entity.py         THE CONTRACT. What a settings object and an     │
│                             artefact must contain. Checked by Python.     │
│                                                                             │
│  config/                                                                  │
│    configuration.py         Reads config.yaml. One small class per stage.    │
│                             Components never touch yaml.                   │
│                                                                             │
│  components/          ← THE WORK HAPPENS HERE                              │
│    data_ingestion.py           load raw data                               │
│    data_cleaning_encoding.py   clean + encode                              │
│    model_training.py           fit and save the model  ← methods go here     │
│    model_evaluation.py         score the model                              │
│                                                                             │
│  utils/                                                                  │
│    common.py             logging, file IO, yaml, save/load, folder naming    │
│                                                                             │
│  pipeline/              ← THE ORDER HAPPENS HERE                           │
│    stage_01_data_ingestion.py                                               │
│    stage_02_data_cleaning_encoding.py                                       │
│    stage_03_model_training.py                                               │
│    stage_04_model_evaluation.py                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐  ┌───────────────────┐  ┌──────────────┐
│  artifacts/  │  │      models/      │  │    logs/     │
│  stage1 out  │  │  model_1/         │  │ 2024-06-12_  │
│  stage2 out  │  │  model_1.pkl      │  │ 0930_run.log │
└──────────────┘  └───────────────────┘  └──────────────┘
        hand-off between stages — saved files, not memory
```

Plus two entry points: **`template.py`** runs the research notebooks.
**`Dockerfile`** (and the `-ml.py` style runner script) runs the whole training
pipeline end to end in a reproducible environment.

---

## The five boxes, in plain language

### 1. `config.yaml` — every setting, in one file

Not code. Just values. This is the file you edit when you want to try something
different.

```yaml
# Paths
data_path: "data/raw/signups.csv"
artifacts_path: "artifacts"
models_path: "models"
report_path: "reports"

# Reproducibility
random_seed: 42

# The decision threshold — decided with the business, with the reason recorded
decision_threshold: 0.62

# Model selection: the name maps to a class in model_training.py
model_name: hist_gradient_boosting

# Hyperparameters for the chosen model
model_params:
  max_iter: 300
  max_depth: 8
  learning_rate: 0.05

# The feature list. The same list is checked at prediction time.
features:
  - tenure_days
  - support_tickets_last_30_days
  - monthly_plan
  - avg_spend_per_visit
```

**Why:** changing a setting is a one-line edit in one file, and the history of
what you ran lives in git. See the configuration rules in `.dev/RULES.md`.

**Rule:** if a value could plausibly change between experiments, it lives here.
If it can never change, it is a constant in `constants/`.

---

### 2. `entity/config_entity.py` — the contract

This is the least obvious box, and the most useful one.

An **entity** is a typed description of what something must contain. It is a
dataclass — a plain container with a type check.

```python
"""Typed descriptions of what each stage's settings and outputs must contain.

This file defines the agreement between stages. If stage 1 produces something
that does not match, Python stops the program with a clear message instead of
letting stage 3 fail with a confusing error.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataIngestionConfig:
    """Settings needed to load the raw data."""

    data_path: Path
    artifacts_dir: Path


@dataclass
class DataIngestionArtifact:
    """What stage 1 produced, and where it was saved."""

    raw_data_path: Path
    row_count: int


@dataclass
class ModelTrainerEntity:
    """Settings needed to train the model."""

    train_data_path: Path
    model_dir: Path
    model_name: str
    model_params: dict[str, object]
    random_seed: int
    features: list[str]


@dataclass
class ClassificationMetrics:
    """The scores we report, so every run is measured the same way."""

    recall: float
    precision: float
    f1: float
    roc_auc: float
```

**Why this exists:** without it, stage 1 returns a tuple, stage 2 unpacks it in
the wrong order, and you spend an afternoon on an error that says nothing useful.
With it, the error says `expected 2 fields, got 3`.

**Rule:** if a stage returns something, describe that something in this file with
a dataclass. Never return a bare tuple or dict across a stage boundary.

---

### 3. `config/configuration.py` — the only place that reads `config.yaml`

One small class per stage. Each knows how to find its own settings.

```python
"""Reads config.yaml and hands each stage the settings it needs.

Components never read config.yaml themselves. They ask their config class.
"""

from pathlib import Path

import yaml

from src.entity.config_entity import DataIngestionConfig


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Return the whole configuration as a dictionary."""
    with config_path.open() as file:
        return yaml.safe_load(file)


class DataIngestionConfigReader:
    """Builds the settings object for stage 1."""

    def __init__(self, config_path: Path = Path("config.yaml")) -> None:
        self.config = load_config(config_path)

    def create_config(self) -> DataIngestionConfig:
        """Return the settings stage 1 needs."""
        return DataIngestionConfig(
            data_path=Path(self.config["data_path"]),
            artifacts_dir=Path(self.config["artifacts_path"]),
        )
```

**Why:** yaml parsing, default values and path building live in exactly one file.
A component can be tested by handing it a settings object, with no yaml and no
files involved.

---

### 4. `components/` — where the work happens

One file per job. Each is a plain function: take a settings object, do the work,
return an entity.

```python
"""Loads the raw signup data and saves it for the next stage.

This file does one job: read the raw CSV and save an unchanged copy into
artifacts/. It does not clean anything, and it does not know what comes next.
"""

import logging

import pandas as pd

from src.entity.config_entity import DataIngestionArtifact, DataIngestionConfig

logger = logging.getLogger(__name__)


def run_data_ingestion(config: DataIngestionConfig) -> DataIngestionArtifact:
    """Load the raw data and save it to the artifacts folder.

    Args:
        config: Where the raw data is and where to save the output.

    Returns:
        The path of the saved file and how many rows it holds.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the file is empty. An empty file always means an upstream
            load failed, so the pipeline must stop here.
    """
    if not config.data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {config.data_path}")

    logger.info("[ingestion] reading %s", config.data_path)
    data = pd.read_csv(config.data_path)

    if data.empty:
        raise ValueError(
            f"Raw data is empty: {config.data_path}. Upstream load probably failed."
        )

    output_dir = config.artifacts_dir / "data_ingestion"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_data.csv"
    data.to_csv(output_path, index=False)

    logger.info("[ingestion] wrote %d rows to %s", len(data), output_path)
    return DataIngestionArtifact(raw_data_path=output_path, row_count=len(data))
```

**The rules for a component:**

| Rule | Why |
|---|---|
| One job per file | A file called `everything.py` is where bugs hide |
| It does not know what comes next | So it can be tested alone |
| It does not read `config.yaml` | It receives settings from the caller |
| It logs what it read and what it wrote | So the run can be traced |
| It raises on bad data instead of warning | See `.dev/RULES.md` rule 8 |

**This is where the library methods go.** `model_training.py` is where you call
`HistGradientBoostingClassifier`. `data_cleaning_encoding.py` is where you build
the `ColumnTransformer`. See [`02-D-METHODS-CATALOGUE.md`](02-D-METHODS-CATALOGUE.md)
for which function to use.

---

### 5. `pipeline/` — where the order lives

Each stage file is thin. It calls one component and logs what happened.

```python
"""Stage 1 of the pipeline: load the raw data.

Run with: python -m src.pipeline.stage_01_data_ingestion
"""

import logging

from src.config.configuration import DataIngestionConfigReader

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Load the raw data and save it for stage 2."""
    logging.basicConfig(level=logging.INFO)
    logger.info("[stage 01] starting data ingestion")

    config = DataIngestionConfigReader().create_config()
    artifact = __import__(
        "src.components.data_ingestion", fromlist=["run_data_ingestion"]
    ).run_data_ingestion(config)

    logger.info(
        "[stage 01] finished: %d rows at %s",
        artifact.row_count,
        artifact.raw_data_path,
    )


if __name__ == "__main__":
    run_pipeline()
```

Replace that ugly import with a normal one — it is only written oddly here to
keep the example self-contained. In real code:

```python
from src.components.data_ingestion import run_data_ingestion
```

**Why stages are separate files:** each one can be run on its own while you work
on it. `python -m src.pipeline.stage_03_model_training` and you have a training
run in seconds, without re-running ingestion.

**Rule:** a stage contains **ordering and logging only**. If a stage starts
containing real logic, that logic belongs in a component.

---

## How the pieces hand over to each other

Stages never pass data in memory. Each one saves a file, and the next reads it.

```text
stage 01  ──writes──▶  artifacts/data_ingestion/raw_data.csv
stage 02  ──reads────▶  artifacts/data_ingestion/raw_data.csv
           ──writes──▶  artifacts/data_cleaning_encoding/train.csv
                        artifacts/data_cleaning_encoding/test.csv
                        artifacts/data_cleaning_encoding/features.json
stage 03  ──reads────▶  .../train.csv, .../test.csv, .../features.json
           ──writes──▶  models/model_1/model.pkl
                        models/model_1/config.json
                        models/model_1/model_card.md
stage 04  ──reads────▶  models/model_1/model.pkl
           ──writes──▶  reports/model_1_evaluation.md
```

**Why files instead of memory:**

- Any stage can be re-run alone, without running the ones before it.
- You can look at what a stage produced with any tool, not just Python.
- A failed run leaves the last good output in place, so you can compare.

**Why that matters for debugging:** when stage 3 gives a weird result, you open
the CSV it read and check it. You are not guessing what stage 2 passed it.

---

## The research layer

`research/` holds notebooks: `data_ingestion.ipynb`,
`data_cleaning_encoding.ipynb`, `model_training.ipynb`,
`model_evaluation.ipynb`. The same four jobs as production, but free to be
messy.

**The rule that makes this safe:**

> A notebook may be ugly. **Its conclusions must move into `src/`.**

The moment you find yourself re-running the same code twice, it is no longer
research — it is production code. Move it into a component, give it a name, add
a docstring, and call it from a stage.

```text
Notebook: "I tried 6 models and HistGradientBoosting won"
   ↓
src/components/model_training.py:  build_model(name, params, seed)
   ↓
config.yaml:  model_name: hist_gradient_boosting
```

**Never** the other way around. Production code must not depend on a notebook.

---

## Wiring a method in: the single dispatch point

This is the part the tutorials never show. When `config.yaml` says
`model_name: hist_gradient_boosting`, something has to turn that string into a
class. **One dictionary, in one file.**

```python
"""Maps a model name from config.yaml to the class that implements it.

This is the only place in the project where a model name becomes a class.
Adding a model means adding one line here, and nothing else changes.
"""

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Key: the name used in config.yaml. Value: the class to use.
MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_model(model_name: str, params: dict[str, object], seed: int):
    """Create the model named in the configuration.

    Args:
        model_name: Key from MODEL_REGISTRY, e.g. "random_forest".
        params: Hyperparameters for the model.
        seed: Random seed, so the model is reproducible.

    Returns:
        An unfitted model object, ready for .fit().

    Raises:
        KeyError: If the name is not in the registry, listing the names that are.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(random_state=seed, **params)
```

Then the training component becomes method-agnostic:

```python
def run_model_training(config: ModelTrainerEntity) -> Path:
    """Train the model named in the configuration and save it."""
    train_data = pd.read_csv(config.train_data_path)

    # Fail early rather than training on the wrong columns.
    check_columns("train", set(train_data.columns), set(config.features))

    model = build_model(config.model_name, config.model_params, config.random_seed)
    logger.info("[train] fitting %s", config.model_name)
    model.fit(train_data[config.features], train_data["churned"])

    model_dir = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    return model_path
```

**Now every method question is answered in three places:**

| Question | Answered by |
|---|---|
| Which method are we using? | `config.yaml` → `model_name` |
| How do I run it? | `src/components/model_training.py` |
| What else is available? | `MODEL_REGISTRY` in the file above |

**To try a different method, you change one line in `config.yaml`.** Nothing else
touches. That is what "wiring" means, and it is why this architecture exists.

### Is this a "factory", which our rules forbid?

Yes, technically — and it is allowed here for one specific reason: **we compare
several methods before choosing one.** That is the proven need.

It stays honest because it is:

- one dictionary, in one file, with a plain `if` check for a bad name,
- not a plugin system, not dynamic discovery, not a class hierarchy,
- not wrapped in an abstraction with one implementation.

If the project ever only ever uses one model, delete `MODEL_REGISTRY` and
instantiate that model directly. See `.dev/RULES.md` rule 7.

---

## Where each library plugs in

| Stage | Component | Library calls that belong there |
|---|---|---|
| 01 ingestion | `data_ingestion.py` | `pandas.read_csv` / `read_parquet`, `sqlalchemy` for databases, Feast `get_historical_features`, Great Expectations `validate` |
| 02 cleaning & encoding | `data_cleaning_encoding.py` | `sklearn.pipeline.Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `TargetEncoder`, `StandardScaler`, `function_transformer` |
| 03 training | `model_training.py` | `MODEL_REGISTRY` + the estimator, `fit`, `joblib.dump` |
| 04 evaluation | `model_evaluation.py` | `sklearn.metrics.*`, `sklearn.model_selection.cross_validate`, Evidently `Report`, MLflow `log_metrics` |
| utils | `utils/common.py` | `logging`, `yaml.safe_load`, `joblib`, `datetime` for folder names |
| research | `research/*.ipynb` | Anything, then migrate to the above |

**The rule:** a library is imported in exactly one component. If two components
import the same estimator, that belongs in a shared place — usually the
registry.

---

## Folder list, complete

```text
project/
├── config.yaml                  every setting
├── run_pipeline.py              THE ONLY entry point — no arguments
├── Dockerfile                   reproducible training environment
├── requirements.txt
├── setup.py                     makes src/ importable
│
├── research/                    exploration — never imported by src/
│   ├── data_ingestion.ipynb
│   ├── data_cleaning_encoding.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── constants/__init__.py    values that never change
│   ├── entity/config_entity.py  the contract
│   ├── config/configuration.py  reads config.yaml
│   ├── utils/
│   │   ├── __init__.py
│   │   └── common.py            logging, IO, yaml, save/load
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_cleaning_encoding.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   └── pipeline/
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_data_cleaning_encoding.py
│       ├── stage_03_model_training.py
│       └── stage_04_model_evaluation.py
│
├── artifacts/                   intermediate outputs, one folder per stage
│   ├── data_ingestion/
│   └── data_cleaning_encoding/
├── models/                      one folder per model version
│   └── model_1/model.pkl
├── dev-logs/                    what we changed and why — see .dev/DEV-LOG.md
├── docs/                        problem statement, decisions, runbooks
├── logs/                        what the code did — one file per run
│   └── dev/                     experiments, kept, never deleted
├── data/                        raw data
├── tests/                       one test file per component
└── .lead/ .dev/                 documentation
```

### Entry points

| Command | Purpose | Arguments |
|---|---|---|
| `python run_pipeline.py` | **The only production entry point.** Runs every stage in order | **None. Ever.** |
| `python -m src.pipeline.stage_03_model_training` | One stage, while debugging | None |
| `pytest tests/ -q` | Tests | Allowed |
| `python research/compare_candidates.py` | Method comparison | Allowed |
| `template.py` | Runs the research notebooks | Allowed |

**Why the main command takes no arguments:** everything that varies is in
`config.yaml`. Changing a setting is a committed edit, so the git history always
shows which settings produced which result. If the command could take flags,
somebody would eventually run it with flags nobody wrote down. See
[`.dev/DEBUGGING.md`](../.dev/DEBUGGING.md) section 8.5.

### The two logs, both required

| | Dev log | Run log |
|---|---|---|
| Where | `dev-logs/` | `logs/` |
| Who writes it | You, about your reasoning | The code, about what it did |
| Answers | Why did we do this? | What happened, exactly? |
| Editing | Corrections appended, dated | Never — append only |

Rules: every change gets a dev log entry, including one-line fixes and failed
experiments. Every run and every test gets a run log with the git commit, the
config hash, the seed and the exit code. Nothing is ever deleted. See
[`.dev/DEV-LOG.md`](../.dev/DEV-LOG.md).

---

## How a new developer reads this codebase

This is the actual test of an architecture. A new person should be able to
answer these without being told:

| Question | Found by reading |
|---|---|
| What settings exist? | `config.yaml` |
| What must a stage's output contain? | `entity/config_entity.py` |
| What does the project actually do? | `pipeline/stage_01` → `stage_04`, in order |
| Where is the real logic? | `components/` |
| Where do the logs come from? | `utils/common.py` |
| Which methods are available? | `MODEL_REGISTRY` |

Six files, and the whole project is legible. If that is not true, the structure
is wrong.

---

## Gate — you have understood the architecture when all of these are true

- [ ] You can name what each of the five boxes is for, in one sentence each
- [ ] You can say where a new library call goes, without checking
- [ ] You know that changing a method means editing one line in `config.yaml`
- [ ] You know why stages pass files instead of memory
- [ ] You know that a notebook's conclusions must move into `src/components/`
- [ ] You can trace one row of data from `config.yaml` to a prediction
- [ ] You can answer the six questions in the section above

## Common mistakes with this structure

- **Putting real logic in a pipeline stage.** Stages orchestrate. Logic belongs
  in components.
- **Reading `config.yaml` inside a component.** Then the component cannot be
  tested without files.
- **Passing a bare tuple between stages.** Use an entity from
  `config_entity.py`, so a mistake is caught with a readable message.
- **Notebooks in production.** If `src/` imports a notebook, the project cannot
  be run without Jupyter.
- **Hardcoding a setting inside a component** instead of in `config.yaml`. Now
  changing it requires a code change.
- **An entity per stage but never used.** It is decoration. Either return it, or
  delete it.
- **Trusting `config.yaml` values without checking them.** If `model_name` is
  wrong, fail with a clear message listing the valid names.