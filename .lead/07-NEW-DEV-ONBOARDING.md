# 07 — Onboarding: Your First Days on This Project

**Who this is for:** a developer joining the project, or the lead who is doing
this work alone for the first time and does not know where to start.

**The promise:** you will never have to guess what to do next. Work down this
file in order. Each step says what to do, and what "done" looks like.

---

## Read this first

If you are new, you will feel two things, and both are normal:

1. **"I don't know what I'm supposed to be doing."** This is a documentation
   problem, not a you problem. This file exists to fix it. If you finish a step
   and do not know what the next one is, that is a gap in these files — write it
   down and tell the lead.
2. **"I don't know enough to start."** You are not expected to know enough. Step
   1 below exists purely to make you able to start.

**The one rule for your first month:** never guess. When something is unclear,
write the question down and ask. Every hour spent asking is cheaper than a day
spent building the wrong thing. This is an explicit project rule, not a personal
preference — see `.dev/RULES.md`.

---

## Day 1 — Orient, and get the machine working

**Do this in order:**

1. **Read the problem.** `.lead/00-PROBLEM-AND-BASELINE.md`, then the one-paragraph
   problem statement your lead wrote. If they have not written one yet, that is
   the first thing to do together.

2. **Read the rules.** `.dev/README.md` — the 14 golden rules. Then `.dev/RULES.md`.

3. **Read where things live.** `.dev/PROJECT-STRUCTURE.md` for the quick list,
   then `.lead/02-A-ARCHITECTURE.md` for what each part is actually for. Thirty
   minutes, and you know where to put any file and who calls what.

4. **Get the code running.** Clone the repo, create the virtual environment,
   install the requirements.

   ```bash
   git clone <repo-url>
   cd <project>
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Install the checks.** These run on every commit. Without this, your first
   commit will be rejected.

   ```bash
   pip install ruff pre-commit
   pre-commit install
   ```

6. **Run everything once.** Do not change anything yet. You are learning what the
   project does when it works.

   ```bash
   ruff check .
   pytest
   python -m src.pipeline.stage_01_data_ingestion
   python run_pipeline.py
   ```

7. **Read the runbook.** `.lead/05-PACKAGE-SERVE-DEPLOY.md` Step 5.6. This tells
   you what the project produces, in plain English.

**Done means:** you can run the project, you know what it outputs, and you have
not changed anything yet.

**Write down:** every command that did not work as written here, and fix these
files. That is genuinely valued work, not nitpicking.

---

## Day 2 — Understand the data

**Do this in order:**

1. **Read the data docs.** `.lead/01-DATA.md`. Pay attention to Step 1.3 and
   1.4 — what each column means, and what exactly the label is.

2. **Look at the actual data.** Open the reports from the last exploration, or
   load a small sample yourself:

```python
import pandas as pd

data = pd.read_csv("data/processed/features.csv", nrows=1000)
print(data.shape)
print(data.dtypes)
print(data.isna().sum())
```

3. **Read the column documentation.** Find the docstrings in
   `src/components/data_cleaning_encoding.py`. They exist so you do not have to ask what a column
   means.

4. **Write down what you still do not understand.** Every column you cannot
   explain in one sentence. This list is your task for the week.

5. **Ask about those columns.** One message to the data owner, not one hundred.

**Done means:** you can explain what the data contains and what the model
predicts, in plain language, without guessing.

---

## Day 3 — Follow one prediction end to end

The fastest way to understand a project is to trace a single row through it.

1. **Pick one customer.** Get their ID.
2. **Follow them through:** raw data → cleaned → features → model input →
   prediction → the file it ends up in.
3. **Write down what happened at each step**, in order, in plain language.

```text
Customer 10087
  raw:        2 support tickets on 2024-01-14, plan monthly, tenure 41 days
  clean:      kept, no duplicates found
  features:   tenure_band = "new"  tickets_last_30_days = 2  monthly_plan = 1
  model:      churn_v3 gives 0.71
  threshold:  0.62 -> above, so predicted to churn
  output:     row 118 of reports/churn_v3_predictions.csv
```

4. **Do this for one customer who churned and one who did not.**

**Done means:** you can explain exactly how a row of data becomes a prediction.

**If you get stuck** — a step is unclear, or the data does not match the
documentation — you have found a real bug or a real documentation gap. Write it
down and report it. That is a valuable find, not a failure.

---

## Week 1 — Know the tools

Now the machine learning stack. Read these in order:

| Order | File | What you learn |
|---|---|---|
| 1 | `.lead/02-A-ARCHITECTURE.md` | How the code is wired, and where a method gets plugged in |
| 2 | `.dev/TOOLS.md` section 1 (MLflow) | Where every experiment is recorded |
| 3 | `.lead/02-B-FINDING-SIGNALS.md` | How features were chosen |
| 4 | `.lead/02-C-CHOOSING-THE-METHOD.md` | How the method was chosen, and what else was tried |
| 5 | `.dev/TOOLS.md` section 2 (Feast) | Where features are defined |
| 6 | `.dev/TOOLS.md` section 3 (Great Expectations) | How data quality is checked |
| 7 | `.dev/TOOLS.md` section 5 (Optuna) | How settings were chosen |

**Then do something small and real:**

- Open MLflow and find the run for the current model. Read its settings and its
  score. Compare to the model card. They should match.
- Open a Feast feature definition and find one of the features you traced on Day
  3. Check the name matches what you saw in the data.
- Run the data quality checks yourself. Watch them pass.

**Done means:** you know which tool does what, and you have seen each one working.

---

## Week 2 — Make a first change

Do not start with anything important. Start small.

**Good first tasks:**

- Fix a typo or a confusing name in a docstring.
- Add a missing type hint that `ruff` flagged.
- Fix or add one test.
- Improve a log line so it answers a question you had.
- Fix a documentation gap you found in week 1.

**Before you start, get the change reviewed as too small.** Then follow the
whole path:

```text
1. Make the change.
2. Run ruff check --fix .  and  ruff format .
3. Run pytest.
4. Run the script you changed, and read the log.
5. Write the dev log entry — this is part of the change, not extra work.
6. Update the documentation if behaviour changed.
7. Commit with a clear message.
```

Step 5 has its own file: `.dev/DEV-LOG.md`. Even for a one-line fix. The entry
needs a header, what you did, **why**, how you checked it, what you did *not*
do, and what is still open.

**The point of this task is not the change.** It is to go through the whole path
once — check, test, run, log, **write it down**, commit — so that when you do
something important, you already know the routine.

---

## Month 1 — Own something small

By the end of the first month you should have one small thing that is yours. Good
candidates:

- Owning one feature in the feature pipeline.
- Owning the data quality checks for one table.
- Owning the weekly monitoring review (`.lead/06-MONITOR-AND-MAINTAIN.md` Step
  6.1).
- Owning the retraining procedure for one model.

**Owning** means: you are the person asked when it breaks, you have written down
what to do, and someone else could pick it up from your notes.

---

## Your first questions — the ones people are afraid to ask

All of these are normal. All of them should be asked.

| Question | Why it is fine to ask |
|---|---|
| "What is this column?" | Column meanings are not written down everywhere. You are not expected to know. |
| "Why is the threshold 0.62?" | It was decided by a specific person for a specific reason. Find them. |
| "Why does the split work this way?" | Time-based splits surprise everyone the first time. |
| "Can I add a feature?" | Yes — check for leakage first, then yes. |
| "Is this the right approach?" | Sometimes the answer is no. The go/no-go rule exists for that. |
| "How do I run this in production?" | It should be in the runbook. If it is not, that is a gap worth fixing. |

**Nobody expects you to know how a project works on day one.** They expect you to
ask, and to write down the answers so the next person does not have to.

---

## Where everything is

| I need to... | Read |
|---|---|
| Know what to do next | The gate at the end of the file you are in |
| Know what I am supposed to be building | `.lead/00-PROBLEM-AND-BASELINE.md`, step 0.1 |
| Know where my code goes, who calls what | `.lead/02-A-ARCHITECTURE.md` |
| Know which method to use, or which function | `.lead/02-C-CHOOSING-THE-METHOD.md`, `.lead/02-D-METHODS-CATALOGUE.md` |
| Write code | `.dev/RULES.md` |
| Know where to put a file | `.dev/PROJECT-STRUCTURE.md` |
| Add logs, or find a bug | `.dev/DEBUGGING.md` |
| Set up formatting, typing, seeds | `.dev/CODE-STANDARDS.md` |
| Know which tool to use | `.dev/TOOLS.md` |
| Explore data, find signal | `.dev/EDA.md` |
| Test a claim statistically | `.dev/STATISTICS.md` |
| Write something down | `.lead/09-TEMPLATES.md` |
| Find out why a project failed | `.lead/08-TRAPS.md` |

---

## If you are doing this alone

You have to play every role: analyst, developer, data owner, and the person who
makes decisions.

**What changes when there is no team:**

- **You must write things down.** Your memory is not a reliable record. Every
  decision, every column meaning, every number goes into a file the same day.
- **You must be your own reviewer.** Set a one-day gap before any big decision,
  then re-read it with fresh eyes.
- **You must name a stakeholder**, even if that is a specific person outside the
  project, and check the decision with them. A decision confirmed by nobody is
  an assumption.
- **You must schedule the reviews** from `.lead/06-MONITOR-AND-MAINTAIN.md`, or
  they will not happen.
- **You must stop at the gates**, even when nobody is watching. Gates are how a
  solo project stays honest.

If a solo developer is doing this, the risk is not technical failure. It is
quietly drifting for four months on a project nobody re-checked. The gates and
the written decisions are the defence.