# Coding Rules

These rules apply to every file we write. They exist so the code stays easy to
read, easy to fix, and easy to hand over.

The test for everything below: **would a new team member understand this on day
one?** If not, it is not finished.

---

## 1. Readability comes first

Readable code is worth more than short code.

```python
# BAD - clever, nobody knows what this does
d = [x["t"] for x in r if x["c"] == 1 and x["v"] > 0.5]

# GOOD - says exactly what it means
recent_large_transactions = [
    row for row in records if row["category"] == 1 and row["amount"] > 0.5
]
```

Rules:

- Use long, descriptive names. `total_price` beats `tp`.
- One idea per line. Do not chain four operations with semicolons.
- Prefer the simple, obvious way to write something over the compact way.
- If a line needs a comment to explain the syntax, rewrite the syntax.

---

## 2. No hacks

A hack is any code that works but nobody fully understands.

Not allowed:

- Regex that "somehow" matches, with no explanation of the pattern.
- Copy-paste where only one small thing was meant to change.
- Magic numbers and magic strings with no name.
- Ordering that only works by luck (loop over a dict, index math).
- Anything with `# just leave it` or `# TODO fix later` left in it.
- Catching an error and printing nothing.

Why: hacks are cheap to write and expensive to debug. At 3am nobody wants to
be reverse-engineering a regex.

If something cannot be written cleanly, the design is wrong. Fix the design.

---

## 3. No DIY (Do It Yourself)

**If a library already does it, use the library. Never rebuild it.**

| Do not hand-roll | Use |
|---|---|
| Date parsing and timezones | `datetime` / `pandas.to_datetime` |
| HTTP requests | `requests` / `httpx` |
| YAML / JSON reading | `pyyaml` / `json` |
| Random numbers | `random` / `numpy.random` |
| Logging | `logging` |
| Command line arguments | `argparse` |
| Testing | `pytest` |
| Plotting | `matplotlib` / `seaborn` |
| Config files | `pyyaml` |
| Data structures | standard `list`, `dict`, `set` |

The same rule applies inside the ML stack: use MLflow for tracking, Feast for
features, Great Expectations for data checks, Evidently for drift, Optuna for
tuning. See [TOOLS.md](TOOLS.md).

A hand-rolled version is always more code, more bugs and less trusted than the
library everyone already uses.

---

## 4. Document everything

The code says *what*. The documentation says *why*.

Every file starts with a short header:

```python
"""
Removes duplicate customer records from the raw signup feed.

Input:  data/raw/signups_2024_01.csv
Output: data/interim/signups_clean.csv, one row per customer_id

Run with:  python -m src.pipeline.stage_02_data_cleaning_encoding
"""
```

Every function has:

1. **A name that is a sentence.** `calculate_average_spend_per_visit()`,
   not `calc_aspv()`.
2. **A short docstring** saying what it takes in, what it gives back, and any
   gotcha.
3. **Type hints** on inputs and output, so tools can check it.

```python
def calculate_average_spend_per_visit(total_spend: float, visits: int) -> float:
    """Return average spend per visit.

    Args:
        total_spend: Total money spent by the customer, in dollars.
        visits: Number of visits. Must be greater than zero.

    Returns:
        Average spend per visit, in dollars.

    Raises:
        ValueError: If visits is zero or negative.
    """
    if visits <= 0:
        raise ValueError("visits must be greater than zero")
    return total_spend / visits
```

Inside the code, comment only the things the code cannot say itself:

```python
# Cap at 10 because values above 10 are sensor errors, confirmed with the vendor.
```

Never write a comment that just repeats the code. `# increment i` above `i += 1`
is noise. Delete it.

Keep documentation next to the code. A separate Word file about the code is
always out of date. Never write documentation for code that no longer exists —
delete the code or fix the doc, same day.

---

## 5. Make it debuggable

You must be able to find any bug by reading the code and the logs, without
guessing. Full guide: [DEBUGGING.md](DEBUGGING.md).

- **Small functions.** One job each. If a function is longer than about 50
  lines, split it.
- **No hidden state.** A function should not depend on something that changed
  somewhere else. Pass values in, return values out.
- **Log every step.** Row counts and column names in and out. If you cannot see
  the dataflow, you cannot fix it.
- **Validate at the edges.** Check input data once, at the start, and fail with a
  message that says exactly what is wrong.
- **Fail loudly.** Never swallow an error.

```python
# BAD - the error disappears and you find out three days later
try:
    value = float(row["amount"])
except ValueError:
    pass

# GOOD - stops immediately and says which row is broken
try:
    value = float(row["amount"])
except ValueError as error:
    raise ValueError(
        f"Row {row_id}: amount is not a number: {row['amount']!r}"
    ) from error
```

- **Use logging, not print.** Every step logs what it took in, what it did and
  what it produced. Details and shared helpers are in
  [DEBUGGING.md](DEBUGGING.md).
- **Never assume, always check.** If a function needs a specific column, it
  checks for it and stops with a clear message. Never assume a row count, a
  column name, or a unit. Never read hidden global state — pass everything in.
- **Ask when it is unclear.** If a column's meaning, units or edge cases are
  unclear, ask the person who owns the data. Write the answer in the docstring
  so nobody asks twice.
- **Error messages must be actionable.** Say what failed, on which record, with
  the value, and what to do about it.

---

## 6. Simple always wins

- The simplest version that works is the correct version.
- Solve the problem you have today, not the one you might have next year.
- No settings, flags or options "for later". Add them when there is a second real
  case.
- Delete dead code. Git remembers it if we need it back.
- Prefer three clear lines over one clever line. Every time.

---

## 7. Don't overengineer

Not allowed unless there is a proven need:

- Custom frameworks or base classes with only one implementation.
- Plugin systems, registries, or factories for a single product.
- Config files holding values that never change.
- Generic "utils" modules full of unrelated helpers. If a helper belongs to one
  feature, keep it inside that feature.
- Wrappers around libraries that only forward calls and add nothing.
- Deep folder trees. Four levels of nesting is already too deep.

Rule of thumb: if you are writing something "so it will be easy to extend
later", delete it. Extend it later, when you know how.

---

## 8. Configuration and secrets

- Secrets never go in code or in git. Read them from environment variables.
- **Hyperparameters never live inside the training code.** Learning rate,
  dropout, layer counts, epochs and batch size belong in `config.yaml`. They
  are things we tune and compare, not facts. See
  [CODE-STANDARDS.md](CODE-STANDARDS.md).
- Values that change per machine or per run (paths, model version, thresholds)
  live in that same config file, not scattered around.
- Anything else stays a plain, visible constant next to the code that uses it.
  Do not move a constant into a config file just to be tidy.
- Log the config values with every run, so a result can be reproduced later.

---

## 9. Dependencies

- Every dependency is in `requirements.txt`, with a pinned version.
- Before adding a library, check whether the standard library or an existing
  dependency already does it. See rule 3.
- Never use two libraries for the same job.
- If we drop a library, remove it from the requirements file the same day.

---

## 10. Tests

Machine learning code can be broken in three different places: the logic, the
data, or the model itself. So we test all three. Ordinary rules first, then the
ML-specific tests.

### Ordinary rules

Anything with a branch, a loop, a calculation or a data transformation gets one
small test that proves it works. Trivial one-liners do not need tests.

- Test files live in `tests/`, named `test_<file>.py`.
- Test names say what is being checked: `test_raises_when_visits_is_zero`.
- Tests use `pytest`.
- Tests are deterministic. No random data unless you seed the random generator.
- Test data is small and readable. Put a 5-row example inline, in a real table
  you can read.

### Test 1 — unit tests: is each piece correct?

Test one function on purpose-built small input. Check the value it returns, and
check that it fails properly on bad input.

```python
def test_average_spend_per_visit():
    assert average_spend_per_visit(total_spend=100.0, visits=4) == 25.0


def test_raises_when_visits_is_zero():
    with pytest.raises(ValueError, match="greater than zero"):
        average_spend_per_visit(total_spend=100.0, visits=0)
```

For a model layer, also check the **shape** comes out the size you expect. A
wrong shape is the most common deep learning bug.

### Test 2 — the overfitting test: can the model learn at all?

This is the ML test with no ordinary equivalent. If a model cannot drive the
loss to near zero on 16 rows of data, something is broken — usually the
gradients are not flowing, or the learning rate is wrong. No amount of real data
will hide it.

```python
def test_model_can_overfit_tiny_dataset():
    """The model must be able to memorise 16 rows. If it cannot, training is broken."""
    tiny_features, tiny_labels = make_tiny_dataset(rows=16)
    model = build_model(input_size=tiny_features.shape[1])

    train(model, tiny_features, tiny_labels, epochs=200)

    assert model.predict(tiny_features) == tiny_labels
```

Run it as part of the test suite. It takes seconds and it catches the worst
class of bug in machine learning.

### Test 3 — data validation: is the incoming data shaped right?

Before any pipeline runs, check the data has the structure the code expects:
required columns present, right types, no impossible values, freshness.

Write these checks as code that fails loudly (see
[DEBUGGING.md](DEBUGGING.md)). Great Expectations is the right tool when you have
many tables to check — see [TOOLS.md](TOOLS.md). For a few columns, plain
`check_columns()` plus a few `raise ValueError` lines are enough. Do not add a
tool for something five lines can do.

---

## 11. Log every change, and never delete a log

**Assume someone will read this work in six months and check whether it was done
properly.** They will not have your memory. The log is what they get.

Two logs, two audiences, never merged:

| | Dev log | Run log |
|---|---|---|
| Where | `dev-logs/YYYY-MM-DD-name.md` | `logs/YYYY-MM-DD_HHMM_what.log` |
| Written by | You, about your reasoning | The code, about what it did |
| Answers | *Why* did we do this? What did we learn? | *What* happened, exactly? |
| Read by | Leader, reviewer, auditor, future you | Whoever is debugging |

The full format and the rules are in [`DEV-LOG.md`](DEV-LOG.md) and
[`DEBUGGING.md`](DEBUGGING.md) section 9.

Rules:

1. **Every change gets a dev log entry.** There is no size threshold. A renamed
   variable gets a three-line entry. A failed experiment gets an entry saying it
   failed and why.
2. **Write it the same day.** A week later it is a reconstruction.
3. **Never edit or delete an old entry.** Add a dated correction instead.
   Deleting history is the one unforgivable thing here.
4. **Every run and every test is logged** — including failures, with exit codes.
5. **Include the git commit hash and a hash of `config.yaml` in every run log.**
   Without them, a result cannot be checked against the code that produced it.
6. **The main run command takes no arguments.** One command, everything
   configured in `config.yaml`. Arguments are allowed for tests and development
   commands, and those logs stay in `logs/dev/` — never deleted.
7. **When code and log disagree, the code is the truth.** Correct the log with a
   dated note.

Full format, and the ten-minute weekly read for a leader, in
[`DEV-LOG.md`](DEV-LOG.md).

---

## 12. Version control

- Commit small, single-purpose changes with a clear message.
- Each commit has a matching `dev-logs/` entry.
- Never commit `venv/`, model binaries, secrets or large data files. Commit run
  logs and dev logs — they are the project's history.
- Never force-push shared branches.
- If you delete code, delete it in the same commit that stops using it.

---

## Quick self-review before you ask for review

- [ ] Could a new team member read this and know what it does?
- [ ] Is there any line I cannot explain out loud?
- [ ] Does every function have a name that reads like a sentence?
- [ ] Does every function have type hints?
- [ ] Did I use a library instead of writing it myself?
- [ ] Does every file and function explain *why* it exists?
- [ ] Does every step log what it took in and what it produced?
- [ ] Does the code check the data it needs instead of assuming it?
- [ ] Are all hyperparameters in `config.yaml`, not in the training code?
- [ ] Is the random seed set and logged?
- [ ] Does every run log the git commit and the config hash?
- [ ] Is there a `dev-logs/` entry for this change, with why and how I checked it?
- [ ] Does the main run command take no arguments?
- [ ] Does bad input stop the program with a clear message?
- [ ] Is there a test for the non-trivial logic?
- [ ] Does the model pass the overfitting test?
- [ ] Does `ruff check .` pass?
- [ ] Did I delete anything that was no longer needed?
- [ ] Could any of this be simpler?