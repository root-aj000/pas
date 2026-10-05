# The Dev Log

**Every change gets written down. Every single one.**

Not just the big refactors. Not just the features. A renamed variable, a fixed
typo, a one-line comment, a deleted file, a reverted experiment. All of it.

---

## Why this exists

**Assume someone will read this in six months and check whether the work was
done properly.** Someone who was not here, does not know the project, and is
looking for a reason to trust it — or to find the mistake.

That person is usually one of:

| Reader | What they need from the log |
|---|---|
| **The team leader** | What changed, why, what is verified, what is still open |
| **A reviewer or auditor** | Was the work done properly, and can it be checked? |
| **A new developer** | Why is the code like this? What has already been tried? |
| **You, in three months** | Why did I do that, and did it work? |

A project without a dev log forces all four to guess. Guessing is how a codebase
becomes something nobody is willing to touch.

**And the practical case:** the log is how a leader tracks work without reading
every line of code. Ten minutes with the dev log tells you whether a project is
moving, what is stuck, and whether anything has gone wrong quietly.

---

## The two logs, and why they are separate

**They are different things for different readers. Never merge them.**

| | **Dev log** | **Run log** |
|---|---|---|
| File | `dev-logs/YYYY-MM-DD-short-name.md` | `logs/YYYY-MM-DD_HHMM_<what>.log` |
| Written by | A person, about their thinking | The code, about what it did |
| Language | Plain English, narrative | Timestamps, counts, exit codes |
| Audience | Leader, reviewer, auditor, future you | Whoever is debugging at 2am |
| Answers | *Why* was this done? What did we learn? | *What* happened, exactly? |
| Edited later? | Yes — add a correction with a date | **Never.** Append only |
| Detail | The reasoning, the dead ends, the doubts | Row counts, commands, timings |

**A concrete example of the difference:**

```text
DEV LOG (dev-logs/2024-06-12-fix-merge-duplicates.md)
"I found that the training set had 2.1 million rows instead of 21,000. The
 merge on customer_id was fanning out because the transactions table had two
 rows per customer. I added a uniqueness check before the merge, which now
 fails loudly instead of quietly inflating the data."

RUN LOG (logs/2024-06-12_0930_train.log)
"09:30:01 INFO [ingestion] rows_out=15231
 09:30:04 INFO [merge] duplicate customer_id found: 2 rows for 1_200_034
 09:30:04 ERROR [merge] merge would fan out 21,000 -> 2,100,000. Stopping.
 exit_code=1"
```

The dev log explains **why you were looking** and what you concluded. The run log
proves **exactly what the machine did**. Neither replaces the other.

---

## Where it lives

```text
dev-logs/
├── 2024-06-01-project-kickoff.md
├── 2024-06-03-define-churn-label.md
├── 2024-06-05-first-model-baseline.md
├── 2024-06-12-fix-merge-duplicating-rows.md
└── 2024-06-14-choose-gradient-boosting-over-random-forest.md
```

One file per **piece of work**, not one per day. A day with three unrelated
changes gets three small files. Keep them short — nobody reads 900 lines.

**Rules:**

- **Append, never rewrite history.** If you were wrong, add a new entry saying
  so. Deleting an entry is the one unforgivable thing in this folder.
- **Never delete an old log.** Move it to `dev-logs/archive/YYYY/` at most.
- **Write it the same day.** A log written a week later is a reconstruction, and
  the honest details are gone.
- **Plain text Markdown.** No images needed to understand it.

---

## The format

Copy this. Every entry has all six parts.

```markdown
# 2024-06-12 — Fix the duplicate rows in the training merge

**Worked on:** [name]
**Type of change:** bug fix
**Files touched:** src/components/data_cleaning_encoding.py, tests/components/test_data_cleaning_encoding.py
**Status:** done and verified

## What I did

The training set had 2.1 million rows instead of the expected 21 thousand. The
cause was a merge on `customer_id` against the transactions table, which has two
rows for some customers — so every one of those customers was duplicated.

I added a uniqueness check before the merge. It now stops the pipeline with a
clear message naming the duplicated IDs, instead of silently producing an
inflated dataset.

## Why

A duplicated row makes a customer look more active than they are, so the model
over-predicts churn for anyone with multiple purchases. This was quietly
inflating the training data and would have made every score meaningless.

The check was missing because the merge assumed `customer_id` was unique, and
that assumption was never written down.

## How I checked it

- Ran the pipeline: it now stops with a clear message instead of training.
- Ran the fixed merge on real data: 21,000 rows, 0 duplicates.
- Added a test that fails when `customer_id` is not unique.
- Compared the model score before and after: recall 0.41 -> 0.63. The old score
  was an artifact of the duplicates.

## What I did not do

- I did not deduplicate the data. The duplicates are real — customers can have
  two open subscriptions. Aggregating them properly is a separate piece of work.
- I did not investigate the 1,200 customers with duplicate rows. Noted as an
  open question in `docs/open_questions.md`.

## Still open

- Why do 1,200 customers have two open subscriptions? Ask the data team.
- Should the uniqueness check be on every table, not just this one?
```

---

## The six parts, explained

| Part | Why it is mandatory |
|---|---|
| **Header** — date, name, type, files, status | The leader reads only this. It must be enough to know what happened without opening the file |
| **What I did** | Plain English. No code, no jargon. A non-technical person must understand it |
| **Why** | The most valuable part, and the one always skipped. Six months later this is the only thing that cannot be re-derived |
| **How I checked it** | Proves the work is real. What you ran, what you observed, what number changed |
| **What I did not do** | Stops the same thing being attempted twice, and surfaces scope decisions |
| **Still open** | Hands over cleanly. Nothing dies quietly in a log |

---

## What counts as "a change"

**All of it. There is no threshold.**

| Change | Log it? |
|---|---|
| New feature or stage | Yes |
| Bug fix | Yes |
| Renamed a variable or file | Yes |
| Deleted dead code | Yes |
| Changed a threshold or hyperparameter | Yes |
| Fixed a typo in a docstring | Yes |
| Added a test | Yes |
| Installed or removed a dependency | Yes |
| Updated a dependency version | Yes |
| Changed a config default | Yes |
| Reverted a change | Yes, saying what you reverted and why |
| Tried an approach that failed | **Yes — especially this** |
| Changed a file's location or name | Yes |
| Added or removed a log line | Not on its own — it is part of the change it belongs to |
| Fixed formatting only (`ruff format`) | No — unless it touched logic |

**A one-line fix gets a three-line log.** That is the whole point. The cost is
seconds; the value is that nothing is unexplained.

### Failed attempts are the most valuable entries

```markdown
## What I tried that did not work

- **Adding a churn reason column as a feature.** Recall went DOWN (0.63 ->
  0.51). Cause: the reason is recorded when the customer cancels, so it is
  leakage. Dropped. See `.lead/01-DATA.md` Step 1.5.
- **SMOTE oversampling.** Recall rose to 0.71, but precision collapsed to 0.04,
  so the retention team would have made 18,000 calls a month. Reverted in favour
  of `class_weight="balanced"` plus a higher threshold.
- **Feast.** Set up correctly, but we only have one feature table, so it adds a
  whole service for no benefit yet. Left configured, unused. Revisit when we
  have a second table.
```

Nobody remembers what was tried. Without this, the next person tries the same
thing, wastes a week, and probably concludes the approach is impossible.

---

## Rules that make the log trustworthy

1. **Write it the same day.** Tomorrow you will have forgotten the doubt that
   mattered.
2. **One file per piece of work.** Not one per day, not one per week.
3. **Never edit an old entry to make it look better.** Add a correction, dated.
4. **Plain English.** Assume the reader does not write code.
5. **Record numbers with units and dates.** "recall 0.63 on the 2024-06 test set"
   is useful six months later. "recall improved" is not.
6. **Name the files you touched.** So a reviewer can check your work in minutes.
7. **Say what you did not do.** The most-skipped section, and the one that stops
   duplicated work.
8. **Link to the rule or doc that made you do it.** "Per `.dev/RULES.md` rule 8"
   is how a reviewer knows you were following the standard, not improvising.
9. **Honest status.** `done`, `in progress`, `blocked`, `abandoned`. A log that
   only contains successes is a sign of a log that is not being written honestly.
10. **No secrets, ever.** No passwords, keys, tokens, customer names, or personal
    data. Reference the environment variable name instead.

---

## What the leader does with it

**A ten-minute weekly read.** This is the whole point of writing it.

```text
WEEKLY, [day]:

1. Skim the headers of this week's files. That alone tells you what moved.
2. Look at every entry whose status is not "done".
3. Look at "Still open" and "What I did not do".
4. Notice anything attempted three times. That is the signal to change approach,
   not to try harder.
5. Notice anything that is quiet. No entries for a week on an active project
   means the log is not being written.
```

Four questions a leader should be able to answer from the log alone:

| Question | Where the answer is |
|---|---|
| What did we do this week? | The headers |
| Why did we do it? | "Why" in each entry |
| What is stuck, and on whom? | "Still open" + status |
| What have we already tried? | "What I tried that did not work" |

If any of those four needs a meeting to answer, the log is not doing its job.

---

## When someone reviews the code

**Assume it will happen, and assume they will start with this folder.**

A reviewer's first three questions are always:

1. Does the code do what the dev log says it does?
2. Are the claims in the log verifiable? ("recall went up" — where is that
   recorded?)
3. Was anything changed without being logged?

`How I checked it` is what makes a log entry verifiable. "Tested and works" is
not a check. "Ran `pytest tests/`, 14 passed; ran the pipeline, 21,000 rows out"
is.

**If the code and the log disagree, the log is updated — not deleted.** Add a
dated correction saying what the code actually does.

---

## Self-check before you commit

Run through this. It takes twenty seconds.

- [ ] Is there a `dev-logs/` entry for this change?
- [ ] Does the header say what changed, in plain English?
- [ ] Have I written **why**, not just what?
- [ ] Have I written how I checked it, with actual commands and numbers?
- [ ] Have I written what I did **not** do?
- [ ] Is there anything still open?
- [ ] No secrets or personal data in the text?
- [ ] Have I **not** edited or deleted an older entry?

If the answer to the first one is no, the change is not finished.