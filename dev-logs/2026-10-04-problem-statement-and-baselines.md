# 2026-10-04 — Complete the problem statement and measure the baselines

**Worked on:** [name]
**Type of change:** documentation, plus one new measurement script
**Files touched:**
- `docs/problem_statement.md` — completed Steps 0.2 to 0.5 and the Gate
- `docs/open_questions.md` — **new file.** It was linked from three documents
  and did not exist
- `research/02_baselines.py` — **new file.** Measures the two baselines
- `logs/dev/2026-10-04_2129_measure_baselines.log` — a failed run, kept
- `logs/dev/2026-10-04_2131_measure_baselines.log` — the successful run
**Status:** seven of ten Gate boxes met. **Blocked**, waiting on a person.

## What I did

I was asked to write the problem statement for "predicting airline satisfaction
for test.csv" into `docs/`, following `.lead/00-PROBLEM-AND-BASELINE.md`.

There was already a `docs/problem_statement.md` from earlier the same day. It
held the template with four boxes marked OPEN, and it referred to
`docs/open_questions.md` — which did not exist, along with `split_plan.md`. So
before writing anything I checked what was already there.

What I wrote:

1. **The two baselines `.lead/00` Step 0.4 asks for**, measured for real. The
   "do nothing" rule scores 0.5564 accuracy. The simple rule I picked scores
   0.7203. This closes two Gate boxes that were open.
2. **Steps 0.2, 0.3 and 0.5** of `.lead/00`, filled in as far as the data and my
   own judgement allow.
3. **`docs/open_questions.md`**, with the ten questions that are still unanswered
   and which gate each one blocks.

## Why

`.lead/00` says the baseline step is the most skipped step in the whole process,
and that skipping it means you cannot tell a good model from a lucky one. Until
those two numbers exist, every later score is unjudgeable — including a model
that scores worse than a single column.

Three other Gate boxes were open for one reason only: they need a person, not
more analysis. Filling them in myself would have meant inventing a business
context that does not exist. `.lead/00` Step 0.1 is explicit that this statement
is written *with the person who will use the model, not with the data*. So I
measured everything measurable, wrote down the reasoning that is genuinely mine
to make, and listed the rest as questions with names against them.

I also created `open_questions.md` because three documents pointed at it. A
dangling link means the next reader goes looking for a file that is not there,
which is the gap `.lead/README.md` warns about.

## How I checked it

- **Ran the script.** `python research/02_baselines.py`, exit 0, 3.9 seconds,
  699,635 rows read. Full output in
  `logs/dev/2026-10-04_2131_measure_baselines.log`.
- **The self-check failed on the first run, and it was right to.**
  `logs/dev/2026-10-04_2129_measure_baselines.log` has the traceback. My own
  comment claimed the test case should score 0.5 precision. It scores 2/3. The
  code was correct and my hand arithmetic in the comment was wrong. Fixed the
  comment, not the code. Worth noting because it is exactly the mistake the
  assertion exists to catch, and it caught it before the number reached a
  document.
- **Cross-checked the base rate.** 310,339 of 699,635 = 0.4436, which matches
  the 0.44357272006117476 constant already sitting in `sample_submission.csv`.
  Two independent paths, same number.
- **Checked every cross-reference.** All three documents that cite open question
  numbers now cite the same list, and the numbering in `open_questions.md`
  preserves the one `.lead/01` documents were already written against.

## What I did not do

- **I did not swap the registered baseline for the better rule.** I declared
  `Inflight entertainment == 5` before running, and printed all thirteen
  ratings. `Online boarding` is better (0.7203 against 0.6455). The pre-declared
  rule stays registered, because choosing a goal after seeing the scores is what
  Step 0.4 exists to prevent — but I wrote down that the honest bar is 0.7203, so
  nobody can use the weaker number as a target.
- **I did not write `docs/split_plan.md`**, though it is linked. It is
  `.lead/01` Step 1.7 work and depends on the metric being settled first.
  Recorded in `open_questions.md` under "not a question, but unfinished" so the
  gap is visible.
- **I did not invent an owner, a current practice, a cost of error, or a
  success target.** All four are OPEN, with proposed answers where I could
  honestly propose one.
- **I did not install pandas.** The venv has no scientific stack at all. Reading
  700,000 rows takes 3.9 seconds with the standard library, so nothing is gained
  by adding it before the pipeline exists.
- **I did not write a `requirements.txt`.** Nothing is imported yet.
- **I did not commit anything.** The rules say commit small single-purpose
  changes, and this is one — but the branch has no first commit and that is
  question 10 for the owner to settle.

## What I tried that did not work

- **Confirming the competition metric from the web.** Searched for the S6E10
  evaluation metric; Kaggle returned a JavaScript error on every page. I have
  evidence for ROC-AUC (the submission template wants continuous scores, filled
  with the base rate) but not proof, so it is written as a proposal and logged as
  question 6, not as a fact.

## Still open

- Four questions block the `.lead/00` gate: who acts on the output, whether
  after-the-fact prediction is acceptable, the survey cut point, and the metric.
  See `docs/open_questions.md`.
- **The risk nobody has costed:** if the goal was ever to predict satisfaction
  *before* a flight, 14 of the 21 features are only known after the survey is
  answered, and this dataset is the wrong one. That is the owner's answer to
  give, and it is the cheapest possible time to give it.