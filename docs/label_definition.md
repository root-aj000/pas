# Label Definition, and How Time Is Handled

**Written:** 2026-10-04
**Covers:** Steps 1.4 and 1.5 of `.lead/01-DATA.md`.

`.lead/01-DATA.md` Step 1.4: *"This is the single most important definition in
the whole project, and the one most often skipped."* So it is written out in
full, including the parts that do not apply — because "does not apply" is only
worth anything when you can show why.

---

## Step 1.4 — The label

```text
LABEL:           satisfaction
DEFINITION:      The value in the `satisfaction` column of train.csv. It is
                 True or False. There is no derivation, no window, and no
                 joining: the column is given to us.
WINDOW:          None. The label describes the same flight as the features in
                 the same row. See below.
BASELINE RATE:   310,339 of 699,635 rows are True = 44.36%
                 389,296 of 699,635 rows are False = 55.64%
MEASURED ON:     2026-10-04, from data/train.csv
```

### Detail 1 — What exactly counts as the event?

**Known:** the column is `True` or `False`, with no blanks, no third value, and
no spelling variants.

**Not known:** what the passenger was actually asked. Every version of this
survey that circulates asks some form of "overall, how satisfied were you with
this flight?", but the exact wording, the scale it was answered on, and where
the cut between satisfied and dissatisfied sits are all unconfirmed. There is no
data dictionary file in `data/` — only the three CSVs.

This is not pedantry. The cut point sets the base rate, and the base rate sets
what a good score looks like. Open question 3.

### Detail 2 — When is it decided?

The label is recorded **after the flight**, when the passenger answers the
survey. That much follows from the column names and from the fact that 13 other
columns in the same row are also answers from that passenger.

What is not known is **how long after the flight**. If answers arrive over a
week, the newest rows have had less time to be collected. Nothing in the files
records a date, so this cannot be checked. Open question 4.

### Detail 3 — What time window does the prediction cover?

**None, and this is the important finding.**

`.lead/01-DATA.md` is written for churn: "does this customer leave within 30
days?" There, the label is about the *future* relative to the features, and
getting the window wrong quietly ruins everything.

This dataset has no such structure. Each row is one flight by one passenger. The
21 features and the label are all about **the same flight, at the same moment in
time**. So the window is zero, there is no horizon to get wrong, and the usual
30-day arithmetic does not apply.

### Detail 4 — What about rows that have not had time to become positive yet?

**Does not apply**, for the same reason. There is no "later" to wait for. Every
row in `train.csv` already has its answer.

The only version of this problem that could exist here is a delay between the
flight and the survey being answered, which is unknown (detail 2). If some rows
were answered much faster than others, and something about fast answers
correlates with satisfaction, that would be a bias we cannot see. Recorded as a
risk, not a finding.

---

## Step 1.5 — Time, honestly

`.lead/01-DATA.md` Step 1.5 is called "the hardest part of this file". Here is
what it actually amounts to for this dataset.

### Trap 1 — the answer arrives late

**The fix the file prescribes:** work out the label window, exclude rows whose
window has not closed, put the window in `config.yaml`, log it every run.

**What we can do:** nothing, because there is no date column in any file. There
is no "today", no row timestamp, and no way to order rows by when they
happened. A label window cannot be calculated, so there is nothing to exclude
and nothing to put in `config.yaml`.

**Why that is legitimate here rather than a shortcut:** the trap exists to stop
us training on rows whose answer does not exist yet. Every row in `train.csv`
ships with its answer already filled in. There are no unlabelled rows.

**What we cannot rule out:** rows whose answers were still arriving when the
file was generated. Nothing detects this. If it is a problem, we will not find
it from the data.

### Trap 2 — information from after the prediction moment

The question the file asks, for every column: *"At 09:00 on the day of the
prediction, did I know this value?"*

Applied to this dataset, with the full per-column answers in
[column_dictionary.md](column_dictionary.md):

| Column group | Known after the survey? | Known at boarding? | Verdict |
|---|---|---|---|
| `Age`, `Gender`, `Customer Type`, `Type of Travel`, `Class`, `Flight Distance` | Yes | Yes | Safe under both readings |
| `Departure Delay in Minutes` | Yes | Yes | Safe under both readings |
| `Arrival Delay in Minutes` | Yes — the flight has landed | **No** | Safe only under the after-survey reading |
| The 13 service ratings | Yes — same questionnaire as the label | **No** | Safe only under the after-survey reading |
| `satisfaction` | **This is the answer** | Not yet known | Must never be a feature |
| `id` | Yes | Yes | Must never be a feature. It is a row number |

**Under the after-survey reading, none of the 21 candidate features leaks.**
Each one is recorded at or before the moment the label is recorded, so the
model never sees a value written after the answer it is trying to predict.

**Under the at-boarding reading, 14 of the 21 features leak**, and the whole
task changes: only 7 features are usable.

### The honest summary of this section

There is no leakage problem to solve here. There is a **usefulness** problem,
and it is worth stating plainly rather than hiding behind a clean leakage table:

> Under the reading we are proceeding with, the model answers *"given what this
> passenger rated, were they satisfied?"* The passenger's own ratings are part
> of the input. So the model is not predicting a reaction — it is inferring one
> answer from the others on the same form.

That is a legitimate thing to model, and it is what the competition scores. It
is **not** a system that can warn a passenger, or an airline, that a flight is
going to go badly, because nothing is known until after the flight is over.

If the goal is the second thing, this dataset is the wrong dataset, and no
amount of modelling fixes it. That decision belongs to the owner of the
problem, and it is question 2 in [open_questions.md](open_questions.md).

### What goes into `config.yaml` as a result

| `.lead/01-DATA.md` asks for | Value here |
|---|---|
| Label window | None exists. Not applicable. |
| Observation period | None exists. Not applicable. |
| Rows to exclude for incomplete labels | None. Every row is labelled. |
| Prediction moment | After the survey is submitted. **Assumed — open question 2.** |

`.lead/01-DATA.md` Step 1.5 says to log the label window on every run. There is
no window, so there is nothing to log. Stating that here means the absence is a
decision on the record rather than an oversight nobody notices in six months.

---

## What has changed

Nothing. First version, written 2026-10-04. If the survey wording, the cut point
between `True` and `False`, or the prediction moment is ever confirmed, add a new
dated section here rather than editing the text above.