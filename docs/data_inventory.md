# Data Inventory

**Written:** 2026-10-04
**Covers:** Steps 1.1 and 1.2 of `.lead/01-DATA.md`.
**Measured by:** `python research/01_data_inventory.py`

Every number on this page was measured from the files in `data/`. Nothing is
estimated. Where something could not be measured, the row says so.

---

## Step 1.1 — Access, and what we are allowed to do

`.lead/01-DATA.md` Step 1.1 says to get these answers in writing **before**
writing any code, and that it is the most common reason projects lose weeks.
They are listed here with honest status.

| Question | Status | Answer / who to ask |
|---|---|---|
| Who owns this data? | **OPEN** | Kaggle, via the Playground Series S6E10 competition. Ask the project owner to confirm we may use it outside the competition. |
| What approval is needed? | **OPEN** | Unknown. Ask the project owner. |
| Can it leave our systems? | **OPEN** | The files are already downloaded to this machine, so they have. Confirm that is allowed. |
| How fresh is it? | Not applicable | It is a static snapshot of a past competition. It does not update. Nothing about the world is "current" here. |
| How is it updated? | Not applicable | One-off download. There is no feed and no refresh job. |
| What happens if the schema changes? | **OPEN** | It can only change if Kaggle changes the files. See the file hashes below — a different hash means different data. |
| Who do we ask when something is wrong? | **OPEN** | Not yet named. |

### Personal data

| Column | Personal? | Action |
|---|---|---|
| `Age` | Yes, weakly | A number in years. Not a name, not a contact detail. Keep. |
| `Gender` | Yes, weakly | Two values only. No names, addresses, phone numbers, health, or income anywhere in the file. |

No other column describes a person. The dataset is machine-generated (a
"playground" dataset is synthesised by the competition organisers), so it
contains no real passenger records. That lowers the privacy risk but does not
remove the need for the owner to confirm the licence — see
[open_questions.md](open_questions.md) question 1.

**Practical rules we are following:**

- No names or emails exist in this data, so none can leak into a feature table
  or a log.
- We log column names, counts and ranges only — never row contents.

### Gate within Step 1.1

> "You have written permission, or a named person has confirmed in writing that
> you do not need it."

**NOT MET.** No permission has been confirmed in writing. Recorded as
open question 1.

---

## Step 1.2 — Inventory of the data

There is **one** table. There are no joins to get wrong, and no second table
whose `id` might mean something different.

```text
| File                       | What it holds                     | Rows    | Columns | Contains people data? |
|----------------------------|-----------------------------------|---------|---------|-----------------------|
| data/train.csv             | Past flights, with the answer     | 699,635 | 23      | Weakly: Age, Gender  |
| data/test.csv              | Flights to predict, no answer     | 299,844 | 22      | Weakly: Age, Gender  |
| data/sample_submission.csv | Template for the answer           | 299,844 | 2       | No                    |
```

### Row count and growth

| Fact | Value |
|---|---|
| Rows in `train.csv` | 699,635 |
| Rows in `test.csv` | 299,844 |
| Total rows available | 999,479 |
| Share that is training data | 70.0% |
| Growth per day | None. Static files. |
| Last modified on disk | 2025-08-26 (from the filesystem) |

The 70 / 30 split is the organisers', not ours. They shuffled all 999,479 rows
and cut them. `id` confirms this: training ids run 0–699,634 and test ids run
699,635–999,478, with no overlap and no gaps.

### Primary key

| Fact | `train.csv` | `test.csv` |
|---|---|---|
| Column | `id` | `id` |
| Unique? | Yes — 699,635 distinct values, 0 duplicates | Yes — 299,844 distinct values, 0 duplicates |
| Counts upwards? | Yes, with no gaps | Yes, with no gaps |
| Range | 0 to 699,634 | 699,635 to 999,478 |
| Shared ids between the two files | 0 | |

**`id` is a row number, not a passenger.** Each row is one flight by one
passenger, and the same person can fly many times. There is no passenger
identifier anywhere in this data.

**Therefore `id` must not be used as a feature.** Two reasons:

1. It carries no information about satisfaction. It is a position in a
   shuffled list.
2. Every training id is smaller than every test id. A model that saw `id` would
   learn "small number means training row", which is an artefact of the
   organisers' cut, not a pattern in air travel.

### Join key

Not applicable. One table, no joins. The usual failure — two tables with a
`customer_id` that means two different things — cannot happen here.

### File fingerprints

If these hashes change, we are working with different data and every result so
far is void.

```text
2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36  data/train.csv
ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6  data/test.csv
6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71  data/sample_submission.csv
```

---

## What is missing from this page

`.lead/01-DATA.md` Step 1.2 asks how the data is *updated*. For a competition
download the honest answer is "never", and that has a consequence worth stating
plainly: **there is no fresh data arriving, so there is no monitoring problem
to solve and no retraining schedule to set.** `.lead/06-MONITOR-AND-MAINTAIN.md`
assumes a live feed. It does not apply to this project unless someone finds a
real flight survey feed later.

See [column_dictionary.md](column_dictionary.md) for what each column contains,
[label_definition.md](label_definition.md) for the answer we are predicting, and
[split_plan.md](split_plan.md) for how the rows are divided.