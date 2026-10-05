# Column Dictionary

**Written:** 2026-10-04
**Covers:** Step 1.3 of `.lead/01-DATA.md`.
**Measured by:** `python research/01_data_inventory.py`
**Rule for this file:** the "Known at prediction time?" column is never left
blank. That column is the leakage check from Step 1.5.

`train.csv` has 23 columns: 1 row number, 21 candidate features, and 1 answer.
`test.csv` has the same 22 minus the answer.

---

## The one thing to read before the table: when is the prediction made?

`.lead/01-DATA.md` Step 1.5 asks, for every column: *"At the moment of the
prediction, did I know this value?"* On this dataset the answer depends on a
decision nobody has made yet, so both answers are given.

**The survey.** Thirteen of the columns are ratings the passenger gives, and
`satisfaction` is the outcome the passenger reports. These almost certainly come
from **one questionnaire, filled in after the flight**. There is no column in
any file recording when a row was created, so we cannot confirm this — it is
open question 4.

This gives two possible prediction moments:

| | **After the survey** | **At boarding** |
|---|---|---|
| 13 service ratings | Known | **Not known** |
| `Arrival Delay in Minutes` | Known | **Not known** (flight has not landed) |
| `Departure Delay in Minutes` | Known | Known |
| `Flight Distance`, `Age`, `Gender`, `Customer Type`, `Type of Travel`, `Class` | Known | Known |
| `satisfaction` (the answer) | Not yet answered | Not yet answered |
| Rows usable as features | 21 | 7 |

**We are proceeding on the "after the survey" reading**, because it is the only
one under which the dataset can be learned at all, and because the task as
given is to predict `test.csv`, which is scored after the fact. Under that
reading the model answers: *given what this passenger rated, will they be
satisfied?* — which is a real question, but it is **not** "which flights will
produce unhappy passengers", and it cannot be used to warn anyone at boarding.

If the intended use is at boarding, 14 of the 21 features must be dropped and
the task changes shape. This is the most consequential open question in the
project. It is question 2 in [open_questions.md](open_questions.md).

---

## The dictionary

`n` is the number of rows holding each most-common value, out of 699,635.
"Rating" means the 0–5 scale described under the type column.

| Column | Meaning | Type | Units | Example | Known at prediction time? | Trust it? |
|---|---|---|---|---|---|---|
| `id` | Row number assigned by the organisers after shuffling. Not a passenger. | int | — | 0 | N/A — excluded from features. See below | Yes, as a row number. **Never a feature.** |
| `Age` | Age of the passenger | int | years | 36 | Yes, at booking, from the passenger's own details | Mostly. Range 7–85. See note 4 |
| `Flight Distance` | Length of the flight | int | **Unconfirmed** — miles or km? | 694 | Yes, from the booking | **Unsure.** See note 2. The units trap from Step 1.3 |
| `Inflight wifi service` | Rating of the wifi available on board | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | Value range yes. The meaning of 0 is unclear — note 3 |
| `Departure/Arrival time convenient` | Rating of how convenient the departure and arrival times were | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | As above |
| `Ease of Online booking` | Rating of how easy booking online was | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | As above |
| `Gate location` | Rating of the gate. **Unclear whether this is the gate's size or how easy it was to reach** | int | Rating 0–5 | 3 | After survey: yes. At boarding: **no** | **Unsure what it measures** — note 5 |
| `Food and drink` | Rating of the food and drink on board | int | Rating 0–5 | 1 | After survey: yes. At boarding: **no** | As above |
| `Online boarding` | Rating of online check-in and boarding | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | As above |
| `Seat comfort` | Rating of the seat | int | Rating 0–5 | 1 | After survey: yes. At boarding: **no** | As above |
| `Inflight entertainment` | Rating of the entertainment on board | int | Rating 0–5 | 1 | After survey: yes. At boarding: **no** | As above |
| `On-board service` | Rating of the cabin crew service | int | Rating 0–5 | 3 | After survey: yes. At boarding: **no** | As above |
| `Leg room service` | Rating of the leg room | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | As above |
| `Baggage handling` | Rating of how the baggage was handled | int | Rating 0–5 | 4 | After survey: yes. At boarding: **no** | As above. **The only rating column that never holds 0** — note 3 |
| `Checkin service` | Rating of the check-in service | int | Rating 0–5 | 3 | After survey: yes. At boarding: **no** | As above |
| `Cleanliness` | Rating of cleanliness. **Unclear whether this means the cabin or the aircraft** | int | Rating 0–5 | 1 | After survey: yes. At boarding: **no** | **Unsure what it measures** — note 5 |
| `Departure Delay in Minutes` | How late the flight left | int | minutes | 0 | Yes — known shortly before departure | Yes. 0–489, never negative |
| `Arrival Delay in Minutes` | How late the flight arrived | float | minutes | 0.0 | After landing: yes. At boarding: **no** | Yes, but **292 rows are blank** — note 1 |
| `Gender` | Passenger gender | text | — | Female | Yes, from the booking | Yes. Two values only: Female, Male |
| `Customer Type` | Whether the passenger is a loyal customer | text | — | Loyal Customer | Yes, from the loyalty programme | Yes, but note the spelling — note 6 |
| `Type of Travel` | Reason for the flight | text | — | Personal Travel | Yes, from the booking | Yes |
| `Class` | Travel class | text | — | Eco | Yes, from the booking | Yes, but very unevenly spread — note 7 |
| `satisfaction` | **The answer.** Whether the passenger reported being satisfied | text (True/False) | — | False | N/A — this is the label, never a feature | Yes. 44.36% `True`. See [label_definition.md](label_definition.md) |

---

## Notes on the columns that need a decision

### Note 1 — `Arrival Delay in Minutes` has 292 blank rows

| File | Blank rows | Share |
|---|---|---|
| `train.csv` | 292 | 0.042% |
| `test.csv` | 130 | 0.043% |

**What a blank means here is unknown.** Three readings, with different correct
answers:

1. The flight was on time, so no delay was recorded — the blank is really a 0.
2. The value was never captured — the blank is missing data.
3. The flight was cancelled — the blank is a different event entirely.

This matters because Step 1.6 requires the pipeline to stop on bad data, and we
cannot yet say whether these 292 rows are bad. Open question 3.

**Also a type quirk:** this column is written as a float (`0.0`) while
`Departure Delay in Minutes` is written as an integer (`0`). Both mean the same
kind of thing. Anything reading these columns must handle the difference rather
than assume it.

### Note 2 — `Flight Distance` has no units anywhere

This is the units trap from Step 1.3, happening for real. The column name does
not say. No file in `data/` says. The range is 67 to 4,983.

If the unit is miles, 4,983 miles is a long-haul route. If it is kilometres,
4,983 km is the same kind of trip. **A model does not care.** A human deciding
whether a route is plausible does, so we cannot check the column against
anything until this is answered. Open question 3.

### Note 3 — the ratings run 0 to 5, but the source dataset documented 1 to 5

Measured ranges:

| Rating column | min | max | Rows holding 0 | Rows holding 1 | Rows holding 5 |
|---|---|---|---|---|---|
| `Inflight wifi service` | 0 | 5 | 13,102 | 107,739 | 79,352 |
| `Departure/Arrival time convenient` | 0 | 5 | 25,751 | 95,072 | 156,933 |
| `Ease of Online booking` | 0 | 5 | 16,073 | 110,934 | 91,798 |
| `Gate location` | 0 | 5 | 1,455 | 104,611 | 92,504 |
| `Food and drink` | 0 | 5 | 1,014 | 70,923 | 155,387 |
| `Online boarding` | 0 | 5 | 9,147 | 57,588 | 152,870 |
| `Seat comfort` | 0 | 5 | 838 | 63,130 | 196,288 |
| `Inflight entertainment` | 0 | 5 | 1,020 | 70,131 | 182,068 |
| `On-board service` | 0 | 5 | 801 | 61,164 | 173,895 |
| `Leg room service` | 0 | 5 | 1,693 | 53,094 | 182,369 |
| `Baggage handling` | **1** | 5 | **0** | 30,460 | 202,673 |
| `Checkin service` | 0 | 5 | 929 | 66,468 | 147,550 |
| `Cleanliness` | 0 | 5 | 945 | 72,518 | 162,964 |

The original dataset described this scale as 1 (worst) to 5 (best). This file
contains 0 in twelve of the thirteen columns, and never in `Baggage handling`.

**So `0` is unexplained.** Two possibilities, and they need opposite handling:

- `0` is a valid lowest rating. Then it is real data and must be kept as 0.
- `0` means "not applicable" — wifi not offered on this short flight, no food
  served. Then it is a missing value in disguise, and treating it as a number
  teaches the model something false.

The `Baggage handling` column having no 0 at all is a clue: whatever produced
these files treated that column differently from the other twelve.

**Do not convert 0 to a missing value until this is answered.** It is the
cleanest example in this project of a decision that looks like data cleaning
and is actually a question for a person. Open question 3.

### Note 4 — `Age` goes down to 7

Range 7 to 85. A 7-year-old is possible on a family trip, so this is not
obviously an error. Noting it rather than "fixing" it.

### Note 5 — two column names do not say what they measure

`Gate location` and `Cleanliness` are ambiguous in the source dataset itself:

- `Gate location` — is a high score a big gate, or a gate close to the
  aircraft? These pull in opposite directions.
- `Cleanliness` — the cabin, the toilets, or the whole aircraft?

Both are usable as signals either way, so this does not block modelling. It
does block the sentence "the model found that clean cabins make passengers
happy", which is the kind of claim a model card is supposed to be able to make.
Open question 3.

### Note 6 — `Customer Type` has inconsistent capitalisation

The two values are `Loyal Customer` and `disloyal Customer` — lowercase "d" on
the second. Recorded because splitting or grouping on this string without
noticing will produce a third, empty group. 576,990 rows are `Loyal Customer`,
122,645 are `disloyal Customer`.

### Note 7 — `Class` is very unevenly spread

| Value | Rows | Share |
|---|---|---|
| Business | 342,212 | 48.9% |
| Eco | 327,404 | 46.8% |
| Eco Plus | 30,019 | 4.3% |

`Eco Plus` is 4.3% of the data. `.lead/04-EVALUATION-AND-GO-NO-GO.md` Step 4.5
requires results broken down by segment, so 30,019 rows is enough to measure
that segment separately — but it is small, and any score for `Eco Plus` will be
noisy. Noted now so it is not a surprise later.

---

## What has changed

Nothing. This is the first version of this file, written 2026-10-04.

Per the rule in `.lead/09-TEMPLATES.md`: a row is added the moment a column is
added, and if a definition changes, a new dated row is added rather than
editing the old one.