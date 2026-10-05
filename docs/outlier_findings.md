# Outlier Findings

**Written:** 2026-10-04
**Question asked:** are there outliers, and have we removed them or plan to?
**Answer:** there are extremes, **none of them are errors, and no, we are not
removing them.** The decision is measured, not judged.
**Evidence:** Phase 2e of [`research/02_eda.ipynb`](../research/02_eda.ipynb),
which was executed to produce every number on this page.
Run log: `logs/dev/2026-10-04_*.log`.

---

## The short version

| Question | Answer |
|---|---|
| How many outliers? | Depends on the rule. **0 rows are outside a plausible range.** 16,175 are beyond the IQR fence on `Flight Distance` and are real long-haul flights |
| Are they data-entry errors? | **No.** Their behaviour is internally consistent and moves the target in the direction reality predicts |
| Have we removed them? | **No** |
| Will we? | **No.** Clipping and dropping were both fitted and scored. Neither helped. Change is in the 4th decimal place |

---

## 1. How many — and why the answer depends on the rule

The most common outlier rule is "beyond 1.5 × the interquartile range". **On two
of the four continuous columns that rule is invalid**, and applying it blindly
would have produced a nonsense finding.

| Column | min | max | Share exactly 0 | IQR | IQR rule valid? | Beyond IQR fence | Outside domain bounds |
|---|---|---|---|---|---|---|---|
| `Age` | 7 | 85 | 0.0000 | 23.0 | yes | 2 | **0** |
| `Flight Distance` | 67 | 4,983 | 0.0000 | 1,181.0 | yes | 16,175 | **0** |
| `Departure Delay in Minutes` | 0 | 489 | **0.9080** | **0.0** | **no** | not meaningful | **0** |
| `Arrival Delay in Minutes` | 0 | 491 | **0.9155** | **0.0** | **no** | not meaningful | **0** |

**Why the rule breaks.** 90.8% of departures and 91.6% of arrivals are exactly
zero minutes late. That makes the interquartile range **zero**, and a fence of
`Q3 + 1.5 × 0` equals `Q3`, which equals zero. The rule would therefore declare
**all 64,483 delayed departures to be outliers.** That is not a finding about the
data. It is a property of the rule meeting a column where three quarters of the
values are identical.

This is the same class of mistake as `.lead/01-DATA.md` Step 1.3's units trap: a
number that is correct in isolation and meaningless in context.

**So each column gets a rule that fits it** — the IQR fence where the IQR is
non-zero, and a *domain bound* everywhere else. A domain bound is the widest
value that is still a real flight on Earth.

| Column | Domain bound | Rows outside |
|---|---|---|
| `Age` | 0 to 120 | 0 |
| `Flight Distance` | 1 to 20,000 | 0 |
| `Departure Delay in Minutes` | 0 to 1,440 (24 h) | 0 |
| `Arrival Delay in Minutes` | 0 to 1,440 (24 h) | 0 |

**Zero rows in the entire dataset fall outside a plausible range.** A flight
489 minutes late is a bad day, not a typo.

`Flight Distance`'s bound is deliberately loose because **no unit is documented
anywhere** — miles or kilometres, both pass. That gap is
[`column_dictionary.md`](column_dictionary.md) Note 2, still open.

### Columns where an outlier is impossible

| Group | Why |
|---|---|
| The 13 service ratings | Bounded 0–5 by definition. `Baggage handling` never holds a 0, which is the unexplained quirk in [`column_dictionary.md`](column_dictionary.md) Note 3 |
| The 4 categorical columns | 2 to 3 values each: `Gender`, `Customer Type`, `Type of Travel`, `Class` |

---

## 2. Are they real records, or errors?

A data-entry error usually breaks a row's internal consistency. A flight delayed
by eight hours that the passenger then rated 5 out of 5 on every service is not a
real record. So this looks at whether the extreme rows behave like the rest of
the file.

| Extreme 0.1% | n | Satisfied | Ordinary rows | Whole train |
|---|---|---|---|---|
| `Age` ≥ 70 | 2,156 | **0.1071** | 0.4464 | 0.4436 |
| `Flight Distance` ≥ 3,995 | 505 | **0.7366** | 0.4436 | 0.4436 |
| `Departure Delay` ≥ 52 min | 549 | **0.3060** | 0.3994 | 0.4436 |
| `Arrival Delay` ≥ 49 min | 518 | **0.2857** | 0.3528 | 0.4436 |

**They are real, and the delays are the proof.** The most delayed flights are
satisfied at 0.306 and 0.286 — **below** the ordinary rows. Longer delay, less
satisfaction, which is the direction customer service predicts. If those values
were entry errors the relationship would be random, or inverted. It is neither.

The oldest 0.1% are satisfied at 0.107 against 0.446 for everyone else, and the
longest 0.1% of flights at 0.737. Both are strong, consistent signals — which is
the fourth reason not to throw them away.

---

## 3. The measurement that decides it

Calling extreme values "real" is an opinion until a model has been fitted both
ways. `HistGradientBoostingClassifier`, default settings, seed 42, trained on
489,743 rows and scored on the 104,946-row validation split.

| Version | Accuracy | ROC-AUC |
|---|---|---|
| extremes kept (raw) | 0.924218 | 0.957312 |
| extremes clipped at 0.1 / 99.9 percentile | 0.923989 | 0.957387 |
| extreme rows dropped by hard caps | 0.924361 | 0.957398 |

Clipping changes accuracy by **−0.000229** and ROC-AUC by **+0.000076**. Fourth
decimal place. Noise.

**A caveat on the third row, so it is not over-read:** those hard caps
(`Age` ≤ 100, `Flight Distance` ≤ 6,000, delays ≤ 300) removed only **3 rows**,
because the real maxima are 85, 4,983, 489 and 491. That row is therefore *not* a
fair test of dropping extremes — there was almost nothing to drop. The decision
rests on the first two rows, where clipping moves 0.1% of every column.

---

## 4. The decision

> **Keep the extremes exactly as they are. No clipping, no deletion, no winsorising,
> no imputation of the extremes.**

| # | Reason | Weight |
|---|---|---|
| 1 | **It is measured.** Clipping was fitted and scored. It did not help | Decisive |
| 2 | **Tree models do not care.** `HistGradientBoosting` splits on order, not distance, so an extreme value changes which bucket a row falls into and nothing else. Scaling the input changes nothing either — that is a property of trees, and one reason they are the right family for this data | Strong |
| 3 | **Clipping would invent a ceiling the world does not have.** A flight delayed 400 minutes happened. Capping at 52, because that is the 99.9th percentile of *this sample*, teaches the model something false | Strong |
| 4 | **The extremes are among the strongest signals in the data.** The oldest 0.1% sit at 0.107 satisfaction and the longest flights at 0.737 | Supporting |

`.dev/EDA.md` says never delete an outlier silently. Nothing has been deleted,
and this page is the record of why.

---

## 5. What this does *not* cover

| Item | Why it is separate |
|---|---|
| **The 204 blank arrival delays** | A missing-value question, not an outlier question. Still open — [`column_dictionary.md`](column_dictionary.md) Note 1. They are kept as their own `unknown` group, not filled in |
| **The unexplained `0` ratings** | `0` may be a valid lowest rating or a hidden "not applicable". Opposite handling. Open — [`column_dictionary.md`](column_dictionary.md) Note 3 |
| **Duplicate rows** | Not checked. `id` is unique in the train split, which means there are no duplicate *ids*, but two identical flights with different ids would not be caught. Not checked because the data is synthetic and each row is generated independently |

---

## What has changed

Nothing. First version, written 2026-10-04.

An earlier draft of this page was going to be backed by a standalone script,
`research/03_outliers.py`. That was wrong: `.dev/EDA.md` keeps the exploration in
the numbered notebooks, and splitting one question across two files means two
places to keep in sync. The code was moved into Phase 2e of the EDA notebook and
the script deleted in the same sitting.