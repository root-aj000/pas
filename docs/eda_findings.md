# EDA Findings — Complete

**Written:** 2026-10-04
**Source:** [`research/02_eda.ipynb`](../research/02_eda.ipynb), executed end to end, seed 42.
**Covers:** every finding from every phase. One page, so nothing has to be
re-derived by reading code.

**How to reproduce:** run the notebook top to bottom. Every number below comes out
of it.

**Related:** [`outlier_findings.md`](outlier_findings.md) is the long-form treatment
of outliers. [`method_plan.md`](method_plan.md) turns these findings into a method
choice.

---

## The headline, in five lines

1. **The bar is `Class == Business` at 0.7770 accuracy / 0.7786 AUC.** One column, one boolean condition, no machine learning.
2. **An untuned `HistGradientBoosting` already reaches 0.9242 accuracy / 0.9573 AUC** on validation.
3. **Cabin class matters more than any service rating.** Business class is satisfied at 4.3× the rate of Eco.
4. **The 13 service ratings dominate everything else**, and `Online boarding` dominates them.
5. **Nothing was removed.** No outliers, no rows, no columns. Everything below is a measurement, not a preference.

---

## 1. Shape, split and quality

| Fact | Value |
|---|---|
| `train.csv` | 699,635 rows, 23 columns |
| `test.csv` | 299,844 rows, 22 columns (no label) |
| Train / validation / test split | 489,743 / 104,946 / 104,946 |
| Base rate, identical in all three | **0.4436 satisfied** |
| Seed | 42 |
| Missing values, train split | 204, all in `Arrival Delay in Minutes` |
| Duplicate ids | 0 |
| Rows outside a plausible domain range | **0** |

### The eight quality checks, all passing

| Check | Result |
|---|---|
| Required columns present | all 21 candidates present |
| Row count as expected | 489,743 vs ~489,744 expected (1% tolerance) |
| Service ratings within 0–5 | all within range |
| `id` unique | 489,743 distinct of 489,743 |
| No column more than 1% blank | none over the limit |
| Blank columns recorded for review | `Arrival Delay in Minutes`: 204 |
| No future timestamps | **not applicable** — no date column exists |
| Label rate stable across the file | spread 0.0054 across 10 id-deciles |

### Two structural facts that shape everything downstream

| Fact | Consequence |
|---|---|
| **No date column anywhere** | No time axis. No time-based split, no label window, no drift monitoring, no retraining schedule. `.lead/01-DATA.md` Step 1.5 has nothing to apply to |
| **No passenger identifier** | Each row is an independent flight. No entity spans rows, so a stratified random split is correct rather than a shortcut |

---

## 2. Distributions

### The 13 service ratings are left-skewed

Passengers give 4s and 5s far more than 0s. Counts on the train split:

| Rating | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Online boarding | 6,437 | 40,224 | 83,582 | 104,807 | 147,600 | **107,093** |
| Inflight entertainment | 719 | 49,142 | 79,234 | 85,919 | 147,091 | 127,638 |
| Seat comfort | 592 | 44,320 | 64,487 | 84,005 | 158,733 | 137,606 |
| Inflight wifi service | 9,131 | 75,376 | 130,948 | 129,092 | 89,575 | 55,621 |
| Departure/Arrival time convenient | 18,018 | 66,585 | 83,276 | 85,182 | 126,899 | **109,783** |
| Ease of Online booking | 11,246 | 77,734 | 122,432 | 122,760 | 91,283 | 64,288 |
| Gate location | 1,009 | 73,277 | 91,348 | 141,548 | 117,947 | 64,614 |
| Food and drink | 729 | 49,782 | 104,013 | 107,767 | 118,433 | 109,019 |
| On-board service | 558 | 42,809 | 60,991 | 105,540 | 158,014 | 121,831 |
| Leg room service | 1,202 | 37,136 | 88,676 | 92,252 | 142,680 | 127,797 |
| Baggage handling | **0** | 21,288 | 40,988 | 91,397 | 194,238 | 141,832 |
| Checkin service | 672 | 46,304 | 52,517 | 140,194 | 146,724 | 103,332 |
| Cleanliness | 652 | 50,864 | 71,664 | 118,691 | 133,636 | 114,236 |

`Baggage handling` never holds a 0; twelve columns do. If `0` means "not
applicable" rather than "worst possible", twelve columns carry a hidden missing
value and treating it as a number teaches the model something false.
[`column_dictionary.md`](column_dictionary.md) Note 3 — **open, and it affects the
feature list.**

### The continuous columns

| Column | mean | median | min | max | Shape |
|---|---|---|---|---|---|
| `Age` | 39.01 | 40 | 7 | 85 | Roughly symmetric. 2 rows beyond the IQR fence — nothing |
| `Flight Distance` | 1,352.9 | 954 | 67 | 4,983 | **Right-skewed.** 16,175 rows beyond the IQR fence, all real long-haul routes |
| `Departure Delay in Minutes` | 1.17 | 0 | 0 | 489 | **90.8% exactly 0.** IQR is zero |
| `Arrival Delay in Minutes` | 1.13 | 0 | 0 | 491 | **91.55% exactly 0.** IQR is zero |

### The categoricals — one prediction was wrong

| Column | Values and shares |
|---|---|
| `Gender` | Male 50.31%, Female 49.69% |
| `Customer Type` | Loyal 82.45%, **disloyal 17.55%** |
| `Type of Travel` | Business travel 71.12%, Personal Travel 28.88% |
| `Class` | **Business 48.87%**, Eco 46.84%, Eco Plus 4.29% |

> **Expectation written before looking: "`Eco` should dominate `Class` — it is the
> cheapest cabin."** Wrong. Business is the largest class at 48.87%, and it is also
> by far the most satisfied.

---

## 3. Relationships

### Every feature against the target

| Feature | Spearman | Pearson | Reading |
|---|---|---|---|
| **`Online boarding`** | **0.6028** | 0.5603 | Strongest rating by a wide margin |
| `Inflight entertainment` | 0.4323 | 0.4301 | Second |
| `Seat comfort` | 0.4131 | 0.4017 | Third |
| `On-board service` | 0.3539 | 0.3465 | |
| **`Flight Distance`** | **0.3371** | 0.3703 | **Strongest continuous feature.** Not a rating at all |
| `Cleanliness` | 0.3350 | 0.3394 | |
| `Leg room service` | 0.3320 | 0.3296 | |
| `Inflight wifi service` | 0.2874 | 0.2917 | |
| `Baggage handling` | 0.2811 | 0.2589 | |
| `Checkin service` | 0.2397 | 0.2488 | |
| `Age` | 0.2136 | 0.2024 | Small but real |
| `Food and drink` | 0.2134 | 0.2180 | |
| `Ease of Online booking` | 0.1922 | 0.1913 | |
| `Gate location` | 0.0245 | 0.0233 | **Nothing** |
| `Departure Delay in Minutes` | −0.0305 | −0.0364 | **Nothing** |
| `Departure/Arrival time convenient` | −0.0462 | −0.0487 | **Nothing** |
| `Arrival Delay in Minutes` | −0.0576 | −0.0517 | **Nothing** |

Pearson and Spearman agree closely on every row, so no monotone relationship is
being missed by the rank measure.

### The categoricals separate the target more than any rating

| Feature | Value | Satisfied | Cramér's V |
|---|---|---|---|
| **`Class`** | Business | **0.7257** | **0.3931** |
| | Eco | **0.1676** | |
| | Eco Plus | 0.2427 | |
| **`Type of Travel`** | Business travel | **0.5841** | **0.3140** |
| | Personal Travel | **0.0974** | |
| **`Customer Type`** | Loyal Customer | 0.4953 | 0.1595 |
| | disloyal Customer | 0.2006 | |
| `Gender` | Male | 0.4492 | **0.0080** |
| | Female | 0.4379 | |

**Business class passengers are satisfied at 4.3× the rate of Eco.**
Business travel at 6× the rate of personal travel. Those gaps are larger than
anything the service ratings produce alone.

### Two columns are nearly one column

| Row-normalised | Business | Eco | Eco Plus |
|---|---|---|---|
| **Business travel** | 0.679 | 0.285 | 0.035 |
| **Personal Travel** | 0.019 | 0.919 | 0.062 |

| Row-normalised | Business travel | Personal Travel |
|---|---|---|
| **Loyal Customer** | 0.650 | 0.350 |
| **disloyal Customer** | **0.999** | 0.001 |

**99.9% of disloyal customers travelled for business.** `Customer Type` and
`Type of Travel` are very nearly one variable. `Gender` is independent of `Class`
(0.493 / 0.484) — another reason it carries nothing.

---

## 4. Baselines — the numbers to beat

All measured on the frozen train split, seed 42, no machine learning.

| Rule | Accuracy | ROC-AUC | Precision | Recall | Rows flagged |
|---|---|---|---|---|---|
| **`Class == Business`** | **0.7770** | **0.7786** | 0.7257 | 0.7995 | 239,352 |
| `Online boarding == 5` *(registered)* | 0.7206 | 0.6904 | 0.8754 | 0.4315 | 107,093 |
| `Type of Travel == Business travel` | 0.6761 | 0.7027 | 0.5841 | 0.9366 | 348,315 |
| do nothing | 0.5564 | 0.5000 | 1.0000 | 0.0000 | 0 |
| `Customer Type == Loyal Customer` | 0.5487 | 0.5858 | 0.4953 | 0.9206 | 403,799 |

**`Class == Business` is the bar: 0.7770 accuracy, 0.7786 AUC.** It beats the
registered rule on accuracy *and* recall at comparable precision.

Both metrics are reported because open question 6 — AUC or accuracy? — is still
open. **The two agree on the ranking**, so the bar is settled either way.

### The reference point that matters most

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **`HistGradientBoosting`, default settings, no tuning** | **0.924218** | **0.957312** |

An untuned default gradient-boosted tree is **14.7 accuracy points** and
**17.9 AUC points** past the one-column rule. That is the number a candidate has to
be judged against — not 0.7770.

---

## 5. Derived features

| Derived feature | What it captures | Cliff's δ | Size |
|---|---|---|---|
| **`mean_service_rating`** | Mean of all 13 ratings | **0.6029** | **large** |
| `count_of_ratings_at_or_below_2` | How many services were rated badly | −0.4482 | medium |
| `worst_service_rating` | The single worst service | 0.3234 | small |
| `arrival_delay_status` | on_time / delayed / unknown | — | categorical |
| `flight_distance_band` | under 1000 / 1000–3000 / 3000–5000 / over 5000 | — | categorical |

### The hypothesis that was wrong

The reasoning was: *a passenger remembers the worst part of a flight, not the
average, so the worst single service should beat the average.*

| Feature | Cliff's δ |
|---|---|
| `mean_service_rating` | **0.6029** |
| `worst_service_rating` | 0.3234 |

**The average beats the worst case by nearly double.** The hypothesis was
reasonable and the data rejected it. The average of all thirteen ratings is the
single most useful derived column in the analysis.

### Satisfaction rises monotonically with flight distance

| `Flight Distance` band | Rows | Satisfied |
|---|---|---|
| under 1,000 | 261,026 | 0.2979 |
| 1,000 to 3,000 | 182,579 | 0.5624 |
| 3,000 to 5,000 | 46,138 | **0.7974** |

Monotone across three bands. Longer flights are more satisfying — a long-haul
passenger expects more and gets more services to rate — but nobody would predict
it from the column name.

### `count_of_ratings_at_or_below_2` is not monotonic, and we cannot explain why

| Bad ratings | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Satisfied | 0.775 | 0.559 | 0.367 | 0.312 | **0.548** | 0.393 | 0.199 | 0.135 | 0.136 | 0.119 | 0.086 | 0.077 | 0.093 | 0.075 |

Falls steadily to three, then **jumps back up at exactly four**, then falls again.
**87,553 rows** sit in that group. Either something real happens at exactly four
bad services, or the column measures something other than its name says.
**Flagged, not used, not explained away.**

---

## 6. Statistics

`.dev/STATISTICS.md` followed throughout: hypothesis first, `alpha = 0.05` fixed
before any result, and p-value + effect size + 95% CI all reported. Cliff's delta
interpretation: below 0.147 negligible, to 0.33 small, to 0.474 medium, above
large.

### Every p-value is meaningless, and that is the finding

All 21 tests return `p < 0.001`, with confidence intervals roughly ±0.001 wide.
At `n = 489,743` that is arithmetic, not evidence. `.dev/STATISTICS.md` section 5
predicts exactly this under multiple comparisons.

**9 of 21 features have a medium or large effect. The other 12 are noise dressed
up by sample size.**

### Mann-Whitney U, ranked by effect size

| Feature | δ | 95% CI | Size |
|---|---|---|---|
| **`Online boarding`** | **0.6816** | 0.6804 – 0.6828 | large |
| **`mean_service_rating`** | **0.6029** | 0.6016 – 0.6042 | large |
| `Inflight entertainment` | 0.4882 | 0.4868 – 0.4897 | large |
| `Seat comfort` | 0.4644 | 0.4629 – 0.4658 | medium |
| `On-board service` | 0.3984 | 0.3970 – 0.3999 | medium |
| **`Flight Distance`** | **0.3918** | 0.3903 – 0.3933 | medium |
| `Cleanliness` | 0.3791 | 0.3776 – 0.3806 | medium |
| `Leg room service` | 0.3750 | 0.3735 – 0.3765 | medium |
| `count_of_ratings_at_or_below_2` | −0.4482 | −0.4497 – −0.4467 | medium |
| `Inflight wifi service` | 0.3257 | 0.3242 – 0.3273 | small |
| `worst_service_rating` | 0.3234 | 0.3219 – 0.3250 | small |
| `Baggage handling` | 0.3110 | 0.3095 – 0.3126 | small |
| `Checkin service` | 0.2698 | 0.2682 – 0.2717 | small |
| `Age` | 0.2482 | 0.2466 – 0.2497 | small |
| `Food and drink` | 0.2421 | 0.2406 – 0.2437 | small |
| `Ease of Online booking` | 0.2184 | 0.2168 – 0.2200 | small |
| `Gate location` | 0.0277 | 0.0261 – 0.0294 | **negligible** |
| `Departure Delay in Minutes` | −0.0178 | −0.0194 – −0.0162 | **negligible** |
| `total_delay_minutes` | −0.0303 | −0.0319 – −0.0287 | **negligible** |
| `Arrival Delay in Minutes` | −0.0322 | −0.0339 – −0.0306 | **negligible** |
| `Departure/Arrival time convenient` | −0.0525 | −0.0542 – −0.0509 | **negligible** |

### Chi-square, with Cramér's V

| Feature | χ² | dof | p | Cramér's V |
|---|---|---|---|---|
| **`Class`** | 151,360.70 | 2 | 0.000e+00 | **0.3931** |
| `Type of Travel` | 96,562.67 | 1 | 0.000e+00 | 0.3140 |
| `Customer Type` | 24,929.62 | 1 | 0.000e+00 | 0.1595 |
| `Gender` | 63.19 | 1 | 1.874e-15 | **0.0080** |

---

## 7. What a simple model says

| Feature | Logistic coefficient | Forest impurity | **Permutation** |
|---|---|---|---|
| **`Online boarding`** | **1.1346** | **0.2639** | **0.0806** |
| `Flight Distance` | 0.5639 | 0.0803 | 0.0138 |
| `count_of_ratings_at_or_below_2` | 0.5603 | 0.0209 | 0.0030 |
| `Departure/Arrival time convenient` | −0.5074 | 0.0354 | 0.0193 |
| `Inflight wifi service` | 0.4702 | 0.1215 | 0.0594 |
| `mean_service_rating` | 0.4085 | 0.0682 | 0.0075 |
| `Leg room service` | 0.3891 | 0.0582 | 0.0196 |
| `Inflight entertainment` | 0.2800 | 0.0986 | 0.0107 |
| `Seat comfort` | 0.1953 | 0.0635 | 0.0078 |
| `Age` | 0.0763 | 0.0282 | 0.0100 |
| `Gate location` | 0.2415 | 0.0206 | 0.0068 |
| `Arrival Delay in Minutes` | −0.1002 | 0.0008 | 0.0005 |
| `Departure Delay in Minutes` | 0.0368 | 0.0006 | 0.0006 |
| `total_delay_minutes` | −0.0317 | 0.0009 | 0.0011 |

**Permutation importance is the one to trust.** Impurity importance is biased
towards high-cardinality features — which is why `count_of_ratings_at_or_below_2`
looks useful to the forest (0.0209) and nearly useless to permutation (0.0030).

`Online boarding` is first on all three. `Inflight wifi service` is second on
permutation, well above its Cliff's δ — the tree models find interaction in it
that the rank statistic cannot see.

---

## 8. Outliers

Full treatment in [`outlier_findings.md`](outlier_findings.md). In brief:

- **The 1.5×IQR rule is invalid on both delay columns** — IQR is zero, so it would call all 64,483 delayed departures outliers
- **0 rows** fall outside a plausible domain range
- Extremes are **real and informative**: oldest 0.1% at 0.107 satisfaction, longest flights at 0.737, most delayed at 0.306 and 0.286 — *below* ordinary rows, the direction reality predicts
- **Nothing removed.** Clipping changes ROC-AUC by +0.000076. Noise

---

## 9. Leakage

| Check | Result |
|---|---|
| `satisfaction` used as a feature | **No** — banned by assertion |
| `id` used as a feature | **No** — banned by assertion |
| Features used | 27 |
| `train.csv` id range | 0 – 699,634 |
| `test.csv` id range | 699,635 – 999,478 |
| Ranges overlap | **No.** Asserted, so the run stops if it ever does |

`id` must stay out for a specific reason: **every training id is lower than every
test id**, so a model that saw it would learn the organisers' 70/30 cut rather
than anything about air travel.

### The honest caveat about what this model is

Under the reading we are using — predict `satisfaction` from the other answers on
the same form — nothing leaks. Every feature is recorded at or before the moment
the label is recorded.

But the features *are* the passenger's own ratings on the same questionnaire as
the answer. So the model **infers one survey answer from the others on the same
form.** It does not predict a reaction, and it cannot tell an airline a flight is
about to go badly, because nothing is known until the flight is over. That is a
property of the dataset, not something modelling fixes.

---

## 10. Feature shortlist

| Keep | Reason |
|---|---|
| `Class`, `Type of Travel`, `Customer Type` | Cramér's V 0.39, 0.31, 0.16 — the strongest separators |
| `Online boarding` | δ 0.68, first on all three importance rankings |
| `mean_service_rating` *(derived)* | δ 0.60, the best derived column |
| `Inflight entertainment`, `Seat comfort`, `On-board service`, `Cleanliness`, `Leg room service` | δ 0.49 – 0.38 |
| `Flight Distance` | δ 0.39, strongest continuous feature, monotone effect |
| `Inflight wifi service` | δ 0.33 but 2nd on permutation — interactions the rank statistic misses |
| `Age` | δ 0.25, small but real |

| Drop | Effect size |
|---|---|
| `Gender` | Cramér's V 0.0080 |
| `Gate location` | δ 0.0277 |
| `Departure Delay in Minutes` | δ −0.0178 |
| `Arrival Delay in Minutes` | δ −0.0322 |
| `Departure/Arrival time convenient` | δ −0.0525 |
| `total_delay_minutes` *(derived)* | δ −0.0303, and duplicates the two delay columns |

**Six columns carry no usable signal.** `.dev/EDA.md` says delete a feature that
stops being useful rather than keep it "just in case" — but dropping features is a
**decision**, and `.lead/02-B` Step 2.7 wants it explained rather than assumed.

### Two open questions that bear on the feature list

| Question | Why it matters |
|---|---|
| Is `0` in twelve of the thirteen ratings a valid lowest rating, or a hidden "not applicable"? | The two need **opposite** handling. As a number it is real signal; as a missing value it is a distortion. `Baggage handling` never holds a 0, which is the clue |
| Does `count_of_ratings_at_or_below_2` mean what its name says? | 87,553 rows break the monotonic pattern at exactly four bad ratings |

Neither blocks the method comparison — the raw ratings can be used as they are.
Both are recorded so the choice stays visible.

---

## What has changed

- **2026-10-04:** first version, consolidating every phase of the notebook.
- **2026-10-04, corrected:** the leakage check originally compared the frozen
  *validation* split against the frozen *test* split — both from `train.csv` — so
  it printed `ranges overlap: True` and then claimed "confirmed" anyway. It now
  reads `data/test.csv` directly and asserts the ranges do not overlap. The
  variable holding our frozen test split was renamed to `holdout_test_data` so it
  cannot be confused with the competition's `test.csv` again.