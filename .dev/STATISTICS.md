# Statistics

Use this to turn a pattern you saw in [EDA.md](EDA.md) into a fact you can rely
on. A pattern that only exists in your one dataset is noise.

---

## 1. Write the hypothesis down first

- **Null hypothesis (H0):** there is no real relationship. Whatever you see is
  random.
- **Alternative hypothesis (Ha):** there is a real relationship.

Example, in plain English:

> H0: customers who churned and customers who stayed spent the same amount per
> visit.
> Ha: they do not.

Write it before you run the test. Choosing the test after seeing the result is
how people fool themselves.

---

## 2. Pick the right test

Match the test to the type of data you have.

| Your data | Normal / clean | Skewed or messy |
|---|---|---|
| Number vs. number (linear) | Pearson correlation | Spearman rank |
| Number vs. number (curved) | Scatter plot + fit a curve | Same |
| Category vs. category | Chi-square test | Chi-square test |
| Number across 2 groups | t-test | Mann-Whitney U |
| Number across 3+ groups | ANOVA | Kruskal-Wallis |

Short version:

- **Chi-square** — do two categories belong together?
- **t-test** — do two groups have different averages?
- **ANOVA** — do three or more groups have different averages?
- **Mann-Whitney / Kruskal-Wallis** — the same questions, for data that is
  skewed or has outliers.

---

## 3. Read the p-value correctly

The **p-value** answers: *if there were no real relationship, how likely would a
result this strong show up by chance?*

- The usual threshold is `0.05`.
- `p < 0.05` → reject H0. The relationship is probably real.
- `p >= 0.05` → cannot reject H0. You found nothing, not proof of no effect.

The most common mistake: reading `p < 0.05` as "there is a 95% chance I am
right". It does not mean that. It only means the result is hard to explain by
chance alone.

---

## 4. Check the size of the effect, not just the p-value

A tiny difference can be "significant" with a big enough dataset, and still be
useless. Report three numbers every time:

- **p-value** — is it real?
- **Effect size** — how big is it? (difference in averages, correlation
  coefficient)
- **Confidence interval** — the range the true value likely sits in.

---

## 5. The rules that keep you honest

- **Pick the threshold before you start.** Do not use 0.01 after 0.05 failed.
- **Run the test on the training data only.** Testing on data you also trained
  on produces flattering, useless answers.
- **A p-value does not prove a cause.** It only shows a relationship. "Ice
  cream sales correlate with drownings" does not mean ice cream drowns people.
- **Multiple comparisons.** If you test 50 columns, some will look significant
  by luck alone. Be suspicious of the single surprise result out of many.
- **Correlation is not cause.** Something else may be behind it. Domain
  knowledge decides which explanation makes sense.
- **Do not p-hack.** If you keep changing the data, the columns or the threshold
  until `p < 0.05`, the result is meaningless. Report what you actually found.

---

## 6. Sample

Reporting format, so every result looks the same:

> **Question:** Do churned customers spend less per visit?
> **Test:** Mann-Whitney U (skewed data, outliers present).
> **Result:** churned 4.20 USD/visit vs stayed 6.85 USD/visit.
> Difference -2.65 USD (95% CI -3.10 to -2.20), p < 0.001.
> **Conclusion:** yes, churned customers spend less per visit.

---

## Checklist

- [ ] Hypothesis written in plain English, before testing
- [ ] Test chosen to match the data type and distribution
- [ ] p-value, effect size and confidence interval all reported
- [ ] Tested on training data only
- [ ] Confounders considered
- [ ] Result written up in the format above