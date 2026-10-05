# Exploratory Data Analysis (EDA)

EDA means looking at the data before you build anything.

The goal is simple: **understand the data, clean it, and find the patterns that
really predict the answer** — and throw away the patterns that are just noise.

Do not skip this step. Every hour spent here saves a week of debugging later.

Work in `research/`, in numbered files, so they run in order:

```
01_load_data.ipynb
02_eda_basics.ipynb
03_eda_one_column.ipynb
04_eda_relationships.ipynb
05_find_signals.ipynb
```

**Anything you discover that must be repeated must move into `src/` afterwards.**
Notebooks are scratch paper, not the application.

---

## Phase 1: Get your bearings

Before any analysis, answer: *what am I actually looking at?*

- **Size and types.** How many rows and columns? Which columns are numbers and
  which are categories?
- **Missing values.** Which columns have blanks? Decide what to do with each one:
  drop the row, fill in a sensible value, or add a "was missing" flag column.
- **Basic summary.** Use `.describe()` for the average, spread, smallest,
  largest, and the middle percentiles of every numeric column.

```python
print(data.shape)
print(data.dtypes)
print(data.isna().sum())
print(data.describe())
```

Read the output before moving on. Do not paste and forget.

---

## Phase 2: Look at one column at a time

Understand each column on its own and spot anything strange.

- **Numeric columns** — draw a histogram.
  - *Skewed* means a long tail on one side. Most income data looks like this.
  - *Multiple peaks* means there are distinct groups hiding inside your data.
- **Category columns** — draw a count bar chart.
  - Watch for extreme imbalance. A column that is 99% "yes" and 1% "no"
    usually cannot answer your question.
- **Outliers** — draw a box plot.
  - An outlier is either a data entry mistake, or the most interesting record in
    the file (for example, fraud). Decide which one it is. Never delete it
    silently.

---

## Phase 3: Look for relationships

This is where you find which columns actually matter.

**Number vs. number**

- Draw a correlation heatmap. Use Pearson for straight-line relationships and
  Spearman for anything that only goes up and down.
- A correlation close to +1 or -1 means a strong relationship. Close to 0 means
  none.
- Also draw a scatter plot. A correlation number misses curves and shapes that
  are just as real.

**Category vs. number**

- Draw a box plot or violin plot grouped by the category.
- If the distributions barely overlap, that category is a strong signal.

**Category vs. category**

- Draw a crosstab and a stacked bar chart.
- Some combinations appearing far more often than expected is a signal.

---

## Phase 4: Find the signals

A **signal** is a pattern in the data that reliably predicts the target.

In order of how much they usually help:

### 1. Use domain knowledge to build new columns

Raw data rarely holds the best signals. Combine what you already know:

- Birth date → age.
- Total spend + number of visits → average spend per visit.

### 2. Compare against the target

Split the data into two groups: the ones that got the answer you want
("churned") and the ones that did not ("stayed"). Compare every other column
across the two groups.

A column that looks completely different in the two groups is one of your
strongest signals. Start there.

### 3. Check grouped clusters (only if you have many columns)

If you have hundreds of columns, use PCA or t-SNE to squash them into a 2D
plot. Colour each point by the target. If you see separate groups, a strong
signal exists.

Skip this unless you really have too many columns. Two charts usually do the
job.

### 4. Let a simple model tell you

The fastest way to rank your columns is a simple model: Random Forest or
XGBoost, trained and scored in a few lines. Then read the feature importance
scores.

**The top of that list is your shortlist.** Do not trust it blindly — a tree
model can rank a leaky or useless column highly. Confirm each one by looking at
the data.

### Watch out for leaks

A **leak** is a column that tells the answer away. `account_closed_date` will
predict churn perfectly — and will not exist at prediction time.

Leakage gives you a beautiful score and a useless model. Check every strong
signal and ask: *would I actually know this value at the moment I need to
predict?* If not, remove it.

---

## Phase 5: Write down what you found

Turn patterns into plain claims, and check them statistically. Continue in
[STATISTICS.md](STATISTICS.md).

Then write the findings where the next person will find them:

- A short summary in the notebook, in plain English.
- Anything that becomes code, moved into `src/`.
- Anything suspicious, recorded in the issue tracker with the row count.

An EDA you do not write down was not done.

---

## Checklist

- [ ] Size, types and missing values checked
- [ ] Every column looked at on its own
- [ ] Relationships between columns plotted
- [ ] Target compared against each feature
- [ ] Strongest signals listed with evidence
- [ ] Leaks checked for
- [ ] Findings written down in plain English
- [ ] Repeatable steps moved from the notebook into `src/`