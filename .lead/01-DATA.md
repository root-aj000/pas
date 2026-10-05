# 01 — Get the Data, Define It, Check It, and Split It

**Before this file:** [`00-PROBLEM-AND-BASELINE.md`](00-PROBLEM-AND-BASELINE.md)
is done. You know the decision the model supports.
**After this file:** you have clean, understood, trusted data, and a split that
will not lie to you.
**Time it usually takes:** a few days. This is where most projects spend real
time.

---

## Why this file exists

Four things go wrong in real projects, and all four live in this file:

1. You were not allowed to use the data. Or you were allowed, three weeks late.
2. Nobody knew what a column actually meant, and you discovered it after building
   everything.
3. The "answer" you are predicting was recorded using information that will not
   exist at the moment you need to predict — so the model looked perfect and
   failed in production.
4. The data was split randomly across past and future, so the model saw the
   future during training. Every score was inflated.

Read all four. Each one is common and each one is expensive.

---

## Step 1.1 — Get access, and find out what you are allowed to do

Do this first, before writing any code. It is the most common reason projects
lose weeks.

**Find out, in writing:**

| Question | Why |
|---|---|
| Who owns this data? | Only they can say yes |
| What approval is needed? | Security, privacy, legal, compliance |
| Can it leave our systems? | Decides where the code runs |
| How fresh is it? | Decides whether the problem is even solvable |
| How is it updated? | Manual weekly export vs automatic hourly load |
| What happens if the schema changes? | Usually the thing that breaks us later |
| Who do we ask when something is wrong? | You need a human, not a ticket queue |

**Handle personal data carefully.** If the data contains anything about real
people — names, addresses, phone numbers, health, income — you need privacy
review before it goes anywhere, including your own machine.

Practical rules:

- Prefer IDs over names. Never copy names or emails into a features table.
- If you must keep personal data, keep it in the raw layer only, and never in
  features or logs.
- Delete the data when the project ends, if that was agreed.

**Gate within this step:** you have written permission, or a named person has
confirmed in writing that you do not need it.

---

## Step 1.2 — Build an inventory of the data

Do not explore blindly. Write down what tables exist.

```text
| Table            | What it holds            | Rows      | Updated     | Contains people data? |
|------------------|--------------------------|-----------|-------------|-----------------------|
| customers        | Account details          | 1.2M      | Daily 02:00 | Yes - name, email     |
| transactions     | Each purchase            | 48M       | Hourly      | No                    |
| subscriptions    | Plan and renewal dates   | 1.2M      | Daily 02:00 | No                    |
| support_tickets  | Complaints               | 300k      | Daily 02:00 | Yes - free text       |
```

Then, for **every table you plan to use**, answer these three things:

1. **Row count and growth.** How many rows now, how many per day.
2. **Primary key.** What uniquely identifies a row.
3. **Join key.** How you link it to the other tables.

If two tables both have a `customer_id` but they mean different things, stop and
find out. Silent mismatched joins are the most common source of silently wrong
numbers in this job.

---

## Step 1.3 — Define every single column

For every column you might use, fill this in:

| Column | What it means | Type | Units | Example value | Trust it? |
|---|---|---|---|---|---|
| `lifetime_value` | Total money paid | float | USD | 1450.75 | Yes, but excludes refunds — see note |
| `plan_name` | Current plan | text | — | `pro_monthly` | Yes |
| `tenure_days` | Days since signup | int | days | 412 | **Unsure — from first payment or from signup page visit?** |

Now the actual work: **go and ask the owner.** Not "ask the data scientist", not
"look at the wiki". The person or team who writes that data.

**The questions that prevent disasters:**

- What are the units? Seconds, milliseconds or days? Dollars or cents?
- What time zone are the timestamps in?
- What does a blank mean? Never happened? Not recorded? Failed to load?
- When is this value written? At the moment of the event, or later during a
  nightly job?
- Has this definition **ever changed**? If so, when? Old rows may mean something
  different from new rows.
- How many rows were missing this value, and when?
- Is this value ever corrected after the fact? If so, what date is the
  correction?

**Write the answers into the code as comments and docstrings**, in the same file
that uses the column. See the documentation rules in `.dev/RULES.md`. The next
developer should never have to ask again.

### The units trap — a real example

```text
Question: "How long has this customer been with us?"
Column:   tenure
Value:    0.0012

If the unit is days  -> 103 seconds. Nonsense.
If the unit is years -> 4.4 days. Plausible.
```

Same number. Completely different meaning. A model does not care. You do.

---

## Step 1.4 — Define the label (the thing you are predicting)

This is the single most important definition in the whole project, and the one
most often skipped.

A **label** is the column that holds the answer you want to predict.

```text
"Churned" = the customer cancelled their subscription.
```

That sentence must be completed with four details, and every one of them will
cause problems later if you guess:

**1. What exactly counts as the event?**
"Cancelled" could mean: clicked cancel, or actually stopped paying. Those are
different events, happening days apart, and only one of them is a real churn.

**2. When is it decided?**
If you count "churned in March", you cannot know that until April, when the
payment fails. See Step 1.5.

**3. What time window does the prediction cover?**
"Predict churn in the next 30 days" — is the window measured from today, from
the last login, or from the billing date? Write it down. Every row must have the
same window, or your data is inconsistent.

**4. What about customers who have not had time to churn yet?**
See Step 1.5. This one is subtle and it will bite you.

### Worked example

```text
LABEL:  customer_churned
DEFINITION: A customer counts as churned on day D if a cancellation event with
           status = 'confirmed' exists with cancel_date <= D + 30.
WINDOW:  30 days from D.
BASELINE RATE: 300 of 120,000 active customers = 0.25%.
```

---

## Step 1.5 — Handle time correctly (the hardest part of this file)

Real data has two time traps. Both are silent. Both ruin models.

### Trap 1: the answer arrives late

If you are predicting churn in the next 30 days, the label for a customer is only
known 30 days later.

So the most recent 30 days of your data **do not have labels yet**. Training on
them teaches the model that recent customers did not churn — which is exactly
backwards, because recent customers have not had time to churn.

**The fix:** only use rows whose label window has fully closed.

```text
Today is            2024-06-01
Label window is     30 days
So only use data up to 2024-05-02.

Rows from 2024-05-03 onward: EXCLUDE. Labels not complete yet.
```

This is called the **label window**, or **observation period**. Calculate it, put
it in `config.yaml`, and log it on every run.

### Trap 2: information from after the prediction moment

Every feature must be a value that existed **at the moment you make the
prediction**. If a column was written *after* that moment, it does not exist in
real life — and using it is **data leakage**, the single most common cause of a
model that fails in production.

**Example of leakage:**

```text
Predicting churn on 2024-01-15.
Column: last_payment_attempt_result
That value was written on 2024-01-18, when the payment failed.

Using it -> the model sees the answer -> looks perfect -> useless in production.
```

**The question to ask about every column:**

> "At 09:00 on the day of the prediction, did I know this value?"

If the answer is no, it cannot be a feature. It can still be the label, or it can
be used to **build the label**.

### The rule that prevents this

Build features as-of a timestamp, not as-of today. Always:

```python
# Correct: only data available before the prediction moment
features = get_features(customer_id, as_of_date="2024-01-15")

# Wrong: uses whatever the table holds now
features = get_features(customer_id)
```

---

## Step 1.6 — Check the data quality

Now check it is actually correct, before building anything on top of it.

Use the `check_columns()` helper from `.dev/DEBUGGING.md` for structure, and
Great Expectations (`.dev/TOOLS.md` section 3) when you have several tables.

**The checks worth writing for every dataset:**

| Check | What wrong data looks like | What it usually means |
|---|---|---|
| Required columns present | `missing columns ['tenure_days']` | Someone renamed something |
| Row count is as expected | 4,000 rows instead of 1,200,000 | A partial load. **Never continue.** |
| Values in a valid range | `age = 250` | A unit or join error |
| Key is unique | 12 duplicate `customer_id`s | A join fanned out, or double counting |
| Missing values are reasonable | 60% missing in a key column | The load failed |
| Timestamps are not in the future | Dates next year | Clock or timezone problem |
| The label rate is stable over time | 0.25% last month, 12% this month | Data or definition changed |

**The last row is the one people skip.** Plot the base rate over time. If it
jumps, something changed in the world or in the data, and you must find out
which before training.

**Rules:**

- A failed check **stops the pipeline**. Not a warning — a stop.
- Every check says what failed, on how many rows, with an example value.
- Write the checks as code that runs every time. Do not check by hand once.

---

## Step 1.7 — Split the data correctly

Do this **before** exploring, not after. If you explore first, you have already
seen the test data in your head, and your test score will be optimistic.

You need three sets:

| Set | What it is for | How much |
|---|---|---|
| **Train** | Learning the patterns | ~70% |
| **Validation** | Choosing settings and features | ~15% |
| **Test** | The one honest measurement, used once at the end | ~15% |

**The rules that matter — get these wrong and everything is invalid:**

**1. Split by time, not randomly, when the data is time-ordered.**

```python
# WRONG for time-dependent data - lets the model learn from the future
train, test = train_test_split(data, test_size=0.2, random_state=42)

# RIGHT - train on the past, test on the future, exactly like reality
train = data[data["date"] < "2024-01-01"]
test = data[data["date"] >= "2024-01-01"]
```

**2. Keep the same entity out of both sets.**

If the same customer appears in train and test, the model recognises the
customer rather than learning the pattern — and looks far better than it is.

```python
# Split by customer, not by row
for customer_id, rows in data.groupby("customer_id"):
    assign_whole_customer_to_one_set_only(rows)
```

**3. Never touch the test set until the end.** Not once, not "just to look".
Every decision gets made on train and validation only.

**4. Freeze it and write it down.** Which rows, which date, which seed. In the
experiment log. Otherwise "the test set" quietly changes and results stop being
comparable.

**5. Never scale using the test set.** Fit any scaling or encoding on the training
data only, then apply it to the others.

**6. Check the split looks like production.** Roughly the same base rate, same
time range shape, same customer mix. If the test month is an unusual month, your
score will be misleading either way.

---

## Gate — do not start exploring until all of these are true

- [ ] Written permission to use the data, and we know what we may not do with it
- [ ] An inventory of every table: rows, growth, primary key, join key
- [ ] Every candidate column documented: meaning, type, units, time zone, blanks,
      and whether it has ever changed meaning
- [ ] Unknowns listed as open questions, with a named person to ask
- [ ] The label is defined precisely: the exact event, the window, the
      observation period
- [ ] The label window is calculated, and recent incomplete rows are excluded
- [ ] Every feature confirmed as known at prediction time (no leakage)
- [ ] Quality checks written as code, and passing
- [ ] Data split by time and by entity, with the test set frozen and recorded
- [ ] Every pipeline step logs what it took in and what it produced

**If any box is unticked: stop.** You cannot trust a single number until all of
these are true.

---

## What to do next

Go to [`02-B-FINDING-SIGNALS.md`](02-B-FINDING-SIGNALS.md): explore the training data
only, find the patterns that actually predict the label, and build the first
model.

## Common mistakes in this phase

- **Starting to code before getting permission.** Weeks lost.
- **Assuming a column's meaning from its name.** `tenure` was our example.
- **Forgetting the label window.** Training on data whose answer did not exist
  yet.
- **Leakage from a column written after the prediction moment.**
- **Randomly splitting time-ordered data.** Every score becomes meaningless.
- **Checking the data once by hand instead of writing checks that always run.**