# 00 — Understand the Problem, and Build a Baseline First

**Before this file:** nothing. This is the first file.
**After this file:** you know exactly what decision the model supports, and you
have a number to beat.
**Time it usually takes:** half a day of conversations, plus a few hours of
analysis.

---

## Why this file exists

Most failed machine learning projects fail here — before a single line of model
code is written. Not because of technology. Because nobody agreed on what the
model was for, so a different model was built than the one that was needed.

This file is mostly **conversations with people**, not code. That is normal and
correct.

---

## Step 0.1 — Write the problem in one paragraph

Copy this template and fill in the blanks. Do not skip it. One paragraph, no
technical words.

```text
We want to predict <WHAT> for <WHO>, so that <PERSON OR TEAM> can <SPECIFIC
ACTION>.

Today they <WHAT THEY DO INSTEAD TODAY>, which costs <TIME / MONEY / RISK>.

We will know this model is useful if <SPECIFIC MEASURABLE TARGET>.
```

**Worked example:**

```text
We want to predict which customers will cancel their subscription in the next
30 days, so that the retention team can call them before they leave.

Today they call every customer who has not logged in for 14 days, which means
they waste time on customers who were never going to leave, and miss the ones
who cancel quietly.

We will know this model is useful if we catch at least 60% of the customers who
really do leave, while calling fewer than 2000 customers a month.
```

Why the last line matters: it turns "we want better predictions" into a number
you can argue about and later measure. Without it, you cannot tell success from
failure, and the conversation at the end becomes personal.

**Do this with the person who will use the model, not with the data.** If you
write this alone and later present it, you have built the wrong thing politely.

---

## Step 0.2 — Answer these seven questions

Write the answers down. If you cannot answer one, that is your next task — not
a detail to skip.

| # | Question | Why it matters |
|---|---|---|
| 1 | Who makes the decision using this model's output? | If nobody does, the model is useless |
| 2 | What do they do today instead? | That is your real baseline |
| 3 | What does a wrong prediction cost? | Tells you how careful to be |
| 4 | What is the base rate? What % of cases are positive? | Decides whether accuracy is even a useful measure |
| 5 | Is the decision reversible? | If yes, be fast. If no, be careful |
| 6 | How often does the world change? | Decides how often the model must be retrained |
| 7 | What is the smallest useful version? | Stops you building too much |

### Question 4 in more detail, because it surprises people

The **base rate** is how often the thing you are predicting actually happens. If
1% of customers churn, then a model that always answers "nobody will leave" is
99% accurate — and completely useless.

This is the most common trap in real projects. Always ask for the base rate
before you agree to any accuracy target.

---

## Step 0.3 — Decide whether machine learning is even the right tool

Ask this honestly. Sometimes the answer is no, and finding that out now saves
months.

**Use a simple rule or a lookup instead when:**

- There are fewer than about 20 rules a person could write down.
- An existing system already produces the answer.
- You have very little data (roughly under a few hundred examples of the
  outcome).
- The answer must be 100% correct every time and can be checked directly.
- The pattern changes every week.

**Use machine learning when:**

- The answer depends on many inputs at once, in ways nobody can write as rules.
- There is a lot of historical data with the outcome already recorded.
- Being right most of the time is valuable.
- The pattern is stable enough to be worth learning.

**Write down which you chose, and why.** This is the "no" you can point at later
if the project gets reopened.

---

## Step 0.4 — Build the baseline first

This is the most skipped step in the entire process, and it is the one that
saves projects.

A **baseline** is the simplest possible answer to the problem, using no machine
learning. It gives you a number to beat, and it sometimes turns out to be good
enough.

Build at least these three:

| Baseline | How to build it | What it tells you |
|---|---|---|
| **Do nothing** | Always answer "no" / take no action | The floor. Any model must beat this |
| **Current practice** | Whatever the team does today | The number you must beat to be worth building |
| **One simple rule** | Use the single most obviously related number you have | Whether the data even holds the signal |

**Worked example:**

```text
Do nothing:        catch 0 of 300 leavers, call 0 people
Current practice:  catch 90 of 300 leaves (30%), call 11,000 people a month
Simple rule:       "no login for 14 days" catches 180 of 300 (60%), calls 4,000
```

Now you know three things for certain:

1. Any model must beat **60% catch on 4,000 calls** to be worth building.
2. The data does contain signal — the simple rule found it.
3. If a model gets to 65%, that is a **5 point** improvement, and you can
   honestly ask whether that is worth the complexity. Sometimes the answer is
   no. That is a successful project too.

Record the baseline numbers in the experiment log — see
[`09-TEMPLATES.md`](09-TEMPLATES.md).

---

## Step 0.5 — Write down the risks

List the ways this could go wrong. Keep it short and honest.

| Risk | Likelihood | What you will do about it |
|---|---|---|
| Not enough data | | |
| The answer arrives too late to be useful | | |
| The team does not trust the output | | |
| The model is right but the team ignores it | | |
| The decision it supports is itself cancelled | | |

The third and fourth rows are the ones people forget, and they are the most
common reasons real projects die after launch. A model nobody trusts is worth
nothing, no matter how accurate it is.

---

## Gate — do not start anything until all of these are true

- [ ] The problem is written in one paragraph, in plain language, with a number
      in the last line
- [ ] The person who will act on the output is named
- [ ] The current practice is written down, and we know how well it does
- [ ] The base rate is known
- [ ] We have decided ML is the right tool, and written down why
- [ ] At least the "do nothing" and one simple rule baselines are measured
- [ ] Risks are listed
- [ ] The problem statement has been shown to, and agreed by, the person who
      asked for the model

**If any box is unticked: stop here.** The next file is
[`01-DATA.md`](01-DATA.md). Do not read ahead, and do not start downloading data
because it is tempting. Data collected before the problem is defined is data you
cannot use.

---

## What to do next

Go to [`01-DATA.md`](01-DATA.md): get access to the data, define every column,
define the label, check quality, and split the data properly.

## Common mistakes in this phase

- **Skipping the problem statement** because everyone "already knows" what is
  wanted. They do not — they know a symptom, not the problem.
- **Agreeing to an accuracy target** without knowing the base rate.
- **Building the fancy model first** to look impressive, then discovering there
  is no signal to find.
- **Treating "no" as failure.** If a simple rule wins, that is valuable
  information delivered cheaply.