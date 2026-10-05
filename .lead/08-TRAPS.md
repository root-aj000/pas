# 08 — Traps: How Real Projects Fail

**Who this is for:** anyone who has to spot trouble before it costs months.

Every trap here has killed a real project. Each one is written as: what it looks
like, why it happens, how to spot it early, and what to do about it.

**Read this file when:** a project feels like it is going nowhere, a result looks
too good, a deadline is close, or someone proposes skipping a step.

---

## The pattern behind all of them

Almost every failure below shares one cause:

> **A decision was made without writing down the reasoning.**

Six months later nobody can tell whether a choice was an accident or a decision.
So nobody dares to change it, and nobody can improve it. This is the reason
documentation here is not optional politeness — it is a working tool.

---

## Trap 1 — The problem was never defined

**Looks like:** a request for "a model to predict churn". Everyone agrees it is
a good idea. Three months later there is an accurate model and nobody knows what
to do with it.

**Why it happens:** the request sounds obvious, so nobody writes it down.

**Spot it early:** nobody can answer "what decision does this model support?" in
one sentence.

**Fix:** Stop. Do `.lead/00-PROBLEM-AND-BASELINE.md` Step 0.1. Everything
downstream is wasted until this exists.

**Cost of fixing later:** the whole project.

---

## Trap 2 — No baseline, so nothing can be judged

**Looks like:** "the model is 94% accurate!" and nobody can say whether that is
good.

**Why it happens:** the baseline was never measured, so there is no number to
compare against.

**Spot it early:** nobody has tried a simple rule.

**Fix:** Build the three baselines in `.lead/00-PROBLEM-AND-BASELINE.md` Step
0.4. It takes an hour. Without them you cannot tell a good model from a lucky
one, and you cannot decide whether to ship.

**Cost of fixing later:** you ship something that does not beat a one-line rule,
and nobody notices until the budget is spent.

---

## Trap 3 — Leakage, the quiet one

**Looks like:** the best model ever built. Then it fails completely in production.

**Why it happens:** a feature was used that would not exist at the moment of
prediction.

**The four usual sources:**

| Source | Example |
|---|---|
| A column written after the decision | "cancellation reason", recorded when they cancel |
| Future information inside a total | "total spend this year" when predicting mid-year |
| The label leaking through a copy | A `churned` copy in the feature table |
| Random split of time-ordered data | Training on the future to test on the past |

**Spot it early:** accuracy is suspiciously high. Near-perfect scores are
leakage until proven otherwise.

**Fix:** apply the test from `.lead/01-DATA.md` Step 1.5 to every feature —
*"at the moment of the prediction, did I know this?"* Then check the model card
lists no label-derived features.

**Cost of fixing later:** total. The model is worthless and everything built on
it must be redone.

---

## Trap 4 — Nobody knew what a column meant

**Looks like:** a feature that makes no sense but has a large coefficient. Or
values that are off by a factor of 1000.

**Why it happens:** the analysis ran before anyone asked the data owner.

**Spot it early:** a value that looks impossible. Tenure of 0.0012 days.

**Fix:** `.lead/01-DATA.md` Step 1.3. One conversation with the data owner.
Document the answer in the code.

**Cost of fixing later:** days of work, and a model built on a wrong assumption
that produces confident nonsense.

---

## Trap 5 — The test set was used for tuning

**Looks like:** great results during development, disappointing results in
production.

**Why it happens:** it is easy to do. Each of the 30 tuning decisions "used the
test set a little bit", and the total effect is large.

**Spot it early:** validation and test scores are very close together while the
real-world result is much worse.

**Fix:** freeze the test set. Every decision on train and validation only. See
`.lead/01-DATA.md` Step 1.7.

**Cost of fixing later:** you have no honest measurement, and no way to tell a
real improvement from noise.

---

## Trap 6 — Accuracy on an imbalanced problem

**Looks like:** "99.8% accuracy!" with a 0.25% churn rate.

**Why it happens:** accuracy is the default metric, so it is the metric that
gets reported.

**Spot it early:** the base rate was never mentioned.

**Fix:** confusion matrix, then cost each error. See
`.lead/04-EVALUATION-AND-GO-NO-GO.md` Steps 4.2 and 4.3.

**Cost of fixing later:** a project celebrated internally and delivering nothing.

---

## Trap 7 — It works for the average, and fails a group

**Looks like:** an acceptable overall score, and a group that the model is
completely blind to.

**Why it happens:** only the average was measured.

**Spot it early:** nobody has broken the results down by any segment.

**Fix:** `.lead/04-EVALUATION-AND-GO-NO-GO.md` Step 4.5. Every model has a
group it is bad at. Finding it early is cheap.

**Cost of fixing later:** you find out from an unhappy stakeholder, and the
damage to trust is far worse than the performance gap.

---

## Trap 8 — The threshold was never decided

**Looks like:** the model outputs scores, nobody knows which ones to act on. The
default 0.50 is in use because it is the default.

**Why it happens:** converting a score into a decision feels like someone else's
job, so nobody does it.

**Spot it early:** the code has `if score > 0.5` and no reason.

**Fix:** `.lead/04-EVALUATION-AND-GO-NO-GO.md` Step 4.4 — decide it with the
person who owns the decision, and write the reason next to the number.

**Cost of fixing later:** far too many false alarms, or far too many missed
cases. Both get the model switched off.

---

## Trap 9 — Overfitting, and the gap nobody noticed

**Looks like:** excellent training score, poor validation score.

**Why it happens:** the model learned the specific rows, including their noise.

**Spot it early:** a large gap between training and validation scores.

**Fix:** fewer features, more data, a simpler model, or regularisation. And
always run the overfitting test from `.lead/03-TRAIN-AND-TUNE.md` Step 3.3 — if
the model cannot memorise 16 rows, training itself is broken.

---

## Trap 10 — The model cannot be reproduced

**Looks like:** the best version of the model exists only on someone's laptop,
and that person has left.

**Why it happens:** the model file was saved, but not the settings, the feature
list, the seed, or the code version.

**Spot it early:** ask "can you reproduce last quarter's result?" The answer is
a pause, then "probably".

**Fix:** `.lead/03-TRAIN-AND-TUNE.md` Step 3.6 — model, features, config, seed,
log and commit hash, saved together.

**Cost of fixing later:** the project cannot be maintained, and every future
change is guesswork.

---

## Trap 11 — Deployed in one step, no fallback

**Looks like:** it goes live on Monday, breaks on Monday, and everyone stops
working.

**Why it happens:** the deadline. Everyone is tired. The shadow stage would take
another two weeks.

**Spot it early:** no fallback exists, or nobody has tested what happens when
the model is unavailable.

**Fix:** `.lead/05-PACKAGE-SERVE-DEPLOY.md` Steps 5.3 and 5.4 — fallback first,
then shadow, then canary.

**Cost of fixing later:** trust. And a week of the team's time.

---

## Trap 12 — Nobody owns it, so nobody watches it

**Looks like:** the model quietly gets worse for months. Everyone assumes someone
is checking.

**Why it happens:** the project was launched, the task was completed, and
monitoring was never anyone's explicit job.

**Spot it early:** no named owner, no monitoring schedule, no alert recipient.

**Fix:** `.lead/06-MONITOR-AND-MAINTAIN.md` Step 6.1 — five things watched, each
with a number, a threshold and a named person.

**Cost of fixing later:** months of bad decisions made on bad predictions.

---

## Trap 13 — Alerts nobody reads

**Looks like:** a dashboard full of red. Everyone has muted it.

**Why it happens:** the alert fires for things that are not actually problems,
often for 3am.

**Spot it early:** more than a few alerts a week, or alerts with no action
attached.

**Fix:** one alert for things that require action, with the action written in the
alert. Everything else goes in a weekly report someone reads with their coffee.

---

## Trap 14 — Complexity added because it felt impressive

**Looks like:** a deep learning model on tabular data, a feature store for one
table, a plugin system for one plugin.

**Why it happens:** it feels like progress, and "we're using the modern approach"
is hard to argue against.

**Spot it early:** a complex thing that has not beaten the simple version.

**Fix:** require every layer of complexity to be justified by a measured number.
`.dev/RULES.md` rule 7.

**Cost of fixing later:** months maintaining something that a simple rule does
just as well.

---

## Trap 15 — Scope grew, and nobody said so

**Looks like:** the original problem is solvable, but six extra requests arrived,
and the deadline has not moved.

**Why it happens:** each extra request is small and reasonable on its own.

**Spot it early:** the thing being built no longer matches the one-paragraph
problem statement.

**Fix:** re-read the original problem statement. Confirm the new scope in
writing. If the deadline cannot move, say plainly what will be cut. Then cut
something and record it.

---

## The one-page summary

| Trap | Early warning | Cost if missed |
|---|---|---|
| Problem undefined | Nobody can say what decision it supports | Total |
| No baseline | Nobody tried a simple rule | Total |
| Leakage | Score is suspiciously high | Total |
| Column meaning unknown | Values look impossible | High |
| Test set used for tuning | Validation and test match, reality does not | High |
| Accuracy on imbalance | "99.8%!" | High |
| Fails a group | Only averages measured | High, plus trust |
| Threshold undecided | `if score > 0.5` with no reason | High |
| Not reproducible | "Probably, but..." | High |
| No fallback | Nothing planned for failure | High |
| Nobody owns it | No name attached | Slow and quiet |
| Alerts ignored | Dashboard all red | Medium |
| Complexity for show | Complex, and not better | Medium |
| Scope creep | No longer matches the statement | Medium |

**Total** means the project has to be redone. Everything else is a cost, not a
catastrophe — which is the argument for catching the top four early.