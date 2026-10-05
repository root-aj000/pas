# Lead Playbook

**Who this is for:** the person responsible for getting the project finished.
Usually one person doing several jobs at once — developer, analyst, project
manager, and the person everyone asks "what now?".

**The promise:** if you follow these files in order, you will never be left
thinking "what is the next step?".

---

## Why this folder exists

Almost everything written about machine learning on the internet is a summary.
It reads like this:

```
1. Gather data
2. Train a model
3. Deploy it
```

That is not wrong. It is just missing the 13 steps in between, and those steps
are where projects actually die. A team that follows the internet version will
build a model that looks great in a notebook and cannot be trusted in real life,
and nobody will know why.

The real path looks like this:

```
00  Understand the problem        -> can we even build this?
01  Baseline first                -> is ML the right tool?
02  Get access to the data        -> are we allowed to use it?
03  Define every column           -> what does this number mean?
04  Define the label              -> what exactly are we predicting?
05  Check data quality            -> is it even correct?
06  Split the data                -> how do we know the result is real?
07  Explore and find signals      -> which features actually predict it?
08  Choose and wire the method    -> which algorithm, from which library?
09  Train and tune                -> settings, seeds, reproducibility
10  Evaluate properly             -> segments, costs, thresholds
11  Decide: ship or stop          -> a real decision, in writing
12  Package for reproducibility
13  Design how it will be served
14  Deploy gradually
15  Monitor and maintain          -> the part everyone forgets
```

Note step 8. Most guides go from "explore the data" straight to "train a
model", and leave out the hardest decision in real work: **which method, and how
it gets wired into the code.** Choosing between a decision tree and gradient
boosting — or between a tabular model and a neural network — takes longer than
training either of them, and nobody writes it down. Files
`02-C` and `02-D` cover exactly that.

Each of those is a numbered file in this folder. Read them in order.

Phase 2 is split into four files, because "build the model" is not one step:

| File | What it answers |
|---|---|
| [`02-A-ARCHITECTURE.md`](02-A-ARCHITECTURE.md) | Where does my code go, who calls what, where does a library method get plugged in? |
| [`02-B-FINDING-SIGNALS.md`](02-B-FINDING-SIGNALS.md) | Which features actually carry signal? |
| [`02-C-CHOOSING-THE-METHOD.md`](02-C-CHOOSING-THE-METHOD.md) | Which method should I try, how do I look up one I don't know, how do I compare them fairly? |
| [`02-D-METHODS-CATALOGUE.md`](02-D-METHODS-CATALOGUE.md) | Which function, from which library, for which task? |

---

## The most important idea in this folder

**A project is a chain of gates, not a straight line.**

At the end of each file there is a **Gate** — a short list of things that must be
true before you move on.

- If a gate is not met, you **stop**. You do not proceed "and fix it later".
- Every skipped gate becomes a bug that is expensive to find later.
- Most of the real work in a project happens inside gates, not between them.

If you are ever unsure whether to continue, the answer is: **stop, and write down
the open question.** Then ask the person who owns that answer. Guessing is the
single most expensive habit in this job.

### The most common gap, and where it is covered

The gap between tutorials and real work is nearly always the same one: **the
decision layer.** Tutorials skip from data to a fitted model, so nobody ever
learns to choose, research, compare, or wire up a method.

That is phase 2, files `A` through `D`. If someone on your team asks "which
model should I use?" and nobody can answer in one sentence, that is the phase to
read, not the training phase.

---

## How to use these files

**If you are the lead, starting a new project:**
Read `00` today. Work through it with the person who asked for the model. Do not
read ahead — the order matters and each file assumes the previous one is done.

**If you are a developer, new to the project:**
Start with [`07-NEW-DEV-ONBOARDING.md`](07-NEW-DEV-ONBOARDING.md). It gives you a
day-by-day path so you are never guessing what to do first.

**If something has gone wrong:**
Check [`08-TRAPS.md`](08-TRAPS.md). It lists the ways real projects fail, and
what the warning signs look like.

**If you need to write something down:**
Use [`09-TEMPLATES.md`](09-TEMPLATES.md). Problem statement, experiment log,
model card, go/no-go decision, incident note. Copy the template, fill it in, done.

---

## Which folder answers what

| Folder | Answers |
|---|---|
| `.lead/` (this one) | What do we do, in what order, and what must be decided? |
| `.dev/` | How do we write the code that does it? |

If it is a **process or decision**, it belongs here.
If it is **code**, it belongs in `.dev/`.

---

## The three questions to ask at every gate

1. **Do we know what decision this model supports?** A model with no decision
   attached gets built, and then never used.
2. **What would we do if we had no model at all?** You always have the option of
   doing nothing. That option must be compared against.
3. **Who is responsible for this when it breaks?** If nobody owns it, it will
   break and stay broken.

---

## A note on this documentation

These files are deliberately long and detailed. A guide that leaves a gap
becomes a guessing game for whoever reads it next, and this project has more
than one person.

If something is unclear, do not fill the gap with your own assumption. Write the
question down, and get it answered. Every rule in `.dev/` about "never assume,
always clarify" applies here too — with more force.