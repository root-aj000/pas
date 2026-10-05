# Developer Handbook

Plain-language rules for writing code on this project.
Every file in this folder answers one question. Read the one you need.

> **Not sure what to do next?** The *process* — what step comes after which, and
> what must be decided — lives in [`../.lead/`](../.lead/README.md). This folder is
> only about **how to write the code**. If you are asking "what should I build
> now", you want `.lead/`.

| File | What it answers |
|---|---|
| [RULES.md](RULES.md) | How am I allowed to write code? |
| [DEV-LOG.md](DEV-LOG.md) | How do I write down every change I make? |
| [CODE-STANDARDS.md](CODE-STANDARDS.md) | How is the code formatted, typed, seeded and checked automatically? |
| [DEBUGGING.md](DEBUGGING.md) | How do I add logs and find what broke, and why? |
| [PROJECT-STRUCTURE.md](PROJECT-STRUCTURE.md) | Where does my file go? |
| [TOOLS.md](TOOLS.md) | Which library do I use for tracking, features, data quality, drift, tuning? |
| [EDA.md](EDA.md) | How do I explore the data and find real signals? |
| [STATISTICS.md](STATISTICS.md) | How do I prove a signal is not random noise? |

## The only rule that matters

**Write code that a non-technical person can read and follow.**

If a smart engineer cannot guess what a piece of code does by reading it top to
bottom, it is too clever. Rewrite it until they can.

## The 16 golden rules

1. **Readability first.** Clear code beats short code. Never sacrifice clarity to
   save a line.
2. **No hacks.** No clever one-liners, no regex tricks, no copy-paste-and-hope.
3. **No DIY.** If a library already does it, use the library. Never rebuild
   something that already exists.
4. **Document everything.** Every file, function and tricky line says *why* it
   exists, in plain English.
5. **Make it debuggable.** Log every step, never assume, and make sure any
   bug can be found by reading the logs. See [DEBUGGING.md](DEBUGGING.md).
6. **Simple always wins.** The simplest version that works is the correct version.
7. **Don't overengineer.** No build systems, no plugin layers, no config for
   values that never change.
8. **Fail loudly, fail early.** Bad input stops the program with a clear message.
   Never fail silently.
9. **One file, one job.** If the name needs the word "and", split the file.
10. **Tests for logic, not for ceremony.** Anything with a branch or a loop gets
    one small test that proves it works.
11. **Never assume, always clarify.** If something is unclear, ask the person
    who owns it. A question now beats wrong work later.
12. **Logs are mandatory.** If you cannot see the dataflow, you cannot fix it.
13. **No hardcoded hyperparameters.** Anything we tune lives in one config file.
14. **Same input, same output.** Seed the randomness and log the seed, so any
    result can be repeated exactly.
15. **Log every change.** Even one line of code gets a dev log entry. Assume
    someone will check.
16. **Never delete a log.** Correct it with a dated note. The history is the
    evidence.

## The test for every line of code you write

Ask yourself: *"Would a new team member understand this on day one?"*

If no, it is not finished.