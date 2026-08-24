# Month 2 — Final Project Logs

This file gets appended to as you complete each day/week — notes, comparisons, and the honest `sklearn` vs. from-scratch numbers from Week 4.

No entries yet — generate Day 1 with `/i-am-in` to begin.

---

## Week 2 Day 1 — Maximum Likelihood Estimation (global Day 7) — 2026-08-24

**Topic:** `log_likelihood` + `mle_fit_gaussian` (closed-form vs grid search)

**How it ties to `minisklearn`:** MLE is the training principle behind every Week-3 model. `LinearRegression` and `LogisticRegression` both maximise the same log-likelihood you implement today (MSE and BCE are negative log-likelihoods under Gaussian/Bernoulli noise). The closed-form Gaussian MLE (`mean`/`var`) is your first rehearsal for `sklearn` validation — Week 4 Day 4 repeats this pattern on a real Kaggle dataset, comparing your `minisklearn` losses against `sklearn`'s. The numerical-vs-closed-form agreement check you write today is the prototype for that final honesty table.

**Today's artefacts:** `month_2/week_2/day_1/{learn,roadmap,coding_problems}.md`, `code/log_likelihood.py`, `code/mle_fit_gaussian.py`, `code/tests/` (7 tests), videos via `video-researcher` (StatQuest MLE ×2, verified).
