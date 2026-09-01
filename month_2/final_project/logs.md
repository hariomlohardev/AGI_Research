# Month 2 — Final Project Logs

This file gets appended to as you complete each day/week — notes, comparisons, and the honest `sklearn` vs. from-scratch numbers from Week 4.

No entries yet — generate Day 1 with `/i-am-in` to begin.

---

## Week 2 Day 1 — Maximum Likelihood Estimation (global Day 7) — 2026-08-24

**Topic:** `log_likelihood` + `mle_fit_gaussian` (closed-form vs grid search)

**How it ties to `minisklearn`:** MLE is the training principle behind every Week-3 model. `LinearRegression` and `LogisticRegression` both maximise the same log-likelihood you implement today (MSE and BCE are negative log-likelihoods under Gaussian/Bernoulli noise). The closed-form Gaussian MLE (`mean`/`var`) is your first rehearsal for `sklearn` validation — Week 4 Day 4 repeats this pattern on a real Kaggle dataset, comparing your `minisklearn` losses against `sklearn`'s. The numerical-vs-closed-form agreement check you write today is the prototype for that final honesty table.

**Today's artefacts:** `month_2/week_2/day_1/{learn,roadmap,coding_problems}.md`, `code/log_likelihood.py`, `code/mle_fit_gaussian.py`, `code/tests/` (7 tests), videos via `video-researcher` (StatQuest MLE ×2, verified).

---

## 2026-08-24 — Day 7 done: Maximum Likelihood Estimation (global Day 7, Week 2 Day 1)

- **Result:** 10/10 tests passed. Closed-form `mean / sqrt(mean((x-mean)^2))` and sequential 1-D grid search both converge on synthetic data.
- **Confidence:** `strong` (quiz: 6/6 correct on first ask, no flags). Self-reported difficulty `medium`, time `150m` (2h30m). Note: got stuck on grid `steps` iteration between start/end, resolved via search — classic off-by-one / inclusive-bounds issue the code review also flagged.
- **Advances to:** Day 8 (global 8) — MAP Estimation & Priors (`/i-am-in` next).
- **Badges unlocked today:** `first-day-done`, `five-day-streak` (streak 7: days 1-7 done, including Week 1 fast-forward).
- **Code review:** `month_2/week_2/day_1/code_review.md` written (naming `fst`/`cureent_mu`, `range(len)` idiom, grid docstring vs 2×1-D sweep); `growth-notes.md` updated with first observation.

---

## Week 2 Day 2 — MAP Estimation & Priors (global Day 8) — 2026-09-01

**Topic:** `log_prior_gaussian` + `log_posterior` + `map_fit_gaussian` (closed-form weighted average vs grid over posterior)

**How it ties to `minisklearn`:** Today's prior term is the **regularisation** that Week 3 Day 1-2 will use verbatim. `LinearRegression`'s Ridge penalty `λ||w||²` *is* the Gaussian log-prior `-(μ-prior_mu)²/(2 τ²)` with `λ = 1/(2τ²)` you derive today; Logistic Regression's `λ` is the same. Week 4's `minisklearn` will expose `regularization`/`prior_strength` whose math is exactly this MAP derivation — the prior-strength sweep you build today (weak → MLE, strong → prior) becomes the hyper-parameter sweep on real tabular data. Keep `map_fit_gaussian.py` — its flat-prior collapse (`τ→∞ → MAP→MLE`) is the sanity check Week 4 repeats.

**Today's artefacts:** `month_2/week_2/day_2/{learn,roadmap,coding_problems}.md`, `code/map_fit_gaussian.py`, `code/log_likelihood.py` (clean copy for independence), `code/tests/` (13 tests: prior/posterior + MAP), videos via verified oEmbed (mathematicalmonk MAP, Brunton MAP, StatQuest Ridge).

