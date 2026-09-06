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

---

## 2026-09-05 — Day 8 done: MAP Estimation & Priors (global Day 8, Week 2 Day 2)

- **Result:** 13/13 tests passed. `log_prior_gaussian`, `log_posterior`,
  `map_fit_gaussian_closed_form` (precision-weighted average) and `map_fit_gaussian_grid`
  (2-D grid over the posterior) all implemented. The `__main__` experiments print the
  3-point outlier demo (`mu_mle = 8.0` vs `mu_map = 6.545` with prior `N(0,1)`) and the
  prior-strength sweep, which climbs `0.135 -> 4.882` as `prior_sigma` tightens `10 -> 0.03`.
- **Confidence:** `struggled` — the quiz needed three rounds. Q3's precision arithmetic was
  right first time, but `lambda = 1/(2 tau^2)` lost its factor of 2, the flat-prior argument
  reached for `tau -> infinity` instead of a uniform prior's constant density, the derivative of
  `|z|` came out as `mu`, and the joint-vs-conditional sigma question went unanswered until it was
  decomposed. All five were correct by round three. No flags. Difficulty `hard`
  (self-reported: math in `learn.md` was the hard part); time not given.
- **Bugs found and fixed in the day's own test suite:** two assertions were unreachable with a
  correct implementation — `test_prior_sweep_monotonic`'s endpoint expected `mu_map ~= 5.0` at
  `prior_sigma=0.5` when the data's precision `n/sigma^2 ~= 27.6` makes that impossible (needs
  `tau <= 0.084`), and `test_closed_form_strong_prior_approaches_prior` used a seed whose sample
  variance is 0.36, not 1. Both corrected, plus the matching over-claim in `coding_problems.md`.
  Worth remembering for Week 4: **a tolerance that passes is not the same as a tolerance that
  tests something**, and `sklearn` comparisons will need the same scrutiny.
- **The number that mattered today:** `m = sigma^2/tau^2` — the prior expressed as a count of
  pseudo-observations. For `data=[0.5,-0.3,0.2]` at `prior_sigma=0.5` it is **0.436**, less than
  half a data point against 3 real ones. This is the quantity Week 3's Ridge `lambda` and Week 4's
  `prior_strength` hyper-parameter are really tuning.
- **Advances to:** Day 9 (global 9) — Shannon Entropy (`/i-am-in` next).
- **Badge unlocked today:** `comeback` (Day 7 closed 2026-08-24, Day 8 closed 2026-09-05 — a
  12-day gap, well past the 3-day threshold).
- **Code review:** `month_2/week_2/day_2/code_review.md` (12 items, all style; bare `raise
  ValueError` x4, `np.arange` excluding the endpoint the docstring promises, dead `n = len(data)`).
  Written inline — the `code-evaluator` subagent failed twice with a 403 auth error.
  `growth-notes.md` updated; three categories now stand at 2 days each, one short of promotion.

---

## Week 2 Day 3 — Shannon Entropy (global Day 9) — 2026-09-06

**Topic:** `entropy` + `entropy_of_text` (expected surprise; char- vs word-level text entropy)

**How it ties to `minisklearn`:** Entropy is the impurity measure behind Week 3's
Decision Trees (Day 4, ID3 splits on information gain = entropy drop) and the base unit
of tomorrow's cross-entropy/KL losses — the same losses Week 4 compares against `sklearn`.
The char-vs-word experiment is the first encounter with "choice of outcome space changes
the number", which returns when Week 3 compares per-sample vs per-class losses.

**Today's artefacts:** `month_2/week_2/day_3/{learn,roadmap,coding_problems}.md`,
`code/entropy.py`, `code/text_entropy.py`, `code/tests/` (15 tests), videos via
`video-researcher` (StatQuest entropy, 3Blue1Brown Wordle information theory — both
verified live via oEmbed + creator sites).
