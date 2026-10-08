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

**Topic:** `entropy` + `file_entropy` (exact Shannon entropy of a text file, char vs word level)

**How it ties to `minisklearn`:** Today's `H(P)` is the vocabulary every later loss is written in. Day 10's cross-entropy `H(P,Q)` is the loss Week 3 Day 2's Logistic Regression minimises and every neural net from Month 3 onward trains on. Week 3 Day 4's Decision Trees reuse `entropy()` almost verbatim for information-gain splits. Week 4's evaluation reports cross-entropy alongside accuracy — keep both files.

**Today's artefacts:** `month_2/week_2/day_3/{learn,roadmap,coding_problems}.md`, `code/entropy.py`, `code/file_entropy.py`, `code/tests/` (13 tests: pmf values + file ordering), videos via `video-researcher` (StatQuest entropy, 3Blue1Brown Compression-is-Intelligence Part 1; Part 2 deliberately deferred to Day 10).

---

## 2026-09-06 — Day 9 done: Shannon Entropy (global Day 9, Week 2 Day 3)

- **Result:** 13/13 tests passed. `entropy(pmf, base=2.0)` with explicit `0 * log 0 = 0`
  handling plus `file_entropy(path, level)` at char and word levels. The `__main__`
  experiments print the ordering proof (repetitive < English < random) and the char-vs-word
  pair on the same file.
- **Confidence:** `shaky` — Q2/Q3 correct first-ask, but the fair-die number came out in
  base 10 (`0.7781` instead of `2.585` bits), the word-vs-char reason needed restating in
  terms of alphabet size, and the `if p > 0` guard took three attempts (including one
  backwards proposal to use `>=`). No flags. Difficulty `easy` (self-reported, enjoyed the
  videos); time not given.
- **The number that mattered today:** per-token surprise follows alphabet size — tens of
  distinct characters vs tens of thousands of distinct words is why word-level entropy
  reads higher on the same file. Same reasoning returns tomorrow for why a flat predictive
  distribution is maximally uncertain, and in Week 3 for information-gain splits.
- **Advances to:** Day 10 (global 10) — Cross-Entropy & KL Divergence (`/i-am-in` next).
- **Code review:** `month_2/week_2/day_3/code_review.md` (9 items + 7 nits, all style;
  headline: duplicate `import math`, `pdf` misnomer for a discrete distribution, `1e-9`
  docstring vs `0.001` code, redundant `FileNotFoundError` re-raise). Written by
  `code-evaluator`, which worked this time. `growth-notes.md` updated — still no
  3-day patterns; `sum([...])` streak broken (improvement logged).
## Week 2 Day 4 — Cross-Entropy & KL Divergence (global Day 10) — 2026-09-07

**Topic:** `cross_entropy` + `kl_divergence` + `binary_log_loss` (the cost of the wrong distribution, the classifier's loss in miniature)

**How it ties to `minisklearn`:** This is the loss-function implementation day. `binary_log_loss` is the miniature BCE that Week 3 Day 2's Logistic Regression minimises, and today's functions (or their direct descendants) land in Week 4's `metrics.py` / `linear_model.py`, where evaluation reports cross-entropy alongside accuracy. The KL-asymmetry intuition foreshadows Week 3 Day 4: information gain is a KL-shaped quantity whose direction is what makes it meaningful.

**Today's artefacts:** `month_2/week_2/day_4/{learn,roadmap,coding_problems}.md`, `code/cross_entropy.py`, `code/log_loss.py`, `code/tests/` (15 tests, proven 15/15 vs a throwaway reference impl), videos via `video-researcher` (3Blue1Brown Compression-is-Intelligence Part 2 with KL chapter, StatQuest NN Part 6 Cross Entropy, StatQuest Mutual Information as KL-in-the-wild reinforcement).

---
## Week 2 Day 5 — Mutual Information (global Day 11) — 2026-09-08

**Topic:** `mutual_information` + `rank_features_by_mi` (uncertainty reduction, distance from independence, the feature-selection tool)

**How it ties to `minisklearn`:** This is the feature-selection day. Week 4 Day 2's EDA step decides which columns of a real dataset are worth modelling — `rank_features_by_mi` is the principled version of that decision, replacing correlation tables that go blind on nonlinear structure. The deeper tie is Week 3 Day 4: information gain is MI between feature and label in different notation, so today's function is nearly the split criterion verbatim. Keep both files.

**Today's artefacts:** `month_2/week_2/day_5/{learn,roadmap,coding_problems}.md`, `code/mutual_information.py`, `code/feature_selection.py`, `code/tests/` (21 tests, proven 21/21 vs a throwaway reference impl), videos via `video-researcher` (StatQuest MI, Complexity Explorer MI tutorial, Ben Lambert intro — all verified via oEmbed).

---

## 2026-09-08 — Day 10 done: Cross-Entropy & KL Divergence (global Day 10, Week 2 Day 4)

- **Result:** 15/15 tests passed. `cross_entropy(p, q)` + `kl_divergence(p, q)` with the
  `q(x)==0 where p(x)>0` guard raising `ValueError`, plus `binary_log_loss(y_true, y_pred)`
  as mean binary cross-entropy. Hand values confirmed: `CE ≈ 1.737`, `KL(P||Q) ≈ 0.737`
  vs `KL(Q||P) ≈ 0.531`, confident-wrong `3.322` vs confident-correct `0.152`.
- **Confidence:** `shaky` — all numbers first-ask and Q4/Q5 clean, but the Q1-why (why CE
  must exceed H) and Q2-why (why direction matters) were circular on round 1 and needed
  re-asks; explain-back skipped. No flags. Difficulty `easy` (self-reported); time not given.
- **The number that mattered today:** the extra `0.737` bits — the price of coding with the
  wrong distribution. Same reasoning returns in Week 3 Day 2 (logistic-regression loss) and
  Day 4 (information-gain splits).
- **Advances to:** Day 11 (global 11) — Mutual Information (`/i-am-in` next).
- **Code review:** `month_2/week_2/day_4/code_review.md` (7 items + nits, all style;
  headline: duplicated-then-diverged validation across siblings, exact float `sum != 1`).
  `growth-notes.md` updated — first Theme promotions (leftover scaffold, list-for-reduction,
  docstring-vs-code, whitespace, message wording). Durable process fix from user feedback:
  future stubs carry the full `ValueError` contract in their docstrings.

---

## 2026-09-09 — Day 11 done: Mutual Information (global Day 11, Week 2 Day 5)

- **Result:** 21/21 tests passed. `mutual_information(x, y)` (empirical joint/marginal
  counting + KL-shaped sum, any hashable labels) plus `rank_features_by_mi(features, y)`
  returning `(index, score)` best-first. Hand values confirmed: correlated binary `1.0`
  bit, independent binary `0.0`, `Y = X^2` demo prints covariance `0.0` vs MI `≈ 2.28`.
- **Confidence:** `shaky` — Q1–Q3 first-ask (Q1 with a quadratic-vs-exponential terminology
  fix), but Q5 (ranking order as API contract) took one re-ask and Q4 (MI bounded by
  marginal entropies) took two. No flags. Difficulty `medium` (self-reported); time `2-3
  hours` (self-reported as a range, so `time_spent_minutes` stays `null` — never estimated).
  Confusion note, verbatim-ish: a bit confused in code from overthinking; all fixed by
  the end.
- **The number that mattered today:** `0.0` vs `2.28` — covariance blind, MI not, on the
  same `X, X^2` pair. Same reasoning returns in Week 3 Day 4 (information-gain splits).
- **Advances to:** Day 12 (global 12) — Review + Consolidation (`/i-am-in` next, then
  the Week 2 review quiz at `/done`).
- **Code review:** `month_2/week_2/day_5/code_review.md` (7 items + nits, all style;
  headline: Cartesian-product loop instead of iterating observed pairs, `Counter` from
  `typing`, `range(len(...))` indexing ×3). `growth-notes.md` updated — new Theme
  (UPPERCASE accumulator, days 9–11), spelling-typo count promoted and folded into the
  wording Theme (days 7, 8, 11).

## 2026-09-11 — Day 12 done: Week 2 Review + Consolidation (global Day 12, Week 2 Day 6) + Week 2 review

- **Result:** 12/12 tests passed, plus the `__main__` assertions. Manual
  spot-check caught a sign bug the suite misses (`CE = -1.737`, `KL = -2.737`
  from `output += p log q`); fixed to `-sum` with `base` threaded through
  `kl_divergence`, re-verified at `CE = 1.737`, `KL = 0.737`. Consolidated
  `week2_utils.py` now holds `bayes_update`, `log_likelihood`,
  `mle_fit_gaussian`, `map_fit_gaussian`, `entropy`, `cross_entropy`,
  `kl_divergence`, `mutual_information` behind full `ValueError` contracts.
- **Confidence:** `struggled` — consolidation quiz took five rounds (Q1/Q3
  repeated re-asks; three discipline-log entries for asking to be told the
  answers and to mark Q1 correct as-is, all declined with flagging offered, no
  flags raised). Week 2 review took three rounds (W1/W2 both needed re-asks).
  Self-reported time `4-5 hours` (range, so `time_spent_minutes` stays `null`);
  difficulty `easy to medium` (not a single value, so `difficulty` stays
  `null`). Hand values confirmed: `H = 1.0`, `CE ≈ 1.737`, `KL(P||Q) ≈ 0.737`,
  `KL(Q||P) ≈ 0.531`, correlated-binary MI `1.0`, independent MI `0.0`.
- **Week 2 review:** passed — MLE (`argmax log P(X|theta)`) vs MAP
  (`argmax [log P(X|theta) + log P(theta)]`) coinciding at a uniform prior, and
  `H(P,Q) = H(P) + D_KL(P||Q)` with `H(P)` constant in `Q`, so minimizing CE
  and minimizing KL give the same argmin. `review_status` set to `done`.
- **Advances to:** Day 13 (global 13) — Week 3 Day 1, Linear Regression
  (`/i-am-in` next).
- **Code review:** `month_2/week_2/day_6/code_review.md` (16 findings + nits,
  all style; headline: duplicated `evidence` length check, `range(len(...))`
  ×3, `MI` uppercase accumulator, `rel_tol` vs `abs_tol` drift).
  `growth-notes.md` updated — `range(len(...))` promoted to Theme (days 7, 11,
  12); wording, whitespace, docstring-vs-code, and UPPERCASE-accumulator Themes
  extended to Day 12. No new badge.

---

## Week 3 Day 1 — Linear Regression (Gradient Descent) (global Day 13) — 2026-09-14

**Topic:** `LinearRegressionGD` (batch GD on MSE, `l2` support) + closed-form
`solve_normal_equation` reference via Gaussian elimination.

**How it ties to `minisklearn`:** This is Week 4's `linear_model.py` in
embryo — the `fit`/`predict` interface, the MSE objective, and the L2 knob
are what `LinearRegression` ships with. Day 14's logistic regression reuses
the same GD loop with BCE instead of MSE.

**Today's artefacts:** `month_2/week_3/day_1/{learn,roadmap,coding_problems}.md`,
`code/linear_regression.py`, `code/normal_equation.py`, `code/tests/`
(gradient finite-difference check, GD↔Normal-Equation agreement, L2
shrinkage), videos via `video-researcher` (StatQuest linear regression,
gradient descent step-by-step, multiple regression — titles/channels verified
via YouTube metadata).

---

## Week 2 Day 6 — Review + Consolidation (global Day 12) — 2026-09-10

**Topic:** `week2_utils.py` consolidating Bayes, Gaussian MLE/MAP, entropy, cross-entropy, KL divergence, and mutual information.

**How it ties to `minisklearn`:** This compatibility layer makes Week 2's formulas deliberate shared utilities for the Week 3 models: MAP explains L2 regularisation, cross-entropy becomes logistic-regression loss, entropy and MI support tree splits and feature selection, and the Gaussian estimators provide the parameter-fitting vocabulary.

**Today's artefacts:** `month_2/week_2/day_6/{learn,roadmap,coding_problems}.md`, `code/week2_utils.py`, and `code/tests/test_week2_utils.py`. The day is generated with a guided review sequence covering statistical estimation, Bayesian inference, and information/entropy.

---

## 2026-09-15 — Day 13 done: Linear Regression (Gradient Descent) (global Day 13, Week 3 Day 1)

- **Result:** 14/14 tests passed (`test_linear_regression.py` 11 + `test_normal_equation.py` 3). `predict`, `mse_loss`, `mse_gradients`, `LinearRegressionGD.fit/predict` with L2 tug (`2·l2·w_j`, bias exempt) plus hand-written Gaussian-elimination `solve_normal_equation` as the GD-vs-exact check.
- **Demo fix (day material):** `__main__` Experiments 1/3 used `lr=0.01` on unscaled data and diverged to `OverflowError`; lowered to `lr=0.001` (converges `26816 → ~27 → 25.5`, L2 shrinkage `18.32 → 18.13`). Experiment 2 kept at `lr=1.0` as the intentional divergence demo (`6e9 → 6e14 → 7e19 …`).
- **Confidence:** `struggled` — Q2 took three rounds (iterative-vs-exact), Q3 needed a re-ask + Experiment 2 lookup, Q4 flagged after four attempts (L2 loss-vs-gradient confusion, no prior named, circular `b` reason; logged to `flagged_questions.md`, `has_flags: true`), explain-back shallow (no gradient step, nothing on the Normal Equation). Self-reported time "about 6 to 8 hours" (range, `time_spent_minutes` stays `null`); difficulty "bitt medium to hard" (not a single value, `difficulty` stays `null`).
- **Advances to:** Day 14 (global 14) — Logistic Regression (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_1/code_review.md` written by `code-evaluator` (15 findings, style only); `growth-notes.md` updated (no new promotions; `range(len)`, wording, whitespace, docstring-vs-code, leftover-scaffold Themes extended to Day 13).

---

## Week 3 Day 2 — Logistic Regression (global Day 14) — 2026-09-16

**Topic:** `LogisticRegressionGD` (sigmoid + BCE, batch GD, `l2` support) +
hand-rolled `metrics.py` (confusion matrix, accuracy/precision/recall).

**How it ties to `minisklearn`:** This is Week 4's `linear_model.py`
second half alongside Day 13's regressor, and `metrics.py` is the embryo
of the evaluation step. Day 14 reuses Day 13's GD loop with BCE instead of
MSE; BCE itself is Week 2 Day 4's cross-entropy applied to 2 classes.

**Today's artefacts:** `month_2/week_3/day_2/{learn,roadmap,coding_problems}.md`,
`code/logistic_regression.py`, `code/metrics.py`, `code/tests/`
(15 tests: sigmoid values, BCE hand value, finite-difference gradient
check, GD fit + threshold validation, L2 shrinkage, metric hand values),
videos via `video-researcher` (StatQuest logistic regression, StatQuest GD
step-by-step, 3Blue1Brown DL ch.2 — runtimes unverifiable, omitted).

---

## 2026-09-17 — Day 14 done: Logistic Regression (global Day 14, Week 3 Day 2)

- **Result:** 15/15 tests passed on the first run (`sigmoid`, BCE hand value
  `0.228`, finite-difference gradient check, GD fit with loss decrease +
  accuracy ≥ 0.95, threshold validation, L2 shrinkage; metrics hand values
  `(tp, fp, tn, fn) = (2, 1, 1, 0)`, accuracy `0.75`, precision `2/3`,
  recall `1.0`).
- **Confidence:** `shaky` — Q1 (sigmoid-derivative derivation) and Q2
  (non-convex MSE+sigmoid, local minima) first-ask; Q3 took three rounds
  (uncertainty → "relative to class 1" → on the decision boundary,
  `z = 0`); Q4 two rounds (direction → predicted-positive-set reasoning);
  Q5 two rounds (gradient differences → `p_i(1-p_i)` cancellation, after
  declining a "which cancellation" ask as a hint and re-asking). No flags,
  no discipline entries. Explain-back covered the core with a steer to name
  BCE. Self-reported time "roughly 6 hours" (`time_spent_minutes: 360`);
  difficulty not given as a single easy/medium/hard value ("not that much
  hard" in Q6 feedback, so `difficulty` stays `null`). Day felt too long;
  classification metrics were brand new (needed an outside StatQuest video).
- **The number that mattered today:** `p_i(1-p_i)` — the sigmoid derivative
  that cancels out of the BCE gradient, leaving Day-13-shaped `(p_i − y_i)`.
  Same cancellation is the reason every neural net's output layer trains
  this cleanly from Month 3 on.
- **Advances to:** Day 15 (global 15) — Week 3 Day 3 (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_2/code_review.md` written by
  `code-evaluator` (9 findings, style only; headline: `metrics.py` trio
  re-loops instead of reusing `confusion_matrix`, validation copied across
  four methods). `growth-notes.md` updated — duplicated-validation
  promoted to Theme (days 10, 13, 14); `range(len)`, wording, whitespace,
  docstring-vs-code, leftover-scaffold Themes extended to Day 14. No new
  badge.

---

## Week 3 Day 3 — Support Vector Machines & the Kernel Trick (global Day 15)

**Topic:** `LinearSVMGD` (hinge loss + subgradient descent, `l2` support) +
`polynomial_kernel` + `add_radius_feature` (hand-built ring rescue).

**How it ties to `minisklearn`:** This is the margin-loss member of Week 4's
model loop — same `fit`/`predict` interface as Days 13–14, so a `LinearSVM`
drops into the comparison table unchanged, and the table gets more
interesting for having a non-likelihood loss in it. The ring-vs-radius
lesson (features beat algorithms) returns on Week 4 Day 2 when encoding
decisions on real data matter more than model choice. Note the label
convention shift (`{-1, +1}` vs `{0, 1}`) — Week 4's `metrics.py` must
handle both.

**Today's artefacts:** `month_2/week_3/day_3/{learn,roadmap,coding_problems}.md`,
`code/svm.py`, `code/tests/` (16 tests, proven 16/16 vs a throwaway
reference impl: linear fit 0.775 on ring vs 1.0 rescued, gap-asserted),
videos via `video-researcher` (StatQuest SVM Parts 1–3 + ritvikmath
optional, all verified).

---

## 2026-09-19 — Day 15 done: Support Vector Machines & the Kernel Trick (global Day 15, Week 3 Day 3)

- **Result:** 16/16 tests passed on the first run (`hinge_loss`,
  `hinge_subgradients` incl. kink-==1-nothing, `LinearSVMGD` fit with
  `l2` (bias exempt) + `decision_function`/`predict`, `polynomial_kernel`
  hand value `144.0`, similarity ordering `3.61 > 2.25`,
  `add_radius_feature` rescue; linear SVM separates blobs, visibly fails
  on the ring, radius feature rescues to high accuracy).
- **Confidence:** `shaky` — Q1 took three rounds (claim-without-mechanism,
  then hinge-half-only, then both halves with `max(0, 1 - y*s)` vs
  `-log(p)`); Q2–Q4 first-ask; Q5 correct. No flags. Explain-back gave the
  right skeleton with a steer to name the exact-zero and the
  support-vector-only pull. Self-reported difficulty `hard` ("hard hell as
  hard"); time not given. Material feedback: add longer big-lecture math
  videos as optional references on future days.
- **The number that mattered today:** `144` vs `3.61 > 2.25` — the kernel
  as similarity: neighbour outscores far without ever building the lifted
  vectors. Same "features beat algorithms" lesson returns Week 4 Day 2.
- **Advances to:** Day 16 (global 16) — Week 3 Day 4 (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_3/code_review.md` written by
  `code-evaluator` (12 findings, style only; headline: unused `C`,
  dead-and-broken `magnitude`, 4-way duplicated validation).
  `growth-notes.md` updated — 4 new Themes promoted (vague `output`,
  `== None`, redundant-paren `return`, untyped helpers); `range(len)`,
  wording, whitespace, docstring-vs-code, leftover-scaffold,
  duplicated-validation Themes extended to Day 15. No new badge.

---

## Week 3 Day 4 — Decision Trees (ID3 Algorithm) (global Day 16)

**Topic:** `entropy` + `information_gain` + `majority_label` + recursive
`DecisionTreeID3` (`fit`/`predict`/`to_dict`, `max_depth` /
`min_samples_split`) + `format_tree` (readable print).

**How it ties to `minisklearn`:** This is Week 4's `tree.py` in embryo —
the `fit`/`predict` interface matches Days 13–15 exactly, so a
`DecisionTreeClassifier` drops into the model loop unchanged, and Day 17's
`ensemble.py` (Random Forest) is a bagged crowd of today's class with a
feature-subset knob. The depth experiment (`max_depth=1` ≈ 0.714 vs
unlimited = 1.0 train accuracy on tennis) is Week 3's first honest
overfitting demo — Week 4's evaluation repeats the "training accuracy is
not the point" lesson on real data. Note the label convention shift
(arbitrary hashables, no label-range validation) — Week 4's `metrics.py`
must handle string labels too.

**Today's artefacts:** `month_2/week_3/day_4/{learn,roadmap,coding_problems}.md`,
`code/decision_tree.py`, `code/tests/` (18 tests, proven 18/18 vs a throwaway
reference impl: tennis gains Outlook 0.247 > Humidity 0.152 > Wind 0.048 >
Temperature 0.029, textbook tree shape, depth-1 vs full gap-asserted),
videos via `video-researcher` (StatQuest Decision Trees + ML Recipes #8,
both verified; no third long-lecture pick met the verification bar —
video search was down at generation time).

---

## 2026-09-20 — Day 16 done: Decision Trees (ID3 Algorithm) (global Day 16, Week 3 Day 4)

- **Result:** 18/18 tests passed (`entropy` hand values incl. tennis
  `0.940`, IG hand values `1.0` / `≈0.0817`, tennis ordering Outlook
  `0.247` > Humidity `0.152` > Wind `0.048` > Temperature `0.029`, root
  Outlook, overcast pure-`yes`, depth-1 `10/14` vs full `1.0` gap-asserted,
  unseen-value fallback, full `ValueError` contracts).
- **Confidence:** `shaky` — Q1/Q2/Q4/Q5 first-ask; Q3 took one follow-up
  (stated overfitting correctly but didn't connect the `1.0` train accuracy
  to memorisation until asked). No flags. Explain-back gave the core
  (max-gain feature) with a steer toward recursion. Self-reported difficulty
  `hard`; time "6-7 hours" (range, so `time_spent_minutes` stays `null`).
- **Material feedback (second day running):** the two short videos didn't
  land — same note as Day 15 — explicit ask for full-math big lectures as
  the primary resource, plus "today's class is so big" (matches the review's
  ~70-line nested-`build` finding). Future generations should lead with a
  long lecture pick where one verifies, and consider splitting the ID3 class
  across two files once the tree grows past one screen.
- **The number that mattered today:** `1.000` — the unlimited tree reciting
  all 14 tennis rows back. Training accuracy as a memorisation confession,
  not an achievement. Same lens Week 4's evaluation step applies on real data.
- **Advances to:** Day 17 (global 17) — Week 3 Day 5, Random Forests
  (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_4/code_review.md` written by
  `code-evaluator` (10 findings, style only; headline: `Counter` from
  `typing`, informal error-message voice, duplicated validation ripe for a
  `_validate_X_y` helper). `growth-notes.md` updated — no new promotions;
  `range(len)`, `== None`, redundant-`return` streaks all broken (sustained
  fixes). No new badge.

---

## Week 3 Day 5 — Random Forests (global Day 17) — 2026-09-21

**Topic:** `bootstrap_indices` + `bootstrap_sample` (`random.Random` with replacement, `1/e` OOB) + `RandomForestClassifier` (`n_estimators` bagged ID3 trees, per-split `max_features` feature randomness, majority vote)

**How it ties to `minisklearn`:** This is Week 4's `ensemble.py` in embryo — the `RandomForestClassifier` you write today *is* `ensemble.py` once renamed, behind the same `fit`/`predict` interface as `linear_model.py` and `tree.py`. Day 16's `DecisionTreeID3` becomes a bagged crowd with one knob (`max_features`) added; the depth-vs-ensemble comparison (`max_depth=1` vs unlimited vs forest) is the direct precedent for the model-comparison table on Week 4 Day 3. The OOB size check (`≈36.8%` via `(1−1/n)^n → 1/e`) also foreshadows Week 4's hold-out evaluation — OOB is evaluation without an explicit test split.

**Today's artefacts:** `month_2/week_3/day_5/{learn,roadmap,coding_problems}.md`, `code/random_forest.py`, `code/tests/` (bootstrap determinism/OOB + forest fit/predict + `n_estimators` sweep + feature-randomness ablation, 22 tests total), videos via `video-researcher` (StatQuest Random Forests Part 1, Patrick Loeber ML-From-Scratch #10, StatQuest Bootstrapping Main Ideas — all verified via oEmbed).

---

## 2026-09-29 — Day 17 done: Random Forests (global Day 17, Week 3 Day 5)

- **Result:** 21/21 tests passed (`bootstrap_indices` determinism/duplicates,
  OOB `≈632/1000` + mean `≈5.1/14`, forest fit/predict on tennis, single-tree
  equivalence with no-bag-all-features, `max_features` variants, unseen-value
  fallback, `bootstrap=False`, full `ValueError` contracts, `n_estimators`
  sweep). One assistant-side test fix before the quiz:
  `test_n_estimators_sweep_monotonic_stability` demanded B=100 within 0.15 of
  B=5 on a 4-row slice (one row = 0.25) — relaxed to 0.25 with a comment.
- **Confidence:** `struggled` — 6 re-asks total, no flags. Q1 swapped OOB/in-bag
  first attempt (63%/9 → ~36.8%/~5); Q2 circular first attempt, mechanism on
  re-ask; Q3 first-ask and thorough (dominant-feature correlation); Q4 named
  B=20 first attempt but blamed overfitting, slice-noise on re-ask; Q5 took
  three re-asks (pairing-guess → docstring paste → identical-trees mechanism).
  Explain-back skipped (asked, no response). Self-reported difficulty `medium`;
  time "4 to 5 hours" (range, so `time_spent_minutes` stays `null`).
- **Material feedback (first positive since Day 14):** videos "good", class size
  "about right" (contrast Day 15/16 "so big" + short-video complaints).
- **The number that mattered today:** `0.80 → 0.90 → 1.00 → 1.00` train vs
  `0.75 → 0.75 → 0.75 → 0.50` test across B=`1,5,20,100` — plateau by 20,
  B=100 dip is one flipped row on a 4-row slice, not regression.
- **Advances to:** Day 18 (global 18) — Week 3 Day 6, K-Means Clustering
  (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_5/code_review.md` (10 findings, style only;
  headline: `n_estimatore`/`min_smaple_split` typos, `self.estimates` misnomer,
  3-way duplicated validation). `growth-notes.md` updated — two new Themes
  promoted (bare-sibling-import days 9/14/17; attribute-first-assigned-in-`fit`
  days 13/16/17); whitespace, wording, duplicated-validation, `range(len)`,
  vague-accumulator, untyped-helpers Themes extended to Day 17. No new badge.

---

## Week 3 Day 6 — K-Means Clustering (global Day 18) — 2026-09-29

**Topic:** `euclidean` + `assign_clusters` + `recompute_centroids` + `inertia` + `standardize` + `KMeans` (Lloyd's assign/update loop, distinct-point init, `tol`/`max_iters`/`n_iters_`/`cluster_centers_`) + `elbow_inertias` (k=1..8, bend at 3)

**How it ties to `minisklearn`:** This is Week 4's `cluster.py` in embryo — the `KMeans` you write today *is* `cluster.py` once renamed, behind the same `fit`/`predict` interface as the supervised models (with `predict` meaning "assign to nearest centroid"). Unlike Days 13–17 there is no accuracy table for it — Week 4 evaluates it by inertia and by eyeball on the same EDA plots from Day 2. The standardize-then-cluster lesson (features beat algorithms, again after Day 15's radius rescue) returns on Week 4 Day 2 when encoding/scaling decisions on real data matter more than model choice. Last day of Week 3 — `/done` runs the Week 3 review quiz against the week's `done_when` before closing the week.

**Today's artefacts:** `month_2/week_3/day_6/{learn,roadmap,coding_problems}.md`, `code/kmeans.py`, `code/tests/` (helpers hand checks + 3-blob fit + determinism + local-optima seeds + elbow + full `ValueError` contracts), videos via `video-researcher` (StatQuest K-means clustering, freeCodeCamp ML-for-Everybody K-Means sections — both verified via oEmbed; elbow + never-increases argument specced by hand in `learn.md` since neither video covers them).

---

## 2026-10-08 — Day 18 done: K-Means Clustering (global Day 18, Week 3 Day 6) + Week 3 review passed

- **Result:** 20/20 tests passed (`euclidean` hand values, assignment +
  tiebreak, centroid means + empty-cluster keeps-centroid, inertia hand
  value 2.0, standardize incl. zero-variance, 3-blob purity + centroid
  proximity, determinism, k=1 mean, `n_iters_`, full `ValueError` contracts,
  elbow). Three debugging episodes during the build (all logged in-session):
  (1) `assign_clusters` picked `max` (farthest) not `min`, plus inner loop
  `range(n)` over rows instead of centroids — passed the 3×3 unit test by
  coincidence, IndexError on 60-row fit data; (2) my test typo (k=3/label-2
  is valid input) → k=2; (3) learner accidentally deleted the label-range
  check mid-edit (green test turned red) — restored, `git diff` habit taught.
- **Confidence:** `shaky` — Q1/Q4/Q6 first-ask, Q2 one re-ask (minimization
  half before local-optimum half), Q3 accepted via Month-1 PCA sighting, Q5
  three rounds (sklearn's farthest-point first, then mechanism without why —
  combined). No flags. Explain-back given, needed one steer on the
  why-converges half. Self-reported difficulty `hard`; time and class size
  not reported (so `time_spent_minutes` stays `null`).
- **Material feedback:** videos 2/10 (StatQuest K-means + freeCodeCamp
  sections, both watched) — lowest video rating of the month. Follow-up:
  implementation-first picks for Week 4.
- **Week 3 review: passed** — classical ML not outdated (interpretability
  for regulated small-table jobs: readable tree paths vs opaque weights),
  K-Means as the no-labels complement (produces cluster assignments the
  supervised days can't). One discipline-log entry: initial refusal to
  answer the review (LLM/AGI goal cited) — declined, re-asked in smaller
  pieces (R1 took three rounds, R2 two), then passed.
- **Advances to:** Day 19 (global 19) — Week 4 Day 1, Package Structure
  (`minisklearn`) (`/i-am-in` next).
- **Code review:** `month_2/week_3/day_6/code_review.md` (12 findings, style
  only; headline: whitespace clusters, 6× duplicated validation, `== None`
  and redundant-parens reappearing after Day 16–17 fixes). `growth-notes.md`
  updated — one new Theme promoted (`list()`-for-empty-list, days 14/15/18);
  attribute-in-`__init__` fix holds. No new badge.

---

## Week 4 Day 1 — Package Structure (global Day 19) — 2026-10-08

**Topic:** `minisklearn/` skeleton + move map + `metrics.py` (new) + `__init__.py` front door

**How it ties to `minisklearn`:** Today IS the final project, day one of six — the package created here is the artifact Days 2–6 operate on (real data → eval → sklearn showdown → write-up → GitHub). Unlike Weeks 1–3, implementations are COPIES (the week folders are the archive, never edited for Week 4 reasons) and the package lives month-level at `month_2/minisklearn/` so all six days share it. The uniform `.fit/.predict` interface is load-bearing: Day 3's harness loops over models without glue code, Day 4 runs the identical loop against real `sklearn`. Only genuinely new code today is `metrics.py` (adapting Day 14's trio + new `f1_score`); everything else is move + normalize + document.

**Today's artefacts:** `month_2/week_4/day_1/{learn,coding_problems,roadmap}.md` (videos via `video-researcher`: Tech With Tim `__init__.py` deep-dive, 2MinutesPy 3-min API-design nudge, ArjanCodes packaging arc first half — all verified live; deliberately short/build-first after the Day-18 2/10 verdict), `month_2/minisklearn/` (`__init__` + 4 move-map skeletons + `metrics.py` stub with full `Raises` contracts), `code/tests/test_package.py` (imports + top-level exposure + interface uniformity + smoke fits + metrics hand values, ~11 tests; green-on-skeleton: 1 pass, 10 fail on `NotImplementedError`/missing names — the task list, not a regression).
