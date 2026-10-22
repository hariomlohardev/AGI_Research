# Struggle Log

Conceptual sticking points surfaced during quizzes — not code style (that's `growth-notes.md`). A theme becomes a "pattern" only after the same underlying concept causes trouble on 3+ separate days.

## 2026-08-24 — Day 7 (Month 2 Week 2 Day 1) — Maximum Likelihood Estimation (MLE)

- Single-day quiz went clean: 6/6 correct first-ask, no flags, no re-asks. No conceptual struggle flagged today; one self-reported snag was mechanical (grid `steps` iteration between start/end inclusive bounds) which was resolved via external search — not a math misunderstanding. Tagged `math-heavy`/`coding-heavy` day; confidence `strong`.
- No pattern promoted — still below 3-day threshold for any topic.

## 2026-09-05 — Day 8 (Month 2 Week 2 Day 2) — MAP Estimation & Priors

- Six-question quiz took **three rounds**. Round 1: Q1/Q3/Q4 answered with "don't understand the
  question", Q2 and Q5 answered but wrong. Round 2 (questions restated concretely): Q3's arithmetic
  came out fully correct (27.5 / 4.0 / 6.9x), Q1 part 1 correct, Q4's L2 derivative correct — but
  `lambda` was given as `1/tau^2` instead of `1/(2 tau^2)`, the flat-prior argument reached for
  `tau -> infinity` instead of the uniform prior's constant density, the L1 derivative was given as
  `mu`, and Q5 went unanswered. Round 3: all five correct and well-argued.
- Concepts that needed the re-asks, on this day:
  - **"A constant in the parameter cannot move the argmax."** Used correctly on Q1 part 1 (dropping
    `-0.5 log(2 pi tau^2)`) but not transferred to Q2's uniform prior, where the same argument is the
    whole answer. The idea landed locally without generalising.
  - **The Gaussian-prior -> L2 correspondence**, specifically the factor of 2 in
    `lambda = 1/(2 tau^2)` — dropped on first attempt despite being written in `learn.md:28` and in
    the learner's own file docstring.
  - **Derivative of `|z|`** — L1's constant-magnitude gradient vs L2's vanishing one. This is the
    mechanism behind sparsity, so it is worth a second pass.
  - **Joint vs conditional optimisation** (grid over `(mu, sigma)` vs closed form at
    `sigma = sigma_mle`) — no answer at all until the question was decomposed.
- Self-reported, verbatim: *"the thing is today i find math difficult in learn"*. Recorded
  `difficulty: "hard"`; `time_spent_minutes` left `null` (never given).
- Also asked mid-quiz to skip the remaining questions and mark the day done (headache). Declined and
  logged in `discipline-log.md`; the questions were answered afterwards, so no flags were raised and
  `has_flags` stays `false`.
- Confidence recorded as `struggled` (repeated re-asks across four of six questions). No pattern
  promoted — day 7's quiz was clean (6/6 first-ask), so this is the first day any of the above
  concepts caused trouble. Threshold for a pattern is 3 separate days.

## 2026-09-06 — Day 9 (Month 2 Week 2 Day 3) — Shannon Entropy

- Six-question quiz took **two rounds** (one question took three). Round 1: Q2/Q3 correct
  first-ask, Q6 feedback given. Q1 gave `0.7781` for the fair die (log base 10 used instead
  of base 2 — the expected `log2(6) = 2.585` bits); Q4 correctly said word-level entropy is
  higher but explained it as "not that much words to be compacted" rather than alphabet
  size; Q5 named the right failure (`math.log(0)` -> `ValueError: math domain error`) but
  framed the guard as being about negative probabilities. Round 2: Q1 recomputed correctly
  (`1 + 1.585 = 2.585` bits), Q4 restated properly (tens of distinct chars vs tens of
  thousands of distinct words, so per-token surprise is higher); Q5 wobbled — proposed
  changing `>` to `>=` and said the guard filters out `-0.2`, which is backwards (negatives
  are caught by validation at `entropy.py:36-37`; the guard skips zeros; `>=` would *include*
  zero and crash). Round 3: affirmed the operative point (keep `>`, otherwise `log` errors).
- Concepts that needed the re-asks, on this day:
  - **Log base discipline** — bits mean base 2, and the test suite's `log2(6)` expectation
    is the check. First-day occurrence of this slip.
  - **The `if p > 0` guard's actual job** (skip zeros, not negatives) — needed three
    attempts and is still the shakiest item from today; worth a `/spaced-review` probe.
- Explain-back needed one correction: "the rigged coin's information is more than the fair
  one" — reversed (rigged coin gives *less* information per toss, `0.0808` vs `1.0` bits,
  because the outcome is predictable). The "expected surprise" framing itself was right.
- Self-reported difficulty `easy` ("easy pizy", enjoyed the videos); `time_spent_minutes`
  left `null` (answered "NA").
- Confidence recorded as `shaky` (three questions needed re-asks, one of them twice). No
  flags. No pattern promoted — day 7's quiz was clean and day 8's trouble was different
  concepts (MAP derivation, `|z|` derivative, joint-vs-conditional); none of today's items
  has appeared on 3 days.

## 2026-09-08 — Day 10 (Month 2 Week 2 Day 4) — Cross-Entropy & KL Divergence

- Six-question quiz took **three rounds**. Round 1: all hand numbers correct first-ask
  (`1.7369`, `0.7369` / `0.531`, `3.3219`); Q4 (q-zero `ValueError`) and Q5 (`CE − H = KL`
  identity) correct first-ask; Q6 feedback given. Q1-why was circular ("must exceed H(P)
  because H(P) is less and Q is more skewed to one side" — skewedness is not the general
  reason); Q2-why was circular ("swapping changes the answer because D_KL is asymmetric" —
  the name of the phenomenon, not the reason); Q3 accepted (the `y=0` formula walk-through
  contained the mechanism even if wordy). Round 2: Q2 accepted (P is the reference doing
  the weighting, Q the newcomer being judged). Round 3: Q1 accepted (Q isn't fitted to the
  world P describes, so coding P-outcomes with Q wastes bits).
- Explain-back was skipped — `/done` was re-run without answering it. Not a gate, noted
  here instead of stalling the session.
- Self-reported difficulty `easy`; `time_spent_minutes` left `null` (no time given).
  Durable feedback from Q6: generated stubs should list the `ValueError` contract in their
  docstrings (validation cases, not just the formula) — folded into future `/i-am-in` runs.
- Confidence recorded as `shaky` (two questions needed re-asks, one of them twice). No
  flags. No pattern promoted — `CE ≥ H` and KL asymmetry are first-time concepts; none of
  today's items has appeared on 3 days.

## 2026-09-09 — Day 11 (Month 2 Week 2 Day 5) — Mutual Information

- Six-question quiz took **three rounds**. Round 1: Q1 accepted with a terminology fix
  (`X^2` is quadratic, not exponential — covariance is blind to any nonlinear shape, not
  just that one); Q2 accepted (MI *is* a KL, KL can't go below zero); Q3 accepted
  (`I = 0` means independent, `Cov = 0` only means linearly uncorrelated); Q6 feedback
  NA. Q4 ("don't know, teach me") and Q5 ("don't understand it") went to re-ask —
  teaching mid-quiz was declined per the no-answers rule, flagging offered instead.
  Round 2: Q5 accepted (order is the interface — callers take `ranking[0]` as best, so
  ascending silently hands them the worst column). Round 3: Q4 accepted (the
  `I(X;Y) <= min(H(X), H(Y))` bound, so `1.35 > H(x) = 1.0` means the implementation
  is broken).
- Concepts that needed the re-asks, on this day:
  - **MI bounded by the marginal entropies** — needed two re-asks and still arrived as
    a stated rule rather than via `H(X) - H(X|Y)` with non-negative conditional
    entropy; worth a `/spaced-review` probe.
  - **Ranking order as API contract** — one re-ask; the "same information" intuition
    is true for a human reading a list, false for code taking the top entry.
- Explain-back covered the feature-selection use correctly but missed the day's actual
  insight (uncertainty removed / distance from independence, and why covariance can't
  see it) — one-line steer given, non-blocking.
- Confidence recorded as `shaky` (two questions needed re-asks, one of them twice). No
  flags. No pattern promoted — MI bounds and ranking semantics are first-time
  concepts; none of today's items has appeared on 3 days.

## 2026-09-11 — Day 12 (Month 2 Week 2 Day 6) — Week 2 Review and Consolidation

- Twelve consolidation tests passed first run (12/12), but a manual spot-check
  blocked `/done`: `cross_entropy` accumulated `+sum(p log q)` instead of
  `-sum(p log q)`, so `CE = -1.737` and `KL = -2.737`. Fixed by the learner
  before the quiz resumed (sign flip plus `base` threading through
  `kl_divergence`); re-run shows `CE = 1.737`, `KL = 0.737` with 12/12 still
  green. Tests alone did not catch the sign because the suite only checks the
  `CE - H = KL` identity, which holds for the negated values too.
- Six-question consolidation quiz took **five rounds**. Round 1: Q4 hand values
  correct first-ask (`1.0`, `1.737`, `0.737`, `0.531`); Q5 accepted with a
  terminology fix (`Y = X^2` is quadratic, not exponential); Q6 feedback given.
  Q1 (Bayes/MLE/MAP + when MAP becomes MLE), Q2 (entropy/CE/KL + which term is
  constant), and Q3 (flat prior without a limit) went to re-ask. Round 2: Q2
  accepted (`KL = CE - H`, `P` constant); Q1/Q3 still open. Mid-quiz the learner
  asked to be told what the questions wanted, then twice asked for the Q1
  attempt to be marked correct as-is — both declined per the no-answers rule
  and logged in `discipline-log.md`; flagging offered each time, never used.
  Round 3-4: Q3 accepted (uniform prior is a constant scaling factor, so the
  argmax does not move); Q1 accepted on round 5 (posterior vs likelihood
  objectives, uniform prior makes them identical).
- Week 2 review (against the week's `done_when`) took **three rounds**. W1
  accepted on round 3 (MLE `argmax log P(X|theta)` vs MAP
  `argmax [log P(X|theta) + log P(theta)]`, coinciding at a uniform prior;
  the extra asymptotic-infinite-data condition offered is true but beyond what
  was asked). W2 accepted on round 3 (`H(P,Q) = H(P) + D_KL(P||Q)`, `H(P)` is
  constant in `Q` so minimizing CE and minimizing KL give the same argmin).
- Concepts that needed the re-asks, on this day:
  - **Stating both halves of a "relate X, Y, Z" question** — first attempts
    gave one half (e.g. "MAP uses a prior", "KL = CE - H") without the asked
    second half (when MAP becomes MLE; which term is constant and why).
  - **Flat prior as a constant that cannot move an argmax** — needed several
    passes despite being the same argument as Day 8's dropped
    `-0.5 log(2 pi tau^2)` constant. Second occurrence across days (Day 8,
    Day 12), still below the 3-day pattern threshold.
- Explain-back was asked alongside the quiz but never answered separately;
  the quiz answers themselves covered both paragraphs by the end. Non-blocking,
  noted here instead of stalling.
- Self-reported time `4-5 hours` (a range, so `time_spent_minutes` stays
  `null` — never estimated); difficulty `easy to medium` (not a single
  easy/medium/hard value, so `difficulty` stays `null`, verbatim preserved in
  the logs). No flags.
- Confidence recorded as `struggled` (repeated re-asks on Q1/Q3/W1/W2 plus
  three discipline-log entries). No pattern promoted — none of today's items
  has appeared on 3 days.

## 2026-09-15 — Day 13 (Month 2 Week 3 Day 1) — Linear Regression (Gradient Descent)

- Six-question quiz + explain-back. Round 1: Q1 gradient derivation correct
  in essence (chain rule through `ŷ_i`, `2/n` factor) with notation cleanup
  (`Σ_i`, `X[i][j]`); Q2 first attempt wrong (framed GD-vs-Normal-Equation as
  generalization/overfitting), took three rounds to land the iterative-vs-exact
  distinction; Q3 loss-curve shapes partially answered, needed re-ask plus help
  locating Experiment 2 (`linear_regression.py:221-224`) before the exact
  `high-lr losses` values were reported; Q5 huge-init accepted first-ask;
  Q6 feedback given (day felt long, hand-written Normal Equation felt needless
  given numpy later).
- Q4 (L2 loss term + weight-gradient term + Gaussian prior + why `b` exempt)
  never answered correctly across four attempts — gradient line quoted back for
  the loss part, no prior named, `b` reason restated the code. Flagged via the
  escape hatch ("i litteraly dont understand whats going on in this"), logged
  to `month_2/week_3/day_1/flagged_questions.md`; `has_flags` set true.
- Explain-back shallow and half-missing: "learns from errors/loss" with no
  gradient-step mechanism (`w −= lr·grad` from zeros) and nothing on what the
  Normal Equation adds (exact closed-form check). One-line steer given,
  non-blocking.
- Demo bug (day material, not learner code): `__main__` Experiments 1/3 used
  `lr=0.01` on unscaled house-price data (features 8–16, targets ~200), so
  `fit` diverged to `OverflowError` in `mse_loss`. Fixed to `lr=0.001`
  (converges `26816 → ~27 → 25.5`); Experiment 2 kept at `lr=1.0` as the
  intentional divergence demo (`6e9 → 6e14 → 7e19 …`). Tests 14/14 green.
- Self-report verbatim: time "about 6 to 8 hours" (a range, so
  `time_spent_minutes` stays `null` — never estimated); difficulty "bitt
  medium to hard" (not a single easy/medium/hard value, so `difficulty`
  stays `null`, verbatim preserved here and in `progress.md`).
- Confidence recorded as `struggled` (repeated Q2 re-asks, Q3 re-ask + lookup
  help, Q4 flagged after four attempts, shallow explain-back). No pattern
  promoted — L2↔Gaussian-prior trouble previously appeared on Day 8 and
  Day 12, but Day 8 was the factor-of-2/flat-prior form and Day 12 the
  constant-argmax form; today's failure mode (loss-vs-gradient confusion, no
  prior named) is adjacent, not the same underlying slip a third time.
  Worth a `/spaced-review` probe on MAP↔L2 + iterative-vs-exact before Week 3
  builds on the GD loop.

## 2026-09-17 — Day 14 (Month 2 Week 3 Day 2) — Logistic Regression

- Six-question quiz + explain-back; 15/15 tests green on the first run.
  Round 1: Q1 sigmoid-derivative derivation correct first-ask (chain rule,
  split, substitute — the learn.md quota met); Q2 accepted first-ask
  (non-convex MSE+sigmoid surface, local minima) with a terminology fix
  (the surface is *non*-convex; convexity is what BCE preserves); Q6
  feedback given (day felt too long; classification metrics were new, needed
  a StatQuest video on top of the day material). Q3 (what `p = 0.5` means +
  where the point sits) took three rounds: uncertainty landed first, then
  "relative to class 1" for the geometry half, then "lies on the decision
  boundary" (`z = 0`). Q4 (threshold 0.5 → 0.3) took two rounds: direction
  (precision down, recall up) first, predicted-positive-set reasoning second.
  Q5 (BCE vs MSE gradient: two differences + the cancellation) took two
  rounds: differences (factor of 2, raw `ŷ` vs squashed `p`) first, then a
  "which cancellation are you talking about" ask — declined as a hint per
  the no-answers rule, exact question re-asked, `p_i(1-p_i)` named. No flags,
  no discipline-log entries.
- Explain-back covered the core (sigmoid squashes the linear score to
  (0, 1), training minimizes the loss) with a one-line steer to name the
  loss (binary cross-entropy — *which* loss is the point of the day).
  Non-blocking.
- Confidence recorded as `shaky` (three questions needed re-asks, Q3 twice;
  everything landed, no flags). No pattern promoted — second occurrence of
  the Day-12 slip (first attempts answer one half of a two-part question:
  Day 12 "relate X, Y, Z" halves; today Q3/Q4/Q5 each gave the first half
  before the second). Still below the 3-day threshold; if it shows again,
  promote it and probe with explicit "give both halves" framing.
- Learner-noted gap, not a quiz failure: precision/recall/confusion-matrix
  were brand new (needed outside video). Worth keeping in view when Week 4
  builds evaluation on top of `metrics.py`.

## 2026-09-19 — Day 15 (Month 2 Week 3 Day 3) — Support Vector Machines & the Kernel Trick

- 16/16 tests green on the first run (`hinge`, subgradients incl. kink,
  linear fit, ring fail/rescue, kernels).
- Five-question quiz + explain-back + 2 feedback. Q1 (why hinge is exactly 0
  vs CE never 0) took three rounds: round 1 restated the claim without the
  mechanism; round 2 gave the `max(0, ...)` clipping half without the
  `-log(p)` half; round 3 landed both (`max(0, 1 - y*s)` clips at 0 once
  `y*s >= 1`, `-log(p) > 0` for any `p < 1`). Q2 (support vectors as
  `y*s <= 1`, only non-zero subgradient) first-ask with terminology wobble
  (soft/hard margin) but the mechanism present. Q3 (kernel trick: linear
  separator after lifting, trick computes only the dot products) accepted
  first-ask. Q4 hand calc (`s = 1`, margin `1`, loss `0`) first-ask. Q5
  (`(11+1)^2 = 144`, `3.61 > 2.25` similarity ordering) correct. No flags,
  no discipline-log entries.
- Explain-back gave the right skeleton (margin classification, different
  loss, kernel avoids building higher-dim vectors) with a one-line steer to
  name the exact-zero (`y*s >= 1`) and the support-vector-only pull.
  Non-blocking.
- Self-reported difficulty `hard` (verbatim "hard hell as hard");
  `time_spent_minutes` stays `null` (no time given).
- Durable material feedback (F2): day helped, but asked that future days add
  longer big-lecture videos tied to each math piece as optional references.
- Confidence recorded as `shaky` (one question needed two re-asks; everything
  else first-ask, no flags).
- Pattern update: third occurrence of the Day-12/Day-14 slip (first attempts
  answer one half of a two-part question: Day 12 "relate X, Y, Z" halves;
  Day 14 Q3/Q4/Q5 first-half-only; today Q1 hinge-half before the CE half).
  Promoted to pattern — probe future two-part questions with explicit "give
  both halves" framing and a `/spaced-review` pass on hinge-vs-CE vs
  CE-vs-H. Hinge exact-zero itself is first-time; no concept pattern beyond
  the answering shape.

## 2026-09-20 — Day 16 (Month 2 Week 3 Day 4) — Decision Trees (ID3 Algorithm)

- 18/18 tests green on the first run (`entropy` hand values, IG hand values
  `1.0` / `≈0.0817`, tennis ordering Outlook `0.247` > Humidity `0.152` >
  Wind `0.048` > Temperature `0.029`, root Outlook, depth-1 `10/14` vs full
  `1.0`, unseen-value fallback, contracts).
- Five-question quiz + explain-back + 2 feedback. Q1 (IG hand calc) correct
  numbers first-ask; child entropies (`≈0.9183` each) supplied on follow-up.
  Q2 (pure node → gain exactly 0) first-ask. Q3 (unlimited-depth risk) took
  one follow-up: first answer stated overfitting without connecting the
  `1.0` train accuracy to memorisation; follow-up landed it (`1.000` is the
  tree reciting all 14 rows, noise included). Q4 (tennis trace
  `sunny → high → no`) first-ask. Q5 (why strings need no encoding: counting,
  never arithmetic) first-ask. No flags, no discipline-log entries.
- Explain-back gave the core (max-gain feature) with a one-line steer toward
  recursion (same question re-asked at every node). Non-blocking.
- Self-reported difficulty `hard`; time "6-7 hours" (a range, so
  `time_spent_minutes` stays `null` — never estimated).
- Material feedback (not quiz struggle): the two short videos didn't land —
  second day running after Day 15 — explicit ask for full-math big lectures
  as the primary resource; plus "today's class is so big" (matches the
  review's nested-`build` finding). Logged in `final_project/logs.md` for
  future `/i-am-in` runs.
- Confidence recorded as `shaky` (one question needed a follow-up;
  everything else first-ask, no flags).
- Pattern update: fourth occurrence of the Day-12/Day-14/Day-15 slip (first
  attempts answer one half of a two-part question — today Q3's overfitting
  half before the numbers half). Pattern stands; keep the explicit
  "give both halves" framing on multi-part questions.

## 2026-09-29 — Day 17 (Month 2 Week 3 Day 5) — Random Forests

- 21/21 tests green at `/done` (9 bootstrap + 12 forest). One test-suite fix
  by the assistant before the quiz: `test_n_estimators_sweep_monotonic_stability`
  demanded B=100 within 0.15 of B=5 on a 4-row test slice (granularity 0.25 —
  one flipped row fails it); relaxed to 0.25 with a comment. Same class of fix
  as Day 8's unreachable assertions, not learner code.
- Five-question quiz + 2 feedback, no flags, no discipline-log entries.
  Q1 (bootstrap duplicates + OOB fraction/count) first attempt swapped
  in-bag/OOB (63% / 9 for n=14); re-ask landed it (duplicates force omissions,
  ~36.8% OOB via 1/e, ~5 rows for n=14). Q2 (variance down / bias flat)
  first attempt was circular (restated the claim); re-ask landed the mechanism
  (averaging cuts variance toward tree-covariance, deep trees already low-bias).
  Q3 (feature randomness vs dominant-feature correlation) first-ask, thorough.
  Q4 (sweep plateau at B=20, B=100 dip as slice noise) took one re-ask: first
  attempt correctly named B=20 but attributed the dip to overfitting; re-ask
  landed it (bagging doesn't overfit more with B — 4-row slice artifact).
  Q5 (per-tree seed from shared rng vs same seed) took three re-asks
  (guessed "X/Y pairing + determinism", then pasted the docstring, then
  landed it: same seed in → identical resamples → 20 copies of one tree).
- Explain-back skipped (asked, no response) — non-blocking per /done step 5.
- Self-reported difficulty `medium`; time "4 to 5 hours" (a range, so
  `time_spent_minutes` stays `null` — never estimated).
- Material feedback: videos "good" this time (StatQuest RF + Loeber +
  Bootstrapping — first positive video note since Day 14); class size
  "about right" (contrast Day 15/16 "so big").
- Confidence recorded as `struggled` (6 re-asks total, Q5 alone took 4 rounds).
- Pattern update: fifth occurrence of the Day-12/Day-14/Day-15/Day-16 slip
  (first attempts answer one half of a two-part question — today Q4's plateau
  half before the noise-mechanism half, Q5's determinism half before the
  identical-trees half). Pattern stands; keep the explicit "give both halves"
  framing on multi-part questions.

## 2026-10-08 — Day 18 (Month 2 Week 3 Day 6) — K-Means Clustering

- 20/20 tests green at `/done` (2 euclidean + 2 assign + 2 recompute + 2
  inertia + 3 standardize + 8 fit/predict/elbow + 1 elbow-sweep). Two
  assistant-side fixes during the build (both documented in-session, not
  learner code): `assign_clusters` inner loop `range(n)` → `range(len(centroids))`
  (passed 3×3 unit test by coincidence, IndexError on 60-row fit data);
  `test_recompute_centroids_rejects_bad_input` case used k=3/label-2 (valid
  input — empty clusters keep prev centroids) → k=2. One learner regression
  caught live: the label-range check got deleted in an edit (passing test
  turned red); restored, plus the `git diff` habit taught for
  passed-then-fails regressions.
- Six-question quiz + 2 feedback, no flags, no discipline-log entries.
  Q1 (seed sensitivity → local optima) first-ask correct. Q2 (never-increases
  argument + why not global) first attempt gave the minimization half only;
  re-ask landed the local-vs-global half (greedy downhill, no climbing out).
  Q3 (×1000 stretch mechanism + standardize rescue) first-ask, thorough
  (variance ×1e6, blind to feature 1); cited Month-1 PCA for the "seen before"
  half instead of the in-course Week-1-Day-5/Day-15 sightings — accepted, same
  lesson. Q4 (elbow picks vs proves) first-ask correct. Q5 (empty-cluster
  convention in the learner's own code) took three rounds: first attempt
  described sklearn's farthest-point reassignment instead of the learner's
  keep-prev-centroid; second gave the mechanism ("because you said") without
  the why; accepted by combining with the first attempt's why (mean of zero
  members is 0/0, math would happily drop to k−1). Q6 (`n_iters_` meaning +
  max_iters-with-shift warning) first-ask correct.
- Explain-back given (nearest-assignment + converges-but-local); needed a
  one-line steer sharpening the why-converges half (never-increases argument)
  — non-blocking per /done step 5.
- Self-reported difficulty `hard`; time not reported ("ya" to leaving it
  blank, so `time_spent_minutes` stays `null` — never estimated); class size
  not reported.
- Material feedback: videos 2/10 (StatQuest K-means + freeCodeCamp ML-for-
  Everybody sections, both watched) — lowest video rating of the month
  (contrast Day 17 "good"). Follow-up: lean implementation-first picks for
  Week 4 days; the bite-sized from-scratch series format (Loeber-style) has
  the best reception record.
- Confidence recorded as `shaky` (Q2 one re-ask, Q5 three rounds, no flags).
- Pattern update: sixth occurrence of the half-answer slip (Q2's minimization
  half before the local-optimum half; Q5's sklearn-what before the own-code
  what+why). Also second occurrence of answering a *different* question than
  asked (Q5: sklearn's convention vs the learner's own). Pattern stands.

## 2026-10-14 — Day 19 (Month 2 Week 4 Day 1) — Package Structure (minisklearn)

- 11/11 tests green at `/done`, but only after three pre-quiz debug rounds
  on the moved package (all package-side, none test-side): `__init__.py`
  still the empty scaffold (no re-exports → `LinearRegression missing`);
  `tree.py` generator-instead-of-`set` plus `.items()`-instead-of-keys
  regressions from the Day-16 move; `LogisticRegression.fit` returning the
  loss-history list instead of `self` (uniform-interface rule). Each fixed
  and re-run live (8 → 9 → 10 → 11).
- Six-question quiz + 2 feedback, one flag, no discipline-log entries.
  Q1 (uniform interface → model loop) first-ask correct. Q2 (empty vs wired
  `__init__`) first-ask correct. Q3 (relative-import mechanism) accepted
  first-ask with a one-line steer on the why (copies diverge, `sys.path`
  hacks are cwd-fragile). Q4 (hand values `[[2,1],[1,2]]`, 4/6, 2/3 ×3)
  all correct first-ask. Q5 two-parter: the KMeans-no-`y` half first-ask,
  the predict-before-fit half took two re-asks (normal flow described twice
  before landing "refuse — no parameters exist yet", i.e. the `ValueError`
  `test_predict_before_fit_raises` locks in). Q6 (why zero-division
  conventions return `0.0`) never landed across three attempts — circular
  ("because of conventions") then mechanistic ("because it give the zero
  divition error") without reaching the design reason — then flagged via
  the escape hatch ("i dont quiet understnad what you want from me"),
  logged to `month_2/week_4/day_1/flagged_questions.md`; `has_flags` true.
- Explain-back reasonable (same verbs, seamless use, front door eases
  imports) with a one-line steer toward the loop payoff — non-blocking.
- Self-reported time "4 hrs" → `time_spent_minutes: 240`; difficulty
  `easy`.
- Material feedback: videos 4/10 with "i like big videos that explains
  all" — third explicit big-lecture ask (Days 15, 16, 19) against the short
  picks; F2 "NA".
- Confidence recorded as `shaky` (Q5 re-asks + one flag; Q1–Q4 clean).
- Pattern update: seventh occurrence of the half-answer slip (Q5's
  unsupervised half before the predict-before-fit half). The Q6 block is a
  first-time concept (convention-rationale: loop-robustness + honest-zero),
  not a concept pattern — but note the answering shape matches Day 8's
  "don't understand the question" mid-quiz stalls, which resolved there via
  concrete restatement (same move tried here, didn't land). Revisit Q6 via
  `/spaced-review` before Day 3's evaluation harness leans on it.

## 2026-10-22 — Day 20 (Month 2 Week 4 Day 2) — Apply to a Real Dataset

- Build episode before the quiz: 9/10 on the first run, the failure in
  *my* test, not learner code — `test_integration_no_missing_numeric`
  demanded all-`float` while the test's own pipeline helper uses
  `label_encode` (ints, per its contract) for Sex/Embarked. Fixed the
  assertion to int-or-float (bool excluded), re-ran 10/10. Same class of
  assistant-side fix as Day 17's suite relaxation; learner code untouched.
- Four-question quiz + 2 feedback, no flags, no discipline-log entries.
  Q1 (62%-accuracy illusion + precision/recall exposure) took one re-ask:
  first attempt named only a vague "precision metrics" half, retry landed
  the full mechanism (zero predictive power, recall 0/342, precision 0/0).
  Q2 (two fill strategies + one downside each) took one re-ask: strategies
  named first, downsides missing; retry landed both (mean/median
  variance-spike and relationship destruction; KNN cost and noise
  propagation). Q3 (stratification rationale) first-ask correct
  (representativeness), sharpened with the starved-train/noisy-test
  failure mode.
- Explain-back reasonable (dirty data breaks training, EDA inspects first)
  with a one-line steer adding the other two verbs (encode to
  numbers-only, split for an honest test set). Non-blocking.
- Self-reported time "4 hrs" → `time_spent_minutes: 240`; difficulty
  "easy - medium" (a range, so `difficulty` stays `null` — never coerced,
  verbatim preserved here and in the logs). Material feedback: replacement
  videos "good" on length (the 22–39 min swap landed); F2 "NAA".
- Confidence recorded as `shaky` (two questions needed one re-ask each;
  everything landed, no flags).
- Pattern update: eighth occurrence of the half-answer slip (Q1's vague
  one-metric half before the two-metric mechanism). Neutral voice
  observation (no integrity claim; the quiz is open-book by design): both
  retries arrived in a markedly more polished register than the first
  attempts — logged so future `/done` runs can compare, not as a finding.
