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
