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
