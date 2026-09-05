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
