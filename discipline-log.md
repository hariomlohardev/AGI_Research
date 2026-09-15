# Discipline Log

Factual record of requests to bypass a progress gate, and that they were declined.
Not a scolding — just a record, so the honesty of `state.json` stays auditable.

## 2026-09-05 — Month 2, Day 8 (Week 2 Day 2 — MAP Estimation & Priors)

- **Asked:** to skip the remaining quiz questions and mark the day `done`, with the
  stated reason of a headache and an intention to revisit the material at the end of
  the month.
- **Declined.** Tests were genuinely passing (13/13), but 4 of 6 quiz questions still
  had wrong or unanswered parts, and `/done` step 8 requires the quiz to be fully
  answered (correctly, or flagged via the escape hatch) before a day can close. The
  sanctioned escape hatch is flagging a question the user believes is broken — it did
  not apply here, since the request was to defer, not to dispute a question.
- **Outcome:** day left as `status: "current"`. Tests, `code_review.md` and the
  `growth-notes.md` entry all stand; no state was written. `/done` can be re-run to
  resume from the quiz.

## 2026-09-11 — Month 2, Day 12 (Week 2 Day 6 — Review + Consolidation)

- **Asked:** to be told what the quiz wanted for the open questions (Q1 Bayes/MLE/MAP and Q3 flat prior), saying they did not understand what was being asked for.
- **Declined.** Giving the answers or marking the questions correct without a correct answer would break the quiz gate; the sanctioned escape hatch is flagging a question believed to be wrong or unanswerable, not receiving the answer.
- **Outcome:** day left as `status: "current"`. Quiz resumed with Q1 and Q3 still open; flagging offered as the way forward if either question is believed to be broken.

- **Asked:** to accept the latest Q1 attempt as correct as-is.
- **Declined.** The attempt restates the Bayes structure but does not yet answer the two explicitly asked parts, so it cannot be marked correct; the sanctioned alternative remains flagging the question as wrong or unanswerable with a reason.
- **Outcome:** day left as `status: "current"` with Q1 and Q3 still open.

- **Asked (repeat):** to accept the same Q1 attempt as correct.
- **Declined.** Same reason — the required parts of Q1 are still unanswered, so it cannot be marked correct; flagging remains the sanctioned alternative.
- **Outcome:** day left as `status: "current"` with Q1 and Q3 still open.

## 2026-09-15 — Month 2, Day 13 (Week 3 Day 1 — Linear Regression, GD version)

- **Asked:** to skip implementing `solve_normal_equation` on the grounds that
  numpy will handle it later, so hand-writing it wastes time.
- **Declined.** The Normal Equation is this day's independent check on the GD
  implementation (the day's `Done when` is GD-vs-exact agreement), and its
  tests are part of the suite `/done` must run green — skipping it leaves the
  day unclosable. Code guidance offered instead.
- **Outcome:** day left as `status: "current"`; implementation guidance given
  for `normal_equation.py`.
