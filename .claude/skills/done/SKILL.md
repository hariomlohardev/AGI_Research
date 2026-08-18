---
name: done
description: Run the current day's tests, get non-blocking code review from the code-evaluator subagent, quiz the user, and mark the day (and possibly the week) complete in state.json and progress.md. Use when the user invokes /done to close out the current study day.
---

# /done

## Steps

1. If `state.json` doesn't exist, or the current day (`current_month`/`current_day`) has no generated content yet, tell the user clearly what to run first (`/month` and/or `/i-am-in`) and stop.

2. **Run tests.** Run pytest against `code/tests/` for the current day's `code/` (against the user's implementation). If any fail, tell the user exactly which ones and **stop** — do not proceed to code review or the quiz until they pass.

3. **Non-blocking code review.** Once tests pass, delegate to the `code-evaluator` subagent on the current day's `code/` folder (readability, structure, naming, idiomatic style — not correctness). Write its feedback to `day_<D>/code_review.md`. This never blocks marking the day done.
   - After writing `code_review.md`, have `code-evaluator` also read (and create if missing) `growth-notes.md` at the project root and update it. Pass it `current_month`/`current_day` so it can attribute observations correctly. Per the project's overclaiming guard: an issue is only written up as a **recurring pattern** once it has appeared on **3 or more separate days** — below that, log it as a plain single-day observation (e.g. "day 4: inconsistent naming noted"), not as a trend.

4. **Quiz.** Ask a mix of the source plan's own practice questions for this day (adapted) plus a couple of new ones — some coding, some theoretical — plus a couple of feedback questions (the user may answer "NA" on those).
   - **Strict re-ask, with an escape hatch.** If an answer is wrong, re-ask that exact question until answered correctly — no hints. **Exception:** if the user responds with something like "flag this," "I think this question is wrong," or "skip this," do not keep looping. Instead, log it to `day_<D>/flagged_questions.md` (the question + their stated reason) and move on to the rest of the quiz. Set `has_flags: true` on this day's entry in `state.json`.

5. **Mark the day done**, once tests pass and the quiz is fully answered (including any flagged questions logged, not necessarily "correctly" answered):
   - Set this day's `status: "done"` in `state.json`.
   - Advance `current_day` to the next day, auto-skipping any Rest Day per `STATE_SCHEMA.md`'s rules, and set the new current day's `status: "current"`.
   - Append a short summary to `month_<N>/final_project/logs.md`.
   - Update `progress.md`: mark the day done, and if `has_flags` is true, note that it has flagged questions to revisit.

6. **Week review.** If the day just completed was the last day of its week: quiz the user strictly against that week's own `done_when` criteria from `state.json` (sourced from the plan doc) before setting `review_status: "done"` on that week.

7. `git add -A && git commit -m "Complete day <D> (+ week <W> review)"` — name the day, and the week if a review ran.
