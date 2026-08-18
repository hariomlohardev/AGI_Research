---
name: done
description: Run the current day's tests, get non-blocking code review from the code-evaluator subagent, have the user explain the concept back, quiz them, track confidence/time-spent/struggle patterns, and mark the day (and possibly the week) complete in state.json and progress.md. Use when the user invokes /done to close out the current study day.
---

# /done

## Steps

1. If `state.json` doesn't exist, or the current day (`current_month`/`current_day`) has no generated content yet, tell the user clearly what to run first (`/month` and/or `/i-am-in`) and stop.

2. **Run tests.** Run pytest against `code/tests/` for the current day's `code/` (against the user's implementation). If any fail, tell the user exactly which ones and **stop** — do not proceed to code review or the quiz until they pass.

3. **Non-blocking code review.** Once tests pass, delegate to the `code-evaluator` subagent on the current day's `code/` folder (readability, structure, naming, idiomatic style — not correctness). Write its feedback to `day_<D>/code_review.md`. This never blocks marking the day done.
   - After writing `code_review.md`, have `code-evaluator` also read (and create if missing) `growth-notes.md` at the project root and update it. Pass it `current_month`/`current_day` so it can attribute observations correctly. Per the project's overclaiming guard: an issue is only written up as a **recurring pattern** once it has appeared on **3 or more separate days** — below that, log it as a plain single-day observation (e.g. "day 4: inconsistent naming noted"), not as a trend.

4. **Explain it back to me (non-blocking nudge).** Before the quiz, ask the user to summarize today's core idea in their own words, in a couple of sentences. Read their summary and:
   - If it's reasonable, briefly acknowledge it and move on — don't nitpick.
   - If it seems shallow or misses the core idea, say so plainly and give a one-line steer toward what's missing, but **do not block** — proceed to the quiz regardless. This is a nudge, not a gate. If this keeps happening on this topic across days, it will surface later via `struggle-log.md` (step 6), not by stalling this session.

5. **Quiz.** Ask a mix of the source plan's own practice questions for this day (adapted) plus a couple of new ones — some coding, some theoretical — plus a couple of feedback questions (the user may answer "NA" on those).
   - **Strict re-ask, with an escape hatch.** If an answer is wrong, re-ask that exact question until answered correctly — no hints. **Exception:** if the user responds with something like "flag this," "I think this question is wrong," or "skip this," do not keep looping. Instead, log it to `day_<D>/flagged_questions.md` (the question + their stated reason) and move on to the rest of the quiz. Set `has_flags: true` on this day's entry in `state.json`.
   - As you go, informally track how many re-asks and flags happened — you'll use this to set `confidence` in step 7.

6. **Update `struggle-log.md`** (root; create with a short header if missing) based on the quiz just run. This is separate from `growth-notes.md` (that's code style, from `code-evaluator`) — this is about **conceptual** sticking points, and `/done` writes it directly, no subagent needed.
   - Identify any concept(s) the user visibly struggled with this session (multiple re-asks on the same idea, a flagged question, or a shallow self-explanation from step 4).
   - Same non-overclaiming rule as `growth-notes.md`: only write something up as a **recurring pattern** once the same underlying concept has caused trouble on **3 or more separate days** — reference which days. Below that threshold, log it as a plain single-day observation (e.g. `"day 6: shaky on chain rule application"`), no trend language.
   - Identify the day via `current_month`/`current_day` and this day's `topic` field.

7. **Set confidence and time spent**, then **mark the day done**, once tests pass and the quiz is fully answered (including any flagged questions logged, not necessarily "correctly" answered):
   - Ask the user how long today actually took (self-reported, e.g. "about 90 minutes"). Store it verbatim as `time_spent_minutes` — an integer if they give one, or leave `null` if they'd rather skip it. Never estimate this yourself.
   - Set `confidence` on this day: `"strong"` (no/minimal re-asks, no flags), `"shaky"` (some re-asks or one flag), or `"struggled"` (repeated re-asks or multiple flags) — based on what you tracked in step 5.
   - Set this day's `status: "done"` in `state.json`.
   - Advance `current_day` to the next day, auto-skipping any Rest Day per `STATE_SCHEMA.md`'s rules, and set the new current day's `status: "current"`.
   - Append a short summary to `month_<N>/final_project/logs.md`.
   - Update `progress.md`: mark the day done, note its confidence level, note `time_spent_minutes` vs. `expected_minutes` if both are set (otherwise omit the comparison — never invent an expected value), and if `has_flags` is true, note that it has flagged questions to revisit.

8. **Week review.** If the day just completed was the last day of its week: quiz the user strictly against that week's own `done_when` criteria from `state.json` (sourced from the plan doc) before setting `review_status: "done"` on that week.

9. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing — this catches things like an out-of-range `confidence` value or a day left without exactly one `"current"` in the month.

10. `git add -A && git commit -m "Complete day <D> (+ week <W> review)"` — name the day, and the week if a review ran.
