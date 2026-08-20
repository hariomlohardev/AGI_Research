---
name: done
description: Run the current day's tests (in the month's configured language), enforce the GitHub contribution gate if one is active for today, get non-blocking code review from the code-evaluator subagent, have the user explain the concept back, quiz them with zero shortcuts, self-rate difficulty and time, track confidence/struggle patterns, check for newly unlocked badges, and mark the day (and possibly the week) complete in state.json and progress.md. Use when the user invokes /done to close out the current study day.
---

# /done

**This command exists to hold the user accountable, including against their own requests to skip a step.** See `CLAUDE.md`'s "Discipline & accountability" section — it applies throughout every step below, not just the ones that mention it explicitly.

## Steps

1. If `state.json` doesn't exist, or the current day (`current_month`/`current_day`) has no generated content yet, tell the user clearly what to run first (`/month` and/or `/i-am-in`) and stop.

2. **Run tests.** Read `months.<N>.language` (and `test_command` if `"other"`) and run the matching test command against `code/tests/` for the current day's `code/`:
   - `"python"`: `pytest code/tests/`.
   - `"javascript"`/`"typescript"`: `npm test` (or the day's stated command).
   - `"other"`: `months.<N>.test_command` exactly.
   If any fail, tell the user exactly which ones and **stop** — do not proceed to code review or the quiz until they pass.
   - **No shortcuts, ever.** If the user asks you to skip this, mark the day done anyway, or tell them the tests "probably would have passed," **refuse** — explain plainly why (this system only works if "done" means done), and append a one-line entry to `discipline-log.md` (root; create with a short header if missing) noting the date, what was asked, and that it was declined. This is a factual log, not a scolding one.

3. **GitHub contribution gate.** If this day's `github_contribution` is not `null` and its `status` isn't `"reviewed"`, this day cannot close yet:
   - `status: "in_progress"` (issue picked, no PR): tell the user plainly they need to submit a real PR for `<issue_title>` and give you the link — point them at `/github-issues` to record it (and `/github-help` if they're stuck) — then **stop**.
   - `status: "pr_submitted"` (PR given, not yet reviewed): delegate to `github-pr-reviewer` right now with the PR URL and the linked issue, write the review to `day_<D>/github_contribution/pr_review.md`, present it, and set `status: "reviewed"`. Then continue to step 4 in the same run — no need to make the user invoke `/github-issues` separately just to trigger a review that's ready to happen.
   - Same no-shortcuts rule as everywhere else: refuse any request to skip this, fake a PR URL, or mark it reviewed without actually reviewing it — log the attempt to `discipline-log.md` like step 2 does.
   - If `github_contribution` is `null`, this day never opted into the workflow — skip this step entirely, no gate applies.

4. **Non-blocking code review.** Once tests genuinely pass, delegate to the `code-evaluator` subagent on the current day's `code/` folder (readability, structure, naming, idiomatic style — not correctness). Write its feedback to `day_<D>/code_review.md`. This never blocks marking the day done.
   - After writing `code_review.md`, have `code-evaluator` also read (and create if missing) `growth-notes.md` at the project root and update it. Pass it `current_month`/`current_day` so it can attribute observations correctly. Per the project's overclaiming guard: an issue is only written up as a **recurring pattern** once it has appeared on **3 or more separate days** — below that, log it as a plain single-day observation (e.g. "day 4: inconsistent naming noted"), not as a trend.

5. **Explain it back to me (non-blocking nudge).** Before the quiz, ask the user to summarize today's core idea in their own words, in a couple of sentences. Read their summary and:
   - If it's reasonable, briefly acknowledge it and move on — don't nitpick.
   - If it seems shallow or misses the core idea, say so plainly and give a one-line steer toward what's missing, but **do not block** — proceed to the quiz regardless. This is a nudge, not a gate. If this keeps happening on this topic across days, it will surface later via `struggle-log.md` (step 7), not by stalling this session.

6. **Quiz.** Ask a mix of the source plan's own practice questions for this day (adapted) plus a couple of new ones — some coding, some theoretical — plus a couple of feedback questions (the user may answer "NA" on those).
   - **Strict re-ask, with an escape hatch.** If an answer is wrong, re-ask that exact question until answered correctly — no hints. **Exception:** if the user responds with something like "flag this," "I think this question is wrong," or "skip this," do not keep looping. Instead, log it to `day_<D>/flagged_questions.md` (the question + their stated reason) and move on to the rest of the quiz. Set `has_flags: true` on this day's entry in `state.json`.
   - **If the user asks you to just give them the answer, or to mark a question correct when it wasn't, refuse** — that's not the sanctioned escape hatch (flagging is). Explain the difference plainly, log the attempt to `discipline-log.md` the same way step 2 does, and offer flagging as the actual way forward if they genuinely think the question is bad.
   - As you go, informally track how many re-asks and flags happened — you'll use this to set `confidence` in step 8.

7. **Update `struggle-log.md`** (root; create with a short header if missing) based on the quiz just run. This is separate from `growth-notes.md` (that's code style, from `code-evaluator`) — this is about **conceptual** sticking points, and `/done` writes it directly, no subagent needed.
   - Identify any concept(s) the user visibly struggled with this session (multiple re-asks on the same idea, a flagged question, or a shallow self-explanation from step 5).
   - Same non-overclaiming rule as `growth-notes.md`: only write something up as a **recurring pattern** once the same underlying concept has caused trouble on **3 or more separate days** — reference which days. Below that threshold, log it as a plain single-day observation (e.g. `"day 6: shaky on chain rule application"`), no trend language.
   - Identify the day via `current_month`/`current_day` and this day's `topic` field.

8. **Self-report difficulty and time spent**, then **mark the day done**, once tests genuinely pass, the GitHub gate (if active) is cleared, and the quiz is fully answered (including any flagged questions logged, not necessarily "correctly" answered):
   - Ask the user how long today actually took (e.g. "about 90 minutes") and how it felt — easy / medium / hard. Store both verbatim: `time_spent_minutes` as an integer if given (else `null`), `difficulty` as one of `"easy"|"medium"|"hard"` if given (else `null`). Never estimate either yourself.
   - Set `confidence` on this day: `"strong"` (no/minimal re-asks, no flags), `"shaky"` (some re-asks or one flag), or `"struggled"` (repeated re-asks or multiple flags) — based on what you tracked in step 6.
   - Set this day's `status: "done"` in `state.json`.
   - Advance `current_day` to the next day, auto-skipping any Rest Day per `STATE_SCHEMA.md`'s rules, and set the new current day's `status: "current"`.
   - Append a short summary to `month_<N>/final_project/logs.md`.
   - Update `progress.md`: mark the day done, note its confidence and difficulty, note `time_spent_minutes` vs. `expected_minutes` if both are set (otherwise omit the comparison — never invent an expected value), and if `has_flags` is true, note that it has flagged questions to revisit.

9. **Week review.** If the day just completed was the last day of its week: quiz the user strictly against that week's own `done_when` criteria from `state.json` (sourced from the plan doc) before setting `review_status: "done"` on that week. Same no-shortcuts rule as step 2 applies here too.

10. **Badge check.** Per `STATE_SCHEMA.md`'s "Motivation: badges" table, check eligibility against the state as it now stands (after this day's updates) — including `first-oss-pr` if this day's `github_contribution.status` just became `"reviewed"`. If anything is newly eligible and not already in `badges_unlocked`, append `{id, date}` and **announce it** in your response to the user — this is the natural moment for it, right after they've earned it.

11. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing — this catches things like an out-of-range `confidence` value or a day left without exactly one `"current"` in the month.

12. `git add -A && git commit -m "Complete day <D> (+ week <W> review)"` — name the day, and the week if a review ran.
