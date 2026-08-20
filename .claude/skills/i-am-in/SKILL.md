---
name: i-am-in
description: Generate today's study materials (learning resources, coding problem stubs, tests, roadmap) for the current day in state.json. In the default "guided" mode, delegates video research to the video-researcher subagent and requires multiple real video links. If the user says something like "challenge me," switches to "challenge" mode instead, where learn.md gives only the topic with no resource links. Scales stub/problem difficulty to the month's skill_level, generates in the month's configured language, and enforces the discipline gate before starting a new day. Use when the user invokes /i-am-in, optionally with a note about focus or a challenge-mode request.
---

# /i-am-in

**Input (optional):** a note from the user in `$ARGUMENTS`, e.g. "focus more on the proof, less on the library usage," or a challenge-mode trigger like "challenge me."

## Steps

1. If `state.json` doesn't exist, tell the user clearly to run `/month` first and stop.

2. Read `state.json` for `current_month` (`N`) and `current_day` (`D`), and resolve `D`'s `(week, day_in_week)` via the schema in `STATE_SCHEMA.md`. Also read `months.<N>.language` and `months.<N>.skill_level`.

3. **Discipline gate.** Per `STATE_SCHEMA.md` rule 6: if 3 or more `done` days in month `N` have `(has_flags: true OR confidence: "struggled")` **and** `review_history.times_reviewed == 0`, refuse to generate a new day. Say plainly which days are piling up unreviewed, and point the user at `/spaced-review` — running even one review pass on any of those days clears the gate. This is not optional and not something to talk your way around; see `CLAUDE.md`'s "Discipline & accountability" section.

4. If `month_<N>/week_<W>/day_<D>/` already has content, tell the user clearly that this day's material already exists, and offer to answer questions about it instead. Do **not** overwrite or regenerate anything — this includes not being able to switch modes retroactively; if they want challenge mode after already generating a guided day, that's a fresh conversation, not a regeneration.

5. **Check `$ARGUMENTS` for a challenge-mode trigger** — a phrase like "challenge me" (case-insensitive; close variants like "challenge mode" or "no links please" count too). This decides which of the two branches below to follow, and sets this day's `mode` field accordingly (`"guided"` is already the default from `/month`, so only needs to change to `"challenge"`).

6. **Create `month_<N>/week_<W>/day_<D>/`, then branch:**

   **If guided mode (default, no challenge trigger):**
   - **Delegate research to the `video-researcher` subagent.** Pass it the day's topic and the source plan's suggested search terms as a starting point (not a hard limit), plus `$ARGUMENTS` if the user gave a focus note. Do not search inline — use the subagent. It should return **at least two** real, verified videos (not just one) with a note on why each is worth watching — see `video-researcher.md` for the exact bar. Never fabricate a title or link.
   - Write **`learn.md`**: what to learn today, plus the video(s) the subagent found. Do not add reading/article links here — supplementary reading is opt-in only, via `/for-read`, never bundled into base generation.
   - **`roadmap.md`** should reference watching the videos as an early step.

   **If challenge mode (user asked to be challenged):**
   - **Do not call `video-researcher` or any other research subagent.** No resource links of any kind go into this day's materials.
   - Write **`learn.md`** containing only: the topic name/label, and a short framing line making clear this is challenge mode — e.g. "Challenge mode: find and learn solid resources on `<topic>` yourself before starting today's problems." Nothing else.
   - **`roadmap.md`** should open with "find your own resources on `<topic>`" as the first step, before the coding tasks.

   **Either way:**
   - Write **`coding_problems.md`**: the day's coding problems from the source plan. **Scale to `skill_level`**: beginner gets more scaffolding/smaller steps in the problem statements, advanced gets terser statements and expects more independent design; intermediate is the source plan's problems roughly as-is.
   - **Language-appropriate stubs.** Write `code/` in `months.<N>.language`:
     - `"python"`: one `.py` file per task, signature + docstring only, no implementation.
     - `"javascript"`/`"typescript"`: one file per task with the function signature (typed, for TS) and a doc comment, no implementation; ensure a `package.json` exists at the project root (create a minimal one if `/month` didn't already).
     - `"other"`: match the idioms of whatever language the plan implies; note the day's `learn.md` should mention the exact run/test command since `months.<N>.test_command` is what `/done` will use.
   - Write **`code/tests/`** (or the language's equivalent test location) against those stubs, written by you. For tasks that aren't cleanly testable (plots, simulations, exploratory output), use judgment — light sanity checks rather than forcing strict correctness tests.
   - **Auto-detect sample-data needs.** If a task genuinely needs a dataset to be meaningful (not just synthetic random data generated inline), either generate a small synthetic one in a setup fixture, or fetch a small well-known public one via web search/fetch — never fabricate what a fetched dataset contains. Note in `learn.md` where it came from.
   - **Pin new dependencies.** If today's tasks introduce a new dependency, web-search its current stable version and add it to `requirements.txt` (or `package.json`) **pinned** (e.g. `numpy==2.x.x`), not bare — unpinned dependencies drift silently over the life of a multi-month project.

7. **Auto-tag the day.** Set `tags` (array of strings) on this day based on what you actually generated — e.g. `"math-heavy"` if there's real derivation/proof work, `"coding-heavy"` if the bulk is implementation, `"conceptual"` for mostly-reading/understanding days. Keep it to 1-3 tags, only ones genuinely earned by the content, not a checklist applied to every day.

8. Append a short entry to `month_<N>/final_project/structure.md` or `logs.md` tying today's topic into that month's final project.

9. Update `state.json`: this day's entry stays `status: "current"` (unchanged — generating content doesn't mark it done, `/done` does). Set its `topic` field to a short human-readable label for today's core topic (e.g. `"gradient descent"`), derived from the material you just wrote — used later by `/explain` and `/week-recap`. Set `mode` per step 5 and `tags` per step 7.

10. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

11. `git add -A && git commit -m "Generate day <D> (week <W>, month <N>) [<mode>]"`.
