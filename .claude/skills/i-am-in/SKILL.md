---
name: i-am-in
description: Generate today's study materials (learning resources, coding problem stubs, tests, roadmap) for the current day in state.json, delegating video/article research to the video-researcher subagent. Use when the user invokes /i-am-in, optionally with a note about what they want to focus on today.
---

# /i-am-in

**Input (optional):** a note from the user in `$ARGUMENTS`, e.g. "focus more on the proof, less on the library usage."

## Steps

1. If `state.json` doesn't exist, tell the user clearly to run `/month` first and stop.

2. Read `state.json` for `current_month` (`N`) and `current_day` (`D`), and resolve `D`'s `(week, day_in_week)` via the schema in `STATE_SCHEMA.md`.

3. If `month_<N>/week_<W>/day_<D>/` already has content, tell the user clearly that this day's material already exists, and offer to answer questions about it instead. Do **not** overwrite or regenerate anything.

4. Otherwise, create `month_<N>/week_<W>/day_<D>/` and:
   - **Delegate research to the `video-researcher` subagent.** Pass it the day's topic and the source plan's suggested search terms as a starting point (not a hard limit), plus `$ARGUMENTS` if the user gave a focus note. Do not search inline — use the subagent. The subagent returns real, verified resources with a note on why each is worth watching; never fabricate a title or link.
   - Write **`learn.md`**: what to learn today plus the resources the subagent found.
   - Write **`coding_problems.md`**: the day's coding problems from the source plan.
   - Write **`roadmap.md`**: a realistic single-sitting plan for the day.
   - Write **`code/`**: one file per coding task, named for what it does, each containing only the function signature and a docstring (expected input, required output) — no implementation.
   - Write **`code/tests/`**: pytest tests against those stubs, written by you. For tasks that aren't cleanly testable (plots, simulations, exploratory output), use judgment — light sanity checks rather than forcing strict correctness tests.

5. Append a short entry to `month_<N>/final_project/structure.md` or `logs.md` tying today's topic into that month's final project.

6. If today's tasks introduce a new dependency, add it to `requirements.txt`.

7. Update `state.json`: this day's entry stays `status: "current"` (unchanged — generating content doesn't mark it done, `/done` does). Set its `topic` field to a short human-readable label for today's core topic (e.g. `"gradient descent"`), derived from the material you just wrote — this is used later by `/explain` and `/recap`.

8. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

9. `git add -A && git commit -m "Generate day <D> (week <W>, month <N>)"`.
