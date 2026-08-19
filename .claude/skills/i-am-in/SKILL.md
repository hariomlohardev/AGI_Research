---
name: i-am-in
description: Generate today's study materials (learning resources, coding problem stubs, tests, roadmap) for the current day in state.json. In the default "guided" mode, delegates video research to the video-researcher subagent and requires multiple real video links. If the user says something like "challenge me," switches to "challenge" mode instead, where learn.md gives only the topic with no resource links, and the user has to find and learn the material themselves. Use when the user invokes /i-am-in, optionally with a note about focus or a challenge-mode request.
---

# /i-am-in

**Input (optional):** a note from the user in `$ARGUMENTS`, e.g. "focus more on the proof, less on the library usage," or a challenge-mode trigger like "challenge me."

## Steps

1. If `state.json` doesn't exist, tell the user clearly to run `/month` first and stop.

2. Read `state.json` for `current_month` (`N`) and `current_day` (`D`), and resolve `D`'s `(week, day_in_week)` via the schema in `STATE_SCHEMA.md`.

3. If `month_<N>/week_<W>/day_<D>/` already has content, tell the user clearly that this day's material already exists, and offer to answer questions about it instead. Do **not** overwrite or regenerate anything — this includes not being able to switch modes retroactively; if they want challenge mode after already generating a guided day, that's a fresh conversation, not a regeneration.

4. **Check `$ARGUMENTS` for a challenge-mode trigger** — a phrase like "challenge me" (case-insensitive; close variants like "challenge mode" or "no links please" count too). This decides which of the two branches below to follow, and sets this day's `mode` field accordingly (`"guided"` is already the default from `/month`, so only needs to change to `"challenge"`).

5. **Create `month_<N>/week_<W>/day_<D>/`, then branch:**

   **If guided mode (default, no challenge trigger):**
   - **Delegate research to the `video-researcher` subagent.** Pass it the day's topic and the source plan's suggested search terms as a starting point (not a hard limit), plus `$ARGUMENTS` if the user gave a focus note. Do not search inline — use the subagent. It should return **at least two** real, verified videos (not just one) with a note on why each is worth watching — see `video-researcher.md` for the exact bar. Never fabricate a title or link.
   - Write **`learn.md`**: what to learn today, plus the video(s) the subagent found. Do not add reading/article links here — supplementary reading is opt-in only, via `/for-read`, never bundled into base generation.
   - **`roadmap.md`** should reference watching the videos as an early step.

   **If challenge mode (user asked to be challenged):**
   - **Do not call `video-researcher` or any other research subagent.** No resource links of any kind go into this day's materials.
   - Write **`learn.md`** containing only: the topic name/label, and a short framing line making clear this is challenge mode — e.g. "Challenge mode: find and learn solid resources on `<topic>` yourself before starting today's problems." Nothing else.
   - **`roadmap.md`** should open with "find your own resources on `<topic>`" as the first step, before the coding tasks.

   **Either way:**
   - Write **`coding_problems.md`**: the day's coding problems from the source plan.
   - Write **`code/`**: one file per coding task, named for what it does, each containing only the function signature and a docstring (expected input, required output) — no implementation.
   - Write **`code/tests/`**: pytest tests against those stubs, written by you. For tasks that aren't cleanly testable (plots, simulations, exploratory output), use judgment — light sanity checks rather than forcing strict correctness tests.

6. Append a short entry to `month_<N>/final_project/structure.md` or `logs.md` tying today's topic into that month's final project.

7. If today's tasks introduce a new dependency, add it to `requirements.txt`.

8. Update `state.json`: this day's entry stays `status: "current"` (unchanged — generating content doesn't mark it done, `/done` does). Set its `topic` field to a short human-readable label for today's core topic (e.g. `"gradient descent"`), derived from the material you just wrote — used later by `/explain` and `/recap`. Set `mode` per step 4.

9. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

10. `git add -A && git commit -m "Generate day <D> (week <W>, month <N>) [<mode>]"`.

