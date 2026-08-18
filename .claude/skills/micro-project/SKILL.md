---
name: micro-project
description: Design and scaffold a small hands-on project applying the current (or most recently completed) day's topic, optionally pulling in earlier completed days. Use when the user invokes /micro-project, optionally with a note about what kind of project they want.
---

# /micro-project

**Input (optional):** a note from the user in `$ARGUMENTS` about what kind of project they want.

## Steps

1. Read `state.json`. The current day, or the most recently completed day if the current day isn't done yet, must have `status: "done"` — otherwise tell the user clearly to finish and run `/done` first, and stop.

2. Design a small, self-contained project applying that day's topic hands-on. It may deliberately pull in concepts/code from earlier **completed** days where it makes sense — if it does, say explicitly which days it's drawing on. Factor in `$ARGUMENTS` if the user specified a kind of project.

3. Create `day_<D>/micro_projects/<slug>/` (slug named for what the project is; if `/micro-project` is run again for the same day, this is a **new** subfolder/slug, not an overwrite):
   - `brief.md` — what to build and why, and which earlier days it draws on (if any).
   - `code/` — stubs (signature + docstring, no implementation).
   - `code/tests/` — tests where reasonably testable.

4. Add this micro-project to `state.json` under that day's `micro_projects` array: `{slug, name, status: "in_progress", created: <ISO date>}`.

5. **Once tests pass** (when the user later works on it and asks you to check): delegate to the `code-evaluator` subagent on `micro_projects/<slug>/code/`. Write feedback to `micro_projects/<slug>/code_review.md`, non-blocking. Have it update `growth-notes.md` the same way `/done` does (3+ days before calling something a pattern).

6. When the user tells you conversationally that they've finished a micro-project, set its `status: "done"` in `state.json` and note it in `progress.md`.

7. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

8. `git add -A && git commit -m "Add/update micro-project <slug> for day <D>"`.
