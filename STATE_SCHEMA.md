# STATE_SCHEMA.md

This is the **single source of truth** for the shape of `state.json`. Every
skill in `.claude/skills/` and every agent in `.claude/agents/` must read and
write `state.json` strictly according to this schema. Do not invent new
top-level fields in a skill/agent — if a field is missing here, add it here
first, then use it everywhere consistently.

## Top-level shape

```json
{
  "current_month": 2,
  "current_day": 7,
  "months": {
    "1": {
      "weeks": {
        "1": {
          "done_when": "string, the week's own success criteria from the source doc",
          "review_status": "pending | done",
          "days": {
            "1": {
              "day_in_week": 1,
              "global_day": 1,
              "label": "e.g. 'Day 1' or 'Rest Day'",
              "status": "pending | current | done | rest",
              "has_flags": false,
              "micro_projects": [
                {
                  "slug": "kebab-case-folder-name",
                  "name": "human readable name",
                  "status": "in_progress | done",
                  "created": "ISO date"
                }
              ]
            }
          }
        }
      }
    }
  }
}
```

## Field definitions

- **current_month** (int): the month currently being worked through. Set by `/month`.
- **current_day** (int): the *continuous counter within the current month*
  (not reset per week). This is what `/i-am-in`, `/done`, and `/skip-to`
  advance and read. It maps to a `(week, day_in_week)` pair via the
  `months.<N>.weeks.<W>.days.<D>` structure below.
- **months.<N>.weeks.<W>.done_when**: the week's own "Done when" success
  criteria, copied verbatim (or lightly cleaned) from the source learning
  plan. Read by `/done` during a week-review pass. Never edited by anything
  except `/month` when it first parses the plan.
- **months.<N>.weeks.<W>.review_status**: `"pending"` until `/done` runs the
  week-review quiz for the last day of that week and the user passes it, then
  `"done"`.
- **months.<N>.weeks.<W>.days.<D>.day_in_week**: 1-based position within the
  week (e.g. Monday=1).
- **months.<N>.weeks.<W>.days.<D>.global_day**: the same value as
  `current_day` would be when this day is current — i.e. this day's position
  in the month's continuous counter. This is the join key between the
  week/day tree and `current_day`.
- **...days.<D>.label**: display label, e.g. `"Day 3"` or `"Rest Day"`.
- **...days.<D>.status**: one of `pending`, `current`, `done`, `rest`. Exactly
  one non-rest day should be `current` at a time (the one `current_day`
  points to). Rest days are set to `rest` at parse time by `/month` and never
  become `current` — `/i-am-in`, `/done`, and `/skip-to` must all skip over
  them automatically.
- **...days.<D>.has_flags** (bool): set `true` by `/done` if any quiz
  questions were flagged/skipped that day (see the escape-hatch behavior in
  `/done`). Read by `/progress` and noted in `progress.md`.
- **...days.<D>.micro_projects**: array, populated by `/micro-project`. Each
  entry has `slug` (matches the folder name under `micro_projects/`), `name`,
  `status` (`in_progress` or `done`), and `created` (ISO date string).

## What growth-notes.md needs from state.json

`code-evaluator` does not read `state.json` directly for grading, but when it
updates `growth-notes.md` it should be able to identify *which day* each
piece of feedback came from, so it can count occurrences across days
correctly. It gets the day identity (`month N, day D`) from the invoking
skill (`/done` or `/micro-project`), not by re-deriving it from `state.json`
itself — the invoking skill passes `current_month` / `current_day` (and the
micro-project slug, if applicable) into the agent's prompt.

## Rules for all skills/agents

1. Never write a field not listed above without updating this file first.
2. Always read `current_month` / `current_day` to resolve "the current day"
   — never assume month 1 / day 1.
3. Rest days are real entries with `status: "rest"`, not gaps in the day
   numbering. `global_day` still increments through them.
4. `/skip-to` and `/done`'s auto-advance must both skip rest days using the
   same logic: advance `current_day`, and if the resulting day's `status` is
   `rest`, keep advancing until a non-rest day is reached (marking each
   skipped rest day's status unchanged — it's already `rest`, not `done`).
