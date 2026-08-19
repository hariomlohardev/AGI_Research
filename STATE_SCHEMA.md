# STATE_SCHEMA.md

This is the canonical **human-readable** definition of the shape of
`state.json`, plus the rules that a JSON Schema alone can't express (which
fields are computed live vs. stored, the "exactly one current day"
invariant, etc.). Every skill in `.claude/skills/` and every agent in
`.claude/agents/` must read and write `state.json` strictly according to
this file. Do not invent new top-level fields in a skill/agent — if a field
is missing here, add it here first, then use it everywhere consistently.

**This is no longer just a convention.** `state.schema.json` at the project
root is the machine-enforced counterpart of this file, checked by
`scripts/validate_state.py` and gated on every commit by
`.githooks/pre-commit`. Any invented or mistyped field, wrong enum value, or
missing required field in `state.json` gets **rejected at commit time**, not
just flagged by a reader of this doc. If you change the shape here, change
`state.schema.json` to match — they must stay in sync.

## Top-level shape

```json
{
  "current_month": 2,
  "current_day": 7,
  "months": {
    "1": {
      "started": "ISO date, set when /month first creates this month",
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
              "mode": "guided | challenge — set at month creation, may be switched to 'challenge' by /i-am-in",
              "topic": "short human-readable topic label, e.g. 'gradient descent'",
              "time_spent_minutes": null,
              "expected_minutes": null,
              "confidence": null,
              "confusion_notes": [
                {
                  "note": "string, the user's own words",
                  "date": "ISO date"
                }
              ],
              "review_history": {
                "last_reviewed": null,
                "times_reviewed": 0
              },
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
- **...days.<D>.mode** (`"guided" | "challenge"`): defaults to `"guided"`,
  set by `/month` at creation. Controls how `/i-am-in` generates the day:
  - `"guided"` (base/default): `/i-am-in` delegates to `video-researcher`
    and includes real video links in `learn.md`, aiming for **at least two**
    strong videos, not just one.
  - `"challenge"`: triggered when the user includes something like
    "challenge me" in `/i-am-in`'s `$ARGUMENTS`. `/i-am-in` does **not**
    call `video-researcher` and `learn.md` contains only the topic name and
    a short framing line — no resource links at all. The user finds and
    learns the material themselves.
  - Either way, supplementary reading via `reading-researcher`/`/for-read`
    stays **optional and separate** — it's never bundled into the base day
    generation in either mode, only pulled in when the user explicitly runs
    `/for-read`.
  - Set once when `/i-am-in` actually generates the day (not editable
    afterward without regenerating).
- **...days.<D>.micro_projects**: array, populated by `/micro-project`. Each
  entry has `slug` (matches the folder name under `micro_projects/`), `name`,
  `status` (`in_progress` or `done`), and `created` (ISO date string).
- **months.<N>.started** (ISO date string): set once, when `/month` first
  creates month `N`. Used by `/pace` to project a finish date from the
  completion rate so far. Never edited after creation.
- **...days.<D>.topic** (string): a short human-readable label for the day's
  core topic, set by `/i-am-in` when it generates the day's content (derived
  from the source plan / the material it wrote). Used by `/explain <concept>`
  to find which past day(s) a concept belongs to, and by `/recap` to
  summarize a week in plain language.
- **...days.<D>.time_spent_minutes** (int or `null`): self-reported by the
  user during `/done`. Stays `null` if they don't say, or if a day is not
  yet done. Never guessed or estimated by Claude.
- **...days.<D>.expected_minutes** (int or `null`): only set if the source
  plan itself states an expected duration for the day; otherwise stays
  `null` and `/progress`/`/pace` simply omit the comparison for that day
  rather than inventing a number.
- **...days.<D>.confidence** (`"strong" | "shaky" | "struggled" | null`): set
  by `/done` right after the quiz, from how it went — few/no re-asks and no
  flags is `"strong"`, some re-asks or one flag is `"shaky"`, repeated
  re-asks or multiple flags is `"struggled"`. `null` until the day is done.
  Used by `/review` to weight which days' questions resurface first.
- **...days.<D>.confusion_notes**: array of `{note, date}`, appended to by
  `/confused`. Never cleared automatically — these are meant to persist as a
  record and to feed `/review`, not to be marked "resolved" implicitly.
- **...days.<D>.review_history**: `{last_reviewed, times_reviewed}`, updated
  by `/review` each time it actually pulls and re-asks a question from that
  day. `last_reviewed` is an ISO date or `null` if never reviewed;
  `times_reviewed` starts at `0`.

## Fields that are deliberately NOT stored (computed live instead)

To avoid two sources of truth drifting apart, the following are always
**derived at read time**, never written to `state.json`:

- **Study streak** (consecutive study days completed, excluding rest days):
  computed by `/progress` and `/recap` by scanning backward from the most
  recently **completed** day (i.e. skip past today's `current` day if it
  isn't done yet — an in-progress day should never zero out the streak),
  counting consecutive `done` days and passing over `rest` days without
  breaking the count, stopping at the first non-`done`, non-`rest` day. Not
  stored as a counter that could go stale.
- **Pace / projected finish date**: computed by `/pace` from
  `months.<N>.started`, today's date, and the ratio of days marked `done`
  vs. total real (non-rest) days in the month so far. Recomputed fresh every
  time `/pace` runs.

## What growth-notes.md needs from state.json

`code-evaluator` does not read `state.json` directly for grading, but when it
updates `growth-notes.md` it should be able to identify *which day* each
piece of feedback came from, so it can count occurrences across days
correctly. It gets the day identity (`month N, day D`) from the invoking
skill (`/done` or `/micro-project`), not by re-deriving it from `state.json`
itself — the invoking skill passes `current_month` / `current_day` (and the
micro-project slug, if applicable) into the agent's prompt.

## What struggle-log.md needs from state.json

Unlike `growth-notes.md` (code style, written by `code-evaluator`),
`struggle-log.md` tracks **conceptual** sticking points from quiz
performance and gets updated directly by `/done` (no subagent needed — it's
reading its own quiz transcript, not external code). It follows the same
non-overclaiming rule as `growth-notes.md`: a topic-level struggle is only
written up as a pattern once it has shown up on 3+ separate days; below that
it's logged as a plain single-day observation. `/done` identifies the day via
`current_month`/`current_day` and the day's `topic` field, the same way
`code-evaluator` does for `growth-notes.md`.

## What project-history.md needs from state.json

`project-history.md` (root) is updated by `/month` whenever it creates a new
month's `final_project/structure.md`. It reads the previous month's
`final_project/structure.md` (on disk, not from `state.json`) to describe
the throughline, but uses `state.json` only to know which month number is
"previous" (`current_month` before it's bumped to `N`).

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
5. Any skill that writes `state.json` runs `python3 scripts/validate_state.py`
   before committing, and stops (fixes the write, doesn't force the commit)
   if it reports errors. `.githooks/pre-commit` enforces this regardless, but
   checking proactively gives a clearer, closer-to-the-cause error message.
