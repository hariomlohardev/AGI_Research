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
  "schema_version": 2,
  "current_month": 2,
  "current_day": 7,
  "badges_unlocked": [
    { "id": "five-day-streak", "date": "ISO date" }
  ],
  "months": {
    "1": {
      "started": "ISO date, set when /month first creates this month",
      "language": "python | javascript | typescript | other",
      "skill_level": "beginner | intermediate | advanced",
      "test_command": null,
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
              "tags": ["math-heavy"],
              "difficulty": null,
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
              "srs": {
                "ease_factor": 2.5,
                "interval_days": 1,
                "next_review_due": null
              },
              "github_contribution": null,
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

`github_contribution` when active looks like:

```json
{
  "status": "in_progress | pr_submitted | reviewed",
  "repo": "owner/repo",
  "issue_url": "https://github.com/owner/repo/issues/123",
  "issue_title": "string",
  "pr_url": null
}
```

## Field definitions

- **schema_version** (int): version of this schema shape, currently `2`.
  Bumped whenever `state.schema.json`'s shape changes in a way that isn't
  purely additive-with-defaults. See "Schema migrations" below.
- **current_month** (int): the month currently being worked through. Set by `/month`.
- **current_day** (int): the *continuous counter within the current month*
  (not reset per week). This is what `/i-am-in`, `/done`, and `/skip-to`
  advance and read. It maps to a `(week, day_in_week)` pair via the
  `months.<N>.weeks.<W>.days.<D>` structure below.
- **badges_unlocked**: top-level array of `{id, date}`, append-only. See
  "Motivation: badges" below for how eligibility is computed and recorded.
- **months.<N>.language** (`"python" | "javascript" | "typescript" | "other"`):
  set by `/month` (asks the user once, at month creation, defaulting to
  `"python"` if unspecified). Determines the test runner `/done` uses and
  the syntax `/i-am-in` generates for code stubs. If `"other"`,
  `months.<N>.test_command` must be set so `/done` knows how to run tests.
- **months.<N>.skill_level** (`"beginner" | "intermediate" | "advanced"`):
  set by `/month` (asks the user once, defaulting to `"intermediate"` if
  unspecified). Read by `/i-am-in` to scale stub/problem difficulty and by
  `code-evaluator` to calibrate feedback strictness.
- **months.<N>.test_command** (string or `null`): only required when
  `language` is `"other"` — the exact shell command `/done` should run in
  place of `pytest`/`npm test`. Stays `null` otherwise.
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
- **...days.<D>.tags** (array of strings, default `[]`): free-form labels
  like `"math-heavy"` or `"coding-heavy"`, auto-suggested by `/i-am-in` when
  it generates the day based on the material's actual content. Can be
  amended later just by asking Claude conversationally (e.g. "tag today as
  needs-review too") — any such ad hoc edit still runs
  `scripts/validate_state.py` before committing, same as every other write.
- **...days.<D>.difficulty** (`"easy" | "medium" | "hard" | null`):
  self-rated by the user during `/done`, alongside `time_spent_minutes` —
  never assumed or inferred by Claude. `null` until the day is done and the
  user answers (or skips) the prompt.
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
- **...days.<D>.github_contribution** (object or `null`, default `null`):
  tracks an optional real open-source contribution tied to this day. `null`
  means the user hasn't engaged `/github-issues` for this day — the normal
  `/done` flow is completely unaffected. When set, it's an object:
  - `status`: `"in_progress"` (issue picked, no PR yet) →
    `"pr_submitted"` (PR link given, not yet reviewed) → `"reviewed"`
    (`github-pr-reviewer` has given feedback). `/done` will not let the day
    close while `status` is anything other than `"reviewed"` — see
    `CLAUDE.md`'s "Discipline & accountability" section and
    `/github-issues`/`/done`'s own SKILL.md files.
  - `repo`: `"owner/repo"` of the issue's repository.
  - `issue_url` / `issue_title`: the specific issue chosen, real and
    verified by `github-issue-researcher` — never fabricated.
  - `pr_url`: `null` until the user gives their real, submitted PR's URL;
    never filled in speculatively.
- **...days.<D>.micro_projects**: array, populated by `/micro-project`. Each
  entry has `slug` (matches the folder name under `micro_projects/`), `name`,
  `status` (`in_progress` or `done`), and `created` (ISO date string).
- **months.<N>.started** (ISO date string): set once, when `/month` first
  creates month `N`. Used by `/pace` to project a finish date from the
  completion rate so far. Never edited after creation.
- **...days.<D>.topic** (string): a short human-readable label for the day's
  core topic, set by `/i-am-in` when it generates the day's content (derived
  from the source plan / the material it wrote). Used by `/explain <concept>`
  to find which past day(s) a concept belongs to, and by `/week-recap` to
  summarize a week in plain language. Also used by `github-issue-researcher`
  to try to tie a suggested issue to the week's actual topic.
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
  Used by `/spaced-review` to weight which days' questions resurface first.
- **...days.<D>.confusion_notes**: array of `{note, date}`, appended to by
  `/confused`. Never cleared automatically — these are meant to persist as a
  record and to feed `/spaced-review`, not to be marked "resolved" implicitly.
- **...days.<D>.review_history**: `{last_reviewed, times_reviewed}`, updated
  by `/spaced-review` each time it actually pulls and re-asks a question from
  that day. `last_reviewed` is an ISO date or `null` if never reviewed;
  `times_reviewed` starts at `0`.
- **...days.<D>.srs**: `{ease_factor, interval_days, next_review_due}` — a
  simplified SM-2 spaced-repetition schedule, maintained entirely by
  `/spaced-review` (nothing else writes to it). Starts at
  `ease_factor: 2.5`, `interval_days: 1`, `next_review_due: null`
  (not due until the day has been reviewed once). See "Spaced repetition
  scheduling" below for the update formula.

## Fields that are deliberately NOT stored (computed live instead)

To avoid two sources of truth drifting apart, the following are always
**derived at read time**, never written to `state.json`:

- **Study streak** (consecutive study days completed, excluding rest days):
  computed by `/progress` and `/week-recap` by first building one
  chronological list of **every** day across **all** months (sort by month
  number ascending, then `global_day` ascending within each month — a
  new month's day 1 comes immediately after the previous month's last day,
  it does not reset the streak by itself), then scanning that list backward
  from the most recently **completed** day (i.e. skip past today's `current`
  day if it isn't done yet — an in-progress day should never zero out the
  streak), counting consecutive `done` days and passing over `rest` days
  without breaking the count, stopping at the first non-`done`, non-`rest`
  day. Not stored as a counter that could go stale.
- **Pace / projected finish date**: computed by `/pace` from
  `months.<N>.started`, today's date, and the ratio of days marked `done`
  vs. total real (non-rest) days in the month so far. Recomputed fresh every
  time `/pace` runs.
- **Badge eligibility**: `/done`, `/progress`, and `/week-recap` compute
  which badges the current state *qualifies* for live, by scanning
  `state.json` (see "Motivation: badges" below). Only the fact that a badge
  was unlocked — and when — is persisted, in `badges_unlocked`.

## Spaced repetition scheduling (SM-2, simplified)

`...days.<D>.srs` implements a simplified SM-2 algorithm, owned entirely by
`/spaced-review` — no other skill reads or writes it. Every time a day's
question(s) get reviewed:

1. Rate the review quality `q` on a 0-5 scale from how it went: `5` =
   answered correctly, no hesitation; `3` = correct but needed a re-ask or
   two; `1` = flagged/skipped or still wrong when the session ends.
2. Update `ease_factor`: `ease_factor = max(1.3, ease_factor + (0.1 - (5-q) *
   (0.08 + (5-q) * 0.02)))`.
3. Update `interval_days`: if `q < 3`, reset `interval_days = 1` (it wasn't
   really retained — review again soon). Otherwise, on the *first* successful
   review `interval_days = 1`, on the *second* `interval_days = 6`, and on
   every one after that `interval_days = round(interval_days * ease_factor)`.
4. Set `next_review_due = today + interval_days` (ISO date).

`/spaced-review`'s day-selection weighting (recency/confidence/flags) stays
as the *first pass filter* for which days are eligible, but within that,
days with `next_review_due` on or before today are prioritized over days not
yet due — this is what actually makes it "spaced" rather than just
recency-weighted.

## Motivation: badges

Badge *eligibility* is always computed live by scanning `state.json` (never
stored as a running "current status"), but once a badge is newly eligible,
its unlock is a one-time event worth recording permanently — that's what
`badges_unlocked` is for. Whenever `/done`, `/progress`, or `/week-recap`
runs, check eligibility for every badge below; if one is newly eligible and
not already in `badges_unlocked`, append `{id, date}` and announce it in the
response (most naturally from `/done`, right when it happens).

| id                    | Unlocked when...                                                        |
| ---------------------- | ------------------------------------------------------------------------ |
| `first-day-done`      | The first day in the whole project is marked `done`.                    |
| `five-day-streak`     | The live-computed study streak (see above) reaches 5.                   |
| `first-micro-project` | Any micro-project's `status` first becomes `done`.                      |
| `flawless-week`       | A week's `review_status` becomes `done` and every day in it has `confidence: "strong"` and `has_flags: false`. |
| `month-complete`      | Every day in a month is `done` or `rest`, and every week's `review_status` is `done`. |
| `comeback`            | A day is completed after a gap of 3+ real calendar days since the previous `done` day (encourages resuming after a break, not just uninterrupted streaks). |
| `first-oss-pr`        | Any day's `github_contribution.status` first becomes `"reviewed"` — a real pull request, actually submitted and reviewed. |

This list can grow — add new rows here and the eligibility check wherever
badges are computed; no schema change needed since `badges_unlocked` entries
are free-form `{id, date}` pairs.

## Multi-language support

`months.<N>.language` and `months.<N>.test_command` (see field definitions
above) let `/done` and `/i-am-in` work outside Python:

- `"python"` (default): `/i-am-in` writes `code/` + `code/tests/` for
  `pytest`; `/done` runs `pytest code/tests/`.
- `"javascript"` / `"typescript"`: `/i-am-in` writes stubs + Jest-style tests;
  `/done` runs `npm test` (or the project's existing test script) instead.
- `"other"`: `months.<N>.test_command` must be set (asked once, at `/month`
  time) and `/done` runs that exact command instead of guessing.

`requirements.txt` remains the Python dependency file regardless; a
JS/TS month instead gets a `package.json` at the project root the first time
`/i-am-in` needs one. `github-issue-researcher` also reads `language` to
find issues in the right ecosystem.

## Open-source contribution workflow (`github_contribution`)

This is entirely **opt-in per day** — running `/github-issues` for a given
day is what activates it; a day where the user never runs that command
behaves exactly as it always has, with `github_contribution: null` and no
extra gate on `/done`.

1. `/github-issues` delegates to `github-issue-researcher`, which finds
   **3** real, currently-open issues from open-source repos with **50+
   stars**, in `months.<N>.language`, genuinely solvable by the user —
   preferring ones related to the current week's `done_when`/topics if a
   genuine connection exists, never forcing a weak one just to claim
   relevance.
2. Once the user picks one, `github_contribution` is created on the current
   day with `status: "in_progress"` and the chosen issue's real details.
3. When the user has a real PR open, they give `/github-issues` (or `/done`,
   if it's the one prompting) the URL; `pr_url` is set and `status` becomes
   `"pr_submitted"`.
4. `github-pr-reviewer` is delegated to review the actual PR (fetched live —
   never fabricated), giving specific feedback the same way `code-evaluator`
   does for local code. `status` becomes `"reviewed"`.
5. **`/done` will not mark the day complete while `status` is anything other
   than `"reviewed"`** (or the field is `null`, meaning the workflow was
   never started for this day). This is a hard gate, not a nudge — see
   `CLAUDE.md`'s "Discipline & accountability" section.

## Schema migrations

`schema_version` exists so a future shape change doesn't silently corrupt an
older `state.json`. Rule: **additive changes with a sensible default don't
need a version bump** (a skill can just treat a missing field as its
default). A change that removes a field, changes a type, or changes an enum's
meaning **does** need one — bump `schema_version`, and add a short
`scripts/migrate_state.py` step (or a one-off instruction in the commit)
that transforms an old `state.json` forward. Never edit `state.schema.json`
in a breaking way without also handling migration for anyone who already has
a `state.json` on disk.

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
6. **Discipline gate**: `/i-am-in` refuses to generate a new day if 3 or
   more *done* days in the current month have `(has_flags: true OR confidence:
   "struggled")` **and** `review_history.times_reviewed == 0` — i.e. real,
   never-revisited struggle has piled up. It tells the user plainly why and
   points them at `/spaced-review`; running at least one review pass on any
   one of those days clears the gate for the next `/i-am-in` call.
7. **GitHub contribution gate**: `/done` refuses to mark a day complete
   while that day's `github_contribution` is non-`null` and its `status`
   isn't `"reviewed"` — see "Open-source contribution workflow" above and
   `CLAUDE.md`'s "Discipline & accountability" section.
