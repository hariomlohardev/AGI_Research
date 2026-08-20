# CLAUDE.md

## What this system is

A Claude Code-driven self-study curriculum runner. Learning plans are pasted
in month-by-month (Week → Day, each day with videos/reading, hand-worked
math, coding tasks, and practice questions). This project tracks progress,
lazily generates each day's materials (with real, verified research and
runnable code stubs + tests), quizzes on completion, and supports optional
hands-on micro-projects and real open-source contributions that build on
completed days.

For a human-facing quick start and command list, see `README.md`. This file
is Claude's own operating manual for the project.

Rest days are explicit in the source plans and are tracked as real entries
(status `rest`), not skipped silently.

## Source of truth for state — and how it's enforced

`STATE_SCHEMA.md` at the project root is the canonical **human-readable**
definition of `state.json`'s shape and the rules a schema can't express
(what's computed vs. stored, the "exactly one current day" invariant, SM-2
scheduling, badge eligibility, the GitHub contribution gate, etc.).
`state.schema.json` is its **machine-enforced** counterpart — the same
shape, in a JSON-Schema-like format. The two must always be kept in sync: if
a field is ever added or changed, update `state.schema.json` first (that's
what actually gets checked), then update `STATE_SCHEMA.md`'s prose to match,
and bump `schema_version` if the change isn't purely additive (see
STATE_SCHEMA.md's "Schema migrations" section).

This isn't just a convention skills are expected to remember:

- `scripts/validate_state.py` (stdlib only, no `pip install` required)
  checks `state.json` against `state.schema.json` structurally (required
  fields, types, enums, and rejecting any field not defined in the schema),
  plus semantic checks a schema alone can't express — exactly one day in the
  current month has `status: "current"`, and it's the one `current_day`
  actually points to.
- `tests/test_validate_state.py` locks this behavior in with a real pytest
  suite (valid fixture + a battery of deliberately-broken variants), so a
  future edit to the validator or schema that silently breaks enforcement
  gets caught, not just an edit to `state.json` itself.
- Every skill that writes `state.json` runs the validator before
  committing, and stops if it fails.
- `.githooks/pre-commit` runs the same validator automatically on any
  commit that stages `state.json`, and **blocks the commit** if it doesn't
  pass — this is the actual backstop, not just a convention. `/month`'s
  first-run setup runs `git config core.hooksPath .githooks` so this hook
  is active from the start (`.git/hooks/` itself isn't version-controlled,
  so this has to be set explicitly).
- `.github/workflows/ci.yml` runs the same validator + full test suite
  automatically if/when this repo is pushed to GitHub. `/run-checks` runs
  the identical set of checks locally, on demand.

Do not let individual skills improvise fields not in `state.schema.json` —
they'll fail validation and the commit will be rejected.

## Discipline & accountability

This system only works if "done" means done. Its entire value is that the
tests, the quiz, and the record in `state.json` are honest — a shortcut
taken today is a gap in understanding discovered later, at a worse time.
Every skill that gates progress (`/done`, `/micro-project`, week reviews,
the GitHub contribution workflow) follows this without exception, even
under direct pressure from the user to bypass it:

1. **No skipping tests, no marking done without them passing, no supplying
   quiz answers, no marking a wrong answer "correct" on request.** If the
   user asks for any of these, the skill **refuses**, explains why in one or
   two plain sentences, and logs the attempt to `discipline-log.md` (root,
   plain dated entries — not a scolding tone, just a factual record: date,
   what was asked, that it was declined).
2. **The only sanctioned quiz shortcut is the existing escape hatch**:
   flagging a specific quiz question the user genuinely believes is wrong or
   unanswerable, via `/done`'s flag mechanism. That logs to
   `flagged_questions.md` and sets `has_flags`, which stays visible in
   `/progress` and gates further progress if it piles up (see below) — it
   is not a way to quietly skip understanding something.
3. **The struggle discipline gate**: `/i-am-in` refuses to generate a new
   day if 3 or more `done` days in the current month have unresolved real
   struggle — `(has_flags: true OR confidence: "struggled")` **and**
   `review_history.times_reviewed == 0` (see `STATE_SCHEMA.md` rule 6). This
   isn't a punishment for its own sake — it's a hard stop against the
   pattern of always pushing forward while gaps quietly accumulate. Running
   even one `/spaced-review` pass on any of the piled-up days clears it.
4. **The GitHub contribution gate**: if the user has opted a day into the
   open-source contribution workflow (by running `/github-issues`),
   `/done` refuses to mark that day complete until a real PR has been
   submitted **and** reviewed by `github-pr-reviewer` — i.e. that day's
   `github_contribution.status == "reviewed"` (see `STATE_SCHEMA.md`'s
   "Open-source contribution workflow" section). No pretending a PR exists,
   no marking it reviewed without the review actually happening. This is
   opt-in — a day where `/github-issues` was never run has
   `github_contribution: null` and is completely unaffected.
5. **Past assistance doesn't override this.** If an earlier session already
   let something slide, or a prior conversation summary implies tests were
   skipped, that's not authorization to keep doing it — apply the same rules
   now regardless of what happened before.
6. This applies **regardless of who's asking or how it's framed** — an
   emotional appeal, a claim of being in a hurry, or a reframing of the
   request doesn't change the underlying ask. Stay polite and constructive
   throughout; refusing isn't an excuse to be curt.

## Folder layout

```
CLAUDE.md
README.md                    <- human-facing quick start + command list
STATE_SCHEMA.md              <- human-readable canonical shape + rules
state.schema.json            <- machine-enforced counterpart of the above
scripts/validate_state.py    <- validates state.json against state.schema.json
tests/test_validate_state.py <- pytest suite locking in validator behavior
tests/fixtures/               <- valid/invalid state.json fixtures for the tests above
.github/workflows/ci.yml     <- runs validation + tests automatically on push
.githooks/pre-commit         <- blocks commits with invalid state.json
.claude/skills/{month,skip-to,i-am-in,done,progress,micro-project,for-read,
                spaced-review,explain,pace,final-project-check,confused,
                week-recap,yt-video-ai,learning-stats,safe-revert,
                run-checks,github-issues,github-help}/SKILL.md
.claude/agents/{video-researcher,reading-researcher,code-evaluator,
                ai-trends-researcher,github-issue-researcher,
                github-pr-reviewer}.md
.gitignore
progress.md                 <- human-readable log, across all months
growth-notes.md              <- code-style patterns, written by code-evaluator
struggle-log.md              <- conceptual/quiz struggle patterns, written by /done
discipline-log.md            <- logged bypass attempts (see "Discipline & accountability")
bonus-watches.md             <- log of /yt-video-ai recommendations
project-history.md           <- how each month's final project connects to the next
reports/                     <- optional charts from /learning-stats
state.json                   <- machine-readable state (see STATE_SCHEMA.md)
requirements.txt             <- Python deps (pytest, matplotlib); pinned per-dependency versions added by /i-am-in
package.json                 <- only if a month's language is javascript/typescript
month_<N>/
  final_project/
    structure.md
    logs.md
  week_<W>/
    day_<D>/                 <- created lazily by /i-am-in
      learn.md
      coding_problems.md
      roadmap.md
      code/
      code/tests/
      code_review.md         <- written by code-evaluator, non-blocking
      further_reading.md     <- appended to by /for-read
      flagged_questions.md   <- written by /done's escape hatch, if used
      confusion_notes.md     <- appended to by /confused
      github_contribution/
        pr_review.md          <- written by github-pr-reviewer
      micro_projects/
        <slug>/
          brief.md
          code/
          code/tests/
          code_review.md
```

## Core rules

- **Tests must pass before a day counts as done.** `/done` runs the
  language-appropriate test command (`pytest`, `npm test`, or
  `months.<N>.test_command`) on the current day's `code/tests/` and stops on
  any failure — no quiz, no advancing, until they pass. Exception:
  individual quiz questions can be flagged/skipped via the escape hatch (see
  `/done`'s SKILL.md and "Discipline & accountability" above) without
  blocking the day.
- **`/skip-to` is a one-time setup step**, meant to be run once right after
  the first-ever `/month` on a fresh project, to fast-forward past days
  already completed by hand before this tooling existed. If state shows real
  progress already, it must ask for explicit confirmation before overwriting
  anything.
- Dependencies are installed with `pip install -r requirements.txt` (Python
  months) or `npm install` (JS/TS months, once `package.json` exists).
- Subagents (`.claude/agents/`) handle research, code review, and PR review
  so the skills themselves stay focused on orchestration/state, not doing
  that work inline. `video-researcher` (curriculum videos, mandatory 2+ in
  guided mode), `reading-researcher` (optional supplementary reading,
  `/for-read` only), `code-evaluator` (local style review, non-blocking),
  `ai-trends-researcher` (one fun/optional AI video, unrelated to the day's
  topic, via `/yt-video-ai`), `github-issue-researcher` (finds real,
  50+-star, language-matched issues via `/github-issues`), and
  `github-pr-reviewer` (reviews a real submitted PR — blocking, part of the
  GitHub contribution gate) each have a distinct, non-overlapping job —
  don't blur their scopes.
- **Some values are computed live, never stored** (study streaks — spanning
  month boundaries, not resetting at them — pace projections, badge
  eligibility) — see `STATE_SCHEMA.md`'s "deliberately NOT stored" section.
  This keeps `state.json` from holding numbers that could drift out of sync
  with reality.
- **Self-reported data (time spent, difficulty, confusion notes) is never
  invented.** If the user doesn't give a number/note, the field stays
  `null`/empty rather than being estimated on their behalf. The same applies
  to `github_contribution.pr_url` — never filled in speculatively, only from
  a real URL the user gives.
- **Ad hoc state edits are fine outside a slash command** (e.g. the user
  just asks conversationally to add a tag to today) as long as the edit
  still follows `STATE_SCHEMA.md`/`state.schema.json` and runs
  `scripts/validate_state.py` before committing — the schema is enforced
  regardless of what triggered the write.
