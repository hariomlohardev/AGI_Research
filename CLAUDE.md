# CLAUDE.md

## What this system is

A Claude Code-driven self-study curriculum runner. Learning plans are pasted
in month-by-month (Week → Day, each day with videos/reading, hand-worked
math, coding tasks, and practice questions). This project tracks progress,
lazily generates each day's materials (with real, verified research and
runnable code stubs + tests), quizzes on completion, and supports optional
hands-on micro-projects that build on completed days.

Rest days are explicit in the source plans and are tracked as real entries
(status `rest`), not skipped silently.

## Source of truth for state — and how it's enforced

`STATE_SCHEMA.md` at the project root is the canonical **human-readable**
definition of `state.json`'s shape and the rules a schema can't express
(what's computed vs. stored, the "exactly one current day" invariant, etc.).
`state.schema.json` is its **machine-enforced** counterpart — the same shape,
in a JSON-Schema-like format. The two must always be kept in sync: if a field
is ever added or changed, update `state.schema.json` first (that's what
actually gets checked), then update `STATE_SCHEMA.md`'s prose to match.

This isn't just a convention skills are expected to remember:

- `scripts/validate_state.py` (stdlib only, no `pip install` required)
  checks `state.json` against `state.schema.json` structurally (required
  fields, types, enums, and rejecting any field not defined in the schema),
  plus a couple of semantic checks a schema alone can't express — exactly
  one day in the current month has `status: "current"`, and it's the one
  `current_day` actually points to.
- Every skill that writes `state.json` runs this validator before
  committing, and stops if it fails.
- `.githooks/pre-commit` runs the same validator automatically on any
  commit that stages `state.json`, and **blocks the commit** if it doesn't
  pass — this is the actual backstop, not just a convention. `/month`'s
  first-run setup runs `git config core.hooksPath .githooks` so this hook
  is active from the start (`.git/hooks/` itself isn't version-controlled,
  so this has to be set explicitly).

Do not let individual skills improvise fields not in `state.schema.json` —
they'll fail validation and the commit will be rejected.

## Folder layout

```
CLAUDE.md
STATE_SCHEMA.md              <- human-readable canonical shape + rules
state.schema.json            <- machine-enforced counterpart of the above
scripts/validate_state.py    <- validates state.json against state.schema.json
.githooks/pre-commit         <- blocks commits with invalid state.json
.claude/skills/{month,skip-to,i-am-in,done,progress,micro-project,for-read,
                review,explain,pace,final-project-check,confused,recap}/SKILL.md
.claude/agents/{video-researcher,reading-researcher,code-evaluator}.md
.gitignore
progress.md                 <- human-readable log, across all months
growth-notes.md             <- code-style patterns, written by code-evaluator
struggle-log.md             <- conceptual/quiz struggle patterns, written by /done
project-history.md          <- how each month's final project connects to the next
state.json                  <- machine-readable state (see STATE_SCHEMA.md)
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
      micro_projects/
        <slug>/
          brief.md
          code/
          code/tests/
          code_review.md
```

## Core rules

- **Tests must pass before a day counts as done.** `/done` runs pytest on
  the current day's `code/tests/` and stops on any failure — no quiz, no
  advancing, until they pass. Exception: individual quiz questions can be
  flagged/skipped via the escape hatch (see `/done`'s SKILL.md) without
  blocking the day.
- **`/skip-to` is a one-time setup step**, meant to be run once right after
  the first-ever `/month` on a fresh project, to fast-forward past days
  already completed by hand before this tooling existed. If state shows real
  progress already, it must ask for explicit confirmation before overwriting
  anything.
- Dependencies are installed with `pip install -r requirements.txt`.
- Subagents (`.claude/agents/`) handle research and code review so the
  skills themselves stay focused on orchestration/state, not doing that work
  inline.
- **Some values are computed live, never stored** (study streaks, pace
  projections) — see `STATE_SCHEMA.md`'s "deliberately NOT stored" section.
  This keeps `state.json` from holding numbers that could drift out of sync
  with reality.
- **Self-reported data (time spent, confusion notes) is never invented.**
  If the user doesn't give a number/note, the field stays `null`/empty
  rather than being estimated on their behalf.
