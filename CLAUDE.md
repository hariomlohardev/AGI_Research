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

## Source of truth for state

**`STATE_SCHEMA.md` at the project root is the single canonical definition of
`state.json`'s shape.** Every skill and agent must read/write `state.json`
strictly according to that file. If a new field is ever needed, update
`STATE_SCHEMA.md` first, then use it consistently everywhere. Do not let
individual skills improvise their own fields.

## Folder layout

```
CLAUDE.md
STATE_SCHEMA.md
.claude/skills/{month,skip-to,i-am-in,done,progress,micro-project,for-read}/SKILL.md
.claude/agents/{video-researcher,reading-researcher,code-evaluator}.md
.gitignore
progress.md                 <- human-readable log, across all months
growth-notes.md             <- created by code-evaluator on its first run
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
