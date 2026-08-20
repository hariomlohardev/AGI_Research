# Curriculum System

A Claude Code-driven self-study runner. You paste in a month's learning plan
(Week → Day, with videos, reading, math, coding tasks, and practice
questions); it tracks your progress, generates each day's materials as you
reach it, quizzes you honestly, and keeps a real record of how it's going —
confidence, time spent, difficulty, streaks, and a few things worth
celebrating along the way.

## Requirements

- git
- Python 3.9+
- `pip install -r requirements.txt` (installs `pytest` and `matplotlib`)
- For JavaScript/TypeScript months: Node.js + npm

`/month`'s first run checks for these automatically and tells you plainly if
something's missing, instead of failing confusingly later.

## Quick start

1. Open this project in Claude Code.
2. Paste (or attach) your first month's learning plan and run `/month`.
3. Already completed some days by hand before this existed? Run
   `/skip-to <day number>` right after, once, to fast-forward past them.
4. Run `/i-am-in` to generate today's material. Add "challenge me" if you'd
   rather find your own resources than be handed videos.
5. Work through it, then run `/done` to test, quiz, and close out the day.
6. Repeat. Paste the next month's plan with `/month` whenever you finish one.

## Commands

| Command | What it does |
|---|---|
| `/month` | Parse a new month's plan; one-time project setup on first run. |
| `/skip-to <day>` | One-time: fast-forward past days already done by hand. |
| `/i-am-in [note]` | Generate today's material. Add "challenge me" for no-links self-directed mode. |
| `/done` | Test, review, quiz, and close out the current day. No shortcuts — see below. |
| `/progress [all]` | Read-only status: current month, streak, badges, flags, review queue. Pass `all` for every month. |
| `/micro-project [note]` | A small hands-on project applying the current day's topic. |
| `/for-read [note]` | Optional supplementary reading on the current day's topic. Never required. |
| `/spaced-review` | Real spaced-repetition (SM-2) review of past days, prioritizing what's actually due. |
| `/explain <concept>` | On-demand deep dive into any past (or new) concept. |
| `/pace` | Projects a finish date from your completion rate so far. |
| `/final-project-check` | Sanity-checks the month's final project against what's actually been covered. |
| `/confused <note>` | Log something unclear, so it resurfaces in `/spaced-review`. |
| `/week-recap [week]` | Plain-language weekly summary — good for journaling. |
| `/yt-video-ai [note]` | One good, real AI video to watch — a treat, unrelated to today's topic. Only after a day is done. |
| `/github-issues` | Find/pick/track a real open-source issue to solve (50+ star repos, your language). Once started, `/done` requires a reviewed PR to close that day. |
| `/github-help [question]` | Freeform help if you get stuck on a GitHub contribution — git workflow, the codebase, debugging. |
| `/learning-stats` | Whole-project analytics: hours, confidence/difficulty trends, badges. Optional chart. |
| `/safe-revert` | Undo the last commit this tooling made, with a preview and confirmation. Never destructive. |
| `/run-checks` | Run everything CI would run, locally, on demand. |

## How this stays honest

`/done` won't mark a day complete without tests genuinely passing and the
quiz genuinely answered (or explicitly flagged, its one sanctioned
shortcut) — even if you ask it to skip ahead. If you've started a GitHub
contribution for the day via `/github-issues`, `/done` also won't close
until a real PR is submitted and reviewed. If unresolved struggle piles
up across 3+ days without a review pass, `/i-am-in` won't generate a new day
until you run `/spaced-review` on at least one of them. See `CLAUDE.md`'s
"Discipline & accountability" section for the full policy, and
`discipline-log.md` for a plain record of anything it's had to decline.

## How state is kept correct

`state.json` is validated against `state.schema.json` before every commit —
locally by each skill and by a git pre-commit hook, and automatically in CI
if this repo is on GitHub. An invented field, a bad enum value, or an
inconsistent "current day" gets rejected, not silently written. See
`STATE_SCHEMA.md` for the full shape and rules, and `tests/` for the suite
that locks the validator's own behavior in.

## Project layout

See `CLAUDE.md`'s "Folder layout" section for the complete map — this file
is the human quick-start; `CLAUDE.md` is Claude's own detailed operating
manual for the project.
