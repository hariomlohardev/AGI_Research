# AGI Research

Twelve months of building AI from scratch — math worked on paper, code written
without shortcuts, every day logged honestly whether it went well or not.

This repo is my working record for a 12-month roadmap: probability and
information theory, the pre-deep-learning classics built by hand, then
PyTorch, transformers, training my own GPT, fine-tuning and alignment, RL,
MLOps, and a final mini-research project. While I'm learning an algorithm's
internals I don't get to import it — no `sklearn`, no `scipy.stats`, no
`np.cov` — until the day explicitly says "now compare against the real
library."

It runs on a Claude Code-driven curriculum runner that lives in this same repo
(`CLAUDE.md`, `.claude/skills/`). It generates each day's material as I reach
it, quizzes me on it, and refuses to mark a day done when the tests don't
pass — including when I ask it to.

## About me

**Hariom Lohar** — [`@hariomlohardev`](https://github.com/hariomlohardev)

- **GitHub:** https://github.com/hariomlohardev
- **Email:** hariomlohar.new@gmail.com
- **What I'm doing here:** teaching myself machine learning from first
  principles — derive it, then build it, then check it against the real
  library — rather than collecting framework tutorials.
- **Current focus:** Month 2 — probability, information theory, and classical
  ML from scratch, in Python.
- **A constraint I designed around:** zero-cost compute only. Free Colab and
  Kaggle GPUs, free datasets, free hosting. Nothing in this roadmap needs a
  credit card.
- **How I work:** ~2-3 hours a day, math on paper before the editor is open,
  and if I get something wrong it stays in the log (`struggle-log.md`,
  `growth-notes.md`) instead of getting quietly deleted.

## Where I'm at

Snapshot — **2026-09-05**:

| | |
|---|---|
| **Month** | 2 of 12 — Probability, Information Theory & Classical ML from Scratch |
| **Current day** | 8 of 24 — Week 2 Day 2, *MAP Estimation & Priors* |
| **Days completed** | 7 |
| **Language / level** | `python` / `intermediate` |
| **Badges** | `first-day-done`, `five-day-streak` |

This table is a hand-written snapshot and will go stale. The live answer comes
from `/progress` (or `/learning-stats` for the full analytics); `progress.md`
is the human-readable running log, and `state.json` is the machine-readable
source of truth.

## The roadmap

| Month | Focus | Status |
|---|---|---|
| 1 | Math Foundations + Autograd Engine | ✅ done before this repo existed |
| 2 | Probability, Information Theory & Classical ML from Scratch | 🔄 in progress |
| 3 | PyTorch, CNNs & RNNs | planned |
| 4 | Attention & Transformer Fundamentals | planned |
| 5 | Train & Sample From Your Own GPT | planned |
| 6 | Fine-Tuning & Alignment | planned |
| 7 | Modern Architectures & Generative Models | planned |
| 8 | Reinforcement Learning & Reasoning | planned |
| 9 | Systems, MLOps & Deployment | planned |
| 10 | Research Skills & Paper Replication | planned |
| 11 | Mini-Research Project | planned |
| 12 | Launch & Portfolio | planned |

Each month is 4 weeks × 6 days, ending in a capstone that packages the
month's work into something real. Month 2's capstone is **`minisklearn`** —
every algorithm from Weeks 1-3 assembled into one tested library, run on a
genuine dataset, then benchmarked honestly against actual `sklearn`.

Month 1 (Gram-Schmidt/SVD/PCA, Jacobians, chain rule, a hand-built autograd
engine, an MLP, a tweaked Adam) was finished before I set this tooling up, so
it has no day-by-day entries here — only this note, for continuity.

> **Honest note**, carried over from the roadmap itself: finishing this gets me
> a real portfolio and provable work. It does not guarantee a salary, a job, or
> a visa. The outcome to expect is "credible junior ML engineer with proof of
> work," not a number.

## What's in here

```
month_<N>/week_<W>/day_<D>/   <- one folder per study day: learn.md, roadmap.md,
                                 coding_problems.md, code/, code/tests/, reviews
month_<N>/final_project/      <- that month's capstone + running logs
progress.md                   <- human-readable log across all months
struggle-log.md               <- concepts I got wrong, so they resurface later
growth-notes.md               <- recurring code-style patterns, from code review
state.json                    <- machine-readable state (schema-validated)
.claude/skills/               <- the 19 slash commands that run all of this
.claude/agents/               <- research / code-review / PR-review subagents
```

`CLAUDE.md` has the complete folder map and is the detailed operating manual;
this file is the quick start.

## Requirements

- git
- Python 3.9+ (developed and run on 3.13)
- `pip install -r requirements.txt` — currently `pytest`, `matplotlib`, and
  `numpy`, each pinned to an exact version so a multi-month project doesn't
  drift underneath itself
- For JavaScript/TypeScript months: Node.js + npm

`/month`'s first run checks for these and says plainly what's missing, instead
of failing confusingly later. `scripts/validate_state.py` deliberately uses
only the standard library, so state stays checkable with no install at all.

## Quick start

If you want to run this system on your own learning plan:

1. Open the project in Claude Code.
2. Paste (or attach) your first month's plan and run `/month`.
3. Already did some days by hand before this existed? Run
   `/skip-to <day number>` once, right after, to fast-forward past them.
4. Run `/i-am-in` to generate today's material. Add "challenge me" if you'd
   rather hunt down your own resources than be handed videos.
5. Work through it, then run `/done` to test, quiz, and close out the day.
6. Repeat. Paste the next month's plan with `/month` when you finish one.

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

The whole point is that "done" means done, so the record is worth something
later. `/done` won't mark a day complete without tests genuinely passing and
the quiz genuinely answered — or explicitly flagged, which is its one
sanctioned shortcut and stays visible in `/progress` afterwards. If I've
started an open-source contribution for the day via `/github-issues`, `/done`
also won't close it until a real PR exists and has been reviewed. And if
unresolved struggle piles up across 3+ days without a review pass, `/i-am-in`
refuses to generate a new day until I run `/spaced-review` on at least one of
them — a hard stop against pushing forward while gaps quietly accumulate.

This holds even when I'm the one asking to skip. `CLAUDE.md`'s "Discipline &
accountability" section is the full policy, and `discipline-log.md` is the
plain record of anything it's had to decline.

## How state is kept correct

`state.json` is validated against `state.schema.json` before every commit —
by each skill that writes it, by the `.githooks/pre-commit` hook that blocks
the commit outright on failure, and by GitHub Actions on every push. An
invented field, a bad enum value, or an inconsistent "current day" gets
rejected rather than silently written. `.githooks/commit-msg` keeps commit
messages clean of AI-attribution trailers.

Both hooks activate via `git config core.hooksPath .githooks`, which `/month`
sets up on first run. See `STATE_SCHEMA.md` for the full shape and the rules a
schema can't express, and `tests/` for the suite that locks the validator's own
behavior in.
