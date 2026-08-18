---
name: progress
description: Read-only overview of current month/week/day, done/pending status for the month, week-review status, and micro-project status. Use when the user invokes /progress.
---

# /progress

Read-only. Makes no changes to any file.

## Steps

1. If `state.json` doesn't exist, say so and suggest running `/month`. Stop.

2. Read `state.json` and report:
   - Current month and current `(week, day)`.
   - A done/pending/rest overview of every day in the current month, in order (e.g. `Week 1: Day 1 done, Day 2 done, Day 3 rest, Day 4 current, Day 5 pending`).
   - Each week's `review_status`.
   - For any day with `has_flags: true`, note it has flagged questions pending review.
   - Any micro-projects recorded under any day this month, with their `status` (`in_progress`/`done`).
