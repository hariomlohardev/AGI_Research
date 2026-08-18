---
name: progress
description: Read-only overview of current month/week/day, done/pending status for the month, week-review status, micro-project status, study streak, confidence levels, and time tracking. Use when the user invokes /progress.
---

# /progress

Read-only. Makes no changes to any file.

## Steps

1. If `state.json` doesn't exist, say so and suggest running `/month`. Stop.

2. Read `state.json` and report:
   - Current month and current `(week, day)`.
   - A done/pending/rest overview of every day in the current month, in order (e.g. `Week 1: Day 1 done, Day 2 done, Day 3 rest, Day 4 current, Day 5 pending`). Include each done day's `confidence` where set.
   - Each week's `review_status`.
   - For any day with `has_flags: true`, note it has flagged questions pending review.
   - For any day with non-empty `confusion_notes`, note it has open confusion notes.
   - Any micro-projects recorded under any day this month, with their `status` (`in_progress`/`done`).
   - **Current study streak**: computed live (not read from a stored field — see `STATE_SCHEMA.md`). Scan backward starting from the most recently **completed** day — skip past today's `current` day if it isn't `done` yet, since being mid-day shouldn't zero out the streak — then count consecutive `done` days, passing over `rest` days without breaking the count, stopping at the first non-`done`/non-`rest` day.
   - **Time tracking**, only for days that have both `time_spent_minutes` and `expected_minutes` set: a brief actual-vs-expected note. Skip this line entirely if neither is populated — don't imply a comparison that isn't there.
   - A one-line mention of how many days currently have unresolved confusion notes or "struggled"/"shaky" confidence — i.e. roughly how much material is waiting in the `/review` queue — without duplicating `/review`'s own selection logic here.
