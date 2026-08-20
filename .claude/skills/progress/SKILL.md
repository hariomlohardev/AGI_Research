---
name: progress
description: Read-only overview of current month/week/day, done/pending status, week-review status, micro-project status, study streak (spanning month boundaries), confidence/difficulty/tags, badges, and time tracking. Optionally shows all months, not just the current one. Use when the user invokes /progress.
---

# /progress

**Input (optional):** `"all"` in `$ARGUMENTS` to report on every month, not just the current one.

Read-only. Makes no changes to any file.

## Steps

1. If `state.json` doesn't exist, say so and suggest running `/month`. Stop.

2. Read `state.json` and report, for the current month by default (or every month if `$ARGUMENTS` was `"all"`):
   - Current month and current `(week, day)`, plus that month's `language`/`skill_level`.
   - A done/pending/rest overview of every day, in order (e.g. `Week 1: Day 1 done, Day 2 done, Day 3 rest, Day 4 current, Day 5 pending`). Include each done day's `confidence` and `difficulty` where set, and its `tags` if any. For any day that's been generated (has a `topic`), note its `mode` if it's `"challenge"` — guided is the default and doesn't need calling out.
   - Each week's `review_status`.
   - For any day with `has_flags: true`, note it has flagged questions pending review.
   - For any day with non-empty `confusion_notes`, note it has open confusion notes.
   - Any micro-projects recorded under any day, with their `status` (`in_progress`/`done`).
   - **Current study streak**: computed live (not read from a stored field — see `STATE_SCHEMA.md`). Build one chronological list of every day **across all months** (month ascending, then `global_day` ascending), scan backward starting from the most recently **completed** day — skip past today's `current` day if it isn't `done` yet — then count consecutive `done` days, passing over `rest` days without breaking the count, stopping at the first non-`done`/non-`rest` day. This is always computed across the whole project, even when the rest of the report is scoped to one month.
   - **Time tracking**, only for days that have both `time_spent_minutes` and `expected_minutes` set: a brief actual-vs-expected note. Skip this line entirely if neither is populated — don't imply a comparison that isn't there.
   - A one-line mention of how many days currently have unresolved confusion notes, `"struggled"`/`"shaky"` confidence, or an `srs.next_review_due` on or before today — i.e. roughly how much material is waiting in the `/spaced-review` queue — without duplicating `/spaced-review`'s own selection logic here.
   - **Badges**: list everything in `badges_unlocked`. Also check current eligibility per `STATE_SCHEMA.md`'s badges table — if anything is newly eligible and not yet recorded, append it and announce it here too (not just from `/done`), since the user might check `/progress` before running `/done` again.
