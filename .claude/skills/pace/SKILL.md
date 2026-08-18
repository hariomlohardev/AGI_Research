---
name: pace
description: Read-only projection of finish date for the current month, based on months.<N>.started and completion rate so far. Use when the user invokes /pace.
---

# /pace

Read-only. Makes no changes to any file. All numbers here are **computed
live** from `state.json` — none of this is stored, per `STATE_SCHEMA.md`.

## Steps

1. If `state.json` doesn't exist, tell the user to run `/month` first and stop.

2. Read `months.<current_month>.started` (the date this month began) and today's date.

3. Compute:
   - Calendar days elapsed since `started`.
   - Total **real** (non-rest) days in the month so far defined in state, and how many of those are `status: "done"`.
   - A rough completion rate: done days ÷ elapsed calendar days.
   - Remaining real days left in the month (pending + current, excluding rest).
   - A projected finish date: remaining real days ÷ completion rate, added to today.

4. Present this plainly — e.g. "You're N days in, M done, at roughly R days/week, projecting finish around <date>." Be clear this is a rough projection from recent pace, not a guarantee, especially early in a month when the sample is small (say so explicitly if fewer than ~4 days are done so far).

5. No git commit — this command writes nothing.
