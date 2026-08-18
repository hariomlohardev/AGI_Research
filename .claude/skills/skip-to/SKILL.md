---
name: skip-to
description: One-time setup step to fast-forward the current month's progress to a given day number, for days already completed by hand before this tooling existed. Use only when the user invokes /skip-to <day number>, and only as part of initial project setup.
---

# /skip-to

**Input:** a day number in `$ARGUMENTS`, interpreted as the target day's `global_day` (continuous counter) within `current_month`.

## Steps

1. Read `state.json`. If it doesn't exist, tell the user to run `/month` first and stop.

2. **Safety check.** This command is meant to run once, at setup. If either:
   - any day in `months.<current_month>` already has `status: "done"`, or
   - `current_day` is already greater than 1 (i.e. progress has moved past day 1),

   then **stop and ask the user to explicitly confirm** before proceeding, clearly stating that this will overwrite real recorded progress. Do not proceed without an explicit "yes"/confirmation in their next message.

3. Once confirmed (or on a clean first run where neither condition above holds):
   - For every day with `global_day` less than the target: set `status: "done"` if it's a real day, or leave `status: "rest"` unchanged if it's a Rest Day (rest days are auto-skipped, never marked "done").
   - Set the target day's `status: "current"`.
   - Set top-level `current_day` to the target's `global_day`.
   - If the target itself is a Rest Day, advance to the next non-rest day and make that one `"current"` instead — a rest day should never be the "current" day.

4. Update `progress.md` to reflect the new state for this month.

5. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

6. `git add -A && git commit -m "Skip to day <target> in month <N>"`.
