---
name: final-project-check
description: Sanity-check whether the current month's final_project/structure.md still makes sense given what's actually been generated so far, and propose updates if topics have drifted. Use when the user invokes /final-project-check.
---

# /final-project-check

## Steps

1. If `state.json` doesn't exist, or `month_<current_month>/final_project/structure.md` doesn't exist, tell the user clearly what to run first (`/month`) and stop.

2. Read `month_<N>/final_project/structure.md` alongside every generated `day_<D>/learn.md` and `coding_problems.md` so far this month (only days that actually exist on disk — don't assume ungenerated days).

3. Compare: does the final project skeleton still line up with what's actually been covered? Plans sometimes drift once real days are written (a topic goes deeper or shallower than expected, or takes a different turn).

4. **If it still holds up**, say so plainly and make no changes.

5. **If it's drifted**, propose specific edits (don't just say "it's off" — say what changed and what you'd adjust). **Ask for confirmation before writing anything** — this overwrites an existing file the user may have referenced elsewhere (e.g. in `logs.md` entries). Once confirmed, apply the edit to `structure.md` and note the revision (with date) at the bottom of the file rather than silently erasing the original framing.

6. If changes were made, `git add -A && git commit -m "Update final project structure after drift check"`. If not, no commit.
