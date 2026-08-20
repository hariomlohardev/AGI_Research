---
name: learning-stats
description: Read-only whole-project analytics — total hours studied, topics/tags covered, confidence and difficulty distribution, time-spent trend, and badges — across every month, not just the current one. Distinct from /progress, which defaults to the current month. Optionally renders a chart. Use when the user invokes /learning-stats.
---

# /learning-stats

Read-only. Makes no changes to `state.json`. May write a chart image if requested (see step 5) — that's the only file it touches.

## Steps

1. If `state.json` doesn't exist, say so and suggest running `/month`. Stop.

2. Scan **every** month in `state.json` and compute:
   - Total days done, total rest days, total days remaining (pending + current) across the whole project.
   - Total time studied: sum of `time_spent_minutes` across all done days that have it set — report it, but note plainly if some done days have `null` (self-report was skipped) so the total is visibly a lower bound, not a hidden guess.
   - Confidence distribution (`strong`/`shaky`/`struggled` counts) and difficulty distribution (`easy`/`medium`/`hard` counts) across all done days.
   - Most common `tags` across all days, with counts.
   - Micro-projects: total attempted vs. completed.
   - All `badges_unlocked`, in the order earned.
   - The cross-month study streak (same computation as `/progress`).

3. Present this as a compact, readable summary — not a wall of raw numbers. Group related figures together (e.g. time and pace together, confidence and difficulty together).

4. If there's a meaningful trend worth naming (e.g. confidence improving over the last several days, or time-per-day trending down as the user gets faster), say so in plain language — but only if the data actually supports it; don't manufacture a trend from a handful of points.

5. **Optional chart**: if the user asks for a visual (or says something like "show me a chart"), generate one with matplotlib — time-spent-per-day or confidence-over-time are the most natural — save it to `reports/learning-stats-<date>.png`, and mention the file. Skip this by default; it's opt-in since not everyone wants an image file for a text summary. If matplotlib isn't available, say so and offer the text summary instead rather than failing silently.

6. No git commit unless a chart file was created, in which case: `git add -A && git commit -m "Add learning-stats chart"`.
