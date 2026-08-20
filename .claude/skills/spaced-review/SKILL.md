---
name: spaced-review
description: Real spaced-repetition (simplified SM-2) review pulling questions from past completed days, prioritizing whichever are actually due plus ones marked shaky/struggled, flagged, or with open confusion notes. Use when the user invokes /spaced-review.
---

# /spaced-review

**Input (optional):** a note in `$ARGUMENTS` — e.g. a specific topic, day, or "just week 1" to narrow the pool. If absent, pull from the full pool described below.

## Steps

1. If `state.json` doesn't exist, or no day anywhere has `status: "done"` yet, tell the user there's nothing to review yet and stop.

2. **Build the eligible pool**: every day across all months with `status: "done"`. Note that no original quiz transcript is persisted — `/spaced-review` re-derives fresh questions from each candidate day's own materials (`coding_problems.md`, `learn.md`, and, if present, `flagged_questions.md` / `confusion_notes.md`), rather than repeating the exact same wording as the original `/done` quiz.

3. **Prioritize by due date first, then by weight:**
   - Days whose `srs.next_review_due` is today or earlier (or `null`, meaning never reviewed — always due) come first. This is what makes it genuinely spaced, not just recency-sorted.
   - Within that, weight further toward: `confidence: "struggled"` or `"shaky"`, `has_flags: true`, non-empty `confusion_notes`.
   - If `$ARGUMENTS` names a topic/day/week, restrict the pool to matches first, then apply the same prioritization within it.
   - If nothing is due yet and the user didn't name a specific target, say so plainly and ask if they want to review ahead of schedule anyway rather than silently picking something.

4. Pick a handful of questions (aim for 4-8 unless the user's note implies otherwise) spanning a few different days rather than only the single weakest one, so review stays broad. For each, briefly note which day/topic it's from before asking.

5. **Quiz normally** — same re-ask-until-correct behavior as `/done`, with the same escape hatch ("flag this" / "skip this" logs to that day's `flagged_questions.md` and moves on). Same no-shortcuts discipline rule as `/done` applies too — no giving answers, no marking correct on request.

6. **Update SM-2 scheduling** for every day a question was pulled from, per `STATE_SCHEMA.md`'s "Spaced repetition scheduling" section:
   - Rate quality `q` (0-5) from how that day's review question(s) went: `5` = correct, no hesitation; `3` = correct but needed a re-ask; `1` = flagged/skipped or still wrong.
   - `ease_factor = max(1.3, ease_factor + (0.1 - (5-q) * (0.08 + (5-q) * 0.02)))`.
   - `interval_days`: reset to `1` if `q < 3`. Otherwise: `1` on the first successful review ever for that day, `6` on the second, `round(interval_days * ease_factor)` after that.
   - `next_review_due = today + interval_days` (ISO date).
   - Also update `review_history`: `last_reviewed = today`, `times_reviewed += 1`.

7. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

8. `git add -A && git commit -m "Review session across <days>"`.
