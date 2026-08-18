---
name: review
description: Spaced-repetition review pulling questions from past completed days, weighted toward ones marked shaky/struggled, flagged, or with open confusion notes, rather than only quizzing on today. Use when the user invokes /review.
---

# /review

**Input (optional):** a note in `$ARGUMENTS` — e.g. a specific topic, day, or "just week 1" to narrow the pool. If absent, pull from the full pool described below.

## Steps

1. If `state.json` doesn't exist, or no day anywhere has `status: "done"` yet, tell the user there's nothing to review yet and stop.

2. **Build the eligible pool**: every day across all months with `status: "done"`. Note that no original quiz transcript is persisted — `/review` re-derives fresh questions from each candidate day's own materials (`coding_problems.md`, `learn.md`, and, if present, `flagged_questions.md` / `confusion_notes.md`), rather than repeating the exact same wording as the original `/done` quiz.

3. **Weight selection** toward:
   - `confidence: "struggled"` or `"shaky"` over `"strong"`.
   - `has_flags: true` days.
   - Days with non-empty `confusion_notes`.
   - Days with an older (or `null`) `review_history.last_reviewed`, or a lower `times_reviewed` — so review naturally rotates rather than hammering the same day forever.
   - If `$ARGUMENTS` names a topic/day/week, restrict the pool to matches first, then apply the same weighting within it.

4. Pick a handful of questions (aim for 4-8 unless the user's note implies otherwise) spanning a few different days rather than only the single weakest one, so review stays broad. For each, briefly note which day/topic it's from before asking.

5. **Quiz normally** — same re-ask-until-correct behavior as `/done`, with the same escape hatch ("flag this" / "skip this" logs to that day's `flagged_questions.md` and moves on, it does not re-open `has_flags` bookkeeping beyond what's already there).

6. For every day a question was pulled from, update its `review_history`: set `last_reviewed` to today's date and increment `times_reviewed` by 1.

7. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

8. `git add -A && git commit -m "Review session across <days>"`.
