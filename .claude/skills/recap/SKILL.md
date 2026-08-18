---
name: recap
description: Read-only plain-language weekly summary of what was covered, good for journaling or sharing progress. Use when the user invokes /recap, optionally naming a week.
---

# /recap

**Input (optional):** a week number/reference in `$ARGUMENTS`. If absent, default to the most recently completed week if the current week isn't finished yet, otherwise the current week's days done so far.

Read-only. Makes no changes to any file.

## Steps

1. If `state.json` doesn't exist, tell the user to run `/month` first and stop.

2. Resolve the target week and gather its days' `topic`, `status`, and `confidence` fields.

3. Write a short, plain-language recap (a few sentences to a short paragraph, not a bulleted data dump) — what got covered, in what order, how it generally went (referencing confidence levels in plain terms like "came together quickly" vs. "took some re-tries," without just listing the raw enum values), and whether the week's `done_when` criteria were met if `review_status` is `done`.

4. Mention the current study streak (computed live the same way `/progress` does — scan backward from the most recently *completed* day, skipping today's not-yet-done `current` day so it doesn't zero out the streak, counting consecutive `done` days with `rest` days not breaking the count).

5. Offer, but don't do automatically: "want me to add this to `progress.md`?" — only write it there if the user says yes, since this command is read-only by default.
