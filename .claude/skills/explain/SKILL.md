---
name: explain
description: On-demand deep dive into a concept from any past (or current) day, pulling in that day's original resources plus fresh supplementary research if needed. Use when the user invokes /explain <concept>.
---

# /explain

**Input:** a concept/topic in `$ARGUMENTS`. If empty, ask the user (one question) what they want explained.

## Steps

1. If `state.json` doesn't exist, tell the user to run `/month` first and stop.

2. Search across all months/days' `topic` fields for a match to the requested concept. Matching doesn't need to be exact — use judgment on close/related topics.
   - **If one or more days match**: note which day(s) it's from, and pull that day's `learn.md` (and `roadmap.md` if useful) for the original explanation/resources as your starting point.
   - **If nothing matches**: say plainly this concept hasn't come up in the plan yet, and explain it from scratch — don't force a false connection to an unrelated day.

3. Give a clear, self-contained explanation in your own words, building on (not just repeating) whatever the original day's `learn.md` already covered.

4. **If the existing material is thin, or the user is asking for "more" / a different angle**, delegate to the `reading-researcher` subagent (and `video-researcher` too, if the user wants video) for fresh supplementary resources on this specific concept. Never fabricate a source.
   - Append any new resources found to that day's `further_reading.md` (same file `/for-read` uses — create it if missing, otherwise append a new dated section), noting it was triggered by `/explain`. If the concept isn't tied to any existing day, skip this file-write step and just answer conversationally.

5. This command doesn't change quiz/confidence state — it's informational. If it surfaces that this concept has been a recurring point of confusion, mention that the user might want to run `/spaced-review` on it, but don't take that action automatically.

6. If any files were written, `git add -A && git commit -m "Explain: <concept>"`. If nothing was written (pure conversational answer), skip the commit.
