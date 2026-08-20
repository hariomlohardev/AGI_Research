---
name: confused
description: Log a confusion note against the current day for durable tracking, so it resurfaces later in /spaced-review. Use when the user invokes /confused with a note about what's unclear.
---

# /confused

**Input:** a note in `$ARGUMENTS` describing what's unclear. If empty, ask the user (one question) what they're confused about.

## Steps

1. If `state.json` doesn't exist, tell the user to run `/month` first and stop. If the current day has no content yet (`day_<D>/` doesn't exist), still allow logging the note against `current_day` — the folder gets created if needed just to hold `confusion_notes.md`.

2. Append `{note, date}` to this day's `confusion_notes` array in `state.json`, per `STATE_SCHEMA.md`. Never overwrite prior entries — this is a persistent, growing record, not a single-slot field.

3. Append the same note (with date) to `day_<D>/confusion_notes.md`, creating it with a short header if it doesn't exist yet.

4. Optionally, if you can give a quick, accurate clarifying answer right now from the day's own materials (`learn.md`, etc.), do so — but don't treat that as "resolving" the note. The note stays logged regardless, since the primary job here is durable tracking for `/spaced-review`, not necessarily fixing it in the moment.

5. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

6. `git add -A && git commit -m "Log confusion note for day <D>"`.
