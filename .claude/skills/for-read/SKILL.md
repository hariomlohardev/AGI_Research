---
name: for-read
description: Find supplementary reading (articles, papers, explainers) on the current day's topic by delegating to the reading-researcher subagent, and append results to that day's further_reading.md. Use when the user invokes /for-read, optionally with a focus note, any time after that day's material has been generated.
---

# /for-read

**Input (optional):** a focus note in `$ARGUMENTS`, e.g. "more on Laplace smoothing."

## Steps

1. Read `state.json` for `current_month`/`current_day`. If `day_<D>/` doesn't exist yet (i.e. `/i-am-in` hasn't been run for this day), tell the user clearly to run `/i-am-in` first and stop.

2. **Delegate to the `reading-researcher` subagent** for the current day's topic, passing `$ARGUMENTS` as an optional focus note if given. Never fabricate sources.

3. Append the results to `day_<D>/further_reading.md`:
   - Create the file (with a header) if it doesn't exist yet.
   - Otherwise **append** a new dated section — never overwrite prior entries.
   - Each entry should note what was searched for (the topic, and the focus note if one was given).

4. This can be run any number of times per day. Each run adds a new dated section rather than replacing the last.

5. This is **entirely optional** — reading material is never a requirement to run `/done` or mark a day complete, in either `guided` or `challenge` mode. It's just something to reach for whenever the user wants more depth.

6. `git add -A && git commit -m "Add further reading for day <D>"`.
