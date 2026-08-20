---
name: yt-video-ai
description: After a day is marked done, recommends one good, real, current video on something interesting in AI — not necessarily related to today's or this month's curriculum topic. A treat, not required study material. Use when the user invokes /yt-video-ai, optionally with a note about what area of AI they're in the mood for.
---

# /yt-video-ai

**Input (optional):** a note in `$ARGUMENTS` about what area of AI they're interested in right now (e.g. "something on robotics" or "recent research, not product news"). If absent, let `ai-trends-researcher` pick freely.

## Steps

1. Read `state.json`. The current day, or the most recently completed day if the current isn't done yet, must have `status: "done"` — otherwise tell the user clearly to finish and run `/done` first, and stop (mirrors `/micro-project`'s gating).

2. **Delegate to the `ai-trends-researcher` subagent**, passing `$ARGUMENTS` if given. This is deliberately **not** the same as `video-researcher` — it isn't scoped to today's topic, isn't required to return 2+ results, and isn't part of base day generation. Never fabricate a title or link.

3. Present the recommendation conversationally in the response.

4. Append a dated entry to `bonus-watches.md` (root; create with a short header if missing) — title, link, and a one-line note on why. This is a simple running log, not part of `state.json` (no schema field needed — consistent with `growth-notes.md`/`struggle-log.md`/`discipline-log.md`'s pattern of plain markdown logs with no state entanglement).

5. If this pushes the running count of entries in `bonus-watches.md` to a nice milestone (5, 10, 25...), mention it lightly — this is meant to be a fun aside, not another tracked metric with its own badge machinery.

6. This is **entirely optional**, any number of times, never required to progress through the curriculum.

7. `git add -A && git commit -m "Log bonus AI video watch"`.
