---
name: github-help
description: Freeform help for whatever's blocking the user during an open-source contribution — understanding the issue, navigating the unfamiliar codebase, git/GitHub workflow (branches, rebasing, PR etiquette), debugging, or general "I'm stuck" questions. Uses the active github_contribution's context if the current day has one, but works for general git/GitHub questions too. Use when the user invokes /github-help.
---

# /github-help

**Input:** whatever the user asks in `$ARGUMENTS` — a specific error, "how do I rebase," "I don't understand this part of the codebase," etc. This is conversational help, not a structured multi-step workflow like the other skills.

## Steps

1. If the current day has an active `github_contribution` (not `null`), use its `repo`/`issue_url`/`pr_url` as context automatically — fetch the issue/repo/PR as needed to give a grounded answer, don't make the user re-paste links you already have.

2. If there's no active `github_contribution` for today (e.g. they're asking a general git/GitHub question, or helping with a contribution from an earlier day), that's fine too — just help with whatever they're asking, using `WebSearch`/`WebFetch` for anything you need to look up (docs, the specific repo's conventions, an error message).

3. Give direct, practical help — explain the concept or fix, don't just point at docs. If it's a codebase-navigation question and they've mentioned a local clone path, `Read`/`Grep`/`Glob` it for real context rather than guessing at the repo's structure.

4. This is **read-only with respect to `state.json`** — it never advances state, marks anything reviewed, or touches the GitHub contribution gate. It's pure assistance. If the conversation naturally leads to "okay I've got a PR now," point them at `/github-issues` to actually record it, rather than recording it yourself here.

5. No git commit — this command doesn't write project files by default. (If the user explicitly asks you to save something from the conversation, e.g. notes on a tricky part of the codebase, use your judgment on where that belongs — likely appended to `day_<D>/github_contribution/notes.md` — and commit only in that case.)
