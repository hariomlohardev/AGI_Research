---
name: safe-revert
description: Undo the most recent commit made by this tooling (a /done, /i-am-in, /skip-to, etc.) via a non-destructive git revert, with a preview and explicit confirmation before acting. Use when the user invokes /safe-revert because something got recorded wrong or they want to back out the last change.
---

# /safe-revert

This is deliberately conservative: it only ever reverts **one commit at a
time**, via `git revert` (which adds a new commit undoing the last one and
keeps full history), never `git reset --hard` or anything that rewrites or
discards history. Running it again reverts one step further back if needed.

## Steps

1. If there's no `.git` history yet, or nothing to revert, say so and stop.

2. **Preview before acting.** Show the user the most recent commit: its
   message (`git log -1 --oneline`) and a summary of what it touched
   (`git show --stat HEAD`). Explain in plain language what reverting it
   will undo — e.g. "this will undo marking day 4 done and roll `state.json`
   back to day 4 being current again."

3. **Ask for explicit confirmation** before doing anything. Don't proceed on
   an ambiguous or implied "yes" — get a clear go-ahead. If the user declines
   or seems unsure, stop without changing anything.

4. Once confirmed: `git revert --no-edit HEAD`. This creates a new commit
   that undoes the last one; nothing is force-rewritten or destroyed, so the
   original commit is still there in history if they change their mind
   later (they'd need to ask for that explicitly — this command only ever
   moves forward-via-revert, not through raw `git reset`).

5. Run `python3 scripts/validate_state.py` on the result to confirm the
   reverted state is still schema-valid (it should be, since it's returning
   to a previously-committed-and-validated state, but confirm rather than
   assume).

6. Suggest running `/run-checks` afterward to confirm everything (tests
   included) is in a good state post-revert.

7. If the user wants to go back further than one commit, tell them to run
   `/safe-revert` again — each run only ever steps back one commit, on
   purpose, so an accidental multi-step revert can't happen from one command.
