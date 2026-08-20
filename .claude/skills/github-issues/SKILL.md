---
name: github-issues
description: Finds 3 real, currently-open GitHub issues (50+ star repos, matching the month's language, preferring the week's topic if a genuine match exists) via the github-issue-researcher subagent, lets the user pick one, and tracks it against the current day. Once picked, that day's /done requires a real submitted PR to be reviewed before it can close. Use when the user invokes /github-issues to start, check status of, or submit a PR for this day's open-source contribution.
---

# /github-issues

**Entirely opt-in.** Running this is what activates the GitHub contribution
requirement for the current day — a day where this is never run behaves
exactly as it always has (`github_contribution` stays `null`, no extra gate
on `/done`). See `STATE_SCHEMA.md`'s "Open-source contribution workflow"
and `CLAUDE.md`'s "Discipline & accountability" section.

## Steps

1. If `state.json` doesn't exist, tell the user to run `/month` first and stop. Read `current_month` (`N`)/`current_day` (`D`), `months.<N>.language`, `months.<N>.skill_level`, and this day's `topic` and week's `done_when` (for topic-matching).

2. **Branch on this day's current `github_contribution` state:**

   **If `null` (first run for this day):**
   - Delegate to the `github-issue-researcher` subagent with `months.<N>.language`, `months.<N>.skill_level`, and (as a soft preference, not a hard filter) this day's `topic`/the week's `done_when`. It returns exactly 3 real, verified candidates.
   - Present all 3 clearly (repo, stars, issue title/link, what solving it involves, why it fits, topic-relevance note).
   - Ask the user which one they want, if any. If they pick one, set this day's `github_contribution` to `{status: "in_progress", repo, issue_url, issue_title, pr_url: null}`. If they don't want any of the 3, that's fine — leave it `null` and stop; this is optional.

   **If `status: "in_progress"` (issue picked, no PR yet):**
   - Remind the user which issue they picked and its link. Ask if they have a PR to submit yet.
   - If they give a real PR URL, set `pr_url` and `status: "pr_submitted"`, then proceed to step 3.
   - If not yet, just remind them this needs to happen before `/done` can close today, and stop — no need to re-run the researcher or re-pick.

   **If `status: "pr_submitted"` (PR given, not yet reviewed):**
   - Proceed to step 3 directly (the review may not have happened yet, e.g. if `/done` hasn't been run since the PR was submitted).

   **If `status: "reviewed"`:**
   - Tell the user this day's contribution is already reviewed and done; show them where `github_contribution/pr_review.md` is if they want to reread it. No further action.

3. **Trigger the review** (if `pr_url` is set and `status` isn't yet `"reviewed"`): delegate to the `github-pr-reviewer` subagent with the PR URL and the issue it's meant to solve. Write its output to `day_<D>/github_contribution/pr_review.md`. Present the review conversationally too. Set `status: "reviewed"`.

4. **Badge check**: per `STATE_SCHEMA.md`'s badges table, if `status` just became `"reviewed"` for the first time anywhere in the project and `first-oss-pr` isn't already in `badges_unlocked`, append it and announce it.

5. Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the write before committing.

6. `git add -A && git commit -m "GitHub contribution update for day <D>"`.
