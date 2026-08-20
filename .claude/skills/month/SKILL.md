---
name: month
description: Parse a newly pasted/attached month's learning plan (Week -> Day, with Rest Days and weekly "Done when" criteria), run one-time project setup on first use (including an environment check), and record every day of the month as pending/rest in state.json and progress.md. Use when the user pastes or attaches a new month's curriculum text and invokes /month.
---

# /month

**Input:** the plan text pasted or attached in the same message as `$ARGUMENTS` (or the attached file's content).

## Steps

1. **First-run setup** (only if no `.git` directory exists yet at the project root):
   - **Environment check first.** Verify `git --version`, `python3 --version`, and that `pytest` is importable (`python3 -c "import pytest"`). If any is missing, tell the user plainly what's missing and how to install it (e.g. `pip install -r requirements.txt --break-system-packages` for pytest) *before* proceeding — don't let a missing tool surface confusingly three steps later inside `/done`.
   - `git init`
   - Create `.gitignore` (venv, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store`, `*.egg-info/`, `node_modules/`) if missing.
   - Create `requirements.txt` starting with `pytest` if missing.
   - Note in `CLAUDE.md` (create if missing, per the project's `CLAUDE.md` template) that dependencies are installed via `pip install -r requirements.txt`.
   - Run `git config core.hooksPath .githooks` so the schema-enforcement pre-commit hook (`.githooks/pre-commit`, which runs `scripts/validate_state.py`) is actually active for this repo — `.git/hooks/` itself isn't version-controlled, so this has to be set explicitly on first setup.
   - Initialize `state.json`'s top level: `schema_version: 2`, `badges_unlocked: []` (see `STATE_SCHEMA.md`'s "Motivation: badges" section).
   - `git add -A && git commit -m "Initial project setup"`.

2. **Ask about this month, once** (skip anything the user already stated up front):
   - **Language**: what language is this month's coding work in? Default `"python"` if unanswered.
   - **Skill level**: beginner / intermediate / advanced, for scaling stub and problem difficulty. Default `"intermediate"` if unanswered.
   - If language is anything other than `python`/`javascript`/`typescript` (i.e. `"other"`), also ask for the exact test command to run (`months.<N>.test_command`) — otherwise leave it `null`.

3. **Parse the plan text.**
   - Identify Week boundaries and each week's own **"Done when"** success criteria (separate from daily work — do not confuse this with any single day's practice questions).
   - Identify each Day within a week, in order. Detect any day explicitly labeled **"Rest Day"** in the text and mark it as such — do not assign it real learning content.
   - Determine the month number `N`: if `state.json` already has a `current_month`, this new plan is for `N = current_month + 1` unless the user's message says otherwise; if `state.json` doesn't exist yet, this is month 1.

4. **Write state**, following `STATE_SCHEMA.md` exactly:
   - Set `months.<N>.started` to today's date (ISO), set once and never edited again. Set `language`, `skill_level`, `test_command` from step 2.
   - For each week, create `months.<N>.weeks.<W>` with `done_when` set from the parsed criteria and `review_status: "pending"`.
   - For each day, create `months.<N>.weeks.<W>.days.<D>` with `day_in_week`, `global_day` (continuous counter within the month), `label`, and `status`: `"rest"` for Rest Days, otherwise `"pending"` — except day 1 of the month, which is set to `"current"`. Also initialize: `has_flags: false`, `mode: "guided"` (the default — `/i-am-in` may switch a specific day to `"challenge"` if the user asks for it when generating that day), `topic: null`, `tags: []`, `difficulty: null`, `time_spent_minutes: null`, `expected_minutes: null` (only set to a real number if the source plan itself states one for that day — never guessed), `confidence: null`, `confusion_notes: []`, `review_history: {"last_reviewed": null, "times_reviewed": 0}`, `srs: {"ease_factor": 2.5, "interval_days": 1, "next_review_due": null}`, `micro_projects: []`.
   - Set top-level `current_month = N`, `current_day = 1` (the first non-rest day, if day 1 happens to be a rest day, advance to the first real day and mark that `current` instead).
   - Do **not** create `week_<W>/day_<D>/` folders or generate any day content — that is `/i-am-in`'s job, done lazily.

5. **Create month scaffolding on disk:**
   - `month_<N>/final_project/structure.md` — design a real-world project skeleton tying this month's daily topics together, framed as something genuinely worth building. If the source plan already defines a final project, use that; otherwise design one yourself from the month's topics.
   - `month_<N>/final_project/logs.md` — start empty with just a header; it gets appended to as days complete.
   - If `language` is `"javascript"` or `"typescript"` and no `package.json` exists yet at the project root, create a minimal one (with `jest` or the user's stated preference as the test framework) — mirrors what `requirements.txt` does for Python.

6. **Update `progress.md`** (human-readable, spans all months) with a new section for month `N` listing every week/day and its status (pending/rest), and noting the month's `language`/`skill_level`.

7. **Update `project-history.md`** (root; create it with a short header if this is the first month). If a previous month (`N-1`) exists, read its `month_<N-1>/final_project/structure.md` and add a short new entry describing how month `N`'s final project builds on or connects to it (in your own words — do not just copy the file). If this is month 1, just record the starting point.

8. **Do not auto-run `/i-am-in`.** The user's next `/i-am-in` will generate day 1 naturally, or they can run `/skip-to` first to jump ahead.

9. **Validate before committing.** Run `python3 scripts/validate_state.py`. If it reports errors, stop and fix the `state.json` write — do not commit invalid state. (The pre-commit hook enforces this too, but checking proactively surfaces a clearer, closer-to-the-cause error.)

10. `git add -A && git commit -m "Add month <N> plan"`.
