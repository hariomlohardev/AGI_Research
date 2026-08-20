---
name: run-checks
description: Local "run everything CI would run" entrypoint — validates state.json against state.schema.json, runs the tooling's own tests/ suite, and runs pytest (or the configured test_command) across every generated day's and micro-project's code/tests/. Use when the user invokes /run-checks, or before anything that should be shipped/committed as a batch.
---

# /run-checks

Read-only with respect to `state.json` (it only validates, never edits it).
This is the same thing `.github/workflows/ci.yml` runs automatically on
push — this command lets the user run it locally on demand, and it's also
what `/safe-revert` should suggest running after a revert to confirm the
repo landed in a good state.

## Steps

1. **Schema check**: run `python3 scripts/validate_state.py`. Report pass/fail
   clearly. If `state.json` doesn't exist yet, note that plainly (not an
   error — just means no month has been started).

2. **Tooling test suite**: run `pytest tests/ -v` (this exercises
   `scripts/validate_state.py` itself against the fixtures in
   `tests/fixtures/`). Report pass/fail per test, not just a total.

3. **Generated content tests**: for every `month_<N>/week_<W>/day_<D>/code/tests/`
   and `.../micro_projects/<slug>/code/tests/` that exists on disk, run the
   appropriate test command:
   - `months.<N>.language == "python"` (or unset/default): `pytest <path> -v`.
   - `"javascript"`/`"typescript"`: `npm test` from that day's `code/`
     directory, if a `package.json` is present there.
   - `"other"`: `months.<N>.test_command`.
   Report each day/micro-project's result separately — don't just give one
   aggregate pass/fail, since a single failing day shouldn't obscure that
   everything else is fine.

4. **Summary**: a short pass/fail table — schema check, tooling suite, and
   one line per day/micro-project tested. If everything passed, say so
   plainly. If something failed, name exactly what and where, the same way
   `/done` does when a day's own tests fail — don't just say "some tests
   failed."

5. No git commit — this command only reads and reports, never changes state
   or files.
