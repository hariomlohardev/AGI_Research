# What belongs on GitHub

Publish completed code, independent tests, clean synchronized notebooks, public learning explanations, reviewed configs and compact scientific results. Keep the local curriculum controller, personal progress and coaching, scratch work, raw datasets and trained weights outside future commits.

| Public | Local |
|---|---|
| README, docs/system, public Month 3 lessons/reports | .claude, CLAUDE.md, .local backups and plans |
| notebook inputs plus matching py:percent exports | notebook output cells, widget state and checkpoints |
| source modules and independent tests | state.json, progress/confidence/struggle/coaching logs |
| configs, dataset/split/checkpoint manifests | raw/processed datasets, checkpoints, raw runs |
| curated summary CSVs and figures with provenance | reviews, flags, confusion notes, contribution coaching |

## Review before staging

    python scripts/notebooks.py sync --month 3
    python scripts/notebooks.py clean --month 3
    python scripts/notebooks.py check --month 3
    python -m pytest tests -q
    python scripts/check_public.py --worktree
    git status --short

Run the relevant curriculum tests too. Select explicit public paths, for example the current day's lesson, notebook, export and tests, then stage only those paths. Run python scripts/check_public.py against the index and inspect git diff --cached --stat and git diff --cached before committing. The pre-commit hook applies the same staged check. Enable it with git config core.hooksPath .githooks. Do not disable it to bypass a stale export or a private file.

A notebook and .py mirror are one deliverable: include both when inputs change. Clear outputs after preserving scientific evidence in a curated result file or local run artifact. Notebook output clearing is not a substitute for inspecting code/Markdown cells, which can also contain private data. Put credentials in ignored environment files, with only a sanitized .env.example shared if needed.

## Publishing when a study item is completed

The learner authorized automatic completed-work commits and pushes on 25 October 2026. After a successful /done, Claude Code commits the completed day's reviewed public docs, clean notebook, matching Python export and independent tests, then pushes main to origin/main. The same applies to a micro-project only after its completion checks pass. Generating stubs, saving private notes or recording an in-progress item does not publish it. A later explicit request to hold publishing overrides this default.

Capture the completed item identity before progress advances. Sync exports, preserve evidence locally, clear outputs, run its required checks, inspect the index and stage explicit public paths. Include individually reviewed shared helpers/configs/results required for that item. Exclude unfinished next-day work and unrelated staged changes. Run the staged publication guard and the active pre-commit hook, then create a concrete completion commit. Do not make an empty commit if the result is already committed.

Work on main by default. If a task branch was used, integrate only its verified work into main with a normal merge or fast-forward. Fetch first and reconcile remote changes without force, history rewriting, discarding local work or bypassing branch protection. Push with git push origin main and verify the remote main SHA matches the intended HEAD before reporting success. This permission covers this curriculum repository; opening external PRs or messaging others has separate authorization requirements.

If the commit succeeds but the push fails, preserve the completed learning record, save the intended commit/error under ignored .local/publishing and report publication as pending. Repair and retry the Git step only. A push failure does not justify a false success claim, an empty duplicate commit or repeating the quiz. When there are no new public files and main is already up to date, report that no new publication was needed.

## Existing tracked private files

Git ignore rules do not stop tracking a file already committed. This update removes the identified private controller/state/review files from Git's index only, leaving their working copies in place. These appear as staged deletions and take effect in the next reviewed commit. They can be inspected with git diff --cached --name-status. Do not delete local copies to make that list disappear.

This does not remove earlier commits or copies already pushed. Existing history remains unchanged. Removing private files from the current tree does not rewrite history or rotate credentials. If past history needs a separate privacy cleanup, plan that explicitly with the repository owner.

## Enforcement and limits

The staged checker rejects named private paths, bulk artifacts, notebooks with outputs, missing/stale Python mirrors, common credential patterns and unexpectedly large files. It reads staged blobs, so cleaning an unstaged copy cannot hide a dirty staged notebook. It is not a comprehensive secret scanner. Review the diff and use [GitHub secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning) and available push protection for credential detection.

Private state is validated locally by state-writing commands and backed up under .local/state-backups; public CI tests the validator using a sanitized fixture. Reverting a public commit cannot restore ignored study state. A fresh public clone contains the portfolio/tooling; a personal learning dashboard needs its own locally initialized state.

Public dashboard builds now embed only the sanitized test fixture and an example-data label. Live development/preview reads personal state through the local read-only server. Ignoring dist/ alone would not make a deployed snapshot private; do not publish a live local-state export.
