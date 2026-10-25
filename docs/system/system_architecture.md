# Curriculum filesystem and workflow architecture

Updated 25 October 2026.

The existing state schema and root state.json paths are preserved. The file is now a local record, not a public artifact. No Month 3 state entry or day completion is created by this setup. Month 2 coursework remains in its original layout.

Month 3+ uses four sibling roots: docs, notebooks, src/monthN and tests. scripts/curriculum_paths.py resolves week/day_in_week, including eight-day learning weeks. A curriculum.json manifest holds source checksum and layout version independently of state schema.

Notebook inputs are authoritative; scripts/notebooks.py renders deterministic Jupytext py:percent input cells into the package source tree. Tests import exported modules. The export gate checks syntax and conservative import safety without executing learner code; real notebook smoke execution and independent tests supply behavioral evidence. __init__.py and intentional shared modules need no duplicate notebook. Split code and docs do not imply separate copies of each algorithm.

scripts/check_public.py audits either staged blobs or a read-only public worktree inventory. The pre-commit hook prefers the project's actual interpreter and checks the index. Ignore rules protect future additions; an index-only cleanup stops tracking already committed private files, preserving local bytes and existing history.

The dashboard keeps the old Month 2 resolver, uses month_N/docs/days for Month 3+ and reads the new docs hierarchy while excluding code/dependency directories. Its tests use a temporary sanitized fixture rather than a live personal state file. Public CI therefore works without publishing local state. Independent legacy pytest processes prevent identically named day modules from colliding.

Local .claude rules share the layout, notebook and publication contracts across all skills/agents. Generation is lazy and creates stubs, completion verifies evidence, private updates do not create public commits automatically, and recovery distinguishes public Git reverts from private-state backups.

Static dashboard builds intentionally call loadExampleCurriculum and contain no personal documents. The local development/preview endpoint retains loadCurriculum for the learner's live data. The UI labels example builds.

Claude instruction organization follows the [official memory/rules guidance](https://code.claude.com/docs/en/memory): concise root instructions refer to shared topic rules; existing commands retain their individual procedures. Those local teaching settings are deliberately ignored for this personal portfolio.

Completion publishing policy, 25 October 2026: successful /done and verified micro-project completion trigger an explicit-scope public commit and a non-force push to origin/main. The local Claude publication rule supplies the procedure, verifies remote SHA and records Git failures under .local/publishing without rolling back honest study completion. Other private updates do not trigger publication.
