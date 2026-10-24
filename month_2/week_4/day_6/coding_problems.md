# Day 24 — Push to GitHub + Rest: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only (`ast`,
`pathlib`, `sys.stdlib_module_names`). No new math, no new models — today
you verify shippability, then ship.

> The plan lists no practice questions today ("genuinely rest"). Problem 2
> is a manual checklist, not a quiz — `/done` verifies it by looking at
> the repo state with you, not by interrogation.

---

## Problem 1 — `audit_requirements.py` (tested)

Write `code/audit_requirements.py` with two helpers:

```python
def third_party_imports(root: str | Path) -> set[str]:
    """Walk `root` for `.py` files and return top-level third-party
    import names (stdlib and relative imports excluded)."""

def pinned_requirements(path: str | Path) -> dict[str, str]:
    """Parse a requirements file into `{package: version}`; every entry
    must be `==`-pinned."""
```

Full `Raises` contracts in the docstrings (the Day-10 process fix):

- `third_party_imports`: raise `ValueError` if `root` is not a directory.
  Relative imports (`from .x import ...`, `from . import ...`) never count.
  `__future__` imports never count. Submodule imports collapse to the
  top-level name (`from sklearn.cluster import KMeans` → `sklearn`).
- `pinned_requirements`: raise `ValueError` if the file does not exist, or
  on any non-blank, non-comment line that is not exactly
  `name[extras]==version` (an optional `; markers` tail is allowed).
  Return `{}` for an empty (or all-comment) file — an empty requirements
  file is valid, it means "no third-party deps".

Why this exists: the push-day claim "requirements are accurate" should be
checkable. The test suite pins two facts about this repo: the package tree
(`month_2/minisklearn/src/`) imports nothing third-party (it is
stdlib-only by construction), and every line of the root
`requirements.txt` is pinned.

### Running tests

```bash
cd month_2/week_4/day_6 && python -m pytest code/tests/ -v
```

---

## Problem 2 — The push checklist (manual, no stub)

Do these in order; nothing to implement, everything to verify:

1. **Cleanup pass.** `month_2/minisklearn/` reads clean to a stranger:
   formatting consistent, no dead/commented-out code, no leftover
   scaffolding (Day-19 move-map comments should all be landed and gone).
2. **Full green.** From the repo root: `pytest` over everything plus
   `python scripts/validate_state.py`. Do not push on a red suite.
3. **Commit + push.** Clear message for the day's work; push to GitHub.
   History should read one-commit-per-day, not one giant "final commit".
4. **Browser check.** Repo public; README renders cleanly on the GitHub
   page itself (headers, code fences, the Day-22 comparison table) — check
   it in the browser, not just your editor.
5. **Link the blog post** from the README (and commit + push that link).
6. **Rest.**

### Done when (from the source plan)

The repo is public on GitHub, the README renders cleanly on the GitHub
page itself, and you've taken today mostly off.
