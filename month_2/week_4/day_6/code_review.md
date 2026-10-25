# Code review — Day 24 (Dependency audit)

Style-only review of `month_2/week_4/day_6/code/audit_requirements.py`
(`third_party_imports` + `pinned_requirements`). Correctness is established
by the passing tests; this is about readability, structure, naming, and
idiomatic style.

## Strengths

- Excellent contracts: both docstrings state the exact input shape and the
  `Raises: ValueError` cases, with a worked example (`from sklearn.cluster
  import KMeans` contributes `sklearn`).
- Modern hints throughout: `from __future__ import annotations` plus
  `str | Path` params and `set[str]` / `dict[str, str]` returns.
- Deterministic walk: `sorted(root_path.rglob("*.py"))` at `:31`, so results
  do not depend on filesystem order.
- Good error context: `filename=str(path)` for `ast.parse` at `:32`, and
  `{path}:{lineno}: ... {raw!r}` in both `pinned_requirements` failures at
  `:72,75`.
- Right idioms: `line.partition("==")` at `:68`, extras stripped via
  `split("[")[0]` at `:73`, markers stripped via `split(";")[0]` at `:70`,
  `encoding="utf-8"` on both reads.

## Findings (most important first)

1. `audit_requirements.py:29` — `stdlib = set(sys.stdlib_module_names)`
   rebuilt on every call. Hoist to a module constant, e.g.
   `_STDLIB = frozenset(sys.stdlib_module_names)`, and read from it.
2. `audit_requirements.py:37-38` vs `:43-44` — the same
   `not in stdlib` membership test spelled twice, once per `Import` /
   `ImportFrom` branch. Extract a one-line `_is_third_party(top_level)`
   helper so the two branches cannot drift.
3. `audit_requirements.py:64` — the `for lineno, raw in
   enumerate(req_path.read_text(encoding="utf-8").splitlines(), 1):` header
   is ~90 chars. Bind `text = req_path.read_text(encoding="utf-8")` first,
   then loop over `text.splitlines()`.
4. `audit_requirements.py:71` — five conditions packed into one `if`
   (`not sep or not name or not version or " " in version or "=" in
   version`). Split into two guards — missing-pin vs malformed-version —
   with a short comment saying why a space or a second `=` rejects the line.
5. `audit_requirements.py:36,42,73` — `top` / `module` / `base` are three
   names for two concepts. Use one shared `top_level` in both import
   branches and `pkg_name` for the extras-stripped requirement name.
6. `audit_requirements.py:73-75` — after
   `base = name.split("[")[0].strip()`, `base` can never end with `"["`,
   so `base.endswith("[")` never fires. Simplify to `if not base:`.
7. What's working — keep it: the relative-import `continue` with its
   one-line reason at `:41`, the blank/comment skip at `:66-67`, and the
   lowercase `found` / `pinned` accumulators (no UPPERCASE-local shape).

## Recurring vs single-day

No new themes promoted. Findings 1–6 are single-day observations: none
matches an existing Theme's shape (finding 2 is two branches in one
function, not the cross-sibling duplicated-validation Theme). Explicitly
not extended: PEP 8 whitespace (no space-before-comma/colon slips),
error-message wording (both messages typo-free and consistent),
docstring-vs-code (both docstrings match the code), `range(len(...))`,
`== None`, redundant-paren `return`, vague `output`/`out`, untyped
helpers, leftover scaffold, bare sibling import, `list()`, UPPERCASE
accumulator, sum-reduction list, and attribute-first-assigned-in-`fit`.
