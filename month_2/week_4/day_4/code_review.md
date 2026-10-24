# Code review — Day 22 (Compare Against Real `sklearn`)

Style-only review of `month_2/week_4/day_4/code/` (`compare.py` + tests).
Correctness is established by the 5/5 test run; this is about readability,
structure, naming, and idiomatic style.

## Strengths

- Clear contracts: `score_classifier` states the exact return keys and the
  call-once-per-model usage; `load_same_split` says "call it, do not
  reimplement."
- Good reuse structure: `load_same_split` is a thin delegate to the Day
  20/21 helpers instead of copied logic.
- Right guard-clause ordering: validate model/splits/lengths before
  `fit`/`predict`, finite-checks before arithmetic.

## Findings (most important first)

1. `compare.py:10` + `compare.py:98` — duplicate `import math`, second one
   mid-file. Delete line 98, keep the top import.
2. `compare.py:87` — `fit = model.fit(...)` stores an unused result. Call
   bare `model.fit(X_train, y_train)`.
3. `compare.py:13` — `Mapping` imported but never used. Narrow to
   `from collections.abc import Sequence`.
4. `compare.py:107-111`, `124-131` — paired `isnan`/`isinf` checks instead
   of the idiomatic finite check. Use
   `if not math.isfinite(mine) or not math.isfinite(theirs):` (and same
   shape for gap/tolerance).
5. `compare.py:87,92-93` — space before comma (`X_train , y_train`,
   `y_test , y_pred`). Format so args read `f(X_train, y_train)`.
6. `compare.py:136-139` — `else: return` after `return`. Dedent to
   `return "investigate"`.
7. `compare.py:79` — noisy error prefix (`"Execution failed: ..."`).
   Say `"One or more data splits are empty."`
8. `compare.py:77,81` — leftover numbered comments (`# 2.`, `# 3.`, no
   `# 1.`). Drop the numbers, keep plain verbs.
9. `compare.py:74` — single quotes `'.fit'` while the file uses double
   quotes. Use double quotes consistently.
10. `compare.py:100` — informal param names `mine`/`theirs`. Low priority;
    `accuracy_a, accuracy_b` reads better in a public helper.

## Recurring vs single-day

No new themes promoted. Findings 1–3 extend the existing leftover-scaffold
theme; finding 5 extends the PEP 8 whitespace theme; finding 7 extends the
error-message wording theme. The rest are single-day observations.
