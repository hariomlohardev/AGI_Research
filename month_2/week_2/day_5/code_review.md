# Code Review — Month 2, Day 11 (Mutual Information)

Scope: `month_2/week_2/day_5/code/mutual_information.py` (`mutual_information()`),
`month_2/week_2/day_5/code/feature_selection.py` (`rank_features_by_mi()` + `__main__` demos).
Style/readability/structure/naming only — correctness already covered by tests (21/21).

## What is good (sustain this)

- **Error contracts in docstrings** (`mutual_information.py:15-18`, `feature_selection.py:16-21`):
  every `ValueError` case listed with the exact condition. Keep this on every public function.
- **Precise type hints**: `Sequence[Hashable]` (`mutual_information.py:7`) and
  `Sequence[Sequence[Hashable]] -> list[tuple[int, float]]` (`feature_selection.py:8-9`)
  say "any hashable label, any sequence" without over-constraining to `list`. Keep this pattern.
- **Self-demonstrating `__main__`** (`feature_selection.py:58-74`): seeded (`random.Random(11)`),
  two contrasting experiments (informative-vs-noise, `cov(X, X^2)` vs MI), stable prints.
  This is the model for future demo blocks.
- **Deferred demo-only imports** (`feature_selection.py:50-52`): `pathlib, random, sys` imported
  inside `if __name__ == "__main__":` so library import stays light. Keep doing this.
- **Explicit sort intent** (`feature_selection.py:44`, `reverse=True`) makes "best first" obvious.

## 1. Iterate observed pairs, not the Cartesian product (`mutual_information.py:36-40`)

The nested loop over all `|X| * |Y|` combinations plus an `if joint_count > 0` guard forces
the reader to reconstruct "we only care about observed pairs":

```python
for a in count_x:
    for b in count_y:
        joint_count = pairs_count[(a,b)]
        if joint_count > 0:
```

Idiomatic and immediately readable:

```python
for (a_val, b_val), joint_count in pairs_count.items():
    p_xy = joint_count / n
    p_x = count_x[a_val] / n
    p_y = count_y[b_val] / n
    mi += p_xy * math.log(p_xy / (p_x * p_y), base)
```

This also removes one nesting level and the single-letter `a`/`b` loop variables.

## 2. `Counter` imported from the wrong module (`mutual_information.py:5,29-30,33`)

`from typing import Counter` is a typing-only alias; the runtime container lives in
`collections`. It happens to work here, but every reader will pause on it. Fix:

```python
from collections import Counter
```

and leave `collections.abc` for `Hashable, Sequence` only.

## 3. Index-loops where `zip`/`enumerate` is idiomatic

Three instances of the same habit:

- `mutual_information.py:32`: `pairs = ((x[i] , y[i]) for i in range(len(x)))`
  → `pairs = zip(x, y)`.
- `feature_selection.py:33-36`:
  ```python
  columns = []
  for i in range(len(features[0])):
      column = [row[i] for row in features]
      columns.append(column)
  ```
  → `columns = [list(col) for col in zip(*features)]`
- `feature_selection.py:39-43`: `for i in range(n_columns): column = columns[i]; ...`
  → `for i, column in enumerate(columns):` with `mutual_information(column, y, base=base)`.

The current form also builds the full `columns` list and then re-indexes it; `zip` +
`enumerate` fuses intent ("transpose, then score each column with its index") into two lines.

## 4. Uppercase local accumulator `MI` (`mutual_information.py:35,45,54`)

PEP 8 reserves CapWords/all-caps for classes/constants. `MI = 0` then `MI += ...` then
`return MI` reads as a global. Rename to `mi = 0.0` (float initializer matches the
`-> float` return). Same for the demo variable `MI` on line 54 → `mi`.

## 5. Duplicated import propped up by a path hack (`feature_selection.py:4` vs `54-56`)

Line 4 does `from mutual_information import mutual_information` at top level; lines 54-56
then do `sys.path.insert(...)` + import the same name again inside `__main__`. A reader has
to check both to see they are the same object. Pick one structure: keep the top-level import
for tests, and in the demo block drop the re-import (keep only the `sys.path` shim with a
comment), or convert both files to an explicit package-relative import so the shim disappears.

## 6. Error-message typos and cross-file inconsistency

Exact strings a reader/user sees:

- `mutual_information.py:23`: `"the fuction requre equal length"` — `fuction`, `requre`.
- `mutual_information.py:25`: `f"the base is {base} that the fuction doesn't accept"` —
  `fuction`, and unlike the docstring on line 18 it never states the rule
  (`base <= 0 or base == 1`).
- `mutual_information.py:21` vs `feature_selection.py:27`: both say "the size of either ..."
  but one says `x or y`, the other `features or labels(y)` — same shape, different voice.
- `feature_selection.py:31`: `"every features should have same lenth"` — `lenth`, plural mismatch.

Fix spelling and mirror the docstring phrasing, e.g.
`"x and y have different lengths (positions pair up)"` and
`"base must be > 0 and != 1, got {base}"`.

## 7. Systematic space-before-comma / operator spacing

`black`/`ruff format` would flag all of these; they slow scanning:

- `mutual_information.py:32`: `(x[i] , y[i])` → `(x[i], y[i])`
- `mutual_information.py:44`: `p_y  =` (double space); `:45`: `p_xy/(p_x*p_y) , base`
  → `p_xy / (p_x * p_y), base`
- `feature_selection.py:42`: `mutual_information(column , y , base)` → `(column, y, base=base)`
- `feature_selection.py:43`: `(i , mi)` → `(i, mi)`
- `feature_selection.py:44`: `key= lambda pair : pair[1] ,` → `key=lambda pair: pair[1]`
- `mutual_information.py:52`: `x ,y =` → `x, y =`

One formatter run clears the whole category.

## 8. Minor nits

- `mutual_information.py:53` — dead commented-out probe left in. Delete; the
  `feature_selection.py` demo shows how to keep illustrative cases live.
- `feature_selection.py:24-25` — two blank lines with trailing whitespace between docstring
  and first `if`. Body should start immediately after the docstring.
- `feature_selection.py:39` — name `rank` is a verb; `ranked` or `scored` reads better for the
  sorted list being returned (`rank.sort(...)` then `return rank` currently reads as calling a
  method on an action).
- `feature_selection.py:42` — pass `base` as keyword (`base=base`). Positional is legal but hides
  which parameter is being threaded through.
- `feature_selection.py:59-62` — magic literals `Random(11)`, `n = 200`, `> 0.10` deserve names
  (`SEED`, `N`, `FLIP_RATE`) since the demo's point is reproducibility.
- `mutual_information.py:48-61` — trailing blank lines at EOF; PEP 8 wants exactly one newline.
