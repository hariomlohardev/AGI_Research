# Code Review — Month 2 Day 13 (Week 3 Day 1: Linear Regression GD)

Scope: `month_2/week_3/day_1/code/linear_regression.py` + `code/normal_equation.py` (`code/tests/` used for context only). Style only — not correctness.

## 1. What was done well

- Module and method docstrings in `linear_regression.py` state shape contracts (`X` is rows × columns), formulas, and `Raises` cases. `normal_equation.py::solve_normal_equation` has an especially clear docstring plus short section comments (`# Design matrix`, `# Gaussian elimination`, `# Back substitution`).
- `from __future__ import annotations` + `Sequence[...]` hints in signatures is the right modern-stdlib idiom.
- `normal_equation.py` validation reads cleanly: compute `n_features` once, then one explicit `for row in X` ragged check, instead of repeating a generator.
- Keeping the `matplotlib` import deferred inside `__main__`'s `try` is the correct call for an optional demo dependency.
- `LinearRegressionGD` keeps a clean public surface (`fit` returns `history`, `predict` delegates to module-level `predict`).

## 2. Findings

1. `linear_regression.py:12` — wrong return annotation on helper. `def is_empty(data:list) -> float:` returns a `bool`. Fix to `def is_empty(data: Sequence) -> bool:` (and add space after `:`). Docstring-vs-code mismatch of exactly the kind linters/type checkers catch.
2. `linear_regression.py:15` — untyped public helper amid typed code. `def dot_product(X ,Y):` has no hints, wrong spacing before commas, and capital arg names that collide visually with matrix `X` used everywhere else. Suggest `def _dot(xs: Sequence[float], ys: Sequence[float]) -> float:` (private name, lowercase, typed). Same call sites at lines 41, 104 then read unambiguously.
3. `linear_regression.py:30,90,160,198` (+ `normal_equation.py:26-28`) — duplicated ragged-row validation. The `not all(len(x) == len(X[0]) for x in X)` idiom is copy-pasted four times in one file, and `normal_equation.py` hand-rolls a fifth variant. Extract one helper, e.g. `_validate_matrix(X, w_len: int | None)` that checks empty/ragged/width, and call it from `predict`, `mse_gradients`, `fit`, and `LinearRegressionGD.predict`.
4. `linear_regression.py:102-108` — non-idiomatic `range(len(...))` loop. `for i in range(len(X)): x = X[i]; ... y[i]` should be `for x, yi in zip(X, y):`. Inner `for j in range(len(w)): grad_w[j] += error * x[j]` can stay indexed (accumulating), but `for j, x_j in enumerate(x): grad_w[j] += error * x_j` reads better.
5. `normal_equation.py:41-45` — same `range(len(...))` idiom. `rhs[i] = sum(A[k][i] * targets[k] for k in range(len(A)))` is clearer as `sum(row[i] * t for row, t in zip(A, targets))`.
6. `linear_regression.py:192` — `== None` instead of `is None`. `if self.w == None :` should be `if self.w is None:` (also drop the space before `:`). This is the canonical PEP 8 identity-check rule.
7. `linear_regression.py:143,168` — attribute defined outside `__init__`. `self.w = None` is set in `__init__` but `self.b` only appears in `fit` (line 168). Initialise both together, e.g. `self.w: list[float] | None = None` and `self.b: float = 0.0`, so readers (and type checkers) see full instance state up front.
8. `linear_regression.py:27-37,55-62,84-97,154-164,195-202` vs `normal_equation.py:15-28` — inconsistent error-message voice across the two files that ship together. Examples: `"X can't be empty"` vs `"X is empty"` vs `"the X can't be empty"` vs `"data is empty"`; `"w is empty"` vs `"w can't be empty"`; `"len(X) must equal len(y)"` vs `"the length of X and y are different"`. Pick one style (no leading `the`, no contractions) and reuse it; it also makes tests/messages greppable.
9. `linear_regression.py:37,97` — markup leaked into runtime strings. `"a row's feature count differs from ``len(w)``"` carries RST double-backticks into a `ValueError`. Message should be plain, e.g. `"row width (...) != len(w) (...)"`.
10. `linear_regression.py:62,88` — subject-verb agreement in messages. `"the lengths of y_true and y_pred is different"` → `"lengths ... are different"`; `"the length of features(X) and label(y) are different"` → `"len(X) != len(y)"`. Same for `"length of rows are not equal in X"` (lines 31, 91, 161, 199) vs `normal_equation.py:28` `"rows are ragged"` — standardise on the latter, it is shorter and already used.
11. `linear_regression.py:39-42` — case-only name distinction. Loop var `x` (a row) vs matrix `X` differs only by case, and shadows nothing but confuses. Use `row` consistently, matching `normal_equation.py:26` which already does.
12. `linear_regression.py:174-177` — `=` where `+=` is idiomatic. `grad_w[j] = grad_w[j] + (2 * self.l2 * self.w[j])` should be `grad_w[j] += 2 * self.l2 * self.w[j]`. The stale commented-out update on line 176 directly above the live update makes this block easy to misread — delete the comment.
13. `linear_regression.py:180-181` — dead commented code. Line 180 `# history.append(...)` duplicates line 181 exactly. Delete line 180; commented-out code is not version history.
14. `linear_regression.py:114` — redundant parens in `return (grad_w, grad_b)`. Write `return grad_w, grad_b`.
15. `linear_regression.py:217` — redundant start in `range(0, len(history), 50)`. The `0` is the default, so drop it: `range(len(history), 50)` — wait, no: step form needs the stop; correct simplification is `range(0, len(history), 50)` → keep as-is is fine, but the `0` adds nothing: `for i in range(0, len(history), 50)` → `for i in range(0, len(history), 50)` without the `0` reads `range(len(history), 50)` which is wrong — the honest fix is just to leave the explicit `0` or step with `range(0, len(history), 50)`. (Reviewer's original note was garbled; the `0` here is harmless and explicit — lowest priority.)

## 3. Minor nits

- Spacing/PEP 8: `linear_regression.py:12` (`data:list`), `:15` (`X ,Y`), `:16` (`x,y`), `:64` (`y_hat , y`, `y_pred ,y_true`), `:112` (`[ (2/n)`), `:140` (`self.lr  = lr`), `:204` (`X ,self.w ,self.b`). One `black`/`ruff format` pass fixes all of these.
- Blank lines: `linear_regression.py:10-11` has 3 blank lines after imports; lines 46-48 have 3-4 between top-level defs. PEP 8 wants exactly 2 between top-level defs, 1 between methods.
- `linear_regression.py:126` — British `initialised` in docstring while the rest of the file/messages use American spellings. Trivial, but pick one.
- `normal_equation.py:32` — `[1.0, *map(float, row)]` works but mixes `map` into a display expression; `[1.0, *(float(v) for v in row)]` is plainer Python.
- `normal_equation.py:56` — `if pivot == 0.0:` exact float equality is unidiomatic for a singularity guard; `if abs(pivot) < 1e-12:` (or similar tolerance) signals intent. Not a correctness claim, just the conventional way to write the check.
- Duplicated demo data: the same 5-row `X`/`y` literal appears in `linear_regression.py:212-213` and `normal_equation.py:90-91`. Fine for standalone demos, but if the demos grow, hoist to one shared example.
