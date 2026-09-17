# Code Review — Day 14 (Month 2 Week 3 Day 2, Logistic Regression)

Scope: readability, structure, naming, idiomatic style only. Tests already pass; no correctness re-litigation.

## What's good

- Full `Raises: ValueError` contracts on every public function/method, consistent with Day 13.
- `sigmoid()` documents its no-raise stability contract explicitly instead of leaving it implicit.
- Seeded three-experiment `__main__` demo with deferred `matplotlib` import follows the Day 11/13 good pattern.
- `metrics.py:1-5` module docstring states the intended design ("counting lives in exactly one place") — see finding 1 for following through on it.

## Findings

### 1. `metrics.py` re-loops instead of reusing `confusion_matrix` — contradicts its own module docstring

`metrics.py:54-75,78-103,106-132` each hand-roll a `for y_true_i, y_pred_i in zip(...)` loop to recount `tp/fp/tn/fn`, even though `confusion_matrix()` at `:26-51` already returns all four and the module docstring promises "All scores build on `confusion_matrix`".

```python
# accuracy_score, precision_score, recall_score each do this shape:
tp, fp = 0, 0
for y_true_i, y_pred_i in zip(y_true, y_pred):
    ...
```

Idiomatic fix — one of:

```python
def accuracy_score(y_true, y_pred):
    _error_checks(y_true, y_pred)
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return (tp + tn) / len(y_true)
```

Same for precision (`tp / (tp + fp)`) and recall (`tp / (tp + fn)`), keeping only the zero-denominator `0.0` convention in each wrapper. This is the Day 12 good pattern (`kl_divergence` via `entropy` + `cross_entropy`) regressed.

### 2. Same validation copied across four `logistic_regression.py` call sites

`bce_gradients` (`:80-99`), `fit` (`:165-181`), `predict_proba` (`:205-215`), `predict` (`:231-244`) each repeat empty / length / ragged / feature-count checks with slightly different messages. `predict_proba` and `predict` are nearly identical blocks. Suggest one private helper, e.g. `_validate_X(X, n_features)` for the fitted-feature check plus ragged/empty, and have `predict` delegate to `predict_proba` the way Day 13's thin `predict` delegated to module-level `predict`:

```python
probs = self.predict_proba(X)
return [1 if p >= threshold else 0 for p in probs]
```

That also removes finding 3's manual loop.

### 3. `range(len(...))` indexing where `zip`/`enumerate`/comprehension is idiomatic

- `logistic_regression.py:107-115`: `for i in range(len(X)): x = X[i]` → `for x, y_i in zip(X, y)`.
- `logistic_regression.py:187-189`: `for j in range(len(grad_w))` → `for j, g in enumerate(grad_w)`.
- `logistic_regression.py:247-252`: preallocated `output = [0] * len(probs)` plus index assignment → `[1 if p >= threshold else 0 for p in probs]`.
- `logistic_regression.py:113-115`: `grad_w[j] += (1/n) * error * x[j]` divides by `n` every inner iteration; accumulate raw sums and divide once after the loops — clearer and avoids a loop-invariant recomputation.

### 4. Type-hint coverage drifts between helpers and public API

Typed code surrounds untyped helpers: `is_empty(data:list)` (`:14`, missing element type and `-> bool`), `dot_product(a:list,b:list)` (`:17`, missing types and return), `_error_checks(y_true,y_pred)` in `metrics.py:13` (no hints at all), and `accuracy_score` / `precision_score` / `recall_score` (`metrics.py:54,78,106`, no annotations while `confusion_matrix` just above is fully hinted). Same split as Day 13's untyped `dot_product(X, Y)`. Annotate helpers to match, e.g. `def is_empty(data: Sequence) -> bool:` and `def _error_checks(y_true: Sequence[int], y_pred: Sequence[int]) -> None:`.

### 5. Small idiom slips: `== None`, `list()`, redundant parens, dead constant

- `logistic_regression.py:205,231`: `if self.w == None` → `if self.w is None` (repeats Day 13 `:192`).
- `logistic_regression.py:151`: `self.history = list()` → `self.history: list[float] = []`.
- `logistic_regression.py:117`: `return (grad_w,grad_b)` → `return grad_w, grad_b` (same redundant parens as Day 13 `:114`).
- `logistic_regression.py:12`: `_EPS = 1e-15` is never used — `binary_cross_entropy` uses its own `eps` parameter. Delete it or use it as the default (`eps: float = _EPS`).
- `logistic_regression.py:150`: `self.b = 0` (int) while gradients/loss are float; use `0.0` (Day 14 does fix Day 13's worse variant where `b` was not set in `__init__` at all — keep the `__init__` assignment).

### 6. Error-message wording and spelling

- `logistic_regression.py:53`: `"then length of y_true..."` → `"the length..."`.
- `logistic_regression.py:244`: `"threashoud should be between (0,1)"` → `"threshold must be in (0, 1)"`.
- `metrics.py:98,127`: `denomenator` → `denominator` (second appearance after Day 8 `:116`).
- Voice drift for the same condition across one file: `"the y_pred is empty"` / `"the X is empty"` / `"X can't be empty"` / `"the length of input(X) and Output(y)"` (`:46-47,81,165-172,90`). Pick one voice, e.g. `"X must not be empty"`, and reuse it.
- RST backticks leak into messages: `"not in ``{0, 1}``"` vs `"not in ``(0,1)``"` (`:56,99,181`). Messages render as plain text — write `{0, 1}` without backticks and consistently.

### 7. PEP 8 whitespace (recurring theme — many spots, same shapes as Days 12-13)

Representative, not exhaustive: `data:list` (`:14`), `a:list,b:list` (`:17`), `zip(a,b)` (`:18`), `Sequence[float] , eps:float` (`:34`), `output   =` (`:58`), `y_i , p_i` / `zip(y_true , y_pred)` / `(1-y_i )*` / `(1-p_i +eps)` (`:58`), `return (grad_w,grad_b)` (`:117`), `x ,self.w` (`:192,218,246`), `tp , fp` / `tp ,fn` (`metrics.py:65,90,119`), `for y_true_i , y_pred_i` (both files, many lines). `binary_cross_entropy`'s computation at `:58` is also a ~120-char line — break the generator onto its own lines. A formatter pass would clear all of these at once.

### 8. Docstring-vs-code mismatches (recurring theme)

- `metrics.py:1-5` promises DRY via `confusion_matrix`; findings 1 shows three siblings re-looping.
- `logistic_regression.py:38-39` says predictions are "clipped into `[1e-15, 1 - 1e-15]`" but `:58` computes `log(p_i+eps)` / `log(1-p_i+eps)` (a shift, not a clip). Either clip (`min(max(p, eps), 1-eps)`) or reword the docstring to describe the shift.
- `metrics.py:26-33` promises `ValueError` if "any label is not in `{0, 1}`" but `_error_checks` (`:13-24`) only validates `y_true`, never `y_pred`. Validate both or narrow the docstring.
- Ragged checks at `logistic_regression.py:92,174,211,237` read `if all(len(row) != len(X[0]) for row in X)` — `all` over a comparison that is always `False` for the first row is hard to read as "any row is ragged". `if any(len(row) != len(X[0]) for row in X)` states the intent directly.

### 9. Naming: vague `output`, generic cross-module helper

- `output` as accumulator in three places (`logistic_regression.py:58,218,247`) — name what it is: `loss`/`bce`, `probas`, `preds` (Day 12 flagged the same vague `output` ×3).
- `is_empty` (`:14`) is very generic for a public name imported cross-module by `metrics.py:9` (`from logistic_regression import is_empty`, which only resolves with `code/` on `sys.path` — same fragile bare-sibling-import shape as Day 9's `from entropy import entropy`). Either underscore it (`_is_empty`) with a same-package import, or inline `len(...) == 0` at the few call sites.
