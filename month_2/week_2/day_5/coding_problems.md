# Day 11 — Mutual Information: coding problems

## Problem 1 — `mutual_information(x, y)` from scratch

```python
def mutual_information(x: Sequence[Hashable], y: Sequence[Hashable], base: float = 2.0) -> float:
    """..."""
```

- Implement in `code/mutual_information.py` using only `math.log` and counting — no `sklearn.metrics.mutual_info_score`, no `scipy.stats`.
- Estimate empirically: count joint occurrences `P(a, b)` and marginals `P(a)`, `P(b)` directly from the `n` paired samples, then `I = Σ P(a,b) · log(P(a,b) / (P(a)·P(b)))`, skipping pairs that never occur. Labels may be any hashable type (ints, strings, pre-binned values) — never assume numbers.
- Contract (all `ValueError`, all listed in your docstring's `Raises` section):
  - `x` or `y` empty → `ValueError`.
  - different lengths → `ValueError` (positions pair up, so ragged pairs are meaningless).
  - `base <= 0` or `base == 1` → `ValueError` (a base-1 log is division by zero).
- Hand checks (base 2, verify by hand first, then against your function):
  - perfectly correlated binary `x = y = [0, 0, 1, 1]` → exactly `1.0` bit (knowing `y` removes the full 1 bit of uncertainty about `x`).
  - independent binary `x = [0, 0, 1, 1]`, `y = [0, 1, 0, 1]` → exactly `0.0`.
- Properties your tests will check: symmetry (`MI(x, y) == MI(y, x)`), never negative, and bounded by the marginal uncertainties (`MI(x, y) ≤ min(H(x), H(y))` — compute the marginal entropies inline in the test, no cross-day imports).

## Problem 2 — `rank_features_by_mi(features, y)` (the selection tool)

```python
def rank_features_by_mi(
    features: Sequence[Sequence[Hashable]], y: Sequence[Hashable], base: float = 2.0
) -> list[tuple[int, float]]:
    """..."""
```

- Implement in `code/feature_selection.py`, reusing your Problem 1 function (import it — same directory).
- `features` is rows × columns: `features[i][j]` is sample `i`'s value for feature `j`. Return `[(feature_index, mi_score), ...]` sorted best-first (descending score).
- Contract (all `ValueError`, all in the docstring):
  - `features` or `y` empty → `ValueError`.
  - row count `!= len(y)` → `ValueError`.
  - ragged rows (not every row has the same number of features) → `ValueError`.
  - `base <= 0` or `base == 1` → `ValueError` (pass it through to `mutual_information` — one place owns the rule).
- Test it on synthetic data you construct: one feature that copies the label with ~10% flips (informative), one that's pure noise. Your ranking must put the informative feature first by a clear margin — if it doesn't, the bug is in your MI, not in the test.

## Problem 3 — `__main__` experiments (keep runnable)

In `feature_selection.py`'s `__main__` block (import `mutual_information` via a `pathlib`-based `sys.path` insert so it runs from anywhere, not just `code/` as cwd):

1. **Informative vs noise:** build the synthetic pair from Problem 2 (seeded `random` so it prints the same every run), print the full ranking — the informative feature's score should sit near `1 − H(0.1) ≈ 0.531` bits and the noise feature's near 0.
2. **The `Y = X²` demo:** `xs = [-4..4]`, `ys = [v*v for v in xs]`. Print covariance (small inline computation — Week 1's lives in another day's folder, don't import across days) next to `mutual_information(xs, ys)`: expect covariance exactly `0.0` against MI `≈ 2.28` bits. That pair of numbers is the whole day in miniature.

### Running

```bash
cd month_2/week_2/day_5 && python -m pytest code/tests/ -v
```

Stubs raise `NotImplementedError` until you implement them. Tests import from `mutual_information` and `feature_selection` (via a `sys.path` insert, same as Day 10).

### What not to use

- No `sklearn.metrics`, no `scipy.stats`.
- For the math use `math.log` directly plus plain counting (`dict.get` tallies, not `collections.Counter` one-liners you couldn't explain — the counting *is* the estimation step).

### Done when (from the source plan)

Your feature-selection tool correctly ranks a synthetic "informative" feature above a synthetic "noise" feature, and you can explain why MI catches nonlinear relationships that covariance misses.
