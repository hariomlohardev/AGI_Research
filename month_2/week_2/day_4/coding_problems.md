# Day 10 — Cross-Entropy & KL Divergence: coding problems

## Problem 1 — `cross_entropy(p, q)` from scratch

```python
def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Cross-entropy H(P, Q) = -sum(p(x) * log q(x)) in the given log base (bits by default)."""
```

- Implement in `code/cross_entropy.py` using only `math.log` — no `scipy.stats.entropy`, no `sklearn.metrics.log_loss`.
- Both `p` and `q` must be valid pmfs: non-empty, same length, entries `>= 0`, each sums to 1 (within `1e-9`). Raise `ValueError` on any violation (including mismatched lengths or `base <= 0`).
- The `q` zeros need care: where `p(x) > 0` but `q(x) == 0`, the true answer is +infinity (you claimed an occurred event was impossible). Raise `ValueError` in that case — don't silently return `inf`, and don't let `math.log(0)` blow up unhandled. Where `p(x) == 0`, the term contributes `0` regardless of `q(x)` (the `0 * log 0 = 0` convention you handled yesterday).
- Verify by hand first, then against your function: `p = [0.5, 0.5]`, `q = [0.9, 0.1]` → `≈ 1.737` bits. Confirm it is *larger* than yesterday's `H(P) = 1.0`.

## Problem 2 — `kl_divergence(p, q)` from scratch

```python
def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """KL divergence D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))."""
```

- Implement in the same file, with the same validation rules as Problem 1.
- Hand checks with `p = [0.5, 0.5]`, `q = [0.9, 0.1]`: `D_KL(P || Q) ≈ 0.737`, but `D_KL(Q || P) ≈ 0.531` — the two directions differ, so KL is **not** symmetric. Also verify the identity `cross_entropy(p, q) − H(p) == kl_divergence(p, q)` (use your Day 9 `H([0.5, 0.5]) = 1.0`).
- Sanity property your tests will check: `kl_divergence(p, p) == 0` and KL is never negative (Gibbs' inequality — no proof required today, but your function must respect it).

## Problem 3 — log-loss experiments (keep runnable under `__main__`)

```python
def binary_log_loss(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Mean binary cross-entropy over samples: -mean(y*log(p) + (1-y)*log(1-p)), base 2.

    `y_pred` entries must be strictly inside (0, 1) — raise `ValueError` otherwise.
    """
```

- Implement in `code/log_loss.py` (stdlib only). `y_true` entries must be 0/1, same length as `y_pred`.
- Experiments in `__main__`:
  1. **Right vs wrong confidence:** true label 1 with predicted 0.9 → `≈ 0.152`; true label 0 with predicted 0.9 → `≈ 3.322`. Print both — this is *why* log-loss is the classifier's loss: confident errors cost far more than confident correct answers earn.
  2. **Asymmetry demo:** print `D_KL(P||Q)` and `D_KL(Q||P)` for the Problem 2 pair side by side.
  3. **CE ≥ H demo:** print `cross_entropy(p, q)`, `H(p)`, and their difference (= KL) for at least two `(p, q)` pairs.

### Running

```bash
cd month_2/week_2/day_4 && python -m pytest code/tests/ -v
```

Stubs raise `NotImplementedError` until you implement them. Tests import from `cross_entropy` and `log_loss` (via a `sys.path` insert, same as Day 9).

### What not to use

- No `scipy.stats`, no `sklearn.metrics`.
- For the math use `math.log` directly — the one-line formulas are the whole point.

### Done when (from the source plan)

You can explain in one sentence each why cross-entropy is the loss classifiers minimise and why KL divergence being asymmetric means "distance" is the wrong mental model — and your functions produce the hand values above.
