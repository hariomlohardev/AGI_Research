# Day 15 — Support Vector Machines & the Kernel Trick: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only in the tested
functions (plus `matplotlib`/`numpy` for the optional `__main__` boundary plot —
never inside the tested functions). Plain Python lists (`X` is rows × columns),
not numpy — same discipline as Days 13–14, now with a margin objective.

> Labels are `{-1, +1}` today, not `{0, 1}`. The hinge formula
> `max(0, 1 − y·(w·x+b))` needs symmetric labels to work — converting your
> Day-14 datasets means mapping `0 → −1`. The `ValueError` contracts below
> enforce this; a `0` label raises instead of silently training on garbage.
>
> Data note: everything below uses small synthetic datasets generated inline
> (seeded `random`, so runs are reproducible). No external dataset is fetched —
> the point today is the margin + hinge + kernel machinery, not data wrangling.
> Week 4 Day 2 puts these models on a real dataset.

---

## Problem 1 — hinge loss, subgradients, linear SVM

Create `code/svm.py` with a from-scratch implementation.

```python
def hinge_loss(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Return mean hinge `mean(max(0, 1 - y_i*s_i))`."""

def hinge_subgradients(
    X: Sequence[Sequence[float]], y: Sequence[int],
    w: Sequence[float], b: float,
) -> tuple[list[float], float]:
    """Return `(grad_w, grad_b)`, a subgradient of the mean hinge loss."""

class LinearSVMGD:
    """Linear SVM trained by batch subgradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 500, l2: float = 0.01) -> None:
        """Store hyper-parameters; initialise weights lazily in `fit`."""

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
    ) -> list[float]:
        """Run subgradient descent, return the objective history."""

    def decision_function(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw scores `w·x + b` (raise if not fitted)."""

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Return `+1` where score `>= 0`, else `-1` (raise if not fitted)."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises`
contract (docstrings carry the contract — see the Day 10 process fix):

- `hinge_loss`: raise `ValueError` for empty inputs, length mismatch, or any
  label not in `{-1, +1}`.
- `hinge_subgradients`: raise `ValueError` for empty `X`/`y`/`w`,
  `len(X) != len(y)`, ragged rows, rows with zero features, feature-count
  mismatch with `len(w)`, or any label not in `{-1, +1}`. Per-sample rule
  with `s_i = w·x_i + b`, `margin_i = y_i * s_i`: `margin_i < 1` contributes
  `-y_i * x_i` / `-y_i`; `margin_i > 1` contributes nothing; the kink
  `margin_i == 1` contributes nothing (one valid subgradient — document it).
- `LinearSVMGD.__init__`: raise `ValueError` for `lr <= 0`, `epochs <= 0`,
  or `l2 < 0`.
- `LinearSVMGD.fit`: raise `ValueError` for empty data, length mismatch,
  ragged rows, rows with zero features, or any label not in `{-1, +1}`.
  Minimise `mean_hinge + l2 * sum(w_j²)` (bias `b` **not** regularised —
  same MAP-as-regularisation reasoning as Days 13/14). Return the per-epoch
  objective history. Note: subgradient steps need not decrease it strictly
  monotonically — compare early vs late averages.
- `decision_function` / `predict`: raise `ValueError` if called before `fit`
  or on empty/ragged/zero-feature/width-mismatched input.

### Required `__main__` experiments

1. Fit on a synthetic linearly separable 2D blob pair (seeded `random`),
   print the objective every ~10% of epochs, final accuracy, the weights,
   and the support-vector count (`y_i * s_i <= 1`).
2. Build the ring dataset (inner disc `r ∈ [0, 1]` vs outer ring
   `r ∈ [2.5, 3.5]`, seeded angles) — fit the *linear* SVM and print the
   accuracy, which should be poor (the predicted failure; keep the number —
   it is practice question 4's setup).
3. Kernel demo + rescue: print `K(a, a)`, `K(a, neighbour)`, `K(a, far)` for
   the polynomial kernel (neighbour must score above far), then fit a fresh
   SVM on `add_radius_feature(X_ring)` and print the rescued accuracy.

### Running tests

```bash
cd month_2/week_3/day_3 && python -m pytest code/tests/ -v
```

## Problem 2 — polynomial kernel + hand-built radius feature

Same file (`code/svm.py`) — no libraries at all:

```python
def polynomial_kernel(
    x: Sequence[float], y: Sequence[float],
    degree: int = 2, coef0: float = 1.0,
) -> float:
    """Return `(x·y + coef0) ** degree`."""

def add_radius_feature(X: Sequence[Sequence[float]]) -> list[list[float]]:
    """Append `x_0² + x_1²` as a third feature; return the new table."""
```

- `polynomial_kernel` raises `ValueError` for empty inputs, length mismatch,
  or `degree < 1` — documented in the docstring's `Raises` section.
- `add_radius_feature` raises `ValueError` for empty input, ragged rows, or
  any row without exactly 2 features.
- Hand check first: `polynomial_kernel([1, 2], [3, 4]) == 144.0`
  (`dot = 11`, `(11 + 1)²`); `add_radius_feature([[3, 4]]) == [[3, 4, 25]]`.
- Ordering check: `K(a, neighbour) > K(a, far)` for `a = [1, 0]`,
  `neighbour = [0.9, 0.1]`, `far = [-2.5, 0]` — the similarity intuition
  the trick relies on.

### Done when (from the source plan)

Your linear SVM separates the blobs with a clear margin (and you can point
at the support vectors), it visibly fails on the ring, the radius feature
rescues the ring to high accuracy, and you can explain in your own words why
the kernel trick avoids explicitly computing a higher-dimensional
transformation.
