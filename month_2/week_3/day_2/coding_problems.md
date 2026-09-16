# Day 14 — Logistic Regression: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only in the tested
functions (plus `matplotlib` for the optional `__main__` loss-curve demo —
never inside the tested functions). Plain Python lists (`X` is rows ×
columns), not numpy — same discipline as Day 13, now with a sigmoid on top.

> Data note: everything below uses small synthetic datasets generated inline
> (seeded `random`, so runs are reproducible). No external dataset is fetched —
> the point today is the sigmoid + BCE + GD machinery, not data wrangling.
> Week 4 Day 2 puts these models on a real dataset.

---

## Problem 1 — `sigmoid`, `binary_cross_entropy`, GD classifier

Create `code/logistic_regression.py` with a from-scratch implementation.

```python
def sigmoid(z: float) -> float:
    """Return `1 / (1 + exp(-z))`, numerically stable for large |z|."""

def binary_cross_entropy(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Return mean BCE `-mean(y*log(p) + (1-y)*log(1-p))` with epsilon clipping."""

def bce_gradients(
    X: Sequence[Sequence[float]], y: Sequence[int],
    w: Sequence[float], b: float,
) -> tuple[list[float], float]:
    """Return `(grad_w, grad_b)` of the BCE at the current parameters."""

class LogisticRegressionGD:
    """Binary logistic regression trained by batch gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000, l2: float = 0.0) -> None:
        """Store hyper-parameters; initialise weights lazily in `fit`."""

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
    ) -> list[float]:
        """Run gradient descent, return the loss history (one BCE per epoch)."""

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw sigmoid outputs (raise if not fitted)."""

    def predict(
        self, X: Sequence[Sequence[float]], threshold: float = 0.5
    ) -> list[int]:
        """Return `1` where proba `>= threshold`, else `0` (raise if not fitted)."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises`
contract (docstrings carry the contract — see the Day 10 process fix):

- `sigmoid`: no `ValueError` contract — any real `z` maps into `(0, 1)`.
  Implement it stably (no `OverflowError` for `z = ±1000`).
- `binary_cross_entropy`: raise `ValueError` for empty inputs, length
  mismatch, or any label not in `{0, 1}`. Clip predictions into
  `[eps, 1 - eps]` (`eps = 1e-15`) instead of raising on `0.0`/`1.0` —
  document the clipping.
- `bce_gradients`: raise `ValueError` for empty inputs, length mismatch
  (`len(X) != len(y)`), ragged rows, feature-count mismatch, or any label
  not in `{0, 1}`. The gradients are
  `grad_w[j] = (1/n) * sum_i((p_i - y_i) * X[i][j])`,
  `grad_b = (1/n) * sum_i(p_i - y_i)` where `p_i = sigmoid(w·x_i + b)`.
- `LogisticRegressionGD.__init__`: raise `ValueError` for `lr <= 0`,
  `epochs <= 0`, or `l2 < 0`.
- `LogisticRegressionGD.fit`: raise `ValueError` for empty data, length
  mismatch, ragged rows, rows with zero features, or any label not in
  `{0, 1}`. Return the per-epoch loss history. With `l2 > 0`, minimise
  `BCE + l2 * sum(w_j²)` (note: bias `b` is **not** regularised — same
  MAP-as-regularisation reasoning as Day 13).
- `predict_proba` / `predict`: raise `ValueError` if called before `fit`
  or on empty/ragged/width-mismatched input; `predict` additionally raises
  for `threshold` outside `(0, 1)` (exclusive).

### Required `__main__` experiments

1. Fit on a synthetic 2-class blob dataset (seeded `random`, linearly
   separable-ish), print the loss every ~10% of epochs plus final accuracy.
2. Predict at thresholds `0.5` vs `0.3` and print precision/recall for both —
   the direction of change is practice question 4, so keep the numbers.
3. Fit with and without L2 (`l2=0` vs `l2=1.0`) and print both weight norms
   to show shrinkage.

### Running tests

```bash
cd month_2/week_3/day_2 && python -m pytest code/tests/ -v
```

## Problem 2 — metrics by hand (`accuracy`, `precision`, `recall`)

Create `code/metrics.py` — no libraries at all, not even `math`:

```python
def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return `(tp, fp, tn, fn)` counts."""

def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `(tp + tn) / n`."""

def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `tp / (tp + fp)`; `0.0` when the denominator is 0."""

def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `tp / (tp + fn)`; `0.0` when the denominator is 0."""
```

- All four raise `ValueError` for empty inputs, length mismatch, or any
  label not in `{0, 1}` — documented in each docstring's `Raises` section.
- Build the three scores on top of `confusion_matrix` (one place owns the
  counting). The `0.0`-on-empty-denominator rule is a documented convention,
  not a mathematical fact — say so in the docstring.
- Hand check first: `y_true = [0, 0, 1, 1]`, `y_pred = [0, 1, 1, 1]` gives
  `(tp, fp, tn, fn) = (2, 1, 1, 0)`, accuracy `0.75`, precision `2/3`,
  recall `1.0`.

### Done when (from the source plan)

Your classifier trains to a reasonable accuracy on the synthetic binary
data, you can derive the sigmoid's derivative from memory, and you can
explain why accuracy alone misleads on imbalanced data.
