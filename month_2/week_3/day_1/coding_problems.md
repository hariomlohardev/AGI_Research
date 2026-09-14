# Day 13 — Linear Regression (Gradient Descent Version): coding problems

## Problem 1 — multi-feature linear regression trained by gradient descent

Create `code/linear_regression.py` with a from-scratch implementation. Standard
library only (plus `matplotlib` for the optional `__main__` loss-curve demo —
never inside the tested functions). Use plain Python lists (`X` is rows ×
columns), not numpy, so every gradient step is visible.

```python
def predict(X: Sequence[Sequence[float]], w: Sequence[float], b: float) -> list[float]:
    """Return `[w·x + b for x in X]`."""

def mse_loss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean squared error."""

def mse_gradients(
    X: Sequence[Sequence[float]], y: Sequence[float],
    w: Sequence[float], b: float,
) -> tuple[list[float], float]:
    """Return `(grad_w, grad_b)` of the MSE at the current parameters."""

class LinearRegressionGD:
    """Multi-feature linear regression trained by batch gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000, l2: float = 0.0) -> None:
        """Store hyper-parameters; initialise weights lazily in `fit`."""

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> list[float]:
        """Run gradient descent, return the loss history (one MSE per epoch)."""

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict with the fitted weights (raise if not fitted)."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises`
contract:

- `predict`: raise `ValueError` for empty `X`, ragged rows, empty `w`, or
  feature-count mismatch (`len(row) != len(w)`).
- `mse_loss`: raise `ValueError` for empty inputs or length mismatch.
- `mse_gradients`: raise `ValueError` for empty inputs, length mismatch
  (`len(X) != len(y)`), ragged rows, or feature-count mismatch.
- `LinearRegressionGD.__init__`: raise `ValueError` for `lr <= 0`,
  `epochs <= 0`, or `l2 < 0`.
- `LinearRegressionGD.fit`: raise `ValueError` for empty data, length
  mismatch, ragged rows, or rows with zero features. Return the per-epoch loss
  history. With `l2 > 0`, minimise `MSE + l2 * sum(w_j²)` (note: bias `b` is
  **not** regularised) and add the `2 * l2 * w_j` term to the weight gradient.
- `LinearRegressionGD.predict`: raise `ValueError` if called before `fit`.

### Required `__main__` experiments

1. Fit on a small synthetic house-price-like dataset, print the loss every
   ~10% of epochs, and print final `w`, `b`, and MSE.
2. Re-fit with a deliberately too-high learning rate and print the first few
   losses to show divergence/oscillation.
3. Fit with and without L2 (`l2=0` vs `l2=1.0`) and print both weight vectors
   to show shrinkage toward zero.

### Running tests

```bash
cd month_2/week_3/day_1 && python -m pytest code/tests/ -v
```

## Problem 2 — closed-form Normal Equation (the comparison target)

Create `code/normal_equation.py`:

```python
def solve_normal_equation(
    X: Sequence[Sequence[float]], y: Sequence[float]
) -> tuple[list[float], float]:
    """Return `(w, b)` solving least squares via the Normal Equation."""
```

Solve with Gaussian elimination written from scratch (stdlib only) on the
augmented system with a bias column of ones. This is the reference your GD
solution must come close to.

### Full validation contracts

- Raise `ValueError` for empty data, length mismatch, ragged rows, or rows
  with zero features.
- Raise `ValueError` if the system is singular (zero pivot after partial
  pivoting) — e.g. perfectly collinear features.

### Done when (from the source plan)

Your gradient-descent weights land close to the Normal Equation solution, and
your loss curve decreases smoothly. After this day, `/done` quizzes the
gradient derivation, learning-rate behaviour, and the MAP↔L2 connection.
