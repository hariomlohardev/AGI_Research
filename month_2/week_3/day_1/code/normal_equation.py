"""Closed-form least-squares reference via the Normal Equation.

Solves ``(X'X) theta = X'y`` with Gaussian elimination (stdlib only), where
``X'`` is the design matrix with a leading bias column of ones. This is the
comparison target for the gradient-descent solution in
``linear_regression.py``.
"""

from __future__ import annotations

from collections.abc import Sequence


def solve_normal_equation(
    X: Sequence[Sequence[float]], y: Sequence[float]
) -> tuple[list[float], float]:
    """Return ``(w, b)`` solving least squares via the Normal Equation.

    Raises:
        ValueError: if data is empty, ``len(X) != len(y)``, rows are ragged,
            rows have zero features, or the system is singular (zero pivot
            after partial pivoting, e.g. perfectly collinear features).
    """
    raise NotImplementedError


if __name__ == "__main__":
    X = [[8.0, 2.0], [10.0, 3.0], [12.0, 3.0], [14.0, 4.0], [16.0, 4.0]]
    y = [150.0, 200.0, 230.0, 280.0, 320.0]
    w, b = solve_normal_equation(X, y)
    print("w =", w, "b =", b)
