"""Tests for normal_equation.py (closed-form reference)."""

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from linear_regression import LinearRegressionGD, mse_loss
from normal_equation import solve_normal_equation


def test_recovers_exact_weights_on_clean_data():
    # y = 2*x1 - x2 + 3 exactly.
    X = [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0], [2.0, 4.0]]
    y = [2 * r[0] - r[1] + 3 for r in X]
    w, b = solve_normal_equation(X, y)
    assert w == pytest.approx([2.0, -1.0], abs=1e-6)
    assert b == pytest.approx(3.0, abs=1e-6)


def test_agrees_with_gradient_descent():
    rng = random.Random(21)
    true_w = [1.5, -2.5]
    true_b = 0.5
    X = [[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(40)]
    y = [true_w[0] * r[0] + true_w[1] * r[1] + true_b for r in X]
    w_cf, b_cf = solve_normal_equation(X, y)
    model = LinearRegressionGD(lr=0.05, epochs=3000)
    model.fit(X, y)
    assert model.w[0] == pytest.approx(w_cf[0], abs=0.15)
    assert model.w[1] == pytest.approx(w_cf[1], abs=0.15)
    assert model.b == pytest.approx(b_cf, abs=0.15)
    preds_cf = [w_cf[0] * r[0] + w_cf[1] * r[1] + b_cf for r in X]
    assert mse_loss(y, model.predict(X)) == pytest.approx(
        mse_loss(y, preds_cf), abs=1e-3
    )


def test_rejects_bad_input_and_singular_system():
    with pytest.raises(ValueError):
        solve_normal_equation([], [])
    with pytest.raises(ValueError):
        solve_normal_equation([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        solve_normal_equation([[1.0, 2.0], [3.0]], [1.0, 2.0])
    # Perfectly collinear columns: x2 = 2*x1, so X'X is singular.
    with pytest.raises(ValueError):
        solve_normal_equation(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [1.0, 2.0, 3.0]
        )
