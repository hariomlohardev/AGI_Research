"""Tests for linear_regression.py (GD version)."""

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from linear_regression import (
    LinearRegressionGD,
    mse_gradients,
    mse_loss,
    predict,
)


def test_predict_hand_values():
    X = [[1.0, 2.0], [3.0, 4.0]]
    assert predict(X, [2.0, -1.0], 0.5) == pytest.approx([0.5, 2.5])


def test_predict_rejects_bad_input():
    with pytest.raises(ValueError):
        predict([], [1.0], 0.0)
    with pytest.raises(ValueError):
        predict([[1.0, 2.0], [3.0]], [1.0, 1.0], 0.0)
    with pytest.raises(ValueError):
        predict([[1.0]], [], 0.0)
    with pytest.raises(ValueError):
        predict([[1.0, 2.0]], [1.0], 0.0)


def test_mse_loss_hand_value():
    assert mse_loss([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1.0 / 3.0)
    assert mse_loss([5.0], [5.0]) == pytest.approx(0.0)


def test_mse_loss_rejects_bad_input():
    with pytest.raises(ValueError):
        mse_loss([], [])
    with pytest.raises(ValueError):
        mse_loss([1.0], [1.0, 2.0])


def test_gradients_match_finite_differences():
    rng = random.Random(7)
    X = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(12)]
    y = [rng.uniform(-2, 2) for _ in range(12)]
    w = [0.5, -1.0, 0.25]
    b = 0.1
    grad_w, grad_b = mse_gradients(X, y, w, b)
    eps = 1e-6
    for j in range(len(w)):
        w2 = list(w)
        w2[j] += eps
        num = (
            mse_loss(y, predict(X, w2, b)) - mse_loss(y, predict(X, w, b))
        ) / eps
        assert grad_w[j] == pytest.approx(num, rel=1e-3, abs=1e-4)
    num_b = (
        mse_loss(y, predict(X, w, b + eps)) - mse_loss(y, predict(X, w, b))
    ) / eps
    assert grad_b == pytest.approx(num_b, rel=1e-3, abs=1e-4)


def test_gradients_reject_bad_input():
    with pytest.raises(ValueError):
        mse_gradients([], [], [1.0], 0.0)
    with pytest.raises(ValueError):
        mse_gradients([[1.0]], [1.0, 2.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        mse_gradients([[1.0, 2.0]], [1.0], [1.0], 0.0)


def _synthetic(seed=3, n=60):
    rng = random.Random(seed)
    true_w = [3.0, -2.0]
    true_b = 1.0
    X = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(n)]
    y = [true_w[0] * r[0] + true_w[1] * r[1] + true_b for r in X]
    return X, y, true_w, true_b


def test_fit_recovers_weights_on_clean_data():
    X, y, true_w, true_b = _synthetic()
    model = LinearRegressionGD(lr=0.1, epochs=2000)
    history = model.fit(X, y)
    assert history[-1] < 1e-4
    preds = model.predict(X)
    assert mse_loss(y, preds) == pytest.approx(0.0, abs=1e-3)
    assert model.w[0] == pytest.approx(true_w[0], abs=0.1)
    assert model.w[1] == pytest.approx(true_w[1], abs=0.1)
    assert model.b == pytest.approx(true_b, abs=0.1)


def test_loss_trend_decreases_on_well_conditioned_data():
    X, y, _, _ = _synthetic()
    model = LinearRegressionGD(lr=0.1, epochs=500)
    history = model.fit(X, y)
    assert len(history) == 500
    assert history[-1] < history[0]
    # No wild oscillation: the second half stays below the first value.
    assert max(history[250:]) < history[0]


def test_l2_shrinks_weights():
    X, y, _, _ = _synthetic()
    plain = LinearRegressionGD(lr=0.1, epochs=500, l2=0.0)
    plain.fit(X, y)
    shrunk = LinearRegressionGD(lr=0.1, epochs=500, l2=5.0)
    shrunk.fit(X, y)
    norm_plain = sum(v * v for v in plain.w) ** 0.5
    norm_shrunk = sum(v * v for v in shrunk.w) ** 0.5
    assert norm_shrunk < norm_plain


def test_hyperparams_and_predict_before_fit():
    with pytest.raises(ValueError):
        LinearRegressionGD(lr=0.0)
    with pytest.raises(ValueError):
        LinearRegressionGD(lr=-0.1)
    with pytest.raises(ValueError):
        LinearRegressionGD(epochs=0)
    with pytest.raises(ValueError):
        LinearRegressionGD(l2=-1.0)
    model = LinearRegressionGD()
    with pytest.raises(ValueError):
        model.predict([[1.0]])


def test_fit_rejects_bad_data():
    model = LinearRegressionGD()
    with pytest.raises(ValueError):
        model.fit([], [])
    with pytest.raises(ValueError):
        model.fit([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        model.fit([[1.0, 2.0], [3.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        model.fit([[]], [1.0])
