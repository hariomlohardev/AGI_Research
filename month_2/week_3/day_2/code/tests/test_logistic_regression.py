"""Tests for logistic_regression.py."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from logistic_regression import (
    LogisticRegressionGD,
    bce_gradients,
    binary_cross_entropy,
    sigmoid,
)


def test_sigmoid_hand_values():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)
    # Symmetry: sigmoid(-z) == 1 - sigmoid(z).
    for z in (0.3, 1.5, 4.0):
        assert sigmoid(-z) == pytest.approx(1.0 - sigmoid(z))


def test_bce_hand_value():
    # -(ln 0.75 + ln 0.75) / 2 = -ln 0.75.
    assert binary_cross_entropy([0, 1], [0.25, 0.75]) == pytest.approx(
        -math.log(0.75)
    )
    assert binary_cross_entropy([1, 1], [1.0, 1.0]) == pytest.approx(0.0)


def test_bce_rejects_bad_input():
    with pytest.raises(ValueError):
        binary_cross_entropy([], [])
    with pytest.raises(ValueError):
        binary_cross_entropy([0, 1], [0.5])
    with pytest.raises(ValueError):
        binary_cross_entropy([0, 2], [0.5, 0.5])


def test_bce_gradients_finite_difference():
    rng = random.Random(3)
    X = [[rng.uniform(-2, 2), rng.uniform(-2, 2)] for _ in range(6)]
    y = [rng.choice([0, 1]) for _ in range(6)]
    w = [0.4, -0.7]
    b = 0.2
    grad_w, grad_b = bce_gradients(X, y, w, b)
    h = 1e-6
    for j in range(len(w)):
        w_plus = list(w)
        w_minus = list(w)
        w_plus[j] += h
        w_minus[j] -= h
        num = (
            binary_cross_entropy(
                y,
                _sigmoid_list(X, w_plus, b),
            )
            - binary_cross_entropy(y, _sigmoid_list(X, w_minus, b))
        ) / (2 * h)
        assert grad_w[j] == pytest.approx(num, rel=1e-4)
    num_b = (
        binary_cross_entropy(y, _sigmoid_list(X, w, b + h))
        - binary_cross_entropy(y, _sigmoid_list(X, w, b - h))
    ) / (2 * h)
    assert grad_b == pytest.approx(num_b, rel=1e-4)


def _sigmoid_list(X, w, b):
    from logistic_regression import sigmoid as _sig

    return [_sig(sum(xi * wi for xi, wi in zip(row, w)) + b) for row in X]


def test_bce_gradients_reject_bad_input():
    with pytest.raises(ValueError):
        bce_gradients([], [], [1.0], 0.0)
    with pytest.raises(ValueError):
        bce_gradients([[1.0]], [0, 1], [1.0], 0.0)
    with pytest.raises(ValueError):
        bce_gradients([[1.0, 2.0]], [0], [1.0], 0.0)
    with pytest.raises(ValueError):
        bce_gradients([[1.0]], [2], [1.0], 0.0)


def test_fit_loss_decreases_and_classifies():
    rng = random.Random(11)
    X = [[rng.gauss(-2.0, 0.8)] for _ in range(30)] + [
        [rng.gauss(2.0, 0.8)] for _ in range(30)
    ]
    y = [0] * 30 + [1] * 30
    model = LogisticRegressionGD(lr=0.5, epochs=300)
    history = model.fit(X, y)
    assert len(history) == 300
    assert history[-1] < history[0]
    preds = model.predict(X)
    acc = sum(1 for a, p in zip(y, preds) if a == p) / len(y)
    assert acc >= 0.95


def test_fit_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        LogisticRegressionGD(lr=0.0)
    with pytest.raises(ValueError):
        LogisticRegressionGD(epochs=-1)
    with pytest.raises(ValueError):
        LogisticRegressionGD(l2=-0.5)


def test_predict_before_fit_raises():
    model = LogisticRegressionGD()
    with pytest.raises(ValueError):
        model.predict([[1.0]])
    with pytest.raises(ValueError):
        model.predict_proba([[1.0]])


def test_predict_threshold_validation():
    rng = random.Random(5)
    X = [[rng.gauss(-2.0, 1.0)] for _ in range(10)] + [
        [rng.gauss(2.0, 1.0)] for _ in range(10)
    ]
    y = [0] * 10 + [1] * 10
    model = LogisticRegressionGD(lr=0.5, epochs=100)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X, threshold=0.0)
    with pytest.raises(ValueError):
        model.predict(X, threshold=1.0)
    proba = model.predict_proba(X)
    assert all(0.0 < p < 1.0 for p in proba)
    assert model.predict(X, threshold=0.5) == [
        1 if p >= 0.5 else 0 for p in proba
    ]


def test_l2_shrinks_weights():
    rng = random.Random(9)
    X = [[rng.gauss(-1.5, 1.0), rng.gauss(1.5, 1.0)] for _ in range(50)]
    y = [rng.choice([0, 1]) for _ in range(50)]
    plain = LogisticRegressionGD(lr=0.1, epochs=200, l2=0.0)
    plain.fit(X, y)
    shrunk = LogisticRegressionGD(lr=0.1, epochs=200, l2=1.0)
    shrunk.fit(X, y)
    norm = lambda w: math.sqrt(sum(v * v for v in w))
    assert norm(shrunk.w) < norm(plain.w)
