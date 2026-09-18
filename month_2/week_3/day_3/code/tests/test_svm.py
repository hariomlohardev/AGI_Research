"""Tests for svm.py (hinge loss, subgradients, LinearSVMGD)."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from svm import LinearSVMGD, hinge_loss, hinge_subgradients


def test_hinge_hand_values():
    # Margins 0.5, 0.5 -> losses 0.5, 0.5 -> mean 0.5.
    assert hinge_loss([1, -1], [0.5, -0.5]) == pytest.approx(0.5)
    # Comfortable margins -> exactly 0 (never just approaching it, unlike CE).
    assert hinge_loss([1, -1], [2.0, -2.0]) == pytest.approx(0.0)
    # Confidently wrong: y=+1, s=-2 -> margin -2 -> loss 3.0.
    assert hinge_loss([1], [-2.0]) == pytest.approx(3.0)


def test_hinge_rejects_bad_input():
    with pytest.raises(ValueError):
        hinge_loss([], [])
    with pytest.raises(ValueError):
        hinge_loss([1, -1], [0.5])
    with pytest.raises(ValueError):
        hinge_loss([0, 1], [0.5, 0.5])


def test_subgradients_all_violating():
    # w = 0, b = 0 -> every margin is 0 < 1, every point contributes.
    # i1: -y*x = [-1, 0]; i2: -(-1)*[0, 1] = [0, +1]; mean -> [-0.5, +0.5].
    grad_w, grad_b = hinge_subgradients([[1.0, 0.0], [0.0, 1.0]], [1, -1], [0.0, 0.0], 0.0)
    assert grad_w == pytest.approx([-0.5, 0.5])
    assert grad_b == pytest.approx(0.0)


def test_subgradients_mixed_and_kink():
    # Point 1 far beyond its margin (margin 5 -> no contribution);
    # point 2 exactly on the kink (margin == 1 -> 0 by convention).
    grad_w, grad_b = hinge_subgradients(
        [[5.0], [-1.0]], [1, -1], [1.0], 0.0
    )
    # s = [5.0, -1.0]; margins = [5.0, 1.0]; both contribute nothing.
    assert grad_w == pytest.approx([0.0])
    assert grad_b == pytest.approx(0.0)
    # One violator only: margin -1 -> contributes -y*x = +1, -y = +1... check:
    # X=[[2.0]], y=[-1], w=[0.0], b=0 -> s=0, margin=0 <1 -> grad_w=[2.0], grad_b=1.0.
    grad_w2, grad_b2 = hinge_subgradients([[2.0]], [-1], [0.0], 0.0)
    assert grad_w2 == pytest.approx([2.0])
    assert grad_b2 == pytest.approx(1.0)


def test_subgradients_reject_bad_input():
    with pytest.raises(ValueError):
        hinge_subgradients([], [], [1.0], 0.0)
    with pytest.raises(ValueError):
        hinge_subgradients([[1.0]], [1, -1], [1.0], 0.0)
    with pytest.raises(ValueError):
        hinge_subgradients([[1.0, 2.0]], [1], [1.0], 0.0)
    with pytest.raises(ValueError):
        hinge_subgradients([[1.0]], [0], [1.0], 0.0)


def test_fit_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        LinearSVMGD(lr=0.0)
    with pytest.raises(ValueError):
        LinearSVMGD(epochs=0)
    with pytest.raises(ValueError):
        LinearSVMGD(l2=-0.1)


def test_predict_before_fit_raises():
    model = LinearSVMGD()
    with pytest.raises(ValueError):
        model.predict([[1.0, 0.0]])
    with pytest.raises(ValueError):
        model.decision_function([[1.0, 0.0]])


def test_fit_separates_linearly_separable_data():
    rng = random.Random(11)
    X = [[rng.gauss(-2.0, 0.8), rng.gauss(0.0, 0.8)] for _ in range(30)]
    X += [[rng.gauss(2.0, 0.8), rng.gauss(0.0, 0.8)] for _ in range(30)]
    y = [-1] * 30 + [1] * 30
    model = LinearSVMGD(lr=0.1, epochs=500, l2=0.01)
    history = model.fit(X, y)
    assert len(history) == 500
    assert history[-1] < history[0]
    preds = model.predict(X)
    assert set(preds) <= {-1, 1}
    acc = sum(1 for a, p in zip(y, preds) if a == p) / len(y)
    assert acc >= 0.95
    # Margin sanity: most training points sit beyond the margin... at least
    # the fit is not degenerate (both classes predicted).
    assert set(preds) == {-1, 1}


def test_linear_svm_fails_on_ring():
    from svm import add_radius_feature

    rng = random.Random(21)
    X, y = _ring_data(rng)
    model = LinearSVMGD(lr=0.1, epochs=500, l2=0.01)
    model.fit(X, y)
    preds = model.predict(X)
    acc = sum(1 for a, p in zip(y, preds) if a == p) / len(y)
    assert acc < 0.85  # the failure this day's theory predicts
    # ...while the hand-built radius feature rescues it.
    X_lifted = add_radius_feature(X)
    lifted = LinearSVMGD(lr=0.05, epochs=800, l2=0.01)
    lifted.fit(X_lifted, y)
    lifted_preds = lifted.predict(X_lifted)
    lifted_acc = sum(1 for a, p in zip(y, lifted_preds) if a == p) / len(y)
    assert lifted_acc >= 0.90
    assert lifted_acc - acc >= 0.15


def test_l2_shrinks_weights():
    rng = random.Random(9)
    X = [[rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(50)]
    y = [rng.choice([-1, 1]) for _ in range(50)]
    plain = LinearSVMGD(lr=0.1, epochs=200, l2=0.0)
    plain.fit(X, y)
    shrunk = LinearSVMGD(lr=0.1, epochs=200, l2=1.0)
    shrunk.fit(X, y)
    norm = lambda w: math.sqrt(sum(v * v for v in w))
    assert norm(shrunk.w) < norm(plain.w)


def _ring_data(rng):
    X, y = [], []
    for _ in range(40):
        r, t = rng.uniform(0.0, 1.0), rng.uniform(0.0, 2 * math.pi)
        X.append([r * math.cos(t), r * math.sin(t)])
        y.append(-1)
    for _ in range(40):
        r, t = rng.uniform(2.5, 3.5), rng.uniform(0.0, 2 * math.pi)
        X.append([r * math.cos(t), r * math.sin(t)])
        y.append(1)
    return X, y
