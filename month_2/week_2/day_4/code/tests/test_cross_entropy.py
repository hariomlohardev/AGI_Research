"""Tests for cross_entropy() and kl_divergence()."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cross_entropy import cross_entropy, kl_divergence

P = [0.5, 0.5]
Q = [0.9, 0.1]


def test_cross_entropy_hand_value():
    assert cross_entropy(P, Q) == pytest.approx(1.737, abs=1e-3)


def test_cross_entropy_self_equals_entropy():
    # H(P, P) must equal yesterday's H(P) = 1.0 bit for a fair coin.
    assert cross_entropy(P, P) == pytest.approx(1.0, abs=1e-9)


def test_cross_entropy_at_least_entropy():
    # Being wrong about the distribution can only add cost.
    rng = random.Random(0)
    for _ in range(20):
        a = 0.05 + rng.random() * 0.9
        p = [a, 1 - a]
        b = 0.05 + rng.random() * 0.9
        q = [b, 1 - b]
        assert cross_entropy(p, q) >= cross_entropy(p, p) - 1e-9


def test_kl_hand_values_and_asymmetry():
    assert kl_divergence(P, Q) == pytest.approx(0.737, abs=1e-3)
    assert kl_divergence(Q, P) == pytest.approx(0.531, abs=1e-3)
    assert kl_divergence(P, Q) != pytest.approx(kl_divergence(Q, P))


def test_kl_self_is_zero_and_nonnegative():
    assert kl_divergence(P, P) == pytest.approx(0.0, abs=1e-12)
    rng = random.Random(1)
    for _ in range(20):
        a = 0.05 + rng.random() * 0.9
        p = [a, 1 - a]
        b = 0.05 + rng.random() * 0.9
        q = [b, 1 - b]
        assert kl_divergence(p, q) >= -1e-9


def test_ce_minus_h_equals_kl():
    assert cross_entropy(P, Q) - cross_entropy(P, P) == pytest.approx(
        kl_divergence(P, Q), abs=1e-9
    )


def test_nats_base():
    assert cross_entropy(P, Q, base=math.e) == pytest.approx(
        cross_entropy(P, Q) / math.log2(math.e), rel=1e-9
    )


def test_raises_on_bad_input():
    with pytest.raises(ValueError):
        cross_entropy([], [])
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [0.5])  # mismatched lengths
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [0.5, 0.4])  # q doesn't sum to 1
    with pytest.raises(ValueError):
        cross_entropy([0.6, 0.5], Q)  # p doesn't sum to 1
    with pytest.raises(ValueError):
        cross_entropy([-0.1, 1.1], Q)  # negative entry
    with pytest.raises(ValueError):
        cross_entropy(P, [1.0, 0.0])  # q claims an occurred event impossible
    with pytest.raises(ValueError):
        kl_divergence(P, [1.0, 0.0])
    with pytest.raises(ValueError):
        cross_entropy(P, Q, base=0)


def test_zero_p_term_contributes_zero():
    # p(x) == 0 contributes 0 regardless of q(x) — the 0*log0 convention.
    assert cross_entropy([0.0, 1.0], Q) == pytest.approx(-math.log2(0.1), abs=1e-9)
    assert kl_divergence([0.0, 1.0], Q) == pytest.approx(-math.log2(0.1), abs=1e-9)
