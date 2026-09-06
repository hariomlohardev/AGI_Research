"""Tests for entropy — Week 2 Day 3."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from entropy import entropy


def test_fair_coin_is_one_bit():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0, rel=1e-9)


def test_biased_coin_has_lower_entropy():
    # Hand computation: -(0.99*log2(0.99) + 0.01*log2(0.01)) ~= 0.0808
    expected = -(0.99 * math.log2(0.99) + 0.01 * math.log2(0.01))
    assert entropy([0.99, 0.01]) == pytest.approx(expected, rel=1e-9)
    assert entropy([0.99, 0.01]) < entropy([0.5, 0.5])


def test_fair_die_is_log2_6():
    assert entropy([1 / 6] * 6) == pytest.approx(math.log2(6), rel=1e-9)


def test_uniform_maximizes_entropy():
    assert entropy([0.25] * 4) > entropy([0.7, 0.1, 0.1, 0.1])


def test_degenerate_distribution_is_zero():
    # No surprise when the outcome is certain; zeros must not blow up.
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-12)


def test_nats_base():
    assert entropy([0.5, 0.5], base=math.e) == pytest.approx(math.log(2), rel=1e-9)


def test_raises_on_bad_input():
    with pytest.raises(ValueError):
        entropy([])
    with pytest.raises(ValueError):
        entropy([0.5, -0.1, 0.6])
    with pytest.raises(ValueError):
        entropy([0.5, 0.5, 0.5])  # sums to 1.5, not a pmf
    with pytest.raises(ValueError):
        entropy([0.0, 0.0])
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=0)
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=-2)
