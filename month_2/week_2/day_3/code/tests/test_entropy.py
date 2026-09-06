"""Tests for entropy — Week 2 Day 3."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from entropy import entropy


def test_fair_coin_is_exactly_one_bit():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_nats_base():
    # Same fair coin in nats: ln(2) ~= 0.6931
    assert entropy([0.5, 0.5], base=math.e) == pytest.approx(math.log(2))
    # Uniform-4 in base 4 is exactly 1 "tetrit"
    assert entropy([0.25, 0.25, 0.25, 0.25], base=4.0) == pytest.approx(1.0)


def test_fair_die_is_log2_6():
    assert entropy([1 / 6] * 6) == pytest.approx(math.log2(6))


def test_biased_coin_has_low_entropy():
    # H([0.99, 0.01]) ~= 0.0808 — barely any surprise
    h = entropy([0.99, 0.01])
    assert 0 < h < 0.1
    assert h < entropy([0.5, 0.5])


def test_certain_outcome_has_zero_entropy():
    assert entropy([1.0]) == pytest.approx(0.0)
    # Impossible outcomes contribute 0 * log(0) = 0, not an error
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_uniform_maximises_entropy():
    uniform = entropy([0.25, 0.25, 0.25, 0.25])
    skewed = entropy([0.7, 0.1, 0.1, 0.1])
    assert uniform == pytest.approx(2.0)
    assert skewed < uniform
    assert skewed > 0


def test_entropy_never_negative():
    assert entropy([0.2, 0.3, 0.5]) >= 0
    assert entropy([0.999, 0.0005, 0.0005]) >= 0


def test_raises_on_empty_negative_or_bad_sum():
    with pytest.raises(ValueError):
        entropy([])
    with pytest.raises(ValueError):
        entropy([0.5, 0.6, -0.1])
    with pytest.raises(ValueError):
        entropy([0.5, 0.4])  # sums to 0.9, not 1
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=0)
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=-2.0)
