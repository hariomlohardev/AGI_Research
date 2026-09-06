"""Tests for entropy.py."""

import math
import pytest

from entropy import entropy


def test_entropy_fair_coin():
    pmf = {"H": 0.5, "T": 0.5}
    assert entropy(pmf) == pytest.approx(1.0, abs=1e-9)


def test_entropy_biased_coin():
    pmf = {"H": 0.9, "T": 0.1}
    # -0.9*log2(0.9) - 0.1*log2(0.1)
    expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert entropy(pmf) == pytest.approx(expected, abs=1e-12)


def test_entropy_uniform_die():
    pmf = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    expected = math.log2(6)
    assert entropy(pmf) == pytest.approx(expected, abs=1e-12)


def test_entropy_zero_probability():
    pmf = {"A": 0.5, "B": 0.5, "C": 0.0}
    assert entropy(pmf) == pytest.approx(1.0, abs=1e-12)


def test_entropy_empty_pmf():
    with pytest.raises(ValueError):
        entropy({})


def test_entropy_negative_probability():
    with pytest.raises(ValueError):
        entropy({"A": -0.1, "B": 1.1})


def test_entropy_probabilities_not_sum_to_one():
    with pytest.raises(ValueError):
        entropy({"A": 0.2, "B": 0.3})  # sums to 0.5


def test_entropy_close_to_one():
    pmf = {"A": 0.5000000001, "B": 0.4999999999}
    # Should pass tolerance
    assert entropy(pmf) == pytest.approx(1.0, abs=1e-9)