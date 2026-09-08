"""Tests for mutual_information()."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mutual_information import mutual_information

CORR_X = [0, 0, 1, 1]
CORR_Y = [0, 0, 1, 1]
INDEP_X = [0, 0, 1, 1]
INDEP_Y = [0, 1, 0, 1]


def _entropy(values, base=2.0):
    n = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return -sum((c / n) * math.log(c / n, base) for c in counts.values())


def test_perfectly_correlated_binary_is_one_bit():
    assert mutual_information(CORR_X, CORR_Y) == pytest.approx(1.0, abs=1e-9)


def test_independent_binary_is_zero():
    assert mutual_information(INDEP_X, INDEP_Y) == pytest.approx(0.0, abs=1e-9)


def test_string_labels():
    assert mutual_information(["a", "a", "b", "b"], ["u", "u", "v", "v"]) == pytest.approx(
        1.0, abs=1e-9
    )
    assert mutual_information(["a", "a", "b", "b"], ["u", "v", "u", "v"]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_symmetric():
    rng = random.Random(11)
    x = [rng.choice((0, 1, 2)) for _ in range(60)]
    y = [rng.choice((0, 1, 2)) for _ in range(60)]
    assert mutual_information(x, y) == pytest.approx(mutual_information(y, x), abs=1e-9)


def test_never_negative():
    rng = random.Random(12)
    for _ in range(10):
        x = [rng.choice((0, 1, 2)) for _ in range(40)]
        y = [rng.choice((0, 1)) for _ in range(40)]
        assert mutual_information(x, y) >= -1e-9


def test_bounded_by_marginal_entropies():
    rng = random.Random(13)
    for _ in range(10):
        x = [rng.choice((0, 1, 2)) for _ in range(50)]
        y = [rng.choice((0, 1)) for _ in range(50)]
        mi = mutual_information(x, y)
        assert mi <= min(_entropy(x), _entropy(y)) + 1e-9


def test_quadratic_demo_value():
    xs = list(range(-4, 5))
    ys = [v * v for v in xs]
    assert mutual_information(xs, ys) == pytest.approx(2.281, abs=1e-3)


def test_empty_x_raises():
    with pytest.raises(ValueError):
        mutual_information([], [0, 1])


def test_empty_y_raises():
    with pytest.raises(ValueError):
        mutual_information([0, 1], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mutual_information([0, 0, 1], [0, 1])


@pytest.mark.parametrize("base", [0.0, -2.0, 1.0])
def test_bad_base_raises(base):
    with pytest.raises(ValueError):
        mutual_information(CORR_X, CORR_Y, base=base)
