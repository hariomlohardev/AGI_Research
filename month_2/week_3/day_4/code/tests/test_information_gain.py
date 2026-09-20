"""Tests for entropy / information_gain / majority_label."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from decision_tree import entropy, information_gain, majority_label


def test_entropy_hand_values():
    assert entropy(["yes", "no"]) == pytest.approx(1.0)
    assert entropy(["a", "a", "a"]) == pytest.approx(0.0)
    # Tennis Play distribution (9 yes / 5 no) — the parent entropy of every
    # gain computed on the tennis table.
    assert entropy(["yes"] * 9 + ["no"] * 5) == pytest.approx(0.9403, abs=1e-3)
    # 6-row hand example from learn.md: 3 pos / 3 neg -> exactly 1 bit.
    assert entropy([1, 1, 1, 0, 0, 0]) == pytest.approx(1.0)


def test_entropy_rejects_empty():
    with pytest.raises(ValueError):
        entropy([])


def test_majority_label():
    assert majority_label(["no", "yes", "yes", "no", "yes"]) == "yes"
    assert majority_label([0, 0, 1]) == 0
    # Ties break toward the first-seen label (documented convention).
    assert majority_label(["b", "a"]) == "b"
    with pytest.raises(ValueError):
        majority_label([])


def test_information_gain_perfect_split():
    X = [
        ["x", "p"], ["x", "p"], ["x", "q"],
        ["y", "p"], ["y", "q"], ["y", "q"],
    ]
    y = [1, 1, 1, 0, 0, 0]
    assert information_gain(X, y, 0) == pytest.approx(1.0)
    assert information_gain(X, y, 1) == pytest.approx(0.0817, abs=1e-3)


def test_information_gain_tennis_ordering():
    X, y = _tennis()
    ig_outlook = information_gain(X, y, 0)
    ig_temp = information_gain(X, y, 1)
    ig_humidity = information_gain(X, y, 2)
    ig_wind = information_gain(X, y, 3)
    assert ig_outlook == pytest.approx(0.2467, abs=1e-3)
    assert ig_humidity == pytest.approx(0.1518, abs=1e-3)
    assert ig_wind == pytest.approx(0.0481, abs=1e-3)
    assert ig_temp == pytest.approx(0.0292, abs=1e-3)
    assert ig_outlook > ig_humidity > ig_wind > ig_temp


def test_information_gain_single_valued_feature_is_zero():
    X = [["a", 1], ["a", 2], ["a", 3]]
    y = [0, 1, 0]
    assert information_gain(X, y, 0) == pytest.approx(0.0)


def test_information_gain_rejects_bad_input():
    X = [["x"], ["y"]]
    y = [0, 1]
    with pytest.raises(ValueError):
        information_gain([], [], 0)
    with pytest.raises(ValueError):
        information_gain([["x"]], [0, 1], 0)
    with pytest.raises(ValueError):
        information_gain([["x"], ["y", "z"]], [0, 1], 0)
    with pytest.raises(ValueError):
        information_gain([[]], [0], 0)
    with pytest.raises(ValueError):
        information_gain(X, y, 2)
    with pytest.raises(ValueError):
        information_gain(X, y, -1)


def _tennis():
    X = [
        ["sunny", "hot", "high", "weak"],
        ["sunny", "hot", "high", "strong"],
        ["overcast", "hot", "high", "weak"],
        ["rain", "mild", "high", "weak"],
        ["rain", "cool", "normal", "weak"],
        ["rain", "cool", "normal", "strong"],
        ["overcast", "cool", "normal", "strong"],
        ["sunny", "mild", "high", "weak"],
        ["sunny", "cool", "normal", "weak"],
        ["rain", "mild", "normal", "weak"],
        ["sunny", "mild", "normal", "strong"],
        ["overcast", "mild", "high", "strong"],
        ["overcast", "hot", "normal", "weak"],
        ["rain", "mild", "high", "strong"],
    ]
    y = [
        "no", "no", "yes", "yes", "yes", "no", "yes",
        "no", "yes", "yes", "yes", "yes", "yes", "no",
    ]
    return X, y
