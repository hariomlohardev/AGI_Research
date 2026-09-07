"""Tests for binary_log_loss()."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from log_loss import binary_log_loss


def test_confident_correct_costs_little():
    assert binary_log_loss([1], [0.9]) == pytest.approx(0.152, abs=1e-3)


def test_confident_wrong_costs_a_lot():
    assert binary_log_loss([0], [0.9]) == pytest.approx(3.322, abs=1e-3)


def test_wrong_costs_more_than_right():
    assert binary_log_loss([0], [0.9]) > binary_log_loss([1], [0.9])


def test_perfect_predictions_cost_zero():
    assert binary_log_loss([1, 0], [1.0 - 1e-12, 1e-12]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_mean_over_samples():
    # Mean of the two single-sample losses above.
    assert binary_log_loss([1, 0], [0.9, 0.1]) == pytest.approx(
        (binary_log_loss([1], [0.9]) + binary_log_loss([0], [0.1])) / 2,
        abs=1e-9,
    )


def test_raises_on_bad_input():
    with pytest.raises(ValueError):
        binary_log_loss([1], [0.0])  # log(0) undefined
    with pytest.raises(ValueError):
        binary_log_loss([1], [1.0])
    with pytest.raises(ValueError):
        binary_log_loss([1], [1.5])
    with pytest.raises(ValueError):
        binary_log_loss([1, 0], [0.9])  # mismatched lengths
    with pytest.raises(ValueError):
        binary_log_loss([2], [0.9])  # label must be 0/1
    with pytest.raises(ValueError):
        binary_log_loss([], [])
