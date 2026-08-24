"""Tests for log_likelihood — Week 2 Day 1."""

import math
import pathlib
import sys

import pytest

# Allow `from log_likelihood import ...` when pytest is run from project root
# (e.g. `pytest month_2/week_2/day_1/code/tests`) by adding the sibling `code/` dir to sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from log_likelihood import log_likelihood


def _gaussian_log_pdf(x, mu, sigma):
    return -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def test_log_likelihood_matches_hand_computation():
    data = [0, 1, 2]
    mu, sigma = 1.0, 1.0
    expected = sum(_gaussian_log_pdf(x, mu, sigma) for x in data)
    assert log_likelihood(data, mu, sigma) == pytest.approx(expected, rel=1e-9)


def test_log_likelihood_raises_on_nonpositive_sigma():
    with pytest.raises(ValueError):
        log_likelihood([1, 2, 3], mu=1.0, sigma=0)
    with pytest.raises(ValueError):
        log_likelihood([1, 2, 3], mu=1.0, sigma=-1)


def test_log_likelihood_raises_on_empty_data():
    with pytest.raises(ValueError):
        log_likelihood([], mu=0, sigma=1)


def test_log_likelihood_more_likely_near_mean():
    data = [0, 0.1, -0.1, 0.05]
    ll_good = log_likelihood(data, mu=0, sigma=1)
    ll_bad = log_likelihood(data, mu=10, sigma=1)
    assert ll_good > ll_bad, "data clustered at 0 should be more likely under mu=0 than mu=10"


def test_log_likelihood_sigma_sensitivity():
    data = [0, 0, 0, 0]  # all identical
    ll_narrow = log_likelihood(data, mu=0, sigma=0.5)
    ll_wide = log_likelihood(data, mu=0, sigma=5)
    assert ll_narrow > ll_wide
