"""Tests for the Week 2 consolidation utilities."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from week2_utils import (
    bayes_update,
    cross_entropy,
    entropy,
    kl_divergence,
    log_likelihood,
    map_fit_gaussian,
    mle_fit_gaussian,
    mutual_information,
)


def test_bayes_update_normalizes_and_favors_larger_likelihood():
    posterior = bayes_update([0.5, 0.5], [0.1, 0.4], evidence=0.25)
    assert posterior == pytest.approx([0.2, 0.8])
    assert sum(posterior) == pytest.approx(1.0)


def test_bayes_update_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        bayes_update([], [], 1.0)
    with pytest.raises(ValueError):
        bayes_update([1.0], [1.0, 1.0], 1.0)
    with pytest.raises(ValueError):
        bayes_update([1.1, -0.1], [1.0, 1.0], 1.0)
    with pytest.raises(ValueError):
        bayes_update([0.5, 0.4], [1.0, 1.0], 0.9)
    with pytest.raises(ValueError):
        bayes_update([0.5, 0.5], [1.0, 1.0], 0.0)
    with pytest.raises(ValueError):
        bayes_update([0.5, 0.5], [0.0, 0.0], 1.0)


def test_gaussian_log_likelihood_matches_hand_value():
    expected = sum(
        -0.5 * math.log(2 * math.pi) - 0.5 * (x - 1.0) ** 2
        for x in [0.0, 1.0, 2.0]
    )
    assert log_likelihood([0.0, 1.0, 2.0], 1.0, 1.0) == pytest.approx(expected)


def test_gaussian_log_likelihood_rejects_empty_or_nonpositive_sigma():
    with pytest.raises(ValueError):
        log_likelihood([], 0.0, 1.0)
    with pytest.raises(ValueError):
        log_likelihood([1.0], 0.0, 0.0)


def test_mle_uses_population_variance():
    mu_hat, sigma_hat = mle_fit_gaussian([1.0, 2.0, 3.0])
    assert mu_hat == pytest.approx(2.0)
    assert sigma_hat == pytest.approx(math.sqrt(2.0 / 3.0))
    assert mle_fit_gaussian([4.0]) == pytest.approx((4.0, 0.0))


def test_mle_rejects_empty_data():
    with pytest.raises(ValueError):
        mle_fit_gaussian([])


def test_map_is_between_data_mean_and_prior_and_prior_strength_matters():
    data_mean = sum([1.0, 2.0, 3.0]) / 3
    weak = map_fit_gaussian([1.0, 2.0, 3.0], prior_mu=0.0, prior_sigma=10.0)
    strong = map_fit_gaussian([1.0, 2.0, 3.0], prior_mu=0.0, prior_sigma=0.5)
    assert 0.0 < strong < data_mean
    assert abs(strong) < abs(weak)


def test_map_rejects_empty_data_or_nonpositive_prior_sigma():
    with pytest.raises(ValueError):
        map_fit_gaussian([], 0.0, 1.0)
    with pytest.raises(ValueError):
        map_fit_gaussian([1.0], 0.0, 0.0)


def test_entropy_and_distribution_boundaries():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        entropy([], base=2.0)
    with pytest.raises(ValueError):
        entropy([0.5, -0.1, 0.6])
    with pytest.raises(ValueError):
        entropy([0.5, 0.4])
    with pytest.raises(ValueError):
        entropy([0.5, 0.5], base=1.0)


def test_cross_entropy_kl_identity_and_zero_support():
    p = [0.5, 0.5]
    q = [0.9, 0.1]
    assert cross_entropy(p, q) - entropy(p) == pytest.approx(kl_divergence(p, q))
    assert cross_entropy([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cross_entropy(p, [1.0, 0.0])
    with pytest.raises(ValueError):
        kl_divergence(p, [1.0, 0.0])


def test_cross_entropy_and_kl_reject_bad_distributions():
    with pytest.raises(ValueError):
        cross_entropy([], [])
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        cross_entropy([-0.1, 1.1], [0.5, 0.5])
    with pytest.raises(ValueError):
        cross_entropy([0.5, 0.5], [0.5, 0.5], base=1.0)


def test_mutual_information_core_values_and_boundaries():
    assert mutual_information([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    assert mutual_information(["a", "a"], ["x", "x"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mutual_information([], [])
    with pytest.raises(ValueError):
        mutual_information([0], [0, 1])
    with pytest.raises(ValueError):
        mutual_information([0], [0], base=1.0)


# The final two functions are intentionally imported above so a completed
# implementation is checked as one consolidated public API.
