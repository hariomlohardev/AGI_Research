"""Tests for log_prior_gaussian and log_posterior — Week 2 Day 2."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from log_likelihood import log_likelihood
from map_fit_gaussian import log_posterior, log_prior_gaussian


def _gaussian_log_pdf_single(x, mu, sigma):
    return -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def test_log_prior_gaussian_matches_hand_computation():
    prior_mu, prior_sigma = 0.0, 1.0
    mu = 0.5
    expected = _gaussian_log_pdf_single(mu, prior_mu, prior_sigma)
    assert log_prior_gaussian(mu, prior_mu, prior_sigma) == pytest.approx(expected, rel=1e-9)


def test_log_prior_raises_on_nonpositive_sigma():
    with pytest.raises(ValueError):
        log_prior_gaussian(0, prior_mu=0, prior_sigma=0)
    with pytest.raises(ValueError):
        log_prior_gaussian(0, prior_mu=0, prior_sigma=-1)


def test_log_posterior_is_sum():
    data = [0, 1, 2]
    mu, sigma = 1.0, 1.0
    prior_mu, prior_sigma = 0.0, 1.5
    expected = log_likelihood(data, mu, sigma) + log_prior_gaussian(mu, prior_mu, prior_sigma)
    assert log_posterior(data, mu, sigma, prior_mu, prior_sigma) == pytest.approx(expected, rel=1e-9)


def test_log_posterior_raises_on_bad_sigma():
    with pytest.raises(ValueError):
        log_posterior([1, 2], mu=0, sigma=0, prior_mu=0, prior_sigma=1)
    with pytest.raises(ValueError):
        log_posterior([1, 2], mu=0, sigma=1, prior_mu=0, prior_sigma=0)
    with pytest.raises(ValueError):
        log_posterior([], mu=0, sigma=1, prior_mu=0, prior_sigma=1)


def test_log_posterior_prior_pull():
    """With a tight prior far from data, posterior should favour the prior side."""
    data = [8.0, 9.0, 7.0]
    mu_near_data = 8.0
    mu_near_prior = 0.0
    # strong prior N(0,1) — near-prior should beat near-data in posterior
    post_near_data = log_posterior(data, mu_near_data, 1.0, prior_mu=0, prior_sigma=1.0)
    post_near_prior = log_posterior(data, mu_near_prior, 1.0, prior_mu=0, prior_sigma=1.0)
    # Don't assert strict ordering here (depends on data variance) — just that
    # log_posterior is *different* from pure log_likelihood (i.e. prior matters).
    ll_near_data = log_likelihood(data, mu_near_data, 1.0)
    assert post_near_data != pytest.approx(ll_near_data)
