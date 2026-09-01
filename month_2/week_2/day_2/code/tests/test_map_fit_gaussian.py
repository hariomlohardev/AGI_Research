"""Tests for map_fit_gaussian — Week 2 Day 2."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from log_likelihood import log_likelihood
from map_fit_gaussian import (
    log_posterior,
    map_fit_gaussian_closed_form,
    map_fit_gaussian_grid,
)


def _synthetic_gaussian(mu, sigma, n, seed=0):
    rng = random.Random(seed)
    data = []
    while len(data) < n:
        u1 = rng.random()
        u2 = rng.random()
        u1 = max(u1, 1e-12)
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        data.append(mu + sigma * z0)
        if len(data) < n:
            data.append(mu + sigma * z1)
    return data


def test_closed_form_weighted_average_on_known_data():
    # Known data with mean 5.0; with moderate prior should pull toward prior
    data = [2, 4, 4, 4, 5, 5, 7, 9]  # mean 5.0
    prior_mu, prior_sigma = 0.0, 1.0
    # Manually compute expected weighted average
    n = len(data)
    mu_mle = sum(data) / n  # 5.0
    sigma2_mle = sum((x - mu_mle) ** 2 for x in data) / n  # 4.0
    expected_mu_map = (n * mu_mle / sigma2_mle + prior_mu / prior_sigma**2) / (
        n / sigma2_mle + 1 / prior_sigma**2
    )
    mu_map, sigma_map = map_fit_gaussian_closed_form(data, prior_mu, prior_sigma)
    assert mu_map == pytest.approx(expected_mu_map, rel=1e-9)
    # sigma_map should be sigma_mle (sqrt 4 =2)
    assert sigma_map == pytest.approx(math.sqrt(sigma2_mle), rel=1e-9)


def test_closed_form_weak_prior_approaches_mle():
    data = _synthetic_gaussian(3.0, 1.0, n=50, seed=10)
    prior_mu = 100.0  # absurdly far prior
    # Very weak prior — huge sigma — should barely move from MLE
    mu_map_weak, _ = map_fit_gaussian_closed_form(data, prior_mu, prior_sigma=1e6)
    mu_mle = sum(data) / len(data)
    assert mu_map_weak == pytest.approx(mu_mle, abs=0.01)


def test_closed_form_strong_prior_approaches_prior():
    data = _synthetic_gaussian(0.0, 1.0, n=10, seed=11)
    prior_mu = 5.0
    # Very strong prior — tiny sigma — should be near prior
    mu_map_strong, _ = map_fit_gaussian_closed_form(data, prior_mu, prior_sigma=0.1)
    assert mu_map_strong == pytest.approx(prior_mu, abs=0.5)


def test_grid_agrees_with_closed_form():
    data = _synthetic_gaussian(5.0, 2.0, n=300, seed=12)
    prior_mu, prior_sigma = 0.0, 2.0
    mu_cf, sigma_cf = map_fit_gaussian_closed_form(data, prior_mu, prior_sigma)
    mu_grid, sigma_grid = map_fit_gaussian_grid(
        data, prior_mu, prior_sigma, mu_range=(-1, 8), sigma_range=(0.5, 4.0), steps=80
    )
    assert mu_grid == pytest.approx(mu_cf, abs=0.2)
    assert sigma_grid == pytest.approx(sigma_cf, abs=0.3)


def test_grid_actually_maximises_posterior():
    data = _synthetic_gaussian(0, 1, n=80, seed=13)
    prior_mu, prior_sigma = 0.0, 1.0
    mu_grid, sigma_grid = map_fit_gaussian_grid(
        data, prior_mu, prior_sigma, mu_range=(-2, 2), sigma_range=(0.5, 2.5), steps=60
    )
    best = log_posterior(data, mu_grid, sigma_grid, prior_mu, prior_sigma)
    off = log_posterior(data, mu_grid + 1.5, sigma_grid, prior_mu, prior_sigma)
    assert best > off


def test_prior_sweep_monotonic():
    """MAP estimate must move monotonically from MLE toward prior as prior tightens."""
    data = [0.5, -0.3, 0.2]  # mean ~0.13
    prior_mu = 5.0
    sigmas = [10.0, 5.0, 2.0, 1.0, 0.5]
    mus = [map_fit_gaussian_closed_form(data, prior_mu, s)[0] for s in sigmas]
    # As sigma shrinks, mu_map should move *up* toward 5.0
    for i in range(1, len(mus)):
        assert mus[i] >= mus[i - 1] - 1e-9, f"not monotonic: {mus}"
    #Endpoints sanity
    assert mus[0] == pytest.approx(sum(data) / len(data), abs=0.1)
    assert mus[-1] == pytest.approx(prior_mu, abs=0.8)


def test_map_more_robust_than_mle_on_outliers():
    """Classic 3-point outlier demo: MAP sits between MLE mean and prior."""
    data = [8.0, 9.0, 7.0]
    prior_mu, prior_sigma = 0.0, 1.0
    mu_mle = sum(data) / len(data)  # 8.0
    mu_map, _ = map_fit_gaussian_closed_form(data, prior_mu, prior_sigma)
    # MAP should be between prior (0) and MLE (8)
    assert 0 < mu_map < mu_mle
    # And strictly less than MLE (pulled toward prior)
    assert mu_map < mu_mle


def test_raises_on_empty_and_bad_prior():
    with pytest.raises(ValueError):
        map_fit_gaussian_closed_form([], prior_mu=0, prior_sigma=1)
    with pytest.raises(ValueError):
        map_fit_gaussian_closed_form([1, 2], prior_mu=0, prior_sigma=0)
    with pytest.raises(ValueError):
        map_fit_gaussian_closed_form([1, 2], prior_mu=0, prior_sigma=-1)
