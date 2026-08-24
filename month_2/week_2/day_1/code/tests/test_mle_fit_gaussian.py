"""Tests for mle_fit_gaussian — Week 2 Day 1."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mle_fit_gaussian import mle_fit_gaussian_closed_form, mle_fit_gaussian_grid
from log_likelihood import log_likelihood


def _synthetic_gaussian(mu, sigma, n, seed=0):
    rng = random.Random(seed)
    # Box-Muller for reproducibility without numpy dependency in test itself
    data = []
    while len(data) < n:
        u1 = rng.random()
        u2 = rng.random()
        # avoid log(0)
        u1 = max(u1, 1e-12)
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        data.append(mu + sigma * z0)
        if len(data) < n:
            data.append(mu + sigma * z1)
    return data


def test_closed_form_on_known_data():
    data = [2, 4, 4, 4, 5, 5, 7, 9]  # mean 5.0, known biased sigma
    mu_hat, sigma_hat = mle_fit_gaussian_closed_form(data)
    assert mu_hat == pytest.approx(5.0, rel=1e-9)
    # sigma = sqrt(mean((x-5)^2)) = sqrt((9+1+1+1+0+0+4+16)/8)=sqrt(32/8)=2.0
    assert sigma_hat == pytest.approx(2.0, rel=1e-9)


def test_closed_form_recovers_true_params_approx():
    true_mu, true_sigma = 5.0, 2.0
    data = _synthetic_gaussian(true_mu, true_sigma, n=2000, seed=1)
    mu_hat, sigma_hat = mle_fit_gaussian_closed_form(data)
    assert mu_hat == pytest.approx(true_mu, abs=0.15)
    assert sigma_hat == pytest.approx(true_sigma, abs=0.15)


def test_grid_agrees_with_closed_form():
    true_mu, true_sigma = 5.0, 2.0
    data = _synthetic_gaussian(true_mu, true_sigma, n=500, seed=2)
    mu_cf, sigma_cf = mle_fit_gaussian_closed_form(data)
    mu_grid, sigma_grid = mle_fit_gaussian_grid(data, mu_range=(2, 8), sigma_range=(0.5, 4.0), steps=80)
    assert mu_grid == pytest.approx(mu_cf, abs=0.2)
    assert sigma_grid == pytest.approx(sigma_cf, abs=0.25)


def test_grid_actually_maximises_log_likelihood():
    data = _synthetic_gaussian(0, 1, n=100, seed=3)
    mu_grid, sigma_grid = mle_fit_gaussian_grid(data, mu_range=(-2, 2), sigma_range=(0.5, 2.5), steps=60)
    ll_best = log_likelihood(data, mu_grid, sigma_grid)
    # a clearly off point should be worse
    ll_off = log_likelihood(data, mu_grid + 1.5, sigma_grid)
    assert ll_best > ll_off


def test_raises_on_empty():
    with pytest.raises((ValueError, ZeroDivisionError)):
        mle_fit_gaussian_closed_form([])
