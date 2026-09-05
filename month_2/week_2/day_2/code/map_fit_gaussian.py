"""map_fit_gaussian — Week 2 Day 2: MAP Estimation & Priors.

Build on Week 2 Day 1's log_likelihood.  MAP adds a Gaussian prior on mu:
  posterior ∝ likelihood × prior
  log_posterior = log_likelihood + log_prior + const

A Gaussian prior N(prior_mu, prior_sigma) yields an L2 penalty term
  -0.5*((mu - prior_mu)/prior_sigma)^2
which is exactly L2 / weight-decay when prior_mu == 0
  lambda = 1 / (2 * prior_sigma^2).

The math you derive on paper (see learn.md) is:
  mu_map = (n*mu_mle/sigma^2 + prior_mu/prior_sigma^2)
           / (n/sigma^2 + 1/prior_sigma^2)
with sigma treated as sigma_mle (prior only on mu — the focused case).
"""

from __future__ import annotations

from typing import Sequence
import math
from log_likelihood import log_likelihood
import numpy as np


def log_prior_gaussian(mu: float, prior_mu: float, prior_sigma: float) -> float:
    """Log PDF of mu under N(prior_mu, prior_sigma).

    Args:
        mu: value at which to evaluate the prior density.
        prior_mu: mean of the Gaussian prior.
        prior_sigma: std of the prior — must be > 0.

    Returns:
        log N(mu | prior_mu, prior_sigma).

    Raises:
        ValueError: if prior_sigma <= 0.
    """
    if prior_sigma <=0 : raise ValueError
    log_prior =  (-(0.5) * math.log(2*math.pi *( prior_sigma **2))) - ((mu - prior_mu)**2 /(2*(prior_sigma**2)))

    return log_prior


def log_posterior(
    data: Sequence[float],
    mu: float,
    sigma: float,
    prior_mu: float,
    prior_sigma: float,
) -> float:
    """Log-posterior (up to an additive constant) for a Gaussian likelihood
    with a Gaussian prior on mu.

    log_posterior = log_likelihood(data | mu, sigma) + log_prior_gaussian(mu)

    Args:
        data: 1-D observations.
        mu: likelihood mean hypothesis.
        sigma: likelihood std — must be > 0 (delegated to log_likelihood).
        prior_mu: prior mean on mu.
        prior_sigma: prior std on mu — must be > 0.

    Returns:
        Sum log-likelihood plus log-prior (constant terms may be omitted
        as long as argmax is preserved, but the full log_prior is clearer).

    Raises:
        ValueError: if sigma <= 0 or prior_sigma <= 0 or data empty (via
            log_likelihood).
    """
    if sigma <= 0 or prior_sigma  <=0 or (len(data)==0):
        raise ValueError
    
    Output = log_likelihood(data , mu , sigma) + log_prior_gaussian(mu , prior_mu, prior_sigma)
    return Output

def map_fit_gaussian_closed_form(
    data: Sequence[float],
    prior_mu: float,
    prior_sigma: float,
) -> tuple[float, float]:
    """MAP estimate for (mu, sigma) with Gaussian prior N(prior_mu, prior_sigma) on mu.

    Uses the closed-form weighted-average for mu_map (derive on paper from
    d/dmu log_posterior = 0):

        n = len(data)
        mu_mle = mean(data)
        sigma2_mle = mean((x - mu_mle)^2)
        mu_map = (n*mu_mle/sigma2_mle + prior_mu/prior_sigma^2)
                 / (n/sigma2_mle + 1/prior_sigma^2)

    Sigma is returned as sigma_mle (prior only pulls mu — the focused case
    in this day's derivation).

    Args:
        data: 1-D observations (non-empty).
        prior_mu: prior mean on mu.
        prior_sigma: prior std on mu — must be > 0.

    Returns:
        (mu_map, sigma_map) with sigma_map == sigma_mle.

    Raises:
        ValueError: if data empty or prior_sigma <= 0.
    """

    if len(data) == 0 or prior_sigma <= 0 : raise ValueError
    n = len(data)
    mu_mle = sum(data)/n
    sigma_mle = math.sqrt(sum([(x - mu_mle)**2 for x in data])/n)

    numerator =  (n * mu_mle/sigma_mle ** 2) + (prior_mu/prior_sigma ** 2)
    denomenator = (n/sigma_mle ** 2) + (1/ prior_sigma ** 2)
    mu_map = numerator / denomenator

    return (mu_map, sigma_mle)

def map_fit_gaussian_grid(
    data: Sequence[float],
    prior_mu: float,
    prior_sigma: float,
    mu_range: tuple[float, float],
    sigma_range: tuple[float, float],
    steps: int = 100,
) -> tuple[float, float]:
    """Brute-force MAP via grid search over the log-posterior.

    Evaluates log_posterior on a steps x steps grid spanning
    mu_range x sigma_range and returns the (mu, sigma) with highest posterior.

    Args:
        data: 1-D observations.
        prior_mu: prior mean on mu.
        prior_sigma: prior std on mu — must be > 0.
        mu_range: (mu_min, mu_max) inclusive.
        sigma_range: (sigma_min, sigma_max) with sigma_min > 0.
        steps: grid resolution per axis.

    Returns:
        (mu_hat, sigma_hat) maximising log_posterior on the grid.

    Raises:
        ValueError: if prior_sigma <= 0 or data empty.
    """
    if prior_sigma <= 0 or len(data) == 0 : raise ValueError
    n = len(data)


    mu_step =  (mu_range[1] - mu_range[0])/steps
    sigma_step =  (sigma_range[1] - sigma_range[0])/steps
    best_log_posterior = float('-inf')
    mu_hat = 0
    sigma_hat = 0 
    

    for mu in np.arange(mu_range[0], mu_range[1], mu_step):
        for sigma in np.arange(sigma_range[0], sigma_range[1], sigma_step):
            new_log_posterior = log_posterior(data , mu , sigma , prior_mu ,prior_sigma)
            if new_log_posterior > best_log_posterior:
                best_log_posterior = new_log_posterior
                mu_hat = mu
                sigma_hat = sigma

    
    return (mu_hat, sigma_hat)


# Convenience alias — so `from map_fit_gaussian import map_fit_gaussian` works
# (mirrors the mle_fit_gaussian alias from Day 1).
map_fit_gaussian = map_fit_gaussian_closed_form

# ---------------------------------------------------------------------------
# L1 vs L2 thought experiment — leave your prediction here (coding_problems.md
# Problem 3.4 / practice question 4).  Tests do not grade this, but /done
# will ask you about it.
# ---------------------------------------------------------------------------
# (student): If the prior were Laplace (double-exponential)
#   P(mu) ∝ exp(-|mu - prior_mu| / b)
# then log_prior ∝ -|mu - prior_mu|/b — an L1 penalty.  Predict:
#   How would the estimate differ vs the Gaussian (L2) prior?  (Hint: think
#   sparser / exact-zero shrinkage vs smooth pull.)
#   Write your 1-2 sentence prediction below:
# flat prior check 
# if the prior_sigma -> infinity then the mu_map = mu_mle because the 1/prior_sigma ** 2 = 0 

## L1 vs L2
# the L1 is |x-mu|/b
# the L2 is (x-mu)**2 / b
# the L2 is squared ones and punishes the big mistakes much harder than L1  by sqaring thigns 

if __name__ == "__main__":
    # Required experiments — keep these runnable (coding_problems.md §3):
    # 1) 3-point outlier MLE vs MAP with prior N(0,1)
    data_outliers = [8, 9, 7]
    mu_mle = sum(data_outliers) / len(data_outliers)
    mu_map, _ = map_fit_gaussian_closed_form(data_outliers, prior_mu=0, prior_sigma=1)
    print(f"mu_mle = {mu_mle}")   # 8.0 — trusts the outliers completely
    print(f"mu_map = {mu_map}")   # pulled toward 0, but not all the way
    # 2) prior_sigma sweep 10.0 -> 0.5 with fixed data and prior_mu=5
    data = [0.5, -0.3, 0.2]
    prior_mu = 5.0
    sigmas = [10, 5, 2, 1, 0.5, 0.1, 0.03]
    for s in sigmas:
        mu_map, _ = map_fit_gaussian_closed_form(data, prior_mu, s)
        print(f"prior_sigma={s:>5} -> mu_map={mu_map:.3f}")
 
