"""mle_fit_gaussian — Week 2 Day 1: Maximum Likelihood Estimation.

Two MLE fitters for a Gaussian: closed-form (derived from d/dmu log L = 0)
and numerical grid search (or gradient-ascent if you prefer to reuse Month 1).

Implement without scipy.stats / sklearn.
"""

from __future__ import annotations

from typing import Sequence, Tuple
from log_likelihood import log_likelihood

def mle_fit_gaussian_closed_form(data: Sequence[float]) -> Tuple[float, float]:
    """Return (mu_hat, sigma_hat) via closed-form MLE.

    MLE for Gaussian:
      mu_hat    = mean(data)
      sigma_hat = sqrt(mean((x - mu_hat)^2))   # biased MLE, divides by n

    Do not use np.mean / np.var for the logic — compute the mean/variance
    yourself. Using numpy only to generate synthetic test data is fine.

    Args:
        data: 1-D sequence of observations.

    Returns:
        (mu_hat, sigma_hat) with sigma_hat > 0.

    Raises:
        ValueError: if data is empty.
    """
    if len(data) <= 0:
        raise ValueError("data is empty.")

    mu_hat = sum(data)/ len(data)
    var = sum([(x - mu_hat)**2 for x in data])/len(data)
    sigma_hat = var ** (0.5)

    return (mu_hat, sigma_hat)


def mle_fit_gaussian_grid(
    data: Sequence[float],
    mu_range: Tuple[float, float],
    sigma_range: Tuple[float, float],
    steps: int = 100,
) -> Tuple[float, float]:
    """Return (mu_hat, sigma_hat) via brute-force grid search over log-likelihood.

    Evaluate log_likelihood at each point of a steps x steps grid spanning
    mu_range x sigma_range and return the maximising pair. You must call
    your own log_likelihood from code.log_likelihood — no external optimizer.

    Args:
        data: observations.
        mu_range: (mu_min, mu_max) to search.
        sigma_range: (sigma_min, sigma_max) — both > 0.
        steps: grid resolution per axis.

    Returns:
        (mu_hat, sigma_hat) of the best grid point.
    """
    mu_hat = 0
    sigma_hat = sigma_range[0]
    mu_log_likehd = float('-inf')
    step = (mu_range[1] - mu_range[0])/ steps
    for i in range(steps):
        cureent_mu = mu_range[0] + (i * step)
        log_likehd = log_likelihood(data , cureent_mu ,sigma_hat )
        if mu_log_likehd < log_likehd:
            mu_log_likehd = log_likehd
            mu_hat = cureent_mu

    step = (sigma_range[1] - sigma_range[0])/ steps
    sigma_log_likehd = float('-inf')
    for i in range(steps):
        cureent_sigma = sigma_range[0] + (i * step)
        log_likehd = log_likelihood(data , mu_hat ,cureent_sigma )
        if sigma_log_likehd < log_likehd:
            sigma_log_likehd = log_likehd
            sigma_hat = cureent_sigma

    return (mu_hat ,sigma_hat)
        




    # raise NotImplementedError


# Backwards-compatible alias — some tests/solutions expose a single entry point.
def mle_fit_gaussian(data: Sequence[float]) -> Tuple[float, float]:
    """Alias for mle_fit_gaussian_closed_form (the canonical MLE).

    Keeps coding_problems.md's 'mle_fit_gaussian(data) -> (mu, sigma)'
    wording usable without forcing a grid search.
    """
    return mle_fit_gaussian_closed_form(data)



if __name__ == "__main__":
    print(mle_fit_gaussian_closed_form([1,2,3,4,5,6,6]))
    print(mle_fit_gaussian_grid([1,2,3,4,5,6,6], (1, 5) , (1,5) ,steps=5000))