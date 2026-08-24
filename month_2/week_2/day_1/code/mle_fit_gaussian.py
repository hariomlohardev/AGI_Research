"""mle_fit_gaussian — Week 2 Day 1: Maximum Likelihood Estimation.

Two MLE fitters for a Gaussian: closed-form (derived from d/dmu log L = 0)
and numerical grid search (or gradient-ascent if you prefer to reuse Month 1).

Implement without scipy.stats / sklearn.
"""

from __future__ import annotations

from typing import Sequence, Tuple


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
    raise NotImplementedError


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
    raise NotImplementedError


# Backwards-compatible alias — some tests/solutions expose a single entry point.
def mle_fit_gaussian(data: Sequence[float]) -> Tuple[float, float]:
    """Alias for mle_fit_gaussian_closed_form (the canonical MLE).

    Keeps coding_problems.md's 'mle_fit_gaussian(data) -> (mu, sigma)'
    wording usable without forcing a grid search.
    """
    return mle_fit_gaussian_closed_form(data)
