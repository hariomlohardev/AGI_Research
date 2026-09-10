"""Week 2 probability and information-theory utilities.

Implement each function from scratch using only the standard library. The
file is intentionally a consolidation stub: the assertions in ``__main__``
should pass after the functions are implemented.
"""

from collections.abc import Hashable, Sequence
import math


def bayes_update(
    prior: Sequence[float], likelihood: Sequence[float], evidence: float
) -> list[float]:
    """Return normalized posterior probabilities for multiple hypotheses.

    The update is ``posterior[i] = prior[i] * likelihood[i] / evidence``.

    Raises:
        ValueError: if either sequence is empty, the sequence lengths differ,
            any prior or likelihood is negative, the prior does not sum to 1
            within ``1e-9``, ``evidence <= 0``, or the posterior denominator
            computed from prior and likelihood is zero.
    """
    raise NotImplementedError


def log_likelihood(data: Sequence[float], mu: float, sigma: float) -> float:
    """Return the Gaussian log-likelihood of the observations.

    The observations are modeled as independent draws from ``N(mu, sigma)``.

    Raises:
        ValueError: if ``data`` is empty or ``sigma <= 0``.
    """
    raise NotImplementedError


def mle_fit_gaussian(data: Sequence[float]) -> tuple[float, float]:
    """Return the Gaussian MLE ``(mu_hat, sigma_hat)``.

    The variance MLE divides by ``n`` rather than ``n - 1``. For a one-point
    dataset, the valid Gaussian MLE has ``sigma_hat == 0.0``.

    Raises:
        ValueError: if ``data`` is empty.
    """
    raise NotImplementedError


def map_fit_gaussian(
    data: Sequence[float], prior_mu: float, prior_sigma: float
) -> float:
    """Return the MAP estimate of ``mu`` under a Gaussian prior.

    The likelihood variance is the Gaussian MLE variance, and the prior is
    ``N(prior_mu, prior_sigma)``. The result is a precision-weighted blend of
    the data mean and ``prior_mu``.

    Raises:
        ValueError: if ``data`` is empty or ``prior_sigma <= 0``.
    """
    raise NotImplementedError


def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Return Shannon entropy in the selected logarithm base.

    Raises:
        ValueError: if ``pmf`` is empty, any probability is negative, the
            probabilities do not sum to 1 within ``1e-9``, or ``base <= 0``
            or ``base == 1``.
    """
    raise NotImplementedError


def cross_entropy(
    p: Sequence[float], q: Sequence[float], base: float = 2.0
) -> float:
    """Return cross-entropy ``H(P, Q)`` in the selected logarithm base.

    Raises:
        ValueError: if either input is empty, their lengths differ, an entry
            is negative, either PMF does not sum to 1 within ``1e-9``,
            ``base <= 0`` or ``base == 1``, or ``q[i] == 0`` where
            ``p[i] > 0``.
    """
    raise NotImplementedError


def kl_divergence(
    p: Sequence[float], q: Sequence[float], base: float = 2.0
) -> float:
    """Return ``D_KL(P || Q)`` in the selected logarithm base.

    Raises:
        ValueError: if either input is empty, their lengths differ, an entry
            is negative, either PMF does not sum to 1 within ``1e-9``,
            ``base <= 0`` or ``base == 1``, or ``q[i] == 0`` where
            ``p[i] > 0``.
    """
    raise NotImplementedError


def mutual_information(
    x: Sequence[Hashable], y: Sequence[Hashable], base: float = 2.0
) -> float:
    """Return empirical mutual information for paired discrete observations.

    Labels may be any hashable values. The estimate is computed from joint
    and marginal frequencies of the paired observations.

    Raises:
        ValueError: if either input is empty, their lengths differ, or
            ``base <= 0`` or ``base == 1``. A ``TypeError`` from using an
            unhashable label as a dictionary key is allowed to propagate.
    """
    raise NotImplementedError


if __name__ == "__main__":
    posterior = bayes_update([0.5, 0.5], [0.1, 0.4], evidence=0.25)
    assert math.isclose(sum(posterior), 1.0)
    assert posterior[1] > posterior[0]

    mu_hat, sigma_hat = mle_fit_gaussian([1.0, 2.0, 3.0])
    assert math.isclose(mu_hat, 2.0)
    assert math.isclose(sigma_hat, math.sqrt(2.0 / 3.0))

    weak_map = map_fit_gaussian([1.0, 2.0, 3.0], prior_mu=0.0, prior_sigma=10.0)
    strong_map = map_fit_gaussian([1.0, 2.0, 3.0], prior_mu=0.0, prior_sigma=0.5)
    assert 0.0 < strong_map < 2.0
    assert abs(strong_map) < abs(weak_map)

    assert math.isclose(entropy([0.5, 0.5]), 1.0)

    p = [0.5, 0.5]
    q = [0.9, 0.1]
    assert math.isclose(
        cross_entropy(p, q) - entropy(p), kl_divergence(p, q), abs_tol=1e-12
    )

    assert math.isclose(mutual_information([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
    assert math.isclose(mutual_information([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)
