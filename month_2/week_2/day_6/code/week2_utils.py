"""Week 2 probability and information-theory utilities.

Implement each function from scratch using only the standard library. The
file is intentionally a consolidation stub: the assertions in ``__main__``
should pass after the functions are implemented.
"""

from collections.abc import Hashable, Sequence
from collections import Counter
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
    

    if len(prior) == 0 or len(likelihood) == 0 or len(likelihood) == 0:
        raise ValueError("the length of eigther 'prior' , 'likelihood' or 'evidence' is 0")
    
    if not len(prior) == len(likelihood):
        raise ValueError("The length of the 'prior' , 'likelihood' and 'evidence' is not equal")

    if any(p < 0 for p in prior) :
        raise ValueError("an valude of the prior is negative")
    
    if any(lh <= 0 for lh in likelihood) :
        raise ValueError("an valude of the likelihood is negative")
    
    if not math.isclose(1 ,sum(prior) , rel_tol=1e-9):
        raise ValueError("the prior doesnot sum up to 1")

    if evidence <= 0:
        raise ValueError("evedence is either 0 or negative ")

    posterior = []
    for i in range(len(prior)):
        posterior.append((prior[i] * likelihood[i])/evidence)

    return posterior



def log_likelihood(data: Sequence[float], mu: float, sigma: float) -> float:
    """Return the Gaussian log-likelihood of the observations.

    The observations are modeled as independent draws from ``N(mu, sigma)``.

    Raises:
        ValueError: if ``data`` is empty or ``sigma <= 0``.
    """
    if len(data) == 0:
        raise ValueError("the data is empty")
    
    if sigma <= 0:
        raise ValueError(f"the Standerd Deviation is either 0 or negative")

    output = 0
    for x in data:
        output += -0.5*math.log(2*math.pi*sigma**2 ) - (x-mu)**2/(2*sigma**2)

    return output


def mle_fit_gaussian(data: Sequence[float]) -> tuple[float, float]:
    """Return the Gaussian MLE ``(mu_hat, sigma_hat)``.

    The variance MLE divides by ``n`` rather than ``n - 1``. For a one-point
    dataset, the valid Gaussian MLE has ``sigma_hat == 0.0``.

    Raises:
        ValueError: if ``data`` is empty.
    """
    if len(data) == 0:
        raise ValueError("the data is empty")

    n = len(data)

    mu_hat = sum(data)/n
    sigma_hat = math.sqrt(sum((x-mu_hat)** 2 for x in data)/n)

    return (mu_hat , sigma_hat)


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
    if len(data) == 0:
        raise ValueError("the data is empty")

    if prior_sigma <= 0 :
        raise ValueError("the prior is either 0 or negative")

    n = len(data)
    s_mu , s_sigma = mle_fit_gaussian(data)
    mu = (n*s_mu / s_sigma**2 + prior_mu/prior_sigma ** 2) / (n/s_sigma **2  + 1/prior_sigma **2)
    return mu


def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Return Shannon entropy in the selected logarithm base.

    Raises:
        ValueError: if ``pmf`` is empty, any probability is negative, the
            probabilities do not sum to 1 within ``1e-9``, or ``base <= 0``
            or ``base == 1``.
    """
    if len(pmf) == 0 :
        raise ValueError("the PMF is empty")
    if any(p < 0 for p in pmf):
        raise ValueError("either 1 or more prorbabilites are negative")
    if not math.isclose(1.0, sum(pmf), abs_tol=1e-9) :
        raise ValueError("the probabilities doesnot sum up to 1")
    if base <= 0 or base == 1:
        raise ValueError("the base can't be 0 , negative or 1")

    output = 0
    for p in pmf:
        if p != 0:
            output -= p * math.log(p , base)
    return output


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
    if len(p) == 0 or len(q) == 0:
        raise ValueError("either p or q is empty")
    
    if not len(p) == len(q):
        raise ValueError("the length of p and q should be same")

    if any(_p < 0 for _p in p):
        raise ValueError("1 or more values in the p is negative")
    
    if any(_q < 0 for _q in q):
        raise ValueError("1 or more values in the q is negative")
    
    if not math.isclose(1 ,sum(p) ,abs_tol=1e-9):
        raise ValueError("the sum of p doesnot add up to 1")
    
    if not math.isclose(1 ,sum(q) ,abs_tol=1e-9):
        raise ValueError("the sum of q doesnot add up to 1")

    if base <= 0 or base == 1:
        raise ValueError("the base is can't be 0 , negative or 1")

    output = 0
    for i in range(len(p)):
        if q[i] == 0 :
            if p[i] > 0 :
                raise ValueError("``q[i] == 0`` where ``p[i] > 0``")
        else:
            output -= p[i] * math.log(q[i] , base)

    return output


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

    _entropy = entropy(p,base)
    _cross_entropy = cross_entropy(p,q,base)
    return _cross_entropy - _entropy 


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
    if len(x) == 0 or len(y) == 0:
        raise ValueError("either x or y is empty")
    
    if not len(x) == len(y):
        raise ValueError("the length of x and y should be same")

    if base <= 0 or base == 1:
        raise ValueError("the base is can't be 0 , negative or 1")

    if not isinstance(x, Sequence) or not isinstance(y, Sequence):
        raise TypeError("x and y must be sequences (e.g., lists, tuples)")
        
    if not all(isinstance(item, Hashable) for item in x) or not all(isinstance(item, Hashable) for item in y):
        raise TypeError("the elements inside x and y must be hashable")

    
    n = len(x)

    count_x = Counter(x)
    count_y = Counter(y)

    pairs  = ((x[i] , y[i]) for i in range(len(x)))
    pairs_count = Counter(pairs)

    MI = 0
    for a in count_x:
        for b in count_y:
            joint_count = pairs_count[(a,b)]

            if joint_count > 0:

                p_xy = joint_count/ n
                p_x = count_x[a] / n
                p_y  = count_y[b] / n
                MI += p_xy * math.log(p_xy/(p_x*p_y) , base)

    return MI


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
