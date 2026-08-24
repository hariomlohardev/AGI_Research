"""log_likelihood — Week 2 Day 1: Maximum Likelihood Estimation.

Implement the log-likelihood of data under a Gaussian.
Do not use scipy.stats or sklearn — implement the Gaussian PDF yourself.
"""

from __future__ import annotations

import math
from typing import Sequence


def log_likelihood(data: Sequence[float], mu: float, sigma: float) -> float:
    """Return the log-likelihood of *data* under N(mu, sigma).

    Args:
        data: 1-D sequence of observations (list or np.ndarray).
        mu: mean of the Gaussian.
        sigma: standard deviation — must be > 0.

    Returns:
        Sum over i of log N(x_i | mu, sigma), where
        N(x|mu,sigma) = 1/(sigma*sqrt(2*pi)) * exp(-0.5*((x-mu)/sigma)**2).

    Raises:
        ValueError: if sigma <= 0 or data is empty.

    Example:
        >>> log_likelihood([0, 1, 2], mu=1.0, sigma=1.0)  # doctest: +SKIP
        -3.75...  # compute by hand to verify
    """
    
    # type checking 
    if sigma <= 0 :
        raise ValueError("sigma cant be lessthan or equal to 0")
    if len(data) <= 0:
        raise ValueError("data is empy")


    output = 0
    std = sigma ** 2
    fst  = -0.5 * math.log(2*math.pi*std)
    for i in range(len(data)):
        scd  = -((data[i] - mu) ** 2 / (2 * (std)))
        result  = fst + scd
        output += result

    return output


if __name__ == "__main__":
    print(log_likelihood(data=[0,1,2], mu=1, sigma=1))