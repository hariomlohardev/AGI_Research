"""log_likelihood helper for Day 2 — correct reference.

This is a clean copy of the Day-1 log_likelihood so that Day-2's
map_fit_gaussian can import it without depending on Day-1's file path.
"""

from __future__ import annotations

import math
from typing import Sequence


def log_likelihood(data: Sequence[float], mu: float, sigma: float) -> float:
    """Return the log-likelihood of *data* under N(mu, sigma).

    Raises:
        ValueError: if sigma <= 0 or data is empty.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if len(data) == 0:
        raise ValueError("data must not be empty")
    var = sigma * sigma
    const_term = -0.5 * math.log(2 * math.pi * var)
    total = 0.0
    for x in data:
        total += const_term - ((x - mu) ** 2) / (2 * var)
    return total
