"""Shannon entropy of a probability mass function.

This module provides a function to compute the Shannon entropy (in bits) of a
discrete probability distribution given as a mapping from outcomes to
probabilities.
"""

from __future__ import annotations

import math
from typing import Dict, Union


def entropy(pmf: Dict[Union[str, int, float], float]) -> float:
    """Compute the Shannon entropy H = -∑ p_i * log2(p_i).

    Args:
        pmf: A dictionary where keys are outcomes and values are their
             probabilities. Probabilities should be non-negative and sum to
             approximately 1 (within a small tolerance). Zero probabilities
             contribute 0 to the sum (by convention 0 * log2(0) = 0).

    Returns:
        The entropy in bits (float).

    Raises:
        ValueError: If any probability is negative or if the sum of
                    probabilities is not approximately 1.
        ValueError: If the pmf is empty.
    """
    if not pmf:
        raise ValueError("pmf must not be empty")

    total = sum(pmf.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"probabilities must sum to 1 (got {total})")

    # Check for negative probabilities
    for prob in pmf.values():
        if prob < 0:
            raise ValueError("probabilities must be non-negative")

    # Compute entropy, skipping zero probabilities to avoid log(0)
    ent = 0.0
    for prob in pmf.values():
        if prob > 0:  # log2(0) is -inf, but we skip as term is 0
            ent -= prob * math.log2(prob)
    return ent