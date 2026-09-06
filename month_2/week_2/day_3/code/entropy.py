"""entropy — Week 2 Day 3: Shannon Entropy.

H(X) = -sum_x P(x) * log P(x) — the mathematical definition of "surprise"
or uncertainty. Foundation of every loss function that follows this week
(cross-entropy, KL divergence), and the measure of how compressible
information is.
"""

from __future__ import annotations

import math
from typing import Sequence


def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy of a discrete distribution, in the given log base.

    Args:
        pmf: probabilities, one per outcome. Entries must be >= 0 and sum
            to 1 (within 1e-9). Zero entries are allowed and contribute 0
            (the 0 * log 0 = 0 convention).
        base: log base — 2.0 gives bits (default), e gives nats.

    Returns:
        H = -sum(p * log(p)) in the given base. Always >= 0.

    Raises:
        ValueError: if pmf is empty, any entry is negative, the entries do
            not sum to ~1, or base <= 0.
    """
    raise NotImplementedError
