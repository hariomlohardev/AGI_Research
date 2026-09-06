"""entropy — Week 2 Day 3: Shannon Entropy.

Shannon entropy H = -sum(p * log(p)) is the expected "surprise" of a draw
from a discrete distribution. Default base 2 reports bits; base e reports nats.
"""

from __future__ import annotations

from typing import Sequence


def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy of a discrete probability mass function.

    Args:
        pmf: outcome probabilities; must sum to 1 (within 1e-9).
        base: log base — 2.0 for bits (default), math.e for nats.

    Returns:
        H = -sum(p * log_base(p)), with the convention 0 * log(0) = 0.

    Raises:
        ValueError: if pmf is empty, contains a negative probability,
            does not sum to 1, or base <= 0.
        NotImplementedError: stub — implement this (Problem 1).
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Required anchors — keep these runnable (coding_problems.md Problem 1):
    # 1) entropy([0.5, 0.5]) should print exactly 1.0 (one bit: a fair coin)
    # 2) entropy of a fair 6-sided die should print log2(6) ~= 2.58496
    # 3) entropy([0.99, 0.01]) should print something small (< 0.1): barely any surprise
    pass
