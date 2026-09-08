"""Mutual information from scratch (stdlib only)."""

from collections.abc import Hashable, Sequence
import math


def mutual_information(x: Sequence[Hashable], y: Sequence[Hashable], base: float = 2.0) -> float:
    """Mutual information I(X; Y) in the given log base (bits by default).

    Estimated empirically: joint and marginal probabilities are counted
    directly from the paired samples, then
    I(X; Y) = sum over observed (a, b) of P(a, b) * log(P(a, b) / (P(a) * P(b))).
    Labels may be any hashable type (ints, strings, pre-binned values).

    Raises:
        ValueError: if `x` or `y` is empty.
        ValueError: if `x` and `y` have different lengths (positions pair up).
        ValueError: if `base <= 0` or `base == 1` (a base-1 log is division by zero).
    """
    raise NotImplementedError
