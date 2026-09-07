"""Cross-entropy and KL divergence from scratch (stdlib only)."""

from collections.abc import Sequence


def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Cross-entropy H(P, Q) = -sum(p(x) * log q(x)) in the given log base (bits by default)."""
    raise NotImplementedError


def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """KL divergence D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))."""
    raise NotImplementedError
