"""Cross-entropy and KL divergence from scratch (stdlib only)."""

from collections.abc import Sequence
import math


def _validate_p_q(q,p):
    if len(q) == 0 or len(p) == 0 :
        raise ValueError("both p and q are empty")

    if len(p) != len(q):
        raise ValueError("The length of p and q are not equal")

    if not math.isclose(sum(q) , 1,abs_tol=0.01) or not math.isclose(sum(p) , 1,abs_tol=0.01) :
        raise ValueError("p or q are not sum up of 1 ")

    if any(px < 0 for px in p) or any(qx < 0 for qx in q):
        raise ValueError("All values in p and q must be non-negative.")
    
def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Cross-entropy H(P, Q) = -sum(p(x) * log q(x)) in the given log base (bits by default)."""
    _validate_p_q(q,p)

    Output = 0.0

    for px, qx in zip(p,q):
        if px > 0 :
            if qx == 0:
                raise ValueError("the q(x) contains an value that is equal to 0 where p(x) is more than 0 ")
            Output -= px * math.log(qx ,base)

    return Output


def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """KL divergence D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))."""
    _validate_p_q(q,p)
    
    KL_D = 0.0
    for px, qx in zip(p,q):
        if px > 0 :
            if qx == 0:
                raise ValueError("the q(x) contains an value that is equal to 0 where p(x) is more than 0 ")

            KL_D += px * math.log(px/qx , base)
            
    return KL_D
