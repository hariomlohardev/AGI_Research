"""Mutual information from scratch (stdlib only)."""

from collections.abc import Hashable, Sequence
import math
from typing import Counter

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
    if len(x) == 0 or len(y) == 0:
        raise ValueError("the size of either x or y is 0")
    if len(x) != len(y):
        raise ValueError("`x` and `y` have different lengths , the fuction requre equal length")
    if base <= 0 or base == 1:
        raise ValueError(f"the base is {base} that the fuction doesn't accept")

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
    x ,y = [1,2,1] , [1,1,1]
    # x ,y = ['a', 'a', 'b', 'b'] , ['u', 'u', 'v', 'v']
    MI = mutual_information(x,y)
    print(MI)





