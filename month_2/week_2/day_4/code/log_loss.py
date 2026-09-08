"""Binary log-loss (mean cross-entropy) + runnable demos for Day 10."""

from collections.abc import Sequence
from math import log2

def binary_log_loss(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Mean binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p)), base 2."""

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is not equal to the length of y_true")

    if any(y not in (0, 1) for y in y_true):        
        raise ValueError("label should be 1")

    if len(y_true) == 0:
        raise ValueError("the length of label is 0")
    
    b_log_loss_list = []
    for y , p in zip(y_true, y_pred):
        sol = y*log2(p) + (1-y) * log2(1-p)
        b_log_loss_list.append(sol)

    return - sum(b_log_loss_list) / len(b_log_loss_list)


if __name__ == "__main__":
    raise NotImplementedError
