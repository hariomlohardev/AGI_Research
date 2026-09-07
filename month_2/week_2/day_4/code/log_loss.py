"""Binary log-loss (mean cross-entropy) + runnable demos for Day 10."""

from collections.abc import Sequence


def binary_log_loss(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Mean binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p)), base 2."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
