"""Classification metrics from scratch: confusion matrix + accuracy/precision/recall/F1.

New code (not a move): formalise the trio from Week 3 Day 2
(`month_2/week_3/day_2/code/metrics.py` is your starting reference — adapt,
don't blindly copy) and add `f1_score`. Binary `0`/`1` labels only, positive
class `1`. Stdlib only, plain Python lists.
"""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> list[list[int]]:
    """Return `[[TN, FP], [FN, TP]]` (sklearn layout, positive class 1).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    raise NotImplementedError


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the fraction of exactly-matching labels.

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    raise NotImplementedError


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `TP / (TP + FP)`: of the predicted positives, how many were right.

    A model that predicts no positives scores `0.0` (zero-division guard —
    documented convention, not a mathematical fact).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    raise NotImplementedError


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `TP / (TP + FN)`: of the actual positives, how many were found.

    A dataset (or slice) with no actual positives scores `0.0`
    (zero-division guard — documented convention).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    raise NotImplementedError


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the harmonic mean `2PR / (P + R)` of precision and recall.

    Scores `0.0` when `P + R == 0` (zero-division guard — documented
    convention). Harmonic (not arithmetic) mean: it punishes a model that
    is all-precision-no-recall or vice versa.

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    raise NotImplementedError
