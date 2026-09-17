"""Classification metrics computed by hand — no libraries, not even ``math``.

Labels are binary ints in ``{0, 1}``. All scores build on
:func:`confusion_matrix` so the counting lives in exactly one place.
"""

from __future__ import annotations

from logistic_regression import is_empty
from collections.abc import Sequence


def _error_checks(y_true,y_pred):
    if is_empty(y_true):
        raise ValueError("y_true can't be empty")
    
    if is_empty(y_pred):
        raise ValueError("y_pred can't be empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_true and y_pred is different")

    if not all(y_i in (0,1) for y_i in y_true):
        raise ValueError("one or more label is not in ``{0, 1}``")

def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` counts.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """

    _error_checks(y_true,y_pred)


    tp, fp, tn, fn = 0, 0, 0, 0

    for y_true_i , y_pred_i in zip(y_true,y_pred):
        if y_true_i == 0 and y_pred_i == 0:
            tn += 1            
        elif y_true_i == 0 and y_pred_i == 1:
            fp += 1
        elif y_true_i == 1 and y_pred_i == 0:
            fn += 1
        elif y_true_i == 1 and y_pred_i == 1:
            tp += 1

    return (tp, fp, tn, fn)


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``(tp + tn) / n``.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    _error_checks(y_true,y_pred)

    n = len(y_pred)

    tp , tn = 0, 0

    for y_true_i , y_pred_i in zip(y_true,y_pred):
        if y_true_i == 0 and y_pred_i == 0:
            tn += 1
        elif y_true_i == 1 and y_pred_i == 1:
            tp += 1


    return (tp + tn)/n



def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``tp / (tp + fp)``.

    Returns ``0.0`` when the denominator is 0 (no predicted positives) —
    a documented convention, not a mathematical fact.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """

    _error_checks(y_true,y_pred)
    tp , fp = 0, 0

    for y_true_i , y_pred_i in zip(y_true,y_pred):     
        if y_true_i == 0 and y_pred_i == 1:
            fp += 1
        elif y_true_i == 1 and y_pred_i == 1:
            tp += 1

    denomenator = tp + fp

    if denomenator == 0:
        return 0.0
    else:
        return tp / (tp + fp)


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``tp / (tp + fn)``.

    Returns ``0.0`` when the denominator is 0 (no actual positives) — a
    documented convention, not a mathematical fact.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """

    _error_checks(y_true,y_pred)

    tp ,fn = 0,0

    for y_true_i , y_pred_i in zip(y_true,y_pred):
        if y_true_i == 1 and y_pred_i == 0:
            fn += 1
        elif y_true_i == 1 and y_pred_i == 1:
            tp += 1

    denomenator = tp + fn

    if denomenator == 0:
        return 0.0
    else:
        return tp / (tp + fn)
