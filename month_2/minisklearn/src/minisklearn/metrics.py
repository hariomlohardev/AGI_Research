"""Classification metrics from scratch: confusion matrix + accuracy/precision/recall/F1.

New code (not a move): formalise the trio from Week 3 Day 2
(`month_2/week_3/day_2/code/metrics.py` is your starting reference — adapt,
don't blindly copy) and add `f1_score`. Binary `0`/`1` labels only, positive
class `1`. Stdlib only, plain Python lists.
"""

from __future__ import annotations

from collections.abc import Sequence

from .utils import if_empty_raise


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    """Return `[[TN, FP], [FN, TP]]` (sklearn layout, positive class 1).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """

    if_empty_raise(y_true, "y_true is empty")
    if_empty_raise(y_pred, "y_pred is empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is different from y_true")

    if any(val not in (0,1) for val in y_true):
         raise ValueError("the labels should be binary (1, 0)")
    
    
    TN, FP, FN, TP = 0,0,0,0
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
                
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
                
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1

        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1

    return [[TN, FP], [FN, TP]]

    


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the fraction of exactly-matching labels.

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    if_empty_raise(y_true, "y_true is empty")
    if_empty_raise(y_pred, "y_pred is empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is different from y_true")

    if any(val not in (0,1) for val in y_true):
         raise ValueError("the labels should be binary (1, 0)")

    cm = confusion_matrix(y_true , y_pred)

    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    return (TP + TN)/ len(y_pred)




def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `TP / (TP + FP)`: of the predicted positives, how many were right.

    A model that predicts no positives scores `0.0` (zero-division guard —
    documented convention, not a mathematical fact).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    if_empty_raise(y_true, "y_true is empty")
    if_empty_raise(y_pred, "y_pred is empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is different from y_true")

    if any(val not in (0,1) for val in y_true):
         raise ValueError("the labels should be binary (1, 0)")

    cm = confusion_matrix(y_true , y_pred)

    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    return 0.0 if (TP + FP) == 0 else TP / (TP + FP)


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return `TP / (TP + FN)`: of the actual positives, how many were found.

    A dataset (or slice) with no actual positives scores `0.0`
    (zero-division guard — documented convention).

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """

    if_empty_raise(y_true, "y_true is empty")
    if_empty_raise(y_pred, "y_pred is empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is different from y_true")

    if any(val not in (0,1) for val in y_true):
         raise ValueError("the labels should be binary (1, 0)")


    cm = confusion_matrix(y_true , y_pred)

    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    return 0.0 if (TP + FN) == 0 else TP / (TP + FN)


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return the harmonic mean `2PR / (P + R)` of precision and recall.

    Scores `0.0` when `P + R == 0` (zero-division guard — documented
    convention). Harmonic (not arithmetic) mean: it punishes a model that
    is all-precision-no-recall or vice versa.

    Raises:
        ValueError: if either list is empty, lengths differ, or any label
            is not binary (`0` or `1`).
    """
    if_empty_raise(y_true, "y_true is empty")
    if_empty_raise(y_pred, "y_pred is empty")

    if len(y_pred) != len(y_true):
        raise ValueError("the length of y_pred is different from y_true")

    if any(val not in (0,1) for val in y_true):
         raise ValueError("the labels should be binary (1, 0)")

    cm = confusion_matrix(y_true , y_pred)

    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    P = precision_score(y_true,y_pred)
    R = recall_score(y_true,y_pred)
    
    return 0.0 if P+R == 0 else 2 * P * R / (P + R) 
