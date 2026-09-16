"""Classification metrics computed by hand — no libraries, not even ``math``.

Labels are binary ints in ``{0, 1}``. All scores build on
:func:`confusion_matrix` so the counting lives in exactly one place.
"""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` counts.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``(tp + tn) / n``.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``tp / (tp + fp)``.

    Returns ``0.0`` when the denominator is 0 (no predicted positives) —
    a documented convention, not a mathematical fact.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return ``tp / (tp + fn)``.

    Returns ``0.0`` when the denominator is 0 (no actual positives) — a
    documented convention, not a mathematical fact.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError
