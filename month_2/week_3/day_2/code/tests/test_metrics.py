"""Tests for metrics.py."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def test_confusion_matrix_hand_values():
    # tn=1, fp=1, fn=0, tp=2.
    assert confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1]) == (2, 1, 1, 0)


def test_scores_hand_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)
    assert precision_score(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert recall_score(y_true, y_pred) == pytest.approx(1.0)


def test_perfect_and_worst():
    assert accuracy_score([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert accuracy_score([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert precision_score([0, 1], [1, 0]) == pytest.approx(0.0)
    assert recall_score([0, 1], [1, 0]) == pytest.approx(0.0)


def test_empty_denominator_convention():
    # No predicted positives -> precision 0.0; no actual positives -> recall 0.0.
    assert precision_score([0, 0, 0], [0, 0, 0]) == pytest.approx(0.0)
    assert recall_score([0, 0, 0], [0, 0, 0]) == pytest.approx(0.0)
    assert accuracy_score([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)


def test_rejects_bad_input():
    for fn in (confusion_matrix, accuracy_score, precision_score, recall_score):
        with pytest.raises(ValueError):
            fn([], [])
        with pytest.raises(ValueError):
            fn([0, 1], [0])
        with pytest.raises(ValueError):
            fn([0, 2], [0, 1])
