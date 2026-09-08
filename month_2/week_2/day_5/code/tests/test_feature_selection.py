"""Tests for rank_features_by_mi()."""

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from feature_selection import rank_features_by_mi


def _synthetic(seed=11, n=300, flip=0.10):
    rng = random.Random(seed)
    label = [rng.choice((0, 1)) for _ in range(n)]
    informative = [v if rng.random() > flip else 1 - v for v in label]
    noise = [rng.choice((0, 1)) for _ in range(n)]
    rows = [[informative[i], noise[i]] for i in range(n)]
    return rows, label


def test_informative_beats_noise():
    rows, label = _synthetic()
    ranking = rank_features_by_mi(rows, label)
    assert ranking[0][0] == 0
    assert ranking[0][1] - ranking[1][1] > 0.2


def test_informative_score_near_theory():
    # A 10%-flipped copy of a fair binary label carries ~1 - H(0.1) ~= 0.531 bits.
    rows, label = _synthetic()
    ranking = rank_features_by_mi(rows, label)
    assert ranking[0][1] == pytest.approx(0.531, abs=0.1)


def test_sorted_descending_and_complete():
    rng = random.Random(14)
    n = 200
    label = [rng.choice((0, 1)) for _ in range(n)]
    cols = [[rng.choice((0, 1, 2)) for _ in range(n)] for _ in range(3)]
    rows = [list(r) for r in zip(*cols)]
    ranking = rank_features_by_mi(rows, label)
    assert sorted(i for i, _ in ranking) == [0, 1, 2]
    assert len(ranking) == 3
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)


def test_empty_features_raises():
    with pytest.raises(ValueError):
        rank_features_by_mi([], [0, 1])


def test_empty_y_raises():
    with pytest.raises(ValueError):
        rank_features_by_mi([[0], [1]], [])


def test_row_count_mismatch_raises():
    with pytest.raises(ValueError):
        rank_features_by_mi([[0], [1], [0]], [0, 1])


def test_ragged_rows_raise():
    with pytest.raises(ValueError):
        rank_features_by_mi([[0, 1], [0]], [0, 1])


def test_bad_base_raises():
    rows, label = _synthetic()
    with pytest.raises(ValueError):
        rank_features_by_mi(rows, label, base=1.0)
