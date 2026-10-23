"""Tests for Day 22 sklearn-comparison helpers.

Imports this day's `code/` plus Day 2, Day 3, and month-level `minisklearn`
through sys.path shims (the same reuse pattern as Days 20-21). Everything
here fails on the stubs — each test is a task in coding_problems.md.

The suite checks the comparison machinery and one real-data supervised
integration. It does not grade the learner's final four-model conclusions:
those belong in Problem 3's manual table and honest paragraph.
"""

import math
import pathlib
import sys

import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
MONTH2 = pathlib.Path(__file__).resolve().parents[3]
DAY2CODE = MONTH2 / "week_4" / "day_2" / "code"
DAY3CODE = MONTH2 / "week_4" / "day_3" / "code"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(MONTH2))
sys.path.insert(0, str(DAY2CODE))
sys.path.insert(0, str(DAY3CODE))

import compare
from minisklearn import metrics
from minisklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

DATA = MONTH2 / "week_4" / "day_2" / "code" / "data" / "titanic.csv"


class MajorityModel:
    """Minimal classifier-shaped object for testing score_classifier."""

    def __init__(self, label=0):
        self.label = label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [self.label] * len(X)


def test_load_same_split_matches_day21_shape_and_balance():
    X_train, X_test, y_train, y_test = compare.load_same_split(str(DATA))

    assert len(X_train) + len(y_train) == 2 * len(X_train)
    assert len(X_train) + len(X_test) == 889
    assert len(y_train) + len(y_test) == 889
    assert all(len(row) == 7 for row in (*X_train, *X_test))
    for labels in (y_train, y_test):
        assert set(labels) <= {0, 1}
        assert 0.36 <= sum(labels) / len(labels) <= 0.40

    with pytest.raises(ValueError):
        compare.load_same_split(str(DATA), test_ratio=0.0)


def test_score_classifier_uses_minisklearn_metrics():
    X_train = [[0.0], [1.0], [2.0], [3.0]]
    y_train = [0, 0, 1, 1]
    X_test = [[0.1], [2.9]]
    y_test = [0, 1]

    scores = compare.score_classifier(MajorityModel(1), X_train, y_train, X_test, y_test)

    assert set(scores) == {"accuracy", "precision", "recall", "f1"}
    assert scores["accuracy"] == pytest.approx(metrics.accuracy_score(y_test, [1, 1]))
    assert scores["precision"] == pytest.approx(metrics.precision_score(y_test, [1, 1]))
    assert scores["recall"] == pytest.approx(metrics.recall_score(y_test, [1, 1]))
    assert scores["f1"] == pytest.approx(metrics.f1_score(y_test, [1, 1]))

    with pytest.raises(ValueError):
        compare.score_classifier(object(), X_train, y_train, X_test, y_test)
    with pytest.raises(ValueError):
        compare.score_classifier(MajorityModel(), [], [], X_test, y_test)
    with pytest.raises(ValueError):
        compare.score_classifier(MajorityModel(), X_train, [0, 0], X_test, y_test)


def test_accuracy_gap_and_verdict_contracts():
    assert compare.accuracy_gap(0.80, 0.83) == pytest.approx(0.03)
    assert compare.gap_verdict(0.03) == "match"
    assert compare.gap_verdict(0.0300001) == "investigate"
    assert compare.gap_verdict(0.10, tolerance=0.10) == "match"

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            compare.accuracy_gap(bad, 0.8)
        with pytest.raises(ValueError):
            compare.accuracy_gap(0.8, bad)
        with pytest.raises(ValueError):
            compare.gap_verdict(bad)

    with pytest.raises(ValueError):
        compare.gap_verdict(-0.01)
    with pytest.raises(ValueError):
        compare.gap_verdict(0.01, tolerance=0.0)


def test_logistic_pair_on_real_split():
    X_train, X_test, y_train, y_test = compare.load_same_split(str(DATA))
    mine = LogisticRegression(lr=0.001, epochs=300)
    theirs = SklearnLogisticRegression(max_iter=1000)

    mine_scores = compare.score_classifier(mine, X_train, y_train, X_test, y_test)
    their_scores = compare.score_classifier(theirs, X_train, y_train, X_test, y_test)

    for scores in (mine_scores, their_scores):
        assert 0.5 < scores["accuracy"] < 1.0
        assert 0.0 <= scores["precision"] <= 1.0
        assert 0.0 <= scores["recall"] <= 1.0
        assert 0.0 <= scores["f1"] <= 1.0

    gap = compare.accuracy_gap(mine_scores["accuracy"], their_scores["accuracy"])
    assert 0.0 <= gap <= 0.20
    assert compare.gap_verdict(gap) in {"match", "investigate"}


def test_sklearn_kmeans_runs_on_real_rows():
    X_train, _, _, _ = compare.load_same_split(str(DATA))
    model = SklearnKMeans(n_clusters=2, n_init=10, random_state=0).fit(X_train)

    assert len(model.labels_) == len(X_train)
    assert math.isfinite(float(model.inertia_))
    assert float(model.inertia_) > 0.0
