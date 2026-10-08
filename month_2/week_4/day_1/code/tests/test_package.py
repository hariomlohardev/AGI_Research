"""Tests for the minisklearn package skeleton (imports + interface + metrics).

Imports the month-level package `month_2/minisklearn/` via a sys.path shim
(same harness shape as prior days). Everything here fails on the empty
skeleton — each test is a task in coding_problems.md Problems 1-3.
"""

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

import minisklearn
from minisklearn import cluster, ensemble, linear_model, metrics, tree


def test_submodules_importable():
    for mod in (linear_model, tree, ensemble, cluster, metrics):
        assert mod.__name__.startswith("minisklearn.")


def test_top_level_exposes_classes_and_metrics():
    for name in (
        "LinearRegression",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "KMeans",
        "accuracy_score",
        "precision_score",
        "recall_score",
        "f1_score",
        "confusion_matrix",
    ):
        assert hasattr(minisklearn, name), f"minisklearn.{name} missing"


def _TinyTennis():
    # 8-row play-tennis fragment (Outlook, Temp, Humidity, Wind -> Play).
    X = [
        ["sunny", "hot", "high", "weak"],
        ["sunny", "hot", "high", "strong"],
        ["overcast", "hot", "high", "weak"],
        ["rain", "mild", "high", "weak"],
        ["rain", "cool", "normal", "weak"],
        ["rain", "cool", "normal", "strong"],
        ["overcast", "cool", "normal", "strong"],
        ["sunny", "mild", "high", "weak"],
    ]
    y = [0, 0, 1, 1, 1, 0, 1, 0]
    return X, y


def test_interface_uniformity_fit_returns_self():
    X, y = _TinyTennis()
    assert tree.DecisionTreeClassifier().fit(X, y) is not None
    m = linear_model.LogisticRegressionGD() if hasattr(linear_model, "LogisticRegressionGD") else linear_model.LogisticRegression()
    assert m.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]) is m
    km = cluster.KMeans(n_clusters=2, random_state=0)
    assert km.fit([[0.0], [1.0], [9.0], [10.0]]) is km


def test_linear_regression_smoke():
    m = linear_model.LinearRegression()
    X = [[1.0], [2.0], [3.0], [4.0]]
    m.fit(X, [2.0, 4.0, 6.0, 8.0])
    assert m.predict(X) == pytest.approx([2.0, 4.0, 6.0, 8.0], abs=1.0)


def test_logistic_regression_smoke():
    m = linear_model.LogisticRegression()
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [2.0, 3.0], [3.0, 2.0]]
    y = [0, 0, 0, 1, 1, 1]
    m.fit(X, y)
    assert m.predict(X) == y


def test_tree_and_forest_smoke():
    X, y = _TinyTennis()
    assert tree.DecisionTreeClassifier().fit(X, y).predict(X) == y
    f = ensemble.RandomForestClassifier(n_estimators=5, random_state=0)
    preds = f.fit(X, y).predict(X)
    assert metrics.accuracy_score(y, preds) >= 0.75


def test_kmeans_smoke():
    rng = random.Random(0)
    X = [[rng.gauss(0.0, 0.3), rng.gauss(0.0, 0.3)] for _ in range(10)]
    X += [[rng.gauss(5.0, 0.3), rng.gauss(5.0, 0.3)] for _ in range(10)]
    km = cluster.KMeans(n_clusters=2, random_state=0).fit(X)
    labels = km.predict(X)
    agree = 0
    for block in (labels[:10], labels[10:]):
        agree += max(block.count(l) for l in set(block))
    assert agree / 20 == 1.0


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        linear_model.LogisticRegression().predict([[0.0, 0.0]])
    with pytest.raises(ValueError):
        tree.DecisionTreeClassifier().predict(_TinyTennis()[0])
    with pytest.raises(ValueError):
        cluster.KMeans().predict([[0.0]])


def test_metrics_hand_values():
    y_true = [1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1]
    assert metrics.confusion_matrix(y_true, y_pred) == [[2, 1], [1, 2]]
    assert metrics.accuracy_score(y_true, y_pred) == pytest.approx(4 / 6)
    assert metrics.precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert metrics.recall_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert metrics.f1_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_metrics_zero_division_conventions():
    assert metrics.precision_score([0, 0, 1], [0, 0, 0]) == 0.0
    assert metrics.recall_score([0, 0, 0], [0, 0, 1]) == 0.0
    assert metrics.f1_score([0, 0, 0], [0, 0, 0]) == 0.0


def test_metrics_rejects_bad_input():
    for fn in (
        metrics.confusion_matrix,
        metrics.accuracy_score,
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score,
    ):
        with pytest.raises(ValueError):
            fn([], [])
        with pytest.raises(ValueError):
            fn([0, 1], [0])
        with pytest.raises(ValueError):
            fn([0, 2], [0, 1])
