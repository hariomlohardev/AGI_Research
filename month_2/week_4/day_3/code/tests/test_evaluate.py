"""Tests for Day 21 evaluation helpers (synthetic fixtures + CSV pins + integration).

Imports the day's `code/` via a sys.path shim, `minisklearn` from the
month root, and Day 20's `split.py` for the integration run (reuse, not
a copy — same pattern the learner's own script uses). Everything here
fails on the stubs — each test is a task in coding_problems.md.
"""

import pathlib
import sys

import pytest

CODE = pathlib.Path(__file__).resolve().parents[1]
MONTH2 = pathlib.Path(__file__).resolve().parents[4]
DAY2CODE = MONTH2 / "week_4" / "day_2" / "code"

sys.path.insert(0, str(CODE))
sys.path.insert(0, str(MONTH2))
sys.path.insert(0, str(DAY2CODE))

import evaluate
from minisklearn import metrics
from split import stratified_split

DATA = MONTH2 / "week_4" / "day_2" / "code" / "data" / "titanic.csv"


def test_comparison_table_layout():
    table = evaluate.comparison_table([
        ("logreg", 0.75, 2 / 3, 1.0, 0.8),
        ("forest", 0.8, 0.75, 0.9, 0.818),
    ])
    for token in ("Accuracy", "Precision", "Recall", "F1", "logreg", "forest"):
        assert token in table
    assert "0.750" in table  # 3-decimal rendering
    assert "0.667" in table  # 2/3 rounds, not truncates
    with pytest.raises(ValueError):
        evaluate.comparison_table([])


def test_misclassified_indices():
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 1]
    assert evaluate.misclassified_indices(y_true, y_pred, k=2) == [1, 3]
    assert evaluate.misclassified_indices(y_true, y_pred, k=10) == [1, 3, 5]
    assert evaluate.misclassified_indices([0], [0]) == []
    with pytest.raises(ValueError):
        evaluate.misclassified_indices([], [])
    with pytest.raises(ValueError):
        evaluate.misclassified_indices([0, 1], [0])


def test_metrics_hand_values():
    # TN=1, FP=1, FN=0, TP=2 by inspection.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    assert metrics.confusion_matrix(y_true, y_pred) == [[1, 1], [0, 2]]
    assert metrics.accuracy_score(y_true, y_pred) == pytest.approx(0.75)
    assert metrics.precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert metrics.recall_score(y_true, y_pred) == pytest.approx(1.0)
    assert metrics.f1_score(y_true, y_pred) == pytest.approx(0.8)


def test_load_titanic_xy_pins():
    X, y = evaluate.load_titanic_xy(str(DATA))
    assert len(X) == 889 and len(y) == 889  # 891 minus the 2 Embarked-missing rows
    assert all(len(row) == 7 for row in X)
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for row in X for v in row)
    assert all(v == v for row in X for v in row)  # no NaN
    assert set(y) <= {0, 1}
    assert 0.36 <= sum(y) / len(y) <= 0.40  # ~38% positive, matches Day 20


def test_integration_two_models_one_table():
    from minisklearn.ensemble import RandomForestClassifier
    from minisklearn.linear_model import LogisticRegression

    X, y = evaluate.load_titanic_xy(str(DATA))
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.2, seed=0)
    for part in (y_train, y_test):
        assert abs(sum(part) / len(part) - 0.38) < 0.05

    models = {
        "logreg": LogisticRegression(lr=0.001, epochs=300),
        "forest": RandomForestClassifier(n_estimators=5, max_depth=6, random_state=0),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = [int(v) for v in model.predict(X_test)]
        acc = metrics.accuracy_score(y_test, pred)
        assert 0.5 < acc < 1.0
        results.append((name, acc, metrics.precision_score(y_test, pred),
                        metrics.recall_score(y_test, pred), metrics.f1_score(y_test, pred)))

    table = evaluate.comparison_table(results)
    assert "logreg" in table and "forest" in table

    best_pred = [int(v) for v in models["forest"].predict(X_test)]
    bad = evaluate.misclassified_indices(y_test, best_pred, k=5)
    assert len(bad) <= 5
    assert all(y_test[i] != best_pred[i] for i in bad)
