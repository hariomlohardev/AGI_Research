"""Tests for Day 20 data-prep helpers (synthetic fixtures + real CSV pins).

Imports the day's `code/` via a sys.path shim. Everything here fails on
the stubs — each test is a task in coding_problems.md Problems 1-3. The
Titanic pins (891 rows, Age 177 / Embarked 2 missing, 342 survived) were
verified against the vendored CSV at generation time.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import eda
import encode
import split

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "titanic.csv"


def test_load_csv_pins():
    header, rows = eda.load_csv(str(DATA))
    assert header == [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
        "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    ]
    assert len(rows) == 891
    assert all(len(r) == 12 for r in rows)


def test_missing_counts_pins():
    header, rows = eda.load_csv(str(DATA))
    counts = eda.missing_counts(header, rows)
    assert counts["Age"] == 177
    assert counts["Embarked"] == 2
    assert counts["Survived"] == 0
    assert counts["PassengerId"] == 0


def test_class_balance_pins():
    header, rows = eda.load_csv(str(DATA))
    labels = [int(r[header.index("Survived")]) for r in rows]
    assert eda.class_balance(labels) == {0: 549, 1: 342}


def test_numeric_summary_synthetic():
    s = eda.numeric_summary([1.0, 2.0, 3.0, 4.0])
    assert s["count"] == 4
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(4.0)
    assert s["mean"] == pytest.approx(2.5)
    with pytest.raises(ValueError):
        eda.numeric_summary([])


def test_to_float_column():
    rows = [["1.5"], [""], ["3"]]
    assert encode.to_float_column(rows, 0) == [1.5, None, 3.0]
    with pytest.raises(ValueError):
        encode.to_float_column([["abc"]], 0)


def test_fill_mean():
    filled, mean = encode.fill_mean([1.0, None, 3.0, None])
    assert mean == pytest.approx(2.0)
    assert filled == pytest.approx([1.0, 2.0, 3.0, 2.0])
    with pytest.raises(ValueError):
        encode.fill_mean([None, None])
    with pytest.raises(ValueError):
        encode.fill_mean([])


def test_label_encode_and_one_hot():
    enc, mapping = encode.label_encode(["b", "a", "b"])
    assert mapping == {"b": 0, "a": 1}
    assert enc == [0, 1, 0]
    mat, cats = encode.one_hot(["b", "a", "b"])
    assert cats == ["b", "a"]
    assert mat == [[1, 0], [0, 1], [1, 0]]
    with pytest.raises(ValueError):
        encode.label_encode([])


def test_stratified_split_synthetic():
    X = [[float(i)] for i in range(100)]
    y = [0] * 80 + [1] * 20
    Xtr, Xte, ytr, yte = split.stratified_split(X, y, test_ratio=0.25, seed=0)
    assert len(Xte) == 25 and len(Xtr) == 75
    assert sorted(ytr + yte) == sorted(y)  # nothing lost or duplicated
    assert yte.count(1) == 5  # 25% of each class
    # Deterministic: same seed, same split.
    _, _, ytr2, yte2 = split.stratified_split(X, y, test_ratio=0.25, seed=0)
    assert (ytr2, yte2) == (ytr, yte)
    with pytest.raises(ValueError):
        split.stratified_split([], [], test_ratio=0.2)
    with pytest.raises(ValueError):
        split.stratified_split(X, y, test_ratio=1.5)


def _titanic_xy():
    header, rows = eda.load_csv(str(DATA))
    col = header.index
    # Drop the 2 rows with missing Embarked (documented choice); impute Age.
    rows = [r for r in rows if not eda.is_missing(r[col("Embarked")])]
    age = encode.fill_mean(encode.to_float_column(rows, col("Age")))[0]
    sex, _ = encode.label_encode([r[col("Sex")] for r in rows])
    emb, _ = encode.label_encode([r[col("Embarked")] for r in rows])
    X = [
        [
            float(r[col("Pclass")]), s, a,
            float(r[col("SibSp")]), float(r[col("Parch")]),
            float(r[col("Fare")]), e,
        ]
        for r, s, a, e in zip(rows, sex, age, emb)
    ]
    y = [int(r[col("Survived")]) for r in rows]
    return X, y


def test_integration_no_missing_numeric():
    X, y = _titanic_xy()
    assert len(X) == 889 and len(y) == 889
    assert all(isinstance(v, float) for row in X for v in row)
    assert all(v == v for row in X for v in row)  # no NaN
    assert set(y) <= {0, 1}


def test_stratified_on_titanic():
    X, y = _titanic_xy()
    _, _, ytr, yte = split.stratified_split(X, y, test_ratio=0.2, seed=0)
    overall = sum(y) / len(y)
    for part in (ytr, yte):
        assert abs(sum(part) / len(part) - overall) < 0.02
