"""Tests for DecisionTreeID3 (fit/predict/depth/contracts/readable tree)."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from decision_tree import DecisionTreeID3, format_tree


def test_fit_picks_outlook_and_memorises_tennis():
    X, y = _tennis()
    model = DecisionTreeID3().fit(X, y)
    tree = model.to_dict()
    assert tree["feature"] == 0  # Outlook has the highest gain (learn.md math)
    preds = model.predict(X)
    assert preds == y  # classic ID3 tree fits the 14 training rows exactly


def test_overcast_branch_is_pure_yes():
    X, y = _tennis()
    model = DecisionTreeID3().fit(X, y)
    tree = model.to_dict()
    assert tree["children"]["overcast"] == {"label": "yes"}


def test_pure_node_becomes_single_leaf():
    model = DecisionTreeID3().fit([["a"], ["b"]], ["yes", "yes"])
    assert model.to_dict() == {"label": "yes"}
    assert model.predict([["zzz"]]) == ["yes"]


def test_max_depth_one_underfits_tennis():
    X, y = _tennis()
    shallow = DecisionTreeID3(max_depth=1).fit(X, y)
    deep = DecisionTreeID3().fit(X, y)
    acc = lambda m: sum(1 for a, p in zip(y, m.predict(X)) if a == p) / len(y)
    assert acc(shallow) == pytest.approx(10 / 14)
    assert acc(deep) == pytest.approx(1.0)
    # A depth-1 tree is root + leaves only.
    for child in shallow.to_dict()["children"].values():
        assert "label" in child


def test_large_min_samples_split_stops_at_root():
    X, y = _tennis()
    model = DecisionTreeID3(min_samples_split=100).fit(X, y)
    assert model.to_dict() == {"label": "yes"}


def test_unseen_value_falls_back_to_node_majority():
    X, y = _tennis()
    model = DecisionTreeID3().fit(X, y)
    # "snow" was never seen under Outlook — predict the root majority
    # ("yes", 9 of 14).
    assert model.predict([["snow", "mild", "normal", "weak"]]) == ["yes"]


def test_format_tree_uses_feature_names():
    X, y = _tennis()
    model = DecisionTreeID3().fit(X, y)
    text = format_tree(
        model.to_dict(),
        feature_names=["Outlook", "Temperature", "Humidity", "Wind"],
    )
    assert "Outlook" in text and "yes" in text
    with pytest.raises(ValueError):
        format_tree({})


def test_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        DecisionTreeID3(max_depth=0)
    with pytest.raises(ValueError):
        DecisionTreeID3(min_samples_split=1)
    with pytest.raises(ValueError):
        DecisionTreeID3(min_samples_split=0)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        DecisionTreeID3().predict([["sunny"]])
    with pytest.raises(ValueError):
        DecisionTreeID3().to_dict()


def test_fit_rejects_bad_input():
    model = DecisionTreeID3()
    with pytest.raises(ValueError):
        model.fit([], [])
    with pytest.raises(ValueError):
        model.fit([["sunny"]], ["yes", "no"])
    with pytest.raises(ValueError):
        model.fit([["a"], ["b", "c"]], ["yes", "no"])
    with pytest.raises(ValueError):
        model.fit([[]], ["yes"])


def test_predict_rejects_bad_input():
    model = DecisionTreeID3().fit([["a", "b"]], ["yes"])
    with pytest.raises(ValueError):
        model.predict([["a"]])
    with pytest.raises(ValueError):
        model.predict([])


def _tennis():
    X = [
        ["sunny", "hot", "high", "weak"],
        ["sunny", "hot", "high", "strong"],
        ["overcast", "hot", "high", "weak"],
        ["rain", "mild", "high", "weak"],
        ["rain", "cool", "normal", "weak"],
        ["rain", "cool", "normal", "strong"],
        ["overcast", "cool", "normal", "strong"],
        ["sunny", "mild", "high", "weak"],
        ["sunny", "cool", "normal", "weak"],
        ["rain", "mild", "normal", "weak"],
        ["sunny", "mild", "normal", "strong"],
        ["overcast", "mild", "high", "strong"],
        ["overcast", "hot", "normal", "weak"],
        ["rain", "mild", "high", "strong"],
    ]
    y = [
        "no", "no", "yes", "yes", "yes", "no", "yes",
        "no", "yes", "yes", "yes", "yes", "yes", "no",
    ]
    return X, y
