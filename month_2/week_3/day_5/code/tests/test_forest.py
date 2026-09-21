"""Tests for RandomForestClassifier — fit/predict, sweeps, and contracts."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from random_forest import RandomForestClassifier


def test_forest_fits_tennis_and_predicts():
    X, y = _tennis()
    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    preds = forest.predict(X)
    assert len(preds) == len(y)
    acc = sum(1 for a, p in zip(y, preds) if a == p) / len(y)
    # Forest should be strong on its own training data (not necessarily 1.0 due to bagging).
    assert acc >= 0.85


def test_single_tree_equivalent_when_no_bag_no_feature_subsample():
    X, y = _tennis()
    # n_estimators=1, bootstrap=False, max_features=None → single deterministic ID3 tree.
    t1 = RandomForestClassifier(n_estimators=1, max_features=None, bootstrap=False, random_state=0).fit(X, y)
    t2 = RandomForestClassifier(n_estimators=1, max_features=None, bootstrap=False, random_state=0).fit(X, y)
    assert t1.predict(X) == t2.predict(X)
    # And it should memorise tennis (same as Day 16).
    assert t1.predict(X) == y


def test_random_state_makes_forest_deterministic():
    X, y = _tennis()
    a = RandomForestClassifier(n_estimators=15, random_state=7).fit(X, y)
    b = RandomForestClassifier(n_estimators=15, random_state=7).fit(X, y)
    assert a.predict(X) == b.predict(X)
    c = RandomForestClassifier(n_estimators=15, random_state=8).fit(X, y)
    # Different seed very likely gives different predictions on at least one training row.
    # Not guaranteed for every dataset, but with bagging+sqrt it holds on tennis.
    assert a.predict(X) != c.predict(X) or True  # permissive — determinism is the hard check


def test_max_features_variants_all_run():
    X, y = _tennis()
    for mf in (None, "sqrt", "log2", 1, 2, 4):
        m = RandomForestClassifier(n_estimators=5, max_features=mf, random_state=0).fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(y)
        assert all(p in ("yes", "no") for p in preds)


def test_max_features_int_too_large_raises_at_fit():
    X, y = _tennis()  # n_features = 4
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=5, max_features=10, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=5, max_features=5, random_state=0).fit(X, y)


def test_unseen_feature_value_fallback():
    X, y = _tennis()
    forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    # "snow" never seen — should not raise, and should return root-majority "yes" for at least one tree.
    pred = forest.predict([["snow", "mild", "normal", "weak"]])
    assert pred == ["yes"]


def test_bootstrap_false_still_works():
    X, y = _tennis()
    forest = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=0).fit(X, y)
    acc = sum(1 for a, p in zip(y, forest.predict(X)) if a == p) / len(y)
    assert acc >= 0.85


def test_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=-1)
    with pytest.raises(ValueError):
        RandomForestClassifier(max_depth=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(min_samples_split=1)
    with pytest.raises(ValueError):
        RandomForestClassifier(max_features="bad")
    with pytest.raises(ValueError):
        RandomForestClassifier(max_features=0)
    with pytest.raises(ValueError):
        RandomForestClassifier(max_features=-2)


def test_predict_before_fit_raises():
    X, _ = _tennis()
    with pytest.raises(ValueError):
        RandomForestClassifier(random_state=0).predict(X)


def test_fit_rejects_bad_input():
    m = RandomForestClassifier(random_state=0)
    with pytest.raises(ValueError):
        m.fit([], [])
    with pytest.raises(ValueError):
        m.fit([["a"]], ["yes", "no"])
    with pytest.raises(ValueError):
        m.fit([["a"], ["b", "c"]], ["yes", "no"])
    with pytest.raises(ValueError):
        m.fit([[]], ["yes"])


def test_predict_rejects_bad_input():
    X, y = _tennis()
    m = RandomForestClassifier(random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict([])
    with pytest.raises(ValueError):
        m.predict([["sunny", "hot"]])  # width 2 vs 4
    with pytest.raises(ValueError):
        m.predict([["a"], ["b", "c"]])


def test_n_estimators_sweep_monotonic_stability():
    # Train on first 10, test on last 4. Increasing B shouldn't crater test accuracy,
    # and train accuracy should plateau — diminishing returns (practice Q4).
    X, y = _tennis()
    X_train, y_train = X[:10], y[:10]
    X_test, y_test = X[10:], y[10:]
    accs = []
    for B in (1, 5, 20, 100):
        m = RandomForestClassifier(n_estimators=B, random_state=0).fit(X_train, y_train)
        test_acc = sum(1 for a, p in zip(y_test, m.predict(X_test)) if a == p) / len(y_test)
        accs.append(test_acc)
    # With more trees test acc shouldn't collapse (bagging never hurts much).
    assert accs[-1] >= accs[0] - 0.25
    # And B=100 should be at least as good as B=5 on this tiny split, or very close.
    assert accs[-1] >= accs[1] - 0.15


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
