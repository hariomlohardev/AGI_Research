"""Tests for bootstrap helpers — distribution, determinism, and contracts."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from random_forest import bootstrap_indices, bootstrap_sample


def test_bootstrap_indices_length_and_range():
    for n in (1, 6, 14, 100):
        idx = bootstrap_indices(n, random_state=0)
        assert len(idx) == n
        assert all(0 <= i < n for i in idx)


def test_bootstrap_indices_deterministic_with_seed():
    a = bootstrap_indices(14, random_state=42)
    b = bootstrap_indices(14, random_state=42)
    assert a == b
    c = bootstrap_indices(14, random_state=43)
    assert a != c


def test_bootstrap_indices_has_duplicates_for_typical_seed():
    # n=14, seed=0 is known to contain duplicates (that's the OOB point).
    idx = bootstrap_indices(14, random_state=0)
    assert len(idx) == len(set(idx)) + len(idx) - len(set(idx))
    assert len(set(idx)) < 14  # OOB rows exist
    # n=1 can only be [0] — no duplicate possible, but still deterministic.
    assert bootstrap_indices(1, random_state=0) == [0]


def test_bootstrap_indices_oob_size_large_n_is_about_one_over_e():
    idx = bootstrap_indices(1000, random_state=0)
    uniq = len(set(idx))
    # 63.2% unique → 632 ± 30 for this seed (deterministic, not flaky).
    assert 600 <= uniq <= 670


def test_bootstrap_indices_mean_oob_small_n():
    # Mean OOB size across 500 resamples of n=14 should be ~5.1 (36.8%).
    oob_sizes = []
    for seed in range(500):
        s = set(bootstrap_indices(14, random_state=seed))
        oob_sizes.append(14 - len(s))
    mean_oob = sum(oob_sizes) / len(oob_sizes)
    assert mean_oob == pytest.approx(5.1, abs=0.6)


def test_bootstrap_indices_rejects_bad_n():
    with pytest.raises(ValueError):
        bootstrap_indices(0)
    with pytest.raises(ValueError):
        bootstrap_indices(-3)


def test_bootstrap_sample_preserves_pairing_and_size():
    X = [["a", 1], ["b", 2], ["c", 3], ["d", 4]]
    y = ["yes", "no", "yes", "no"]
    Xs, ys = bootstrap_sample(X, y, random_state=7)
    assert len(Xs) == len(ys) == 4
    # Pairing: every (row, label) in the resample must appear in original at same index.
    orig = list(zip(map(tuple, X), y))
    for row, label in zip(Xs, ys):
        assert (tuple(row), label) in orig


def test_bootstrap_sample_deterministic():
    X = [["a"], ["b"], ["c"]]
    y = [0, 1, 0]
    a_X, a_y = bootstrap_sample(X, y, random_state=123)
    b_X, b_y = bootstrap_sample(X, y, random_state=123)
    assert a_X == b_X and a_y == b_y
    c_X, c_y = bootstrap_sample(X, y, random_state=124)
    # Very likely different (not strictly guaranteed, but with n=3 and different seed,
    # collision probability is tiny — treat as contract for determinism test).
    assert (c_X, c_y) != (a_X, a_y)


def test_bootstrap_sample_rejects_bad_input():
    X = [["a", "b"], ["c", "d"]]
    y = ["yes", "no"]
    with pytest.raises(ValueError):
        bootstrap_sample([], [], random_state=0)
    with pytest.raises(ValueError):
        bootstrap_sample([["a", "b"]], ["yes", "no"], random_state=0)
    with pytest.raises(ValueError):
        bootstrap_sample([["a", "b"], ["c"]], ["yes", "no"], random_state=0)
    with pytest.raises(ValueError):
        bootstrap_sample([[]], ["yes"], random_state=0)
