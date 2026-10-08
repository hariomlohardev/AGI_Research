"""Tests for K-Means helpers + fit/predict + elbow (contracts + behaviour)."""

import math
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kmeans import (
    KMeans,
    assign_clusters,
    elbow_inertias,
    euclidean,
    inertia,
    recompute_centroids,
    standardize,
)


def _blobs(centres, per_blob=20, std=0.5, seed=0):
    rng = random.Random(seed)
    X = []
    for cx, cy in centres:
        for _ in range(per_blob):
            X.append([rng.gauss(cx, std), rng.gauss(cy, std)])
    return X


CENTRES = [[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]]


def test_euclidean_hand_values():
    assert euclidean([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)
    assert euclidean([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert euclidean([0.0], [2.0]) == pytest.approx(2.0)


def test_euclidean_rejects_bad_input():
    with pytest.raises(ValueError):
        euclidean([], [])
    with pytest.raises(ValueError):
        euclidean([1.0, 2.0], [1.0])


def test_assign_clusters_nearest_and_tiebreak():
    X = [[0.1, 0.0], [4.9, 5.0], [0.0, 4.8]]
    C = [[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]]
    assert assign_clusters(X, C) == [0, 1, 2]
    # Equidistant point → lowest cluster index (documented convention).
    assert assign_clusters([[1.0]], [[0.0], [2.0]]) == [0]


def test_assign_clusters_rejects_bad_input():
    with pytest.raises(ValueError):
        assign_clusters([], [[0.0]])
    with pytest.raises(ValueError):
        assign_clusters([[]], [[0.0]])
    with pytest.raises(ValueError):
        assign_clusters([[0.0, 0.0], [1.0]], [[0.0, 0.0]])
    with pytest.raises(ValueError):
        assign_clusters([[0.0, 0.0]], [[0.0]])
    with pytest.raises(ValueError):
        assign_clusters([[0.0, 0.0]], [])


def test_recompute_centroids_means_and_empty_cluster():
    X = [[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 6.0]]
    C = recompute_centroids(X, [0, 0, 1, 1], 2, [[9.0, 9.0], [9.0, 9.0]])
    assert C[0] == pytest.approx([1.0, 0.0])
    assert C[1] == pytest.approx([0.0, 5.0])
    # Empty cluster keeps its previous centroid (documented convention).
    C2 = recompute_centroids([[0.0], [10.0]], [0, 0], 2, [[0.0], [99.0]])
    assert C2[0] == pytest.approx([5.0])
    assert C2[1] == pytest.approx([99.0])


def test_recompute_centroids_rejects_bad_input():
    with pytest.raises(ValueError):
        recompute_centroids([], [], 1, [[0.0]])
    with pytest.raises(ValueError):
        recompute_centroids([[0.0]], [0, 1], 2, [[0.0], [1.0]])
    with pytest.raises(ValueError):
        recompute_centroids([[0.0]], [0], 0, [[0.0]])
    with pytest.raises(ValueError):
        recompute_centroids([[0.0]], [2], 2, [[0.0], [1.0]])
    with pytest.raises(ValueError):
        recompute_centroids([[0.0, 0.0]], [0], 1, [[0.0]])


def test_inertia_hand_value():
    assert inertia([[0.0], [2.0]], [0, 0], [[1.0]]) == pytest.approx(2.0)
    assert inertia([[0.0, 0.0]], [0], [[0.0, 0.0]]) == pytest.approx(0.0)


def test_inertia_rejects_bad_input():
    with pytest.raises(ValueError):
        inertia([], [], [[0.0]])
    with pytest.raises(ValueError):
        inertia([[0.0]], [0, 1], [[0.0]])
    with pytest.raises(ValueError):
        inertia([[0.0], [1.0, 2.0]], [0, 0], [[0.0]])
    with pytest.raises(ValueError):
        inertia([[0.0]], [1], [[0.0]])


def test_standardize_zero_mean_unit_std():
    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    Xs, means, stds = standardize(X)
    assert means == pytest.approx([2.0, 20.0])
    col0 = [r[0] for r in Xs]
    col1 = [r[1] for r in Xs]
    assert sum(col0) / 3 == pytest.approx(0.0, abs=1e-9)
    assert sum(v * v for v in col0) / 3 == pytest.approx(1.0)
    assert sum(col1) / 3 == pytest.approx(0.0, abs=1e-9)
    assert sum(v * v for v in col1) / 3 == pytest.approx(1.0)


def test_standardize_zero_variance_column_becomes_zeros():
    Xs, means, stds = standardize([[5.0, 1.0], [5.0, 3.0]])
    assert [r[0] for r in Xs] == [0.0, 0.0]
    assert means[0] == pytest.approx(5.0)


def test_standardize_rejects_bad_input():
    with pytest.raises(ValueError):
        standardize([])
    with pytest.raises(ValueError):
        standardize([[1.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        standardize([[]])


def test_fit_finds_three_blobs():
    X = _blobs(CENTRES)
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.cluster_centers_ is not None and len(km.cluster_centers_) == 3
    # Each true centre has a fitted centroid nearby (permutation-agnostic).
    for tx, ty in CENTRES:
        best = min(math.dist([tx, ty], c) for c in km.cluster_centers_)
        assert best < 0.6
    # Cluster purity: each true blob is dominated by one fitted label.
    labels = km.predict(X)
    agree = 0
    for b in range(3):
        block = labels[b * 20:(b + 1) * 20]
        agree += max(block.count(l) for l in set(block))
    assert agree / 60 >= 0.95


def test_fit_is_deterministic_with_seed():
    X = _blobs(CENTRES)
    a = KMeans(n_clusters=3, random_state=7).fit(X)
    b = KMeans(n_clusters=3, random_state=7).fit(X)
    assert a.predict(X) == b.predict(X)
    assert a.cluster_centers_ == b.cluster_centers_


def test_different_seeds_can_land_in_different_optima():
    Xo = _blobs(CENTRES, std=2.5, seed=1)
    vals = set()
    for seed in range(5):
        m = KMeans(n_clusters=3, random_state=seed).fit(Xo)
        vals.add(round(inertia(Xo, m.predict(Xo), m.cluster_centers_), 1))
    assert len(vals) > 1  # local optima differ per init on overlapping blobs


def test_k_equals_one_centroid_is_mean():
    X = _blobs(CENTRES)
    km = KMeans(n_clusters=1, random_state=0).fit(X)
    mx = sum(r[0] for r in X) / len(X)
    my = sum(r[1] for r in X) / len(X)
    assert km.cluster_centers_[0] == pytest.approx([mx, my], abs=1e-6)
    assert km.predict(X) == [0] * len(X)


def test_n_iters_respects_max_iters():
    X = _blobs(CENTRES)
    assert KMeans(n_clusters=3, random_state=0).fit(X).n_iters_ >= 1
    assert KMeans(n_clusters=3, max_iters=1, random_state=0).fit(X).n_iters_ == 1
    assert KMeans(n_clusters=3, max_iters=3, random_state=0).fit(X).n_iters_ <= 3


def test_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(max_iters=0)
    with pytest.raises(ValueError):
        KMeans(tol=-0.1)


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        KMeans().fit([])
    with pytest.raises(ValueError):
        KMeans().fit([[0.0], [1.0, 2.0]])
    with pytest.raises(ValueError):
        KMeans().fit([[]])
    with pytest.raises(ValueError):
        KMeans(n_clusters=5).fit([[0.0], [1.0], [2.0]])


def test_predict_before_fit_and_bad_input_raise():
    X = _blobs(CENTRES)
    with pytest.raises(ValueError):
        KMeans().predict(X)
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    with pytest.raises(ValueError):
        km.predict([])
    with pytest.raises(ValueError):
        km.predict([[0.0, 0.0], [1.0]])
    with pytest.raises(ValueError):
        km.predict([[0.0, 0.0, 0.0]])


def test_elbow_decreases_overall_and_validates():
    X = _blobs(CENTRES)
    vals = elbow_inertias(X, [1, 2, 3, 4, 5], random_state=0)
    assert len(vals) == 5
    assert all(v >= 0 for v in vals)
    assert vals[0] > vals[-1]  # k=1 far above k=5 on 3-blob data
    with pytest.raises(ValueError):
        elbow_inertias([], [1, 2])
    with pytest.raises(ValueError):
        elbow_inertias(X, [])
    with pytest.raises(ValueError):
        elbow_inertias(X, [1, 0])
