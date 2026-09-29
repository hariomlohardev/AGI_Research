"""K-Means clustering from scratch (Lloyd's algorithm, Euclidean distance).

Stdlib only in the tested API (``math`` for ``sqrt``/``dist``, ``random``
for initialization). Data are plain Python lists: ``X`` is rows of floats.
No labels anywhere — this is the first unsupervised day. No numpy, no
sklearn. ``matplotlib`` appears only in ``__main__`` demos behind a deferred
import (same pattern as Day 15), never in the tested functions.

Scope (deliberate, documented — not missing): Lloyd's alternating
assign/update loop only. Uniform ``Random.sample`` initialization (no
k-means++ — named as the follow-up), Euclidean distance only (no Mahalanobis
/ cosine), no ``predict_proba``. Empty clusters keep their previous centroid
(documented convention). Week 4's ``cluster.py`` ships this class behind a
clean import.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import random


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two equal-length points.

    Raises:
        ValueError: if either point is empty or their lengths differ.
    """
    raise NotImplementedError


def assign_clusters(
    X: Sequence[Sequence[float]], centroids: Sequence[Sequence[float]]
) -> list[int]:
    """Return one cluster index per row: the nearest centroid's index.

    Ties break toward the lowest cluster index (first ``min`` wins). This
    is a documented convention, not a mathematical fact.

    Raises:
        ValueError: if ``X`` or ``centroids`` is empty, rows of ``X`` are
            ragged, points are zero-dimensional, or a centroid's
            dimensionality differs from the points'.
    """
    raise NotImplementedError


def recompute_centroids(
    X: Sequence[Sequence[float]],
    labels: Sequence[int],
    k: int,
    prev_centroids: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return the mean of each cluster's assigned points.

    A cluster with no assigned points keeps its previous centroid. This
    is a documented convention, not a mathematical fact.

    Raises:
        ValueError: if ``X`` is empty, ``len(X) != len(labels)``, rows of
            ``X`` are ragged, ``k < 1``, any label is outside
            ``0 <= label < k``, or a previous centroid's dimensionality
            differs from the points'.
    """
    raise NotImplementedError


def inertia(
    X: Sequence[Sequence[float]],
    labels: Sequence[int],
    centroids: Sequence[Sequence[float]],
) -> float:
    """Return the within-cluster sum of squared distances (the objective).

    ``sum over rows of euclidean(row, centroid[label]) ** 2``. Lloyd's
    loop never increases this value.

    Raises:
        ValueError: if ``X`` is empty, ``len(X) != len(labels)``, rows of
            ``X`` are ragged, or any label is outside
            ``0 <= label < len(centroids)``.
    """
    raise NotImplementedError


def standardize(
    X: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Return ``(X_std, means, stds)`` with each column at mean 0, std 1.

    Uses the population std (divide by ``n``). A zero-variance column maps
    to all-``0.0`` (internally guards ``std == 0`` as ``1.0``). This is a
    documented convention, not a mathematical fact.

    Raises:
        ValueError: if ``X`` is empty, rows of ``X`` are ragged, or points
            are zero-dimensional.
    """
    raise NotImplementedError


class KMeans:
    """Lloyd's K-Means: alternating assignment + mean-update until convergence.

    Each iteration (1) assigns every point to its nearest centroid, then
    (2) moves each centroid to its assigned points' mean. Step (1) can't
    increase the objective given fixed centroids; step (2) can't increase
    it given fixed assignments — so inertia never increases and the loop
    always converges (to a *local* optimum, not necessarily the global one).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Store hyper-parameters; centroids are initialised lazily in ``fit``.

        Raises:
            ValueError: if ``n_clusters < 1``, ``max_iters < 1``, or
                ``tol < 0``.
        """
        raise NotImplementedError

    def fit(self, X: Sequence[Sequence[float]]) -> KMeans:
        """Run Lloyd iterations to convergence; return ``self``.

        Initialises by choosing ``n_clusters`` distinct data points via
        ``random.Random(random_state).sample``. Iterates until the largest
        centroid shift is ``<= tol`` or ``max_iters`` iterations run.
        Records the iteration count in ``n_iters_`` and the fitted centroids
        in ``cluster_centers_``. Deterministic given ``random_state``.

        Raises:
            ValueError: if ``X`` is empty, rows of ``X`` are ragged, points
                are zero-dimensional, or ``n_clusters`` exceeds the number
                of rows.
        """
        raise NotImplementedError

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Assign each row to its nearest fitted centroid (no re-fitting).

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged, or
                width-mismatched input.
        """
        raise NotImplementedError


def elbow_inertias(
    X: Sequence[Sequence[float]],
    k_range: Sequence[int],
    random_state: int | None = None,
) -> list[float]:
    """Return one fitted inertia per k in ``k_range``, in order.

    Fits ``KMeans(n_clusters=k, random_state=random_state)`` per k and
    reports ``inertia`` on the same ``X``. For the elbow plot in
    ``__main__`` — the bend, not a proof.

    Raises:
        ValueError: if ``X`` is empty or ``k_range`` is empty or contains
            any ``k < 1``.
    """
    raise NotImplementedError


def _make_blobs(
    centres: Sequence[Sequence[float]],
    per_blob: int,
    std: float,
    seed: int,
) -> list[list[float]]:
    """Generate synthetic 2D blobs with ``random.Random(seed).gauss`` (demo/test helper)."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    for cx, cy in centres:
        for _ in range(per_blob):
            X.append([rng.gauss(cx, std), rng.gauss(cy, std)])
    return X


if __name__ == "__main__":
    CENTRES = [[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]]

    # Experiment 1 — three well-separated blobs, k=3.
    print("=== Experiment 1: three blobs ===")
    X3 = _make_blobs(CENTRES, per_blob=20, std=0.5, seed=0)
    km = KMeans(n_clusters=3, random_state=0).fit(X3)
    print(f"  centroids: {[[round(v, 2) for v in c] for c in km.cluster_centers_]}")
    print(f"  n_iters_: {km.n_iters_}")
    print(f"  inertia: {inertia(X3, km.predict(X3), km.cluster_centers_):.2f}")
    try:
        import matplotlib.pyplot as plt

        labels = km.predict(X3)
        colours = ["C0", "C1", "C2"]
        for i, (x, y_) in enumerate(X3):
            plt.scatter(x, y_, c=colours[labels[i]], s=12)
        for cx, cy in km.cluster_centers_:
            plt.scatter(cx, cy, c="black", marker="X", s=120)
        plt.title("K-Means k=3 on synthetic blobs")
        plt.show()
    except ImportError:
        print("  (matplotlib missing — skipping plot)")

    # Experiment 2 — initialization sensitivity on overlapping blobs.
    print("\n=== Experiment 2: init sensitivity (overlapping blobs, std=2.0) ===")
    Xo = _make_blobs(CENTRES, per_blob=20, std=2.0, seed=1)
    for seed in (0, 1, 2):
        m = KMeans(n_clusters=3, random_state=seed).fit(Xo)
        print(f"  seed {seed}: inertia = {inertia(Xo, m.predict(Xo), m.cluster_centers_):.2f}")
    print("  note: inertias differ — Lloyd lands in different local optima per init.")

    # Experiment 3 — elbow k=1..8 (practice Q4).
    print("\n=== Experiment 3: elbow (k=1..8) ===")
    ks = list(range(1, 9))
    vals = elbow_inertias(X3, ks, random_state=0)
    for k, v in zip(ks, vals):
        print(f"  k={k}: inertia = {v:.2f}")
    try:
        import matplotlib.pyplot as plt

        plt.plot(ks, vals, marker="o")
        plt.xlabel("k")
        plt.ylabel("inertia")
        plt.title("Elbow — bend at k=3")
        plt.show()
    except ImportError:
        print("  (matplotlib missing — skipping plot)")

    # Experiment 4 — feature scaling distortion + rescue (practice Q3).
    print("\n=== Experiment 4: x1000 stretch, then standardize ===")
    Xbig = [[x * 1000.0, y_] for x, y_ in X3]
    stretched = KMeans(n_clusters=3, random_state=0).fit(Xbig)
    print(f"  stretched centroids: {[[round(v, 1) for v in c] for c in stretched.cluster_centers_]}")
    print(f"  stretched inertia: {inertia(Xbig, stretched.predict(Xbig), stretched.cluster_centers_):.1f}")
    Xs, _, _ = standardize(Xbig)
    rescued = KMeans(n_clusters=3, random_state=0).fit(Xs)
    print(f"  standardized inertia: {inertia(Xs, rescued.predict(Xs), rescued.cluster_centers_):.2f}")
    print("  note: raw Euclidean distance follows the stretched axis (Week 1 Day 5: variance runs the metric); standardizing restores balance.")
