"""Clustering: Lloyd's K-Means (Week 4 Day 1 scaffolding).

MOVE MAP — copy your tested Week 3 implementation here, then delete this
comment block (see `coding_problems.md` Problem 1):

- `KMeans` + helpers (`euclidean`, `assign_clusters`,
  `recompute_centroids`, `inertia`, `standardize`, `elbow_inertias`) ←
  `month_2/week_3/day_6/code/kmeans.py`

Note the interface asymmetry (document it in the class docstring): the
supervised models take `.fit(X, y)`, but KMeans takes `.fit(X)` — there are
no labels. `.predict(X)` means "assign to nearest centroid".
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .utils import if_empty_raise


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two equal-length points.

    Raises:
        ValueError: if either point is empty or their lengths differ.
    """
    if_empty_raise(a, "a is empty")
    if_empty_raise(b, "b is empty")

    if len(a) != len(b):
        raise ValueError("The lenght of a and b is different")

    distance = math.sqrt(sum((x2 - x1) ** 2 for x1, x2 in zip(a, b)))
    return distance


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
    if_empty_raise(X, "X is empty")
    if_empty_raise(centroids, "centroids is empty")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("rows of X are ragged")

    if len(X[0]) == 0:
        raise ValueError("points are zero-dimensional")

    if len(X[0]) != len(centroids[0]):
        raise ValueError("centroid's dimensionality differs from the points")

    n = len(X)
    out = []
    for i in range(n):
        row1 = X[i]
        temp_dist = []
        for j in range(len(centroids)):
            row2 = centroids[j]
            dist = euclidean(row1, row2)
            temp_dist.append(dist)
        out.append(temp_dist.index(min(temp_dist)))

    return out


def recompute_centroids(
    X: Sequence[Sequence[float]],
    labels: Sequence[int],
    k: int,
    prev_centroids: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return the mean of each cluster's assigned points.

    is a documented convention, not a mathematical fact.

    Raises:
    A cluster with no assigned points keeps its previous centroid. This
        ValueError: if ``X`` is empty, ``len(X) != len(labels)``, rows of
            ``X`` are ragged, ``k < 1``, any label is outside
            ``0 <= label < k``, or a previous centroid's dimensionality
            differs from the points'.
    """

    if_empty_raise(X, "X is empty")

    if len(X) != len(labels):
        raise ValueError("lenths of X and labels doiffer")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("rows of X are ragged")

    if k < 1:
        raise ValueError("k cant be less than 1 ")

    if any(label < 0 or label >= k for label in labels):
        raise ValueError("labels should follow this ``0 <= label < k``")

    dim = len(X[0])

    for i, centroid in enumerate(prev_centroids):
        if len(centroid) != dim:
            raise ValueError(
                f"Dimensionality of prev_centroids[{i}] ({len(centroid)}) "
                f"differs from points ({dim})."
            )

    new_centroids = []
    for j in range(k):
        members = [X[i] for i in range(len(X)) if labels[i] == j]

        if len(members) == 0:
            new_centroids.append(list(prev_centroids[j]))

        else:
            mean = [sum(row[d] for row in members) / len(members) for d in range(dim)]
            new_centroids.append(mean)

    return new_centroids


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

    if_empty_raise(X, "X is empty")

    if len(X) != len(labels):
        raise ValueError("lenths of X and labels doiffer")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("rows of X are ragged")

    if any(label < 0 or label >= len(centroids) for label in labels):
        raise ValueError("labels should follow this ``0 <= label < len(centroids)``")

    output = 0

    for row, label in zip(X, labels):
        output += euclidean(row, centroids[label]) ** 2

    return output


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
    if_empty_raise(X, "X is empty")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("rows of X are ragged")

    if len(X[0]) == 0:
        raise ValueError("length of rows is zero ")

    X_std, means, stds = [],[],[]

    columns = [list(col) for col in zip(*X)]
    for col in columns:
        n = len(col)
        mean = sum(col) / (n)
        std = math.sqrt(sum((x - mean) ** 2 for x in col) / n)
        if std == 0:
            x_std = [(value - mean) / 1 for value in col]
        else:
            x_std = [(value - mean) / std for value in col]

        means.append(mean)
        stds.append(std)
        X_std.append(x_std)

    X_std = [list(row) for row in zip(*X_std)]

    return (X_std, means, stds)


class KMeans:
    """Lloyd's K-Means: alternating assignment + mean-update until convergence.

    NOTE: supervised models take `.fit(X, y)`, but KMeans takes `.fit(X)`

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
        if n_clusters < 1:
            raise ValueError("n_clusters should be more than or equal to 1")

        if max_iters < 1:
            raise ValueError("max_iters should be more than or equal to 1")

        if tol < 0:
            raise ValueError("tol can't be less than 0")

        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.n_iters_ = 0

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

        if_empty_raise(X, "X is empty")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("rows of X are ragged")

        if len(X[0]) == 0:
            raise ValueError("length of rows is zero ")

        if self.n_clusters > len(X):
            raise ValueError("``n_clusters`` exceeds the number of rows.")

        rng = random.Random(self.random_state)

        centroids = [list(p) for p in rng.sample(list(X), self.n_clusters)]
        n_iters = 0
        for _ in range(self.max_iters):
            labels = assign_clusters(X, centroids)
            new_centroids = recompute_centroids(X, labels, self.n_clusters, centroids)
            shift = max(euclidean(o, n) for o, n in zip(centroids, new_centroids))
            centroids, n_iters = new_centroids, n_iters + 1
            if shift <= self.tol:
                break
        self.cluster_centers_, self.n_iters_ = centroids, n_iters

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Assign each row to its nearest fitted centroid (no re-fitting).

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged, or
                width-mismatched input.
        """

        if self.cluster_centers_ == None:
            raise ValueError("call fit() first ")

        if_empty_raise(X, "X is empty")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("rows of X are ragged")

        if len(X[0]) == 0:
            raise ValueError("length of rows is zero ")

        if self.n_clusters > len(X):
            raise ValueError("``n_clusters`` exceeds the number of rows.")

        return assign_clusters(X, self.cluster_centers_)


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
    if_empty_raise(X, "X is empty")
    if_empty_raise(k_range, "k_range is empty")

    if any(k < 1 for k in k_range):
        raise ValueError("k_range must contain only k >= 1")

    values = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state).fit(X)
        values.append(inertia(X, model.predict(X), model.cluster_centers_))
    return values
