# Day 18 — K-Means Clustering: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only in the tested
functions (`math` for `sqrt`/`dist`, `random` for initialization — no `numpy`).
Plain Python lists (`X` is rows of floats), not numpy — same discipline as
Days 16–17, now unsupervised: no labels anywhere. `matplotlib` appears only
in `__main__` demos behind a deferred import (same pattern as Day 15), never
in the tested API.

> Data note: everything below uses tiny synthetic 2D datasets generated inline
> with `random.Random(seed).gauss` (fixed seeds → reproducible). No external
> dataset is fetched — the point today is the assign/update loop, local optima,
> and the elbow, not data wrangling. Week 4 Day 2 puts the supervised models on
> a real dataset; K-Means ships as `cluster.py`.

---

## Problem 1 — `euclidean`, `assign_clusters`, `recompute_centroids`, `inertia`, `standardize`

Create `code/kmeans.py` with from-scratch helpers.

```python
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Euclidean distance between two equal-length points."""

def assign_clusters(X: Sequence[Sequence[float]], centroids: Sequence[Sequence[float]]) -> list[int]:
    """Return one cluster index per row: the index of the nearest centroid."""

def recompute_centroids(
    X: Sequence[Sequence[float]], labels: Sequence[int], k: int, prev_centroids: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Return the mean of each cluster's assigned points."""

def inertia(
    X: Sequence[Sequence[float]], labels: Sequence[int], centroids: Sequence[Sequence[float]]
) -> float:
    """Return the within-cluster sum of squared distances (the K-Means objective)."""

def standardize(X: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Return `(X_std, means, stds)` with each column at mean 0, std 1."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises`
contract (docstrings carry the contract — see the Day 10 process fix):

- `euclidean`: raise `ValueError` for empty points or length mismatch.
  Hand check: `euclidean([0, 0], [3, 4]) == 5.0`.
- `assign_clusters`: raise `ValueError` for empty `X`/centroids, ragged rows,
  centroid dimensionality mismatch, or zero-dimensional points. Ties break
  toward the lowest cluster index (first `min` wins — documented convention).
- `recompute_centroids`: raise `ValueError` for empty `X`, `len(X) !=
  len(labels)`, ragged rows, `k < 1`, a label outside `0 <= label < k`, or
  dimensionality mismatch with `prev_centroids`. **Empty-cluster convention**
  (documented, not mathematical): a cluster with no assigned points keeps its
  previous centroid.
- `inertia`: raise `ValueError` for empty `X`, `len(X) != len(labels)`,
  ragged rows, or out-of-range labels. Hand check: `X = [[0.0], [2.0]]`,
  `labels = [0, 0]`, `centroids = [[1.0]]` → exactly `2.0`.
- `standardize`: raise `ValueError` for empty `X`, ragged rows, or
  zero-dimensional points. Zero-variance columns map to all-`0.0` (guard
  `std == 0 → 1.0` internally — documented convention).

## Problem 2 — `KMeans` + elbow

Same file (`code/kmeans.py`):

```python
class KMeans:
    """Lloyd's K-Means: alternating assignment + mean-update until convergence."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Store hyper-parameters; centroids are initialised lazily in `fit`."""

    def fit(self, X: Sequence[Sequence[float]]) -> KMeans:
        """Run Lloyd iterations; return `self` (sklearn-style)."""

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Assign each row to its nearest fitted centroid."""

def elbow_inertias(
    X: Sequence[Sequence[float]], k_range: Sequence[int], random_state: int | None = None
) -> list[float]:
    """Return one fitted inertia per k in `k_range` (for the elbow plot)."""
```

- `KMeans.__init__`: raise `ValueError` for `n_clusters < 1`, `max_iters < 1`,
  or `tol < 0`. Store `cluster_centers_: list[list[float]] | None = None` and
  `n_iters_: int = 0` up front (attribute lifetime visible — see the Day-17
  review's promoted theme).
- `fit`: raise `ValueError` for empty `X`, ragged rows, zero-dimensional
  points, or `n_clusters > n_rows` (can't sample that many distinct seeds).
  Initialise by choosing `n_clusters` **distinct** data points via
  `random.Random(random_state).sample`. Then alternate
  `assign_clusters`/`recompute_centroids` until the largest centroid shift
  (`euclidean` old vs new) is `<= tol`, or `max_iters` iterations run.
  Record iterations in `n_iters_` and expose fitted centroids as
  `cluster_centers_`. Deterministic given `random_state`.
- `predict`: raise `ValueError` if called before `fit`, or on
  empty/ragged/width-mismatched input. Returns cluster indices (nearest
  fitted centroid — no re-fitting).
- `elbow_inertias`: raise `ValueError` for empty `X`, empty `k_range`, or any
  `k < 1`. Fit one `KMeans(n_clusters=k, random_state=random_state)` per k
  and return the fitted `inertia` values in order. (Inertia is
  non-increasing in k up to local-optima noise — your `__main__` plot should
  show the bend, not a proof.)
- Scope note, documented in the module docstring: Lloyd's algorithm only
  (no k-means++ seeding — uniform `sample` is today's init, named as the
  follow-up), Euclidean distance only, no `predict_proba` /
  `cluster_scores_`. Empty clusters keep their centroid (Problem 1's
  convention). Week 4's `cluster.py` ships this class behind a clean import.

### Required `__main__` experiments

When you run `python code/kmeans.py` it must print all four (keep the numbers):

1. Three blobs: generate 60 points (20 each around `(0, 0)`, `(5, 5)`,
   `(0, 5)`, std `0.5`, seed fixed), fit `k=3`, print centroids (≈ the three
   centres), `n_iters_`, and inertia. Then `matplotlib` scatter (deferred
   import, `try/except ImportError` like Day 15): points coloured by label,
   centroids marked with `X`.
2. Initialization sensitivity (practice Q1's setup): fit `k=3` on *overlapping*
   blobs (same centres, std `2.0`) with `random_state=0,1,2`, print the three
   inertias — they should visibly differ (local optima), with a one-line note.
3. Elbow (practice Q4): for `k = 1..8` print `k` vs inertia via
   `elbow_inertias`, plus the `matplotlib` line plot — the bend sits at `k=3`.
4. Scaling demo (practice Q3–Q4): multiply feature 0 by 1000, refit, print new
   centroids/inertia (clustering follows the stretched axis); then
   `standardize` first and refit (sensible clusters return), with a one-line
   note connecting to Week 1 Day 5's variance discussion.

### Running tests

```bash
cd month_2/week_3/day_6 && python -m pytest code/tests/ -v
```

### Done when (from the source plan)

Your K-Means correctly finds 3 visually obvious clusters on synthetic data,
and your elbow-method plot shows a clear "bend" at the true number of
clusters.
