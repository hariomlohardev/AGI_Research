from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .tree import DecisionTreeClassifier
from .utils import if_empty_raise


def bootstrap_indices(n: int, random_state: int | None = None) -> list[int]:
    """Return ``n`` indices sampled with replacement from ``range(n)``.

    Uses ``random.Random(random_state)`` so the output is deterministic when
    ``random_state`` is not ``None``. Duplicates are expected — the expected
    number of distinct indices is ``n * (1 - 1/e) ≈ 0.632 * n``.

    Raises:
        ValueError: if ``n < 1``.
    """

    if n < 1:
        raise ValueError("n can't be less than 1 ")

    if random_state is not None:
        sample_idxs = random.Random(random_state).choices(range(n), k=n)
        return sample_idxs

    else:
        sample_idxs = random.choices(range(n), k=n)
        return sample_idxs


def bootstrap_sample(
    X: Sequence[Sequence[Any]], y: Sequence[Any], random_state: int | None = None
) -> tuple[list[list[Any]], list[Any]]:
    """Return ``(X_sample, y_sample)`` of size ``n = len(X)`` drawn with replacement.

    Row↔label pairing is preserved: ``X_sample[i]`` comes from the same
    original row as ``y_sample[i]``. Sampling uses ``bootstrap_indices`` /
    ``random.Random(random_state)`` internally.

    Raises:
        ValueError: if ``X``/``y`` is empty, ``len(X) != len(y)``, rows of
            ``X`` are ragged, or rows have zero features.
    """

    if_empty_raise(X, "X can't be empty")
    if_empty_raise(y, "y can't be empty")

    if len(X) != len(y):
        raise ValueError("length of X should be equal to length of y")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("rows of X are ragged")

    if_empty_raise(X[0], "rows of X have 0 features")

    n = len(X)
    idxs = bootstrap_indices(n, random_state)
    X_sample = [X[idx] for idx in idxs]
    y_sample = [y[idx] for idx in idxs]

    return (X_sample, y_sample)


class RandomForestClassifier:
    """Bagged ensemble of ID3 trees with per-split feature randomness.

    Each of the ``n_estimators`` trees is an ID3 tree (entropy /
    information gain, exact categorical splits) fitted on a bootstrap
    resample of the training data. At **each split** the tree considers only
    a random subset of the currently-available features, size derived from
    ``max_features`` — this is the "feature bagging" decorrelation trick.

    ``max_features`` mapping (applied per split):
    - ``None`` → ``k = n_features``
    - ``int`` → ``k = that int`` (clipped to ``len(available_features)``)
    - ``"sqrt"`` → ``k = max(1, int(sqrt(n_features)))``
    - ``"log2"`` → ``k = max(1, int(log2(n_features)))``

    Majority vote uses first-seen tiebreak (``Counter`` order over training
    labels), making predictions deterministic given ``random_state``.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Store hyper-parameters; the forest is built lazily in ``fit``.

        Raises:
            ValueError: if ``n_estimators < 1``, ``max_depth`` is not ``None``
                and is ``< 1``, ``min_samples_split < 2``, ``max_features``
                is not ``None``, an ``int >= 1``, ``"sqrt"``, or ``"log2"``,
                or ``bootstrap`` is not a ``bool``.
        """

        if n_estimators < 1:
            raise ValueError("n_estimatore can't be less than 1")

        if max_depth is not None and max_depth < 1:
            raise ValueError("max depth is less than 1 and not None")

        if min_samples_split < 2:
            raise ValueError("min_smaple_split can't be less than 2")

        valid_features = {"sqrt", "log2"}
        is_valid_int = isinstance(max_features, int) and max_features >= 1
        if (
            max_features is not None
            and not is_valid_int
            and max_features not in valid_features
        ):
            raise ValueError(
                'max_features must be None, an int >= 1, "sqrt", or "log2".'
            )

        if not isinstance(bootstrap, bool):
            raise TypeError("bootstrap must be a boolean value.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimates = []

    def fit(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> RandomForestClassifier:
        """Fit ``n_estimators`` ID3 trees on bootstrap resamples; return ``self``.

        Uses a single ``random.Random(random_state)`` stream for both
        bootstrap draws and per-split feature subsets, so
        ``random_state=0`` is reproducible. When ``bootstrap`` is ``False``,
        every tree sees the full training set (feature randomness still
        applies). Recursion inside each tree stops at: pure node,
        ``depth >= max_depth``, fewer than ``min_samples_split`` rows, or no
        multi-valued feature left.

        Raises:
            ValueError: if ``X``/``y`` is empty, ``len(X) != len(y)``, rows
                of ``X`` are ragged, rows have zero features, or
                ``max_features`` as an ``int`` exceeds ``n_features``.
        """
        if_empty_raise(X, "X can't be empty")
        if_empty_raise(y, "y can't be empty")
        if len(X) != len(y):
            raise ValueError("length of X should be equal to length of y")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("rows of X are ragged")

        if_empty_raise(X[0], "rows of X have 0 features")

        n_features = len(X[0])

        max_features_is_int = isinstance(self.max_features, int)
        if max_features_is_int and self.max_features > n_features:
            raise ValueError("``max_features`` as an ``int`` exceeds ``n_features``")

        self.estimates = []
        self._n_features = n_features
        rng = random.Random(self.random_state)  # one stream, before the loop

        for i in range(self.n_estimators):
            if self.bootstrap:
                seed = rng.randint(0, 2**31 - 1)  # different seed per tree
                X_bootstrap, y_bootstrap = bootstrap_sample(X, y, seed)
            else:
                X_bootstrap = [list(r) for r in X]
                y_bootstrap = list(y)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimates.append(tree)  # <- not local tress

        return self

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Return one majority-voted label per row.

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged, or
                width-mismatched input.
        """
        if_empty_raise(self.estimates, "call fit() first")

        if_empty_raise(X, "X can't be empty")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("rows of X are ragged")

        if len(X[0]) != self._n_features:
            raise ValueError("X has different number of features than training data")

        # 3. collect one prediction per tree, per row
        # all_preds[t][i] = tree t's vote on row i
        all_preds = [tree.predict(X) for tree in self.estimates]

        # 4. majority vote per row, first-seen wins ties
        out = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(len(all_preds))]
            counts = Counter(votes)  # preserves first-seen order
            out.append(max(counts, key=counts.get))

        return out
