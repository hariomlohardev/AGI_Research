"""Random Forests from scratch — bootstrap + feature-random ID3 ensemble.

Stdlib only (``math`` for ``log2``/``sqrt``, ``random`` for bootstrap and
per-split feature sampling, ``collections.Counter``). Data are plain Python
lists: ``X`` is rows × columns of hashable categorical values, ``y`` is a
list of hashable labels. No numpy, no sklearn.

Scope (deliberate, documented — not missing): categorical ID3 splits only
(exact-value matches, no CART midpoint thresholds, no C4.5 gain-ratio),
bagging + per-split feature randomness via ``max_features`` (``None`` /
``int`` / ``"sqrt"`` / ``"log2"``), majority-vote prediction. OOB error
aggregation, ``feature_importances_``, and ``predict_proba`` are out of
scope — you only observe the OOB *size* today. Week 4's ``ensemble.py``
ships this class behind a clean import.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from collections import Counter
import math
import random
from typing import Any


def bootstrap_indices(n: int, random_state: int | None = None) -> list[int]:
    """Return ``n`` indices sampled with replacement from ``range(n)``.

    Uses ``random.Random(random_state)`` so the output is deterministic when
    ``random_state`` is not ``None``. Duplicates are expected — the expected
    number of distinct indices is ``n * (1 - 1/e) ≈ 0.632 * n``.

    Raises:
        ValueError: if ``n < 1``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

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
        raise NotImplementedError

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Return one majority-voted label per row.

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged, or
                width-mismatched input.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Experiment 1 — bootstrap hand demo: duplicates + OOB sizes.
    print("=== Experiment 1: bootstrap ===")
    idx = bootstrap_indices(14, random_state=0)
    print(f"bootstrap_indices(14, seed=0): {idx}")
    oob = sorted(set(range(14)) - set(idx))
    print(f"  OOB indices ({len(oob)} rows): {oob}")
    # Mean OOB size across 500 resamples of n=14
    oob_sizes = []
    for seed in range(500):
        s = set(bootstrap_indices(14, random_state=seed))
        oob_sizes.append(14 - len(s))
    mean_oob = sum(oob_sizes) / len(oob_sizes)
    print(f"  mean OOB size over 500 resamples (n=14): {mean_oob:.2f}  (~5.1 expected)")
    # Large-n unique-count
    large = bootstrap_indices(1000, random_state=0)
    uniq = len(set(large))
    print(f"  n=1000 unique count: {uniq}  (~632 expected)")

    # Shared 14-row tennis data (same as Day 16).
    FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
    X_tennis = [
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
    y_tennis = [
        "no", "no", "yes", "yes", "yes", "no", "yes",
        "no", "yes", "yes", "yes", "yes", "yes", "no",
    ]

    # Experiment 2 — single tree vs forest.
    print("\n=== Experiment 2: single tree vs forest ===")
    single = RandomForestClassifier(
        n_estimators=1, max_features=None, bootstrap=False, random_state=0
    ).fit(X_tennis, y_tennis)
    forest = RandomForestClassifier(
        n_estimators=20, max_features="sqrt", random_state=0
    ).fit(X_tennis, y_tennis)
    for name, model in [("single tree (1, no bag, all feats)", single), ("forest (20, sqrt)", forest)]:
        preds = model.predict(X_tennis)
        acc = sum(1 for a, p in zip(y_tennis, preds) if a == p) / len(y_tennis)
        print(f"  {name}: train accuracy = {acc:.3f}")
    # Stability across seeds
    print("  stability check (train acc across random_state 0/1/2):")
    for seed in (0, 1, 2):
        s_acc = sum(1 for a, p in zip(y_tennis, RandomForestClassifier(n_estimators=1, max_features=None, bootstrap=False, random_state=seed).fit(X_tennis, y_tennis).predict(X_tennis)) if a == p) / len(y_tennis)
        f_acc = sum(1 for a, p in zip(y_tennis, RandomForestClassifier(n_estimators=20, max_features="sqrt", random_state=seed).fit(X_tennis, y_tennis).predict(X_tennis)) if a == p) / len(y_tennis)
        print(f"    seed {seed}: single={s_acc:.3f}  forest={f_acc:.3f}")

    # Experiment 3 — n_estimators sweep (practice Q4).
    print("\n=== Experiment 3: n_estimators sweep (1, 5, 20, 100) ===")
    # Hold out last 4 rows as a tiny test set; train on first 10.
    X_train, y_train = X_tennis[:10], y_tennis[:10]
    X_test, y_test = X_tennis[10:], y_tennis[10:]
    for B in (1, 5, 20, 100):
        m = RandomForestClassifier(n_estimators=B, random_state=0).fit(X_train, y_train)
        train_acc = sum(1 for a, p in zip(y_train, m.predict(X_train)) if a == p) / len(y_train)
        test_acc = sum(1 for a, p in zip(y_test, m.predict(X_test)) if a == p) / len(y_test)
        print(f"  B={B:3d}: train={train_acc:.3f}  test={test_acc:.3f}")

    # Experiment 4 — feature-randomness ablation: bag-only vs real RF.
    print("\n=== Experiment 4: feature randomness ablation (bag-only vs sqrt) ===")
    for mf in (None, "sqrt"):
        m = RandomForestClassifier(n_estimators=30, max_features=mf, random_state=0).fit(X_train, y_train)
        train_acc = sum(1 for a, p in zip(y_train, m.predict(X_train)) if a == p) / len(y_train)
        test_acc = sum(1 for a, p in zip(y_test, m.predict(X_test)) if a == p) / len(y_test)
        # Disagreement rate across trees (proxy for decorrelation)
        # For this demo count per-sample vote entropy: fraction of samples where not all trees agree.
        print(f"  max_features={str(mf):5s}: train={train_acc:.3f}  test={test_acc:.3f}  — "
              f"{'bagging only (all feats each split)' if mf is None else 'real RF (sqrt per split, less correlated)'}")
