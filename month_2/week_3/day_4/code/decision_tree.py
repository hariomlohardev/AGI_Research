"""ID3 decision tree classifier from scratch (categorical features).

Stdlib only (``math`` for ``log2``). Data are plain Python lists: ``X``
is rows × columns of hashable categorical values (strings, ints, ...),
``y`` is a list of hashable labels. No numpy, no sklearn — same
discipline as Days 13–15, now with a recursive splitter instead of
gradient descent.

Scope (deliberate, documented — not missing): exact-value categorical
splits only. No midpoint threshold search for continuous features (that
is CART, not ID3) and no gain-ratio normalisation for many-valued
attributes (that is C4.5). Both are named as follow-ups in ``learn.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any


def entropy(labels: Sequence[Hashable]) -> float:
    """Return the base-2 Shannon entropy (in bits) of a label multiset.

    Pure input gives exactly ``0.0``; a 50/50 split gives exactly ``1.0``.

    Raises:
        ValueError: if ``labels`` is empty.
    """
    raise NotImplementedError


def information_gain(
    X: Sequence[Sequence[Any]], y: Sequence[Any], feature_idx: int
) -> float:
    """Return ``H(y)`` minus the size-weighted child entropies.

    The split groups ``y`` by the values in column ``feature_idx`` of
    ``X``: ``IG = H(parent) - sum_v (|S_v|/|S|) * H(S_v)``.

    Raises:
        ValueError: if ``X``/``y`` is empty, ``len(X) != len(y)``, rows
            of ``X`` are ragged, rows have zero features, or
            ``feature_idx`` is out of range
            (``0 <= feature_idx < n_features``). A single-valued column
            yields exactly ``0.0`` (no split is possible).
    """
    raise NotImplementedError


def majority_label(labels: Sequence[Any]) -> Any:
    """Return the most common label.

    Ties break toward the first-seen label. This is a documented
    convention, not a mathematical fact.

    Raises:
        ValueError: if ``labels`` is empty.
    """
    raise NotImplementedError


def format_tree(
    tree: dict, feature_names: Sequence[str] | None = None
) -> str:
    """Render a ``to_dict`` tree as an indented, human-readable string.

    Without ``feature_names``, splits render as ``x[j] = value``.

    Raises:
        ValueError: if ``tree`` is not a non-empty node dict.
    """
    raise NotImplementedError


class DecisionTreeID3:
    """Recursive ID3 decision tree for categorical features."""

    def __init__(
        self, max_depth: int | None = None, min_samples_split: int = 2
    ) -> None:
        """Store stopping rules; the tree itself is built lazily in ``fit``.

        Raises:
            ValueError: if ``max_depth`` is not ``None`` and is smaller
                than 1, or if ``min_samples_split`` is smaller than 2.
        """
        raise NotImplementedError

    def fit(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> DecisionTreeID3:
        """Build the tree recursively; return ``self`` (sklearn-style).

        At each node, split on the multi-valued feature with the highest
        information gain (ties break toward the lowest feature index —
        deterministic, documented). Recursion stops at: a pure node,
        ``depth >= max_depth``, fewer than ``min_samples_split`` rows,
        or no multi-valued feature left (a used feature is single-valued
        inside each child, so no path reuses a feature).

        Raises:
            ValueError: if ``X``/``y`` is empty, ``len(X) != len(y)``,
                rows of ``X`` are ragged, or rows have zero features.
        """
        raise NotImplementedError

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Return one label per row.

        An **unseen** feature value falls back to that node's majority
        label. This is a documented convention, not a mathematical fact
        (same spirit as Day 14's ``0.0``-on-empty-denominator rule).

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged,
                or width-mismatched input.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Return the learned tree as nested dicts.

        A leaf is ``{"label": <label>}``; an internal node is
        ``{"feature": <idx>, "children": {<value>: <subtree>},
        "majority": <label>}``.

        Raises:
            ValueError: if called before ``fit``.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Experiment 1 — 6-row hand example from learn.md (practice question 1).
    X_hand = [
        ["x", "p"], ["x", "p"], ["x", "q"],
        ["y", "p"], ["y", "q"], ["y", "q"],
    ]
    y_hand = [1, 1, 1, 0, 0, 0]
    print(f"H(S) = {entropy(y_hand):.4f}")
    for j, fname in enumerate(["A", "B"]):
        print(f"IG({fname}) = {information_gain(X_hand, y_hand, j):.4f}")

    # Experiment 2 — classic 14-row Quinlan play-tennis table (inline).
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

    full = DecisionTreeID3().fit(X_tennis, y_tennis)
    print(format_tree(full.to_dict(), FEATURES))

    # Experiment 3 — max_depth=1 vs unlimited (practice question 4).
    for depth in (1, None):
        m = DecisionTreeID3(max_depth=depth).fit(X_tennis, y_tennis)
        preds = m.predict(X_tennis)
        acc = sum(1 for a, p in zip(y_tennis, preds) if a == p) / len(y_tennis)
        print(f"max_depth={depth}: train accuracy = {acc:.3f}")

    # Experiment 4 — unseen value falls back to the node majority.
    fresh = DecisionTreeID3().fit(X_tennis, y_tennis)
    print(
        "unseen Outlook=snow ->",
        fresh.predict([["snow", "mild", "normal", "weak"]]),
    )
