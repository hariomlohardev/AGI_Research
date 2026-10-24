from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Any

from .utils import if_empty_raise


def entropy(labels: Sequence[Hashable]) -> float:
    """Return the base-2 Shannon entropy (in bits) of a label multiset.

    Pure input gives exactly ``0.0``; a 50/50 split gives exactly ``1.0``.

    Raises:
        ValueError: if ``labels`` is empty.
    """
    if_empty_raise(labels, "labels can't be empty")

    n = len(labels)
    count_labels = Counter(labels)
    output: float = 0.0

    for count in count_labels.values():
        prob = count / n
        output -= prob * math.log2(prob)

    return output


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

    if_empty_raise(X, "X can't be empty")
    if_empty_raise(y, "y can't be empty")

    if len(X) != len(y):
        raise ValueError(
            "buddy this will not work then length of X and y should be exectly same :("
        )

    if not all(len(row) == len(X[0]) for row in X):
        raise ValueError("Mann your rows of ``X`` is ragged :((")

    if len(X[0]) == 0:
        raise ValueError("Rows of X has zero features mann , what are you doing ")

    n_features = len(X[0])
    if not 0 <= feature_idx < n_features:
        raise ValueError("``feature_idx`` is out of range")

    groups = {}

    for row, label in zip(X, y):
        feature_value = row[feature_idx]

        if feature_value not in groups:
            groups[feature_value] = []

        groups[feature_value].append(label)

    _entropy = entropy(y)

    for child_labels in groups.values():
        weight = len(child_labels) / len(y)
        entropy_child_labels = entropy(child_labels)
        _entropy -= weight * entropy_child_labels

    return _entropy


def majority_label(labels: Sequence[Any]) -> Any:
    """Return the most common label.

    Ties break toward the first-seen label. This is a documented
    convention, not a mathematical fact.

    Raises:
        ValueError: if ``labels`` is empty.
    """

    if_empty_raise(labels, "labels can't be empty")

    counts = Counter(labels)
    return max(counts, key=counts.get)


def format_tree(tree: dict, feature_names: Sequence[str] | None = None) -> str:
    """Render a ``to_dict`` tree as an indented, human-readable string.

    Without ``feature_names``, splits render as ``x[j] = value``.

    Raises:
        ValueError: if ``tree`` is not a non-empty node dict.
    """
    if not isinstance(tree, dict) or not tree:
        raise ValueError("tree must be a non-empty dictionary")

    lines = []

    def walk(node: dict, indent: int = 0):
        if "label" in node:
            lines.append(" " * indent + f"Predict: {node['label']}")
            return

        feature_idx = node["feature"]

        if feature_names is not None:
            feature = feature_names[feature_idx]
        else:
            feature = f"x[{feature_idx}]"

        for value, child in node["children"].items():
            lines.append(" " * indent + f"{feature} = {value}:")
            walk(child, indent + 2)

    walk(tree)

    return "\n".join(lines)


class DecisionTreeClassifier:
    """Recursive ID3 decision tree for categorical features."""

    def __init__(
        self, max_depth: int | None = None, min_samples_split: int = 2
    ) -> None:
        """Store stopping rules; the tree itself is built lazily in ``fit``.

        Raises:
            ValueError: if ``max_depth`` is not ``None`` and is smaller
                than 1, or if ``min_samples_split`` is smaller than 2.
        """
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth shout be less than 1 and not None")

        if min_samples_split < 2:
            raise ValueError("min_samples_split not be smaller than 2")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> DecisionTreeClassifier:
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
        if_empty_raise(X, "X can't be empty")
        if_empty_raise(y, "y can't be empty")

        if len(X) != len(y):
            raise ValueError(
                "buddy this will not work then length of X and y should be exectly same :("
            )

        if not all(len(row) == len(X[0]) for row in X):
            raise ValueError("Mann your rows of ``X`` is ragged :((")

        if len(X[0]) == 0:
            raise ValueError("Rows of X has zero features mann , what are you doing ")

        self._n_features = len(X[0])

        def build(
            rows: Sequence[Sequence[Any]],
            labels: Sequence[Any],
            depth: int,
            available_features: Sequence[int],
        ) -> dict:

            majority = majority_label(labels)

            if all(label == labels[0] for label in labels):
                return {"label": labels[0]}

            if self.max_depth is not None and depth >= self.max_depth:
                return {"label": majority}

            if len(labels) < self.min_samples_split:
                return {"label": majority}

            best_feature = None
            best_gain = -1.0

            for feature_idx in available_features:
                feature_values = set(row[feature_idx] for row in rows)

                if len(feature_values) <= 1:
                    continue

                gain = information_gain(rows, labels, feature_idx)

                if best_feature is None or gain > best_gain:
                    best_feature = feature_idx
                    best_gain = gain

            if best_feature is None:
                return {"label": majority}

            children_rows = {}
            children_labels = {}

            for row, label in zip(rows, labels):
                value = row[best_feature]

                if value not in children_rows:
                    children_rows[value] = []
                    children_labels[value] = []

                children_rows[value].append(row)
                children_labels[value].append(label)

            remaining_features = [
                idx for idx in available_features if idx != best_feature
            ]

            children = {}

            for value in children_rows:
                children[value] = build(
                    children_rows[value],
                    children_labels[value],
                    depth + 1,
                    remaining_features,
                )

            return {
                "feature": best_feature,
                "children": children,
                "majority": majority,
            }

        self._tree = build(
            X,
            y,
            0,
            list(range(self._n_features)),
        )

        return self

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Return one label per row.

        An **unseen** feature value falls back to that node's majority
        label. This is a documented convention, not a mathematical fact
        (same spirit as Day 14's ``0.0``-on-empty-denominator rule).

        Raises:
            ValueError: if called before ``fit``, or on empty, ragged,
                or width-mismatched input.
        """
        if not hasattr(self, "_tree"):
            raise ValueError("DecisionTreeID3 must be fitted before predict")

        if_empty_raise(X, "X can't be empty")

        if not all(len(row) == len(X[0]) for row in X):
            raise ValueError("Mann your rows of ``X`` is ragged :((")

        if len(X[0]) != self._n_features:
            raise ValueError("X has different number of features than training data")

        def walk(row: Sequence[Any], node: dict) -> Any:
            if "label" in node:
                return node["label"]

            feature_idx = node["feature"]
            value = row[feature_idx]

            if value not in node["children"]:
                return node["majority"]

            return walk(row, node["children"][value])

        return [walk(row, self._tree) for row in X]

    def to_dict(self) -> dict:
        """Return the learned tree as nested dicts.

        A leaf is ``{"label": <label>}``; an internal node is
        ``{"feature": <idx>, "children": {<value>: <subtree>},
        "majority": <label>}``.

        Raises:
            ValueError: if called before ``fit``.
        """
        if not hasattr(self, "_tree"):
            raise ValueError("DecisionTreeID3 must be fitted before to_dict")

        return self._tree
