"""Day 22 — compare minisklearn against real sklearn (see Problems 2-3).

`load_same_split` is provided (not a stub): it reuses Day 21's rebuild and
Day 20's split in place, so nothing from those days is rewritten or copied.
Implement the scoring/gap helpers below it.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

DAY2CODE = pathlib.Path(__file__).resolve().parents[2] / "day_2" / "code"
DAY3CODE = pathlib.Path(__file__).resolve().parents[2] / "day_3" / "code"
MINISKLEARN_SRC = pathlib.Path(__file__).resolve().parents[3] / "minisklearn" / "src"

sys.path.insert(0, str(DAY2CODE))
sys.path.insert(0, str(DAY3CODE))
sys.path.insert(0, str(MINISKLEARN_SRC))

from split import stratified_split  # type: ignore[import-not-found]
from evaluate import load_titanic_xy  # type: ignore[import-not-found]
from minisklearn import metrics


def load_same_split(
    csv_path: str,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Reload Day 21's exact Titanic split from the raw CSV.

    Provided helper (already implemented): rebuilds ``(X, y)`` with Day 21's
    cleaning/feature choices, then applies Day 20's stratified split with
    the supplied ratio and seed. Call it; do not reimplement or copy those
    files.

    Raises:
        ValueError: if the CSV has no usable data rows.
    """
    X, y = load_titanic_xy(csv_path)
    return stratified_split(X, y, test_ratio=test_ratio, seed=seed)


def score_classifier(
    model: Any,
    X_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    X_test: Sequence[Sequence[float]],
    y_test: Sequence[int],
) -> dict[str, float]:
    """Fit one classifier and score it with minisklearn's metrics.

    The same function must work for a ``minisklearn`` classifier and the
    corresponding ``sklearn`` classifier because both expose ``.fit`` and
    ``.predict``. Both sides are scored with ``minisklearn.metrics`` so the
    comparison measures model behavior, not metric-definition drift.

    Returns:
        Exactly one dictionary per call, with exactly these keys:
        ``{"accuracy": float, "precision": float, "recall": float,
        "f1": float}``.
        Call once per model: one dictionary for yours, one for sklearn's.

    Raises:
        ValueError: if ``model`` does not expose ``.fit`` and ``.predict``,
            if either split is empty, or if rows and labels have mismatched
            lengths.
    """

    if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
        raise ValueError("The model must expose both '.fit' and '.predict' methods.")

    # 2. Validate that no splits are empty
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Execution failed: One or more data splits are empty.")

    # 3. Validate rows and labels have matching lengths for both splits
    if len(X_train) != len(y_train):
        raise ValueError(f"Mismatched training lengths: {len(X_train)} rows vs {len(y_train)} labels.")
    if len(X_test) != len(y_test):
        raise ValueError(f"Mismatched testing lengths: {len(X_test)} rows vs {len(y_test)} labels.")

    fit  = model.fit(X_train , y_train)

    y_pred = model.predict(X_test)

    return {
        "accuracy": metrics.accuracy_score(y_test , y_pred),
        "precision": metrics.precision_score(y_test , y_pred), 
        "recall": metrics.recall_score(y_test , y_pred),
        "f1": metrics.f1_score(y_test , y_pred)
        }

import math

def accuracy_gap(mine: float, theirs: float) -> float:
    """Return the absolute difference between two accuracy values.

    Raises:
        ValueError: if either accuracy is NaN or infinite.
    """
    # Check if either value is NaN or Infinite
    if math.isnan(mine) or math.isnan(theirs):
        raise ValueError("Accuracy values cannot be NaN.")
        
    if math.isinf(mine) or math.isinf(theirs):
        raise ValueError("Accuracy values cannot be infinite.")

    return abs(mine - theirs)


def gap_verdict(gap: float, tolerance: float = 0.03) -> str:
    """Classify an accuracy gap as a match or as needing investigation.

    Raises:
        ValueError: if ``gap`` is negative or non-finite, or if
            ``tolerance`` is non-positive or non-finite.
    """
    # Validate gap constraints
    if math.isnan(gap) or math.isinf(gap):
        raise ValueError("Gap cannot be non-finite.")
    if gap < 0:
        raise ValueError("Gap cannot be negative.")

    # Validate tolerance constraints
    if math.isnan(tolerance) or math.isinf(tolerance):
        raise ValueError("Tolerance cannot be non-finite.")
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")

    # Classify the verdict
    if gap <= tolerance:
        return "match"
    else:
        return "investigate"
