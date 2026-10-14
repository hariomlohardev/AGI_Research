"""Day 20 — train/test splitting by hand (stub, see Problem 3)."""

from __future__ import annotations


def stratified_split(
    X: list[list[float]],
    y: list[int],
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split into train/test preserving each class's share (stratified).

    Shuffles each class with a seeded local RNG (`random.Random(seed)`),
    then takes `test_ratio` from each class into the test set.

    Raises:
        ValueError: on empty input, length mismatch, a `test_ratio`
            outside (0, 1), or a class too small to stratify.
    """
