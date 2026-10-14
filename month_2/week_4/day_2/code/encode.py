"""Day 20 — impute + encode helpers (stubs, see Problem 2).

Generic over any string table — the Titanic wiring (which columns, label
vs one-hot) is your documented call, not a helper's default.
"""

from __future__ import annotations


def to_float_column(rows: list[list[str]], idx: int) -> list[float | None]:
    """Parse column `idx` to floats, `None` where missing.

    Raises:
        ValueError: on a non-missing entry that is not numeric.
    """


def fill_mean(column: list[float | None]) -> tuple[list[float], float]:
    """Replace `None`s with the mean of the present values.

    Returns the filled column and the mean used.

    Raises:
        ValueError: if the column is empty or all entries are missing.
    """


def label_encode(values: list[str]) -> tuple[list[int], dict[str, int]]:
    """Map each distinct value to an int (first-seen order).

    Returns the encoded list and the mapping used.

    Raises:
        ValueError: if `values` is empty.
    """


def one_hot(values: list[str]) -> tuple[list[list[int]], list[str]]:
    """One-hot encode `values`; returns the matrix and category order.

    Raises:
        ValueError: if `values` is empty.
    """
