"""Day 20 — EDA helpers (stubs): load, missing report, balance, summaries.

All input rows are raw string rows as returned by `load_csv` (see
`coding_problems.md` Problem 1). Nothing here parses the Titanic file
specifically — the CSV pins live in the tests.
"""

from __future__ import annotations


def load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read `path` and return `(header, rows)` as raw strings.

    Raises:
        ValueError: if the file is empty or rows are ragged.
    """


def is_missing(value: str | None) -> bool:
    """Return True for a missing entry (`None` or blank/whitespace-only).

    Raises:
        ValueError: never — any input classifies, that is the contract.
    """


def missing_counts(header: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Count missing entries per column, keyed by column name.

    Raises:
        ValueError: if rows are empty or a row disagrees with the header.
    """


def class_balance(labels: list[int]) -> dict[int, int]:
    """Count occurrences per label, e.g. `{0: 549, 1: 342}`.

    Raises:
        ValueError: if `labels` is empty.
    """


def numeric_summary(values: list[float]) -> dict[str, float]:
    """Return `count`/`min`/`max`/`mean` over `values`.

    Raises:
        ValueError: if `values` is empty.
    """
