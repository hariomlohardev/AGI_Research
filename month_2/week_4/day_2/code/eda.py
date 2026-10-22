"""Day 20 — EDA helpers (stubs): load, missing report, balance, summaries.

All input rows are raw string rows as returned by `load_csv` (see
`coding_problems.md` Problem 1). Nothing here parses the Titanic file
specifically — the CSV pins live in the tests.
"""

from __future__ import annotations

import csv
from collections import Counter

def load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    """Read `path` and return `(header, rows)` as raw strings.

    Raises:
        ValueError: if the file is empty or rows are ragged.
    """
    with open(path , encoding="utf-8") as file:
        reader  = csv.reader(file)
        try:
            # Safely grab the first row as the header
            header = next(reader)
        except StopIteration:
            raise ValueError("the file is empty")



        rows = []
        expected_length = len(header)
        
        for row in reader:
            # If a row has a different number of columns than the header, it is ragged
            if len(row) != expected_length:
                raise ValueError("rows are ragged")
            rows.append(row)
            
    return header, rows



def is_missing(value: str | None) -> bool:
    """Return True for a missing entry (`None` or blank/whitespace-only).

    Raises:
        ValueError: never — any input classifies, that is the contract.
    """

    if value is None:
        return True

    return value.strip() == ""




def missing_counts(header: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Count missing entries per column, keyed by column name.

    Raises:
        ValueError: if rows are empty or a row disagrees with the header.
    """
    if not rows or len(rows) == 0:
        raise ValueError("rows are empty")

    len_header = len(header)

    for row in rows:
        if len(row) != len_header:
            raise ValueError(f"the row : {row} disagrees with the header.")

    counts = {col: 0 for col in header}

    for row in rows:
        for idx , value in enumerate(row):
            if value is None or value.strip() == "":
                col_name = header[idx]
                counts[col_name] += 1

    return counts






def class_balance(labels: list[int]) -> dict[int, int]:
    """Count occurrences per label, e.g. `{0: 549, 1: 342}`.

    Raises:
        ValueError: if `labels` is empty.
    """

    if not labels:
        raise ValueError("lavels is empty")

    return Counter(labels)


def numeric_summary(values: list[float]) -> dict[str, float]:
    """Return `count`/`min`/`max`/`mean` over `values`.

    Raises:
        ValueError: if `values` is empty.
    """

    if not values:
        raise ValueError("vlaues is empty")

    count = len(values)

    return {
        "count" : count,
        "min" : min(values),
        "max" : max(values),
        "mean" : sum(values)/count
    }
