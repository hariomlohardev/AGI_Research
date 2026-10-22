"""Day 21 — honest evaluation on the Titanic splits (stub, see Problems 2-3)."""

from __future__ import annotations


def load_titanic_xy(csv_path: str) -> tuple[list[list[float]], list[int]]:
    """Rebuild Day 20's (X, y) from the raw CSV: drop Embarked-missing rows, mean-impute Age, label-encode Sex/Embarked.

    Features (7): Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

    Raises:
        ValueError: if the file has no usable data rows.
    """
    raise NotImplementedError


def comparison_table(results: list[tuple[str, float, float, float, float]]) -> str:
    """Format (name, accuracy, precision, recall, f1) rows as a printed table string.

    Floats render at 3 decimals under a header naming all four metrics.

    Raises:
        ValueError: if `results` is empty.
    """
    raise NotImplementedError


def misclassified_indices(
    y_true: list[int], y_pred: list[int], k: int = 5
) -> list[int]:
    """Return indices of the first `k` positions where prediction disagrees with truth.

    Raises:
        ValueError: if either list is empty or lengths differ.
    """
    raise NotImplementedError
