"""Day 21 — honest evaluation on the Titanic splits (stub, see Problems 2-3)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "day_2" / "code"))

from eda import load_csv, is_missing  # type: ignore[import-not-found]
from encode import to_float_column, fill_mean, label_encode  # type: ignore[import-not-found]

import csv


def load_titanic_xy(csv_path: str) -> tuple[list[list[float]], list[int]]:
    """Rebuild Day 20's (X, y) from the raw CSV: drop Embarked-missing rows, mean-impute Age, label-encode Sex/Embarked.

    Features (7): Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

    Raises:
        ValueError: if the file has no usable data rows.
    """
    header, rows = load_csv(csv_path)

    if not rows or not header:
        raise ValueError("the file has no usable data rows.")

    header_map = {name.strip(): idx for idx, name in enumerate(header)}
    
    # Filter out rows where Embarked is missing
    embarked_idx = header_map["Embarked"]
    valid_rows = []
    for row in rows:
        if len(row) > embarked_idx and row[embarked_idx].strip() != "":
            valid_rows.append(row)

    if not valid_rows:
        raise ValueError("the file has no usable data rows.")

    survived_idx = header_map["Survived"]
    y = [int(float(row[survived_idx])) for row in valid_rows]

    pclass = to_float_column(valid_rows, header_map["Pclass"])
    sex, _ = label_encode([row[header_map["Sex"]] for row in valid_rows])
    age_col = to_float_column(valid_rows, header_map["Age"])
    age, _ = fill_mean(age_col)
    sibsp = to_float_column(valid_rows, header_map["SibSp"])
    parch = to_float_column(valid_rows, header_map["Parch"])
    fare = to_float_column(valid_rows, header_map["Fare"])
    embarked, _ = label_encode([row[header_map["Embarked"]] for row in valid_rows])

    X = []
    for i in range(len(valid_rows)):
        X.append([
            pclass[i],
            float(sex[i]),
            age[i],
            sibsp[i],
            parch[i],
            fare[i],
            float(embarked[i])
        ])

    return X, y


def comparison_table(results: list[tuple[str, float, float, float, float]]) -> str:
    """Format (name, accuracy, precision, recall, f1) rows as a printed table string.

    Floats render at 3 decimals under a header naming all four metrics.

    Raises:
        ValueError: if `results` is empty.
    """
    if not results:
        raise ValueError("results cannot be empty.")

    header = f"{'Model':<15} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}"
    separator = "-" * len(header)
    lines = [header, separator]

    for name, acc, prec, rec, f1 in results:
        lines.append(f"{name:<15} | {acc:>10.3f} | {prec:>10.3f} | {rec:>10.3f} | {f1:>10.3f}")

    return "\n".join(lines)


def misclassified_indices(
    y_true: list[int], y_pred: list[int], k: int = 5
) -> list[int]:
    """Return indices of the first `k` positions where prediction disagrees with truth.

    Raises:
        ValueError: if either list is empty or lengths differ.
    """
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred cannot be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    indices = []
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            indices.append(idx)
            if len(indices) == k:
                break

    return indices