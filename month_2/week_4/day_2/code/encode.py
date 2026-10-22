"""Day 20 — impute + encode helpers (stubs, see Problem 2).

Generic over any string table — the Titanic wiring (which columns, label
vs one-hot) is your documented call, not a helper's default.
"""

from __future__ import annotations

from eda import is_missing

import math


def to_float_column(rows: list[list[str]], idx: int) -> list[float | None]:
    """Parse column `idx` to floats, `None` where missing.

    Raises:
        ValueError: on a non-missing entry that is not numeric.
    """
    column = [row[idx] for row in rows]
    for i in range(len(column)):
        if not is_missing(column[i]):
            try:
                column[i] = float(column[i])

            except :
                raise ValueError(f"its not an numeric value on row {i}")
        else:
            column[i] = None

    return column



def fill_mean(column: list[float | None]) -> tuple[list[float], float]:
    """Replace `None`s with the mean of the present values.

    Returns the filled column and the mean used.

    Raises:
        ValueError: if the column is empty or all entries are missing.
    """

    if not column:
        raise ValueError("column is empty")

    if all(val is None or val == "" or (isinstance(val, float) and val != val) for val in column):
        raise ValueError("all entries are missing.")

    clean_col = [x for x in column if x is not None]
    mean  = sum(clean_col) / len(clean_col) if clean_col else 0



    for i in range(len(column)):
        if column[i] == None:
            column[i] = mean

    return column , mean



def label_encode(values: list[str]) -> tuple[list[int], dict[str, int]]:
    """Map each distinct value to an int (first-seen order).

    Returns the encoded list and the mapping used.

    Raises:
        ValueError: if `values` is empty.
    """
    if not values:
        raise ValueError("vlaues is empty")

    mapping = {}
    encoded = []
    
    for val in values:
        if val not in mapping:
            # Assign the next available integer based on current unique count
            mapping[val] = len(mapping)
        encoded.append(mapping[val])
        
    return encoded, mapping


    


def one_hot(values: list[str]) -> tuple[list[list[int]], list[str]]:
    """One-hot encode `values`; returns the matrix and category order.

    Raises:
        ValueError: if `values` is empty.
    """

    if not values:
        raise ValueError("vlaues is empty")
    

    # 1. Determine categories in first-seen order
    categories = []
    seen = set()
    for val in values:
        if val not in seen:
            seen.add(val)
            categories.append(val)
            
    num_categories = len(categories)
    
    # 2. Map each category name to its index position
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # 3. Build the one-hot matrix
    matrix = []
    for val in values:
        # Create a row of zeros matching the number of unique categories
        row = [0] * num_categories
        # Set the matching category index to 1
        row[category_to_idx[val]] = 1
        matrix.append(row)
        
    return matrix, categories

