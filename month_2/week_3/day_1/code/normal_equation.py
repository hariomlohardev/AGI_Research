from __future__ import annotations

from collections.abc import Sequence


def solve_normal_equation(
    X: Sequence[Sequence[float]], y: Sequence[float]
) -> tuple[list[float], float]:
    """Return ``(w, b)`` solving least squares via the Normal Equation.

    Raises:
        ValueError: if data is empty, ``len(X) != len(y)``, rows are ragged,
            rows have zero features, or the system is singular.
    """
    if len(X) == 0:
        raise ValueError("data is empty")

    if len(X) != len(y):
        raise ValueError("len(X) must equal len(y)")

    n_features = len(X[0])

    if n_features == 0:
        raise ValueError("rows must have at least one feature")

    for row in X:
        if len(row) != n_features:
            raise ValueError("rows are ragged")

    # Design matrix A = [1, X]
    # The first column is the bias/intercept.
    A = [[1.0, *map(float, row)] for row in X]
    targets = [float(value) for value in y]

    size = n_features + 1

    # Build X'X and X'y.
    normal = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size

    for i in range(size):
        for j in range(size):
            normal[i][j] = sum(row[i] * row[j] for row in A)

        rhs[i] = sum(A[k][i] * targets[k] for k in range(len(A)))

    # Gaussian elimination with partial pivoting.
    for col in range(size):
        pivot_row = max(
            range(col, size),
            key=lambda row: abs(normal[row][col]),
        )

        pivot = normal[pivot_row][col]

        if pivot == 0.0:
            raise ValueError("system is singular")

        if pivot_row != col:
            normal[col], normal[pivot_row] = normal[pivot_row], normal[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        for row in range(col + 1, size):
            factor = normal[row][col] / normal[col][col]

            for j in range(col, size):
                normal[row][j] -= factor * normal[col][j]

            rhs[row] -= factor * rhs[col]

    # Back substitution.
    solution = [0.0] * size

    for row in range(size - 1, -1, -1):
        remaining = sum(
            normal[row][j] * solution[j]
            for j in range(row + 1, size)
        )

        solution[row] = (rhs[row] - remaining) / normal[row][row]

    # solution[0] is the bias; remaining values are weights.
    b = solution[0]
    w = solution[1:]

    return w, b


if __name__ == "__main__":
    X = [[8.0, 2.0], [10.0, 3.0], [12.0, 3.0], [14.0, 4.0], [16.0, 4.0]]
    y = [150.0, 200.0, 230.0, 280.0, 320.0]

    w, b = solve_normal_equation(X, y)
    print("w =", w, "b =", b)