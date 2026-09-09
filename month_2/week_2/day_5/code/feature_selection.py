"""Feature selection by mutual information (stdlib only)."""

from collections.abc import Hashable, Sequence
from mutual_information import mutual_information


def rank_features_by_mi(
    features: Sequence[Sequence[Hashable]], y: Sequence[Hashable], base: float = 2.0
) -> list[tuple[int, float]]:
    """Rank feature columns by their mutual information with the label, best first.

    `features` is rows x columns: `features[i][j]` is sample `i`'s value for
    feature `j`. Returns `[(feature_index, mi_score), ...]` sorted by score
    descending. MI per column comes from `mutual_information` (same directory).

    Raises:
        ValueError: if `features` or `y` is empty.
        ValueError: if the row count `!= len(y)` (every sample needs its label).
        ValueError: if rows are ragged (not every row has the same feature count).
        ValueError: if `base <= 0` or `base == 1` (passed through to
            `mutual_information` — one place owns the rule).
    """

    

    if len(features) == 0 or len(y) == 0:
        raise ValueError("the size of either features or labels(y) is 0")
    if len(features) != len(y):
        raise ValueError("the features length is not equal to label length")
    if not all(len(row) == len(features[0]) for row in features):
        raise ValueError("every features should have same lenth")

    columns = []
    for i in range(len(features[0])):
        column = [row[i] for row in features]
        columns.append(column)
    n_columns = len(columns)

    rank = []
    for i in range(n_columns):
        column = columns[i]
        mi = mutual_information(column , y , base)
        rank.append((i , mi))
    rank.sort(key= lambda pair : pair[1] , reverse=True)
    return rank
    


if __name__ == "__main__":
    import pathlib
    import random
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

    from mutual_information import mutual_information

    # Experiment 1 — informative vs noise (seeded, so the print is stable).
    rng = random.Random(11)
    n = 200
    label = [rng.choice((0, 1)) for _ in range(n)]
    informative = [v if rng.random() > 0.10 else 1 - v for v in label]
    noise = [rng.choice((0, 1)) for _ in range(n)]
    rows = [[informative[i], noise[i]] for i in range(n)]
    print("ranking (index, MI bits):", rank_features_by_mi(rows, label))

    # Experiment 2 — Y = X^2: covariance blind, MI not.
    xs = list(range(-4, 5))
    ys = [v * v for v in xs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys)) / len(xs)
    print("cov(X, X^2) =", cov)
    print("MI(X, X^2)  =", mutual_information(xs, ys), "bits")
