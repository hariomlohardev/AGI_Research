"""Binary logistic regression trained by batch gradient descent.

Stdlib only. Data are plain Python lists: ``X`` is rows × columns, ``y``
holds binary labels in ``{0, 1}``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-15


def sigmoid(z: float) -> float:
    """Return ``1 / (1 + exp(-z))``, numerically stable for large ``|z|``.

    Any real ``z`` maps into ``(0, 1)``; no ``ValueError`` contract — the
    stable branch avoids ``OverflowError`` for inputs like ``±1000``.
    """
    raise NotImplementedError


def binary_cross_entropy(
    y_true: Sequence[int], y_pred: Sequence[float]
) -> float:
    """Return mean binary cross-entropy.

    ``-mean(y * log(p) + (1 - y) * log(1 - p))`` with predictions clipped
    into ``[1e-15, 1 - 1e-15]`` instead of raising on ``0.0``/``1.0``.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError


def bce_gradients(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    w: Sequence[float],
    b: float,
) -> tuple[list[float], float]:
    """Return ``(grad_w, grad_b)`` of the BCE at the current parameters.

    With ``p_i = sigmoid(w·x_i + b)`` over ``n`` samples,
    ``grad_w[j] = (1/n) * sum_i((p_i - y_i) * X[i][j])`` and
    ``grad_b = (1/n) * sum_i(p_i - y_i)``.

    Raises:
        ValueError: if inputs are empty, ``len(X) != len(y)``, rows are
            ragged, a row's feature count differs from ``len(w)``, or any
            label is not in ``{0, 1}``.
    """
    raise NotImplementedError


class LogisticRegressionGD:
    """Binary logistic regression trained by batch gradient descent.

    With ``l2 > 0`` the objective becomes ``BCE + l2 * sum(w_j^2)`` (the
    bias ``b`` is not regularised) and the weight gradient gains a
    ``2 * l2 * w_j`` term — the Day 13 MAP-as-regularisation insight,
    now on a classifier.
    """

    def __init__(
        self, lr: float = 0.01, epochs: int = 1000, l2: float = 0.0
    ) -> None:
        """Store hyper-parameters; weights are initialised lazily in ``fit``.

        Raises:
            ValueError: if ``lr <= 0``, ``epochs <= 0``, or ``l2 < 0``.
        """
        raise NotImplementedError

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
    ) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, rows have zero features, or any label is not in
                ``{0, 1}``.
        """
        raise NotImplementedError

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw sigmoid outputs for each row.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        raise NotImplementedError

    def predict(
        self, X: Sequence[Sequence[float]], threshold: float = 0.5
    ) -> list[int]:
        """Return ``1`` where proba ``>= threshold``, else ``0``.

        Raises:
            ValueError: if called before ``fit``, if the input is empty,
                ragged, or has the wrong feature count, or if
                ``threshold`` is outside ``(0, 1)`` (exclusive).
        """
        raise NotImplementedError


if __name__ == "__main__":
    import pathlib
    import random
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from metrics import precision_score, recall_score

    # Experiment 1 — synthetic 2-class blobs (seeded, linearly separable-ish).
    rng = random.Random(7)
    X = [
        [rng.gauss(-2.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(40)
    ] + [
        [rng.gauss(2.0, 1.0), rng.gauss(0.0, 1.0)] for _ in range(40)
    ]
    y = [0] * 40 + [1] * 40

    model = LogisticRegressionGD(lr=0.1, epochs=500)
    history = model.fit(X, y)
    for i in range(0, len(history), 50):
        print(f"epoch {i:>4}: bce={history[i]:.4f}")
    print("final bce =", history[-1])

    # Experiment 2 — threshold 0.5 vs 0.3 precision/recall.
    for t in (0.5, 0.3):
        preds = model.predict(X, threshold=t)
        print(
            f"threshold={t}: precision={precision_score(y, preds):.3f} "
            f"recall={recall_score(y, preds):.3f}"
        )

    # Experiment 3 — L2 shrinkage on the weight norm.
    plain = LogisticRegressionGD(lr=0.1, epochs=500, l2=0.0)
    plain.fit(X, y)
    shrunk = LogisticRegressionGD(lr=0.1, epochs=500, l2=1.0)
    shrunk.fit(X, y)
    print("plain w =", plain.w)
    print("l2    w =", shrunk.w)

    # Optional: plot the loss curve (matplotlib is pinned in requirements.txt).
    try:
        import matplotlib.pyplot as plt

        plt.plot(history)
        plt.xlabel("epoch")
        plt.ylabel("BCE")
        plt.title("Logistic regression GD loss curve")
        plt.show()
    except ImportError:
        pass
