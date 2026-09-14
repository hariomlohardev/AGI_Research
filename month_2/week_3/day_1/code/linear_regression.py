"""Multi-feature linear regression trained by batch gradient descent.

Stdlib only. Data are plain Python lists: ``X`` is rows × columns.
"""

from __future__ import annotations

from collections.abc import Sequence


def predict(
    X: Sequence[Sequence[float]], w: Sequence[float], b: float
) -> list[float]:
    """Return ``[w·x + b for x in X]``.

    Raises:
        ValueError: if ``X`` is empty, any row is ragged, ``w`` is empty,
            or a row's feature count differs from ``len(w)``.
    """
    raise NotImplementedError


def mse_loss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean squared error ``(1/n) * sum((yp - yt)^2)``.

    Raises:
        ValueError: if either input is empty or their lengths differ.
    """
    raise NotImplementedError


def mse_gradients(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    w: Sequence[float],
    b: float,
) -> tuple[list[float], float]:
    """Return ``(grad_w, grad_b)`` of the MSE at the current parameters.

    For ``n`` samples ``grad_w[j] = (2/n) * sum_i((pred_i - y_i) * X[i][j])``
    and ``grad_b = (2/n) * sum_i(pred_i - y_i)``.

    Raises:
        ValueError: if inputs are empty, ``len(X) != len(y)``, rows are
            ragged, ``w`` is empty, or a row's feature count differs from
            ``len(w)``.
    """
    raise NotImplementedError


class LinearRegressionGD:
    """Multi-feature linear regression trained by batch gradient descent.

    With ``l2 > 0`` the objective becomes ``MSE + l2 * sum(w_j^2)`` (the
    bias ``b`` is not regularised) and the weight gradient gains a
    ``2 * l2 * w_j`` term — the Week 2 Day 2 MAP-as-regularisation insight.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000, l2: float = 0.0) -> None:
        """Store hyper-parameters; weights are initialised lazily in ``fit``.

        Raises:
            ValueError: if ``lr <= 0``, ``epochs <= 0``, or ``l2 < 0``.
        """
        raise NotImplementedError

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, or rows have zero features.
        """
        raise NotImplementedError

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict with the fitted weights.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        raise NotImplementedError


if __name__ == "__main__":
    # Experiment 1 — synthetic house-price-like data: price depends on
    # size (in 100s of sqft) and bedrooms, plus a small deterministic wobble.
    X = [[8.0, 2.0], [10.0, 3.0], [12.0, 3.0], [14.0, 4.0], [16.0, 4.0]]
    y = [150.0, 200.0, 230.0, 280.0, 320.0]

    model = LinearRegressionGD(lr=0.01, epochs=500)
    history = model.fit(X, y)
    for i in range(0, len(history), 50):
        print(f"epoch {i:>4}: mse={history[i]:.3f}")
    print("w =", model.w, "b =", model.b, "final mse =", history[-1])

    # Experiment 2 — deliberately too-high learning rate.
    wild = LinearRegressionGD(lr=1.0, epochs=10)
    wild_history = wild.fit(X, y)
    print("high-lr losses:", [round(v, 2) for v in wild_history[:5]])

    # Experiment 3 — L2 shrinkage.
    plain = LinearRegressionGD(lr=0.01, epochs=500, l2=0.0)
    plain.fit(X, y)
    shrunk = LinearRegressionGD(lr=0.01, epochs=500, l2=1.0)
    shrunk.fit(X, y)
    print("plain w =", plain.w)
    print("l2    w =", shrunk.w)

    # Optional: plot the loss curve (matplotlib is pinned in requirements.txt).
    try:
        import matplotlib.pyplot as plt

        plt.plot(history)
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title("Linear regression GD loss curve")
        plt.show()
    except ImportError:
        pass
