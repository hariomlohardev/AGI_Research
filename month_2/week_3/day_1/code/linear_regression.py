"""Multi-feature linear regression trained by batch gradient descent.

Stdlib only. Data are plain Python lists: ``X`` is rows × columns.
"""

from __future__ import annotations

from collections.abc import Sequence



def is_empty(data:list) -> float:
    return len(data) == 0

def dot_product(X ,Y):
    return sum(x * y for x,y in zip(X,Y))

def predict(
    X: Sequence[Sequence[float]], w: Sequence[float], b: float
) -> list[float]:
    """Return ``[w·x + b for x in X]``.

    Raises:
        ValueError: if ``X`` is empty, any row is ragged, ``w`` is empty,
            or a row's feature count differs from ``len(w)``.
    """
    if is_empty(X):
        raise ValueError("X can't be empty")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("length of rows are not equal in X")

    if is_empty(w):
        raise ValueError("w is empty")

    if len(w) != len(X[0]):
        raise ValueError("a row's feature count differs from ``len(w)``")

    predictions = []
    for x in X:
        dot_prod = dot_product(x,w)
        predictions.append(dot_prod + b)

    return predictions




def mse_loss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean squared error ``(1/n) * sum((yp - yt)^2)``.

    Raises:
        ValueError: if either input is empty or their lengths differ.
    """
    if is_empty(y_pred):
        raise ValueError("y_pred is empty")

    if is_empty(y_true):
        raise ValueError("y_true is empty")

    if len(y_true) != len(y_pred):
        raise ValueError("the lengths of y_true and y_pred is different")

    error = sum((y_hat - y)**2 for y_hat , y in zip(y_pred ,y_true)) / len(y_true)

    return error

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
    if is_empty(X):
        raise ValueError("X is empty")

    if len(X) != len(y):
        raise ValueError("the length of features(X) and label(y) are different")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("length of rows are not equal in X")

    if is_empty(w):
        raise ValueError("w can't be empty")

    if len(w) != len(X[0]):
        raise ValueError("a row's feature count differs from ``len(w)``")
    grad_w = [0.0] * len(w)
    grad_b = 0.0
    n = len(X)

    for i in range(len(X)):
        x = X[i]
        y_hat = dot_product(x,w) + b
        error = y_hat - y[i]

        for j in range(len(w)):
            grad_w[j] += error * x[j]

        grad_b += error

    grad_w = [ (2/n) * g for g in grad_w]
    grad_b = (2/n) * grad_b
    return (grad_w, grad_b)


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
        if lr <= 0:
            raise ValueError("Learning rate (lr) can never be zero or negative")
        
        if epochs <= 0:
            raise ValueError("epochs can never be zero or negative")
        
        if l2 < 0:
            raise ValueError("l2 can never be negative")

        self.lr  = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, or rows have zero features.
        """
        if is_empty(X):
            raise ValueError("the X can't be empty")
        
        if len(X) != len(y):
            raise ValueError("the length of X and y are different")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("length of rows are not equal in X")

        if any(len(x) == 0 for x in X):
            raise ValueError("one or more rows have zero features")

        k = len(X[0])
        self.w = [0.0] * k
        self.b = 0.0
        history = []

        for _ in range(self.epochs):
            grad_w, grad_b = mse_gradients(X, y, self.w, self.b)
            # L2 tug: bias exempt, weights get + 2*l2*w_j
            for j in range(k):
                grad_w[j] = grad_w[j] + (2 * self.l2 * self.w[j])
                # update: self.w[j] -= self.lr * grad_w[j]; self.b -= self.lr * grad_b
                self.w[j] -= self.lr * grad_w[j]

            self.b -= self.lr * grad_b
            # history.append(mse_loss(y, predict(X, self.w, self.b)))
            history.append(mse_loss(y, predict(X, self.w, self.b)))

        return history

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict with the fitted weights.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        if self.w == None :
            raise ValueError("please call fit() before predict()")

        if is_empty(X):
            raise ValueError("X is empty")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("length of rows are not equal in X")

        if any(len(x) == 0 for x in X):
            raise ValueError("one or more rows have zero features")

        predictions = predict(X ,self.w ,self.b)
        return predictions



if __name__ == "__main__":
    # Experiment 1 — synthetic house-price-like data: price depends on
    # size (in 100s of sqft) and bedrooms, plus a small deterministic wobble.
    X = [[8.0, 2.0], [10.0, 3.0], [12.0, 3.0], [14.0, 4.0], [16.0, 4.0]]
    y = [150.0, 200.0, 230.0, 280.0, 320.0]

    model = LinearRegressionGD(lr=0.001, epochs=500)
    history = model.fit(X, y)
    for i in range(0, len(history), 50):
        print(f"epoch {i:>4}: mse={history[i]:.3f}")
    print("w =", model.w, "b =", model.b, "final mse =", history[-1])

    # Experiment 2 — deliberately too-high learning rate.
    wild = LinearRegressionGD(lr=1.0, epochs=10)
    wild_history = wild.fit(X, y)
    print("high-lr losses:", [round(v, 2) for v in wild_history[:5]])

    # Experiment 3 — L2 shrinkage.
    plain = LinearRegressionGD(lr=0.001, epochs=500, l2=0.0)
    plain.fit(X, y)
    shrunk = LinearRegressionGD(lr=0.001, epochs=500, l2=1.0)
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
