from __future__ import annotations

import math
from collections.abc import Sequence

from .utils import dot_product, if_empty_raise

_EPS = 1e-15


def predict(X: Sequence[Sequence[float]], w: Sequence[float], b: float) -> list[float]:
    """Return ``[w·x + b for x in X]``.

    Raises:
        ValueError: if ``X`` is empty, any row is ragged, ``w`` is empty,
            or a row's feature count differs from ``len(w)``.
    """
    if_empty_raise(X, "X can't be empty")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("length of rows are not equal in X")

    if_empty_raise(w, "w is empty")

    if len(w) != len(X[0]):
        raise ValueError("a row's feature count differs from ``len(w)``")

    predictions = []
    for x in X:
        dot_prod = dot_product(x, w)
        predictions.append(dot_prod + b)

    return predictions


def mse_loss(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean squared error ``(1/n) * sum((yp - yt)^2)``.

    Raises:
        ValueError: if either input is empty or their lengths differ.
    """
    if_empty_raise(y_pred, "y_pred is empty")

    if_empty_raise(y_true, "y_true is empty")

    if len(y_true) != len(y_pred):
        raise ValueError("the lengths of y_true and y_pred is different")

    error = sum((y_hat - y) ** 2 for y_hat, y in zip(y_pred, y_true)) / len(y_true)
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
    if_empty_raise(X, "X is empty")

    if len(X) != len(y):
        raise ValueError("the length of features(X) and label(y) are different")

    if not all(len(x) == len(X[0]) for x in X):
        raise ValueError("length of rows are not equal in X")

    if_empty_raise(w, "w can't be empty")

    if len(w) != len(X[0]):
        raise ValueError("a row's feature count differs from ``len(w)``")
    grad_w = [0.0] * len(w)
    grad_b = 0.0
    n = len(X)

    for i in range(len(X)):
        x = X[i]
        y_hat = dot_product(x, w) + b
        error = y_hat - y[i]

        for j in range(len(w)):
            grad_w[j] += error * x[j]

        grad_b += error

    grad_w = [(2 / n) * g for g in grad_w]
    grad_b = (2 / n) * grad_b
    return (grad_w, grad_b)


def sigmoid(z: float) -> float:
    """Return ``1 / (1 + exp(-z))``, numerically stable for large ``|z|``.

    Any real ``z`` maps into ``(0, 1)``; no ``ValueError`` contract — the
    stable branch avoids ``OverflowError`` for inputs like ``±1000``.
    """
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def binary_cross_entropy(
    y_true: Sequence[int], y_pred: Sequence[float], eps: float = 1e-15
) -> float:
    """Return mean binary cross-entropy.

    ``-mean(y * log(p) + (1 - y) * log(1 - p))`` with predictions clipped
    into ``[1e-15, 1 - 1e-15]`` instead of raising on ``0.0``/``1.0``.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """

    if_empty_raise(y_pred, "the y_pred is empty")

    if_empty_raise(y_true, "the y_true is empty")

    if len(y_true) != len(y_pred):
        raise ValueError("then length of y_true is not equal to the length of y_pred")

    if not all(y in (0, 1) for y in y_true):
        raise ValueError("one or more label is not in ``{0, 1}``")

    output = -sum(
        y_i * math.log(p_i + eps) + (1 - y_i) * math.log(1 - p_i + eps)
        for y_i, p_i in zip(y_true, y_pred)
    ) / len(y_true)
    return output


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

    if_empty_raise(X, "the X is empty")

    if_empty_raise(y, "the y is empty")

    if_empty_raise(w, "the w is empty")

    if len(X) != len(y):
        raise ValueError("the length of input(X) and Output(y)")

    if all(len(row) != len(X[0]) for row in X):
        raise ValueError("rows are ragged")

    if len(X[0]) != len(w):
        raise ValueError("the row's feature count differs from length of weights")

    if not all(_y in (0, 1) for _y in y):
        raise ValueError("one or more label is not in ``(0,1)``")

    k = len(X[0])
    n = len(y)

    grad_w = [0.0] * k
    grad_b = 0.0

    for i in range(len(X)):
        x = X[i]

        p_i = sigmoid(dot_product(x, w) + b)
        error = p_i - y[i]

        for j in range(k):
            grad_w[j] += (1 / n) * error * x[j]
        grad_b += (1 / n) * error

    return (grad_w, grad_b)


class LinearRegression:
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

        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, or rows have zero features.
        """
        if_empty_raise(X, "the X can't be empty")

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

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict with the fitted weights.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        if self.w == None:
            raise ValueError("please call fit() before predict()")

        if_empty_raise(X, "X is empty")

        if not all(len(x) == len(X[0]) for x in X):
            raise ValueError("length of rows are not equal in X")

        if any(len(x) == 0 for x in X):
            raise ValueError("one or more rows have zero features")

        predictions = predict(X, self.w, self.b)
        return predictions


class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent.

    With ``l2 > 0`` the objective becomes ``BCE + l2 * sum(w_j^2)`` (the
    bias ``b`` is not regularised) and the weight gradient gains a
    ``2 * l2 * w_j`` term — the Day 13 MAP-as-regularisation insight,
    now on a classifier.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 1000, l2: float = 0.0) -> None:
        """Store hyper-parameters; weights are initialised lazily in ``fit``.

        Raises:
            ValueError: if ``lr <= 0``, ``epochs <= 0``, or ``l2 < 0``.
        """
        if lr <= 0:
            raise ValueError("the Learning rate(lr) can't be zero or negative")

        if epochs <= 0:
            raise ValueError("epochs can't be negative or zero")

        if l2 < 0:
            raise ValueError("l2 can't be negative")

        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.history = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, rows have zero features, or any label is not in
                ``{0, 1}``.
        """

        if_empty_raise(X, "X can't be empty")

        if_empty_raise(y, "y can't be empty")

        if len(X) != len(y):
            raise ValueError("length of X and y are different")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if all(len(row) == 0 for row in X):
            raise ValueError("one or more rows have zero features")

        if not all(y_i in (0, 1) for y_i in y):
            raise ValueError("one or more labels is not in ``(0,1)``")

        self.w = [0.0] * len(X[0])

        for _ in range(self.epochs):
            grad_w, grad_b = bce_gradients(X, y, self.w, self.b)
            for j in range(len(grad_w)):
                grad_w[j] = grad_w[j] + 2 * self.l2 * self.w[j]
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * grad_b

            y_pred = [sigmoid(dot_product(x, self.w) + self.b) for x in X]
            loss = binary_cross_entropy(y, y_pred)
            self.history.append(loss)

        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw sigmoid outputs for each row.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        if self.w == None:
            raise ValueError("call fit() first")

        if_empty_raise(X, "X can't be empty")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) != len(self.w):
            raise ValueError("wrong feature count")

        output = [sigmoid(dot_product(x, self.w) + self.b) for x in X]
        return output

    def predict(
        self, X: Sequence[Sequence[float]], threshold: float = 0.5
    ) -> list[int]:
        """Return ``1`` where proba ``>= threshold``, else ``0``.

        Raises:
            ValueError: if called before ``fit``, if the input is empty,
                ragged, or has the wrong feature count, or if
                ``threshold`` is outside ``(0, 1)`` (exclusive).
        """
        if self.w == None:
            raise ValueError("call fit() first")

        if_empty_raise(X, "X can't be empty")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) != len(self.w):
            raise ValueError("wrong feature count")

        if not 0 < threshold < 1:
            raise ValueError("threashoud should be between (0,1)")

        probs = [sigmoid(dot_product(x, self.w) + self.b) for x in X]
        output = [0] * len(probs)
        for i in range(len(probs)):
            if probs[i] >= threshold:
                output[i] = 1
            else:
                output[i] = 0

        return output
