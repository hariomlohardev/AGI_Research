"""Binary logistic regression trained by batch gradient descent.

Stdlib only. Data are plain Python lists: ``X`` is rows × columns, ``y``
holds binary labels in ``{0, 1}``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-15

def is_empty(data:list):
    return len(data) == 0

def dot_product(a:list,b:list):
    output = sum(a_i * b_i for a_i , b_i in zip(a,b))
    return output

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
    y_true: Sequence[int], y_pred: Sequence[float] , eps:float = 1e-15
) -> float:
    """Return mean binary cross-entropy.

    ``-mean(y * log(p) + (1 - y) * log(1 - p))`` with predictions clipped
    into ``[1e-15, 1 - 1e-15]`` instead of raising on ``0.0``/``1.0``.

    Raises:
        ValueError: if either input is empty, their lengths differ, or any
            label is not in ``{0, 1}``.
    """

    if is_empty(y_pred):
        raise ValueError("the y_pred is empty")
    
    if is_empty(y_true):
        raise ValueError("the y_true is empty")

    if len(y_true) != len(y_pred):
        raise ValueError("then length of y_true is not equal to the length of y_pred")
    
    if not all(y in (0,1) for y in y_true) :
        raise ValueError("one or more label is not in ``{0, 1}``")

    output   = -sum(y_i * math.log(p_i+eps) + (1-y_i )*math.log(1-p_i +eps) for y_i , p_i in zip(y_true , y_pred)) / len(y_true)
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

    if is_empty(X):
        raise ValueError("the X is empty")
    
    if is_empty(y):
        raise ValueError("the y is empty")
    
    if is_empty(w):
        raise ValueError("the w is empty")

    if len(X) != len(y):
        raise ValueError("the length of input(X) and Output(y)")

    if all(len(row) != len(X[0]) for row in X):
        raise ValueError("rows are ragged")

    if len(X[0]) != len(w):
        raise ValueError("the row's feature count differs from length of weights")

    if not all(_y in (0,1) for _y in y) :
        raise ValueError("one or more label is not in ``(0,1)``")

    k = len(X[0])
    n = len(y)

    grad_w = [0.0] * k
    grad_b = 0.0

    for i in range(len(X)):
        x = X[i]

        p_i = sigmoid(dot_product(x,w) + b)
        error = p_i - y[i]
        
        for j in range(k):
            grad_w[j] += (1/n)*error * x[j]
        grad_b += (1/n)*error 

    return (grad_w,grad_b)


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
        self.history = list()
        

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
    ) -> list[float]:
        """Run batch gradient descent; return the per-epoch loss history.

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, rows have zero features, or any label is not in
                ``{0, 1}``.
        """

        if is_empty(X):
            raise ValueError("X can't be empty")

        if is_empty(y):
            raise ValueError("y can't be empty")

        if len(X) != len(y):
            raise ValueError("length of X and y are different")

        if all(len(row)!=len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if all(len(row) == 0 for row in X):
            raise ValueError("one or more rows have zero features")

        if not all(y_i in (0,1) for y_i in y):
            raise ValueError("one or more labels is not in ``(0,1)``")

        self.w = [0.0] * len(X[0])

        for _ in range(self.epochs):
            grad_w, grad_b = bce_gradients(X,y,self.w,self.b)
            for j in range(len(grad_w)):
                grad_w[j] = grad_w[j] + 2 * self.l2 * self.w[j]
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * grad_b

            y_pred = [sigmoid(dot_product(x ,self.w) + self.b) for x in X]
            loss = binary_cross_entropy(y , y_pred)
            self.history.append(loss)

        return self.history

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw sigmoid outputs for each row.

        Raises:
            ValueError: if called before ``fit`` or if the input is empty,
                ragged, or has the wrong feature count.
        """
        if self.w == None:
            raise ValueError("call fit() first")

        if is_empty(X):
            raise ValueError("X can't be empty")
        
        if all(len(row)!=len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) != len(self.w):
            raise ValueError("wrong feature count")
        

        output = [sigmoid(dot_product(x ,self.w) + self.b) for x in X]
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

        if is_empty(X):
            raise ValueError("X can't be empty")
        
        if all(len(row)!=len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) != len(self.w):
            raise ValueError("wrong feature count")

        if not 0 < threshold < 1:
            raise ValueError("threashoud should be between (0,1)")

        probs = [sigmoid(dot_product(x ,self.w) + self.b) for x in X]
        output = [0] * len(probs)
        for i in range(len(probs)):
            if probs[i] >= threshold:
                output[i] = 1
            else:
                output[i] = 0

        return output


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
