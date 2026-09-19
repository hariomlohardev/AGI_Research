"""Linear SVM (hinge loss + subgradient descent) and kernel helpers.

Stdlib only. Data are plain Python lists: ``X`` is rows × columns.
Labels are ``{-1, +1}`` (the SVM convention — unlike the ``{0, 1}``
logistic-regression days, the hinge formula needs symmetric labels).
"""

from __future__ import annotations

from collections.abc import Sequence

def raise_if_empty(data:list,error_message:str=""):
    if len(data) == 0:
        raise ValueError(error_message)

def dot_product(a:list,b:list):
    return sum(_a*_b for _a,_b in zip(a,b))

def magnitude(w:list):
    return math.sqrt(sum(_w ** 2 for _w in w))


def hinge_loss(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Return mean hinge loss ``mean(max(0, 1 - y_i * s_i))``.

    A point correctly classified with margin ``y_i * s_i >= 1`` pays
    exactly ``0`` — unlike cross-entropy, which only approaches 0
    asymptotically (practice question 1).

    Raises:
        ValueError: if either input is empty, their lengths differ, or
            any label is not in ``{-1, +1}``.
    """

    raise_if_empty(y_true,"y_true is empty")

    raise_if_empty(scores,"scores is empty")

    if len(y_true) != len(scores):
        raise ValueError("the lengths of the y_true and scores are different")

    if not all(y_i in (-1,1) for y_i in y_true):
        raise ValueError("one or more labels are not in (-1,1)")

    output = sum(max(0,(1-y_i*s_i)) for y_i, s_i in zip(y_true,scores)) / len(y_true)

    return output


def hinge_subgradients(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    w: Sequence[float],
    b: float,
    C : float = 1.0
) -> tuple[list[float], float]:
    """Return ``(grad_w, grad_b)``, a subgradient of the mean hinge loss.

    Per sample, with ``s_i = w·x_i + b`` and ``margin_i = y_i * s_i``:
    points with ``margin_i < 1`` contribute ``-y_i * x_i`` (weights) and
    ``-y_i`` (bias); points with ``margin_i > 1`` contribute nothing.
    At the kink ``margin_i == 1`` this implementation contributes
    nothing (one valid subgradient choice — documented, not derived).

    This is the *unregularised* gradient; ``LinearSVMGD.fit`` adds the
    ``2 * l2 * w_j`` term itself (bias ``b`` never regularised).

    Raises:
        ValueError: if ``X``/``y``/``w`` is empty, ``len(X) != len(y)``,
            rows are ragged, rows have zero features, a row's feature
            count differs from ``len(w)``, or any label is not in
            ``{-1, +1}``.
    """
    raise_if_empty(X,"X is empty")

    raise_if_empty(y,"y is empty")

    raise_if_empty(w,"w is empty")

    if len(X) != len(y):
        raise ValueError("the length of X any y are diffrent")

    if not all(len(row) == len(X[0]) for row in X):
        raise ValueError("rows of X are ragged")

    raise_if_empty(X[0], "rows of X have 0 features")

    if len(X[0]) != len(w):
        raise ValueError("row's feature count differs form len(w)")

    if not all(y_i in (-1,1) for y_i in y):
        raise ValueError("one or more labels are not in (-1,1)")

    n_features = len(X[0])
    n_samples = len(X)

    grad_w = [0.0] * n_features
    grad_b = 0.0

    for i in range(n_samples):
        # 1. Compute s_i = w · x_i + b
        s_i = sum(w[j] * X[i][j] for j in range(n_features)) + b
        
        # 2. Compute margin_i = y_i * s_i
        margin_i = y[i] * s_i

        # 3. Check condition (margin_i < 1)
        if margin_i < 1:
            # Accumulate weight subgradient: -y_i * x_i
            for j in range(n_features):
                grad_w[j] += -y[i] * X[i][j]
            # Accumulate bias subgradient: -y_i
            grad_b += -y[i]

    # FIX: Divide by the number of samples to get the MEAN subgradient
    grad_w = [g / n_samples for g in grad_w]
    grad_b = grad_b / n_samples
    return (grad_w, grad_b)



class LinearSVMGD:
    """Linear SVM trained by batch subgradient descent.

    Minimises ``mean_hinge + l2 * sum(w_j^2)`` (bias ``b`` exempt —
    same MAP-as-regularisation reasoning as Days 13/14, now pulling
    the max-margin hyperplane toward small weights instead of toward
    a Gaussian prior mean).
    """

    def __init__(
        self, lr: float = 0.1, epochs: int = 500, l2: float = 0.01
    ) -> None:
        """Store hyper-parameters; weights are initialised lazily in ``fit``.

        Raises:
            ValueError: if ``lr <= 0``, ``epochs <= 0``, or ``l2 < 0``.
        """

        if lr <= 0:
            raise ValueError("lr can't be zero or negative")

        if epochs <= 0:
            raise ValueError("epochs can't be zero or negative")

        if l2 < 0:
            raise ValueError("l2 can't be negative")


        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.history = list()


    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
    ) -> list[float]:
        """Run subgradient descent; return the per-epoch objective history.

        Each entry is ``mean_hinge + l2 * ||w||^2`` at that epoch.
        (Subgradient steps need not decrease it strictly monotonically —
        compare early vs late averages, not adjacent epochs.)

        Raises:
            ValueError: if data is empty, ``len(X) != len(y)``, rows are
                ragged, rows have zero features, or any label is not in
                ``{-1, +1}``.
        """

        raise_if_empty(X, "X can't be empty")

        raise_if_empty(y, "y can't be empty")

        if len(X) != len(y):
            raise ValueError("lengths of X and y should be same")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) == 0 :
            raise ValueError("rows have zero features")

        if not all(y_i in (-1,1) for y_i in y):
            raise ValueError("one or more labels are not in (-1,1)")

        k = len(X[0])
        self.w = [0.0] * k
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w , grad_b = hinge_subgradients(X,y,self.w , self.b)

            for j in range(k):
                grad_w[j]  = grad_w[j] +  2 * self.l2 * self.w[j]

            hinge_sum = 0.0
            for i in range(len(X)):
                margin = y[i] * (sum(self.w[j] * X[i][j] for j in range(k)) + self.b)
                if margin < 1.0:
                    hinge_sum += (1.0 - margin)
            
            mean_hinge = hinge_sum / len(X)
            w_norm_sq = sum(wj ** 2 for wj in self.w)
            epoch_obj = mean_hinge + self.l2 * w_norm_sq
            self.history.append(epoch_obj)

            # 4. Update the weights and bias using the learning rate
            for j in range(k):
                self.w[j] -= self.lr * grad_w[j]
            self.b -= self.lr * grad_b

        return self.history


        

    def decision_function(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Return raw scores ``w·x + b`` (signed distance up to scale).

        Raises:
            ValueError: if called before ``fit`` or if the input is
                empty, ragged, has zero features, or has the wrong
                feature count.
        """

        if self.w == None:
            raise ValueError("call fit() first")

        raise_if_empty(X, "X can't be empty")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) == 0 :
            raise ValueError("rows have zero features")

        if len(X[0]) != len(self.w):
            raise ValueError("has wrong feature count")

        scores = [(dot_product(self.w,x) + self.b) for x in X]
        return scores

        

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Return ``+1`` where the score is ``>= 0``, else ``-1``.

        Raises:
            ValueError: if called before ``fit`` or if the input is
                empty, ragged, has zero features, or has the wrong
                feature count.
        """
        if self.w == None:
            raise ValueError("call fit() first")

        raise_if_empty(X, "X can't be empty")

        if all(len(row) != len(X[0]) for row in X):
            raise ValueError("rows are ragged")

        if len(X[0]) == 0 :
            raise ValueError("rows have zero features")

        if len(X[0]) != len(self.w):
            raise ValueError("has wrong feature count")

        output = []
        scores = self.decision_function(X)

        for i in range(len(X)):
            score = scores[i]

            if score >= 0:
                output.append(+1)
            else:
                output.append(-1)


        return output


def polynomial_kernel(
    x: Sequence[float], y: Sequence[float], degree: int = 2, coef0: float = 1.0
) -> float:
    """Return ``(x·y + coef0) ** degree``.

    The kernel trick's core bargain: this single number equals a dot
    product in a higher-dimensional expanded space, computed without
    ever building the expanded vectors.

    Raises:
        ValueError: if either input is empty, their lengths differ, or
            ``degree < 1``.
    """

    raise_if_empty(x , "x is empty")

    raise_if_empty(y , "y is empty")

    if len(x) != len(y):
        raise ValueError("length of x and y is different")

    if degree < 1:
        raise ValueError("degree can't be less than 1")

    return (dot_product(x,y) + coef0) ** degree


def add_radius_feature(X: Sequence[Sequence[float]]) -> list[list[float]]:
    """Append ``x_0^2 + x_1^2`` (squared radius) as a third feature.

    The hand-built kernel trick for the ring dataset: points the linear
    SVM cannot separate in ``(x, y)`` become separable along the new
    axis (inner ring ~ small radius, outer ring ~ large radius), so a
    *linear* separator in the expanded space solves a nonlinear problem
    (practice question 4 — verify by fitting before/after).

    Raises:
        ValueError: if ``X`` is empty, rows are ragged, or any row does
            not have exactly 2 features.
    """

    raise_if_empty(X , "X is empty")

    if all(len(row) != len(X[0]) for row in X):
        raise ValueError("rows are ragged")


    for point in X:
        if len(point) != 2:
            raise ValueError("Each row must have exactly 2 features")


    output = []
    for point in X:
        x = point[0]
        y = point[1]
        
        # Calculate the squared radius (x^2 + y^2)
        squared_radius = (x ** 2) + (y ** 2)
        
        # Append the new 3D row to our output list
        output.append([x, y, squared_radius])

    return output



if __name__ == "__main__":
    import random

    # Experiment 1 — linearly separable blobs: SVM should separate cleanly.
    rng = random.Random(7)
    X_sep = [[rng.gauss(-2.0, 0.8), rng.gauss(0.0, 0.8)] for _ in range(30)]
    X_sep += [[rng.gauss(2.0, 0.8), rng.gauss(0.0, 0.8)] for _ in range(30)]
    y_sep = [-1] * 30 + [1] * 30

    model = LinearSVMGD(lr=0.1, epochs=500, l2=0.01)
    history = model.fit(X_sep, y_sep)
    for i in range(0, len(history), 50):
        print(f"epoch {i:>4}: objective={history[i]:.4f}")
    print("final objective =", history[-1])
    preds = model.predict(X_sep)
    acc = sum(1 for a, p in zip(y_sep, preds) if a == p) / len(y_sep)
    print(f"separable accuracy = {acc:.3f} (w={model.w}, b={model.b:.3f})")
    scores = model.decision_function(X_sep)
    n_sv = sum(1 for a, s in zip(y_sep, scores) if a * s <= 1.0 + 1e-9)
    print(f"support vectors: {n_sv}/{len(y_sep)}")

    # Experiment 2 — ring data (inner disc vs outer ring): linear SVM fails.
    import math

    rng2 = random.Random(21)
    X_ring: list[list[float]] = []
    y_ring: list[int] = []
    for _ in range(40):
        r, t = rng2.uniform(0.0, 1.0), rng2.uniform(0.0, 2 * math.pi)
        X_ring.append([r * math.cos(t), r * math.sin(t)])
        y_ring.append(-1)
    for _ in range(40):
        r, t = rng2.uniform(2.5, 3.5), rng2.uniform(0.0, 2 * math.pi)
        X_ring.append([r * math.cos(t), r * math.sin(t)])
        y_ring.append(1)

    ring_model = LinearSVMGD(lr=0.1, epochs=500, l2=0.01)
    ring_model.fit(X_ring, y_ring)
    ring_preds = ring_model.predict(X_ring)
    ring_acc = sum(1 for a, p in zip(y_ring, ring_preds) if a == p) / len(y_ring)
    print(f"ring accuracy (linear, expect poor) = {ring_acc:.3f}")

    # Experiment 3 — polynomial kernel demo + hand-feature rescue.
    a, b_pt, c = [1.0, 0.0], [0.9, 0.1], [-2.5, 0.0]
    print(f"K(a, a)        = {polynomial_kernel(a, a):.3f}")
    print(f"K(a, neighbour) = {polynomial_kernel(a, b_pt):.3f}")
    print(f"K(a, far)       = {polynomial_kernel(a, c):.3f}")

    X_lifted = add_radius_feature(X_ring)
    lifted = LinearSVMGD(lr=0.05, epochs=800, l2=0.01)
    lifted.fit(X_lifted, y_ring)
    lifted_preds = lifted.predict(X_lifted)
    lifted_acc = sum(1 for a, p in zip(y_ring, lifted_preds) if a == p) / len(
        y_ring
    )
    print(f"ring accuracy (with radius feature) = {lifted_acc:.3f}")

    # Optional: decision boundary + margins (matplotlib pinned in requirements).
    try:
        import matplotlib.pyplot as plt

        # Separable blobs, colour by true label; boundary w·x+b=0 ± margin lines.
        w, b = model.w, model.b
        xs0 = [p[0] for p, lab in zip(X_sep, y_sep) if lab == -1]
        ys0 = [p[1] for p, lab in zip(X_sep, y_sep) if lab == -1]
        xs1 = [p[0] for p, lab in zip(X_sep, y_sep) if lab == 1]
        ys1 = [p[1] for p, lab in zip(X_sep, y_sep) if lab == 1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(xs0, ys0, label="y=-1")
        axes[0].scatter(xs1, ys1, label="y=+1")
        grid = [-5.0 + 10.0 * i / 99 for i in range(100)]
        if abs(w[1]) > 1e-9:
            for level, style in ((0.0, "-"), (1.0, "--"), (-1.0, "--")):
                axes[0].plot(
                    grid,
                    [(-(w[0] * gx + b) + level) / w[1] for gx in grid],
                    linestyle=style,
                )
        axes[0].set_title("Linear SVM: boundary + margins")
        axes[0].legend()
        xr0 = [p[0] for p, lab in zip(X_ring, y_ring) if lab == -1]
        yr0 = [p[1] for p, lab in zip(X_ring, y_ring) if lab == -1]
        xr1 = [p[0] for p, lab in zip(X_ring, y_ring) if lab == 1]
        yr1 = [p[1] for p, lab in zip(X_ring, y_ring) if lab == 1]
        axes[1].scatter(xr0, yr0, label="inner (y=-1)")
        axes[1].scatter(xr1, yr1, label="outer (y=+1)")
        axes[1].set_title("Ring data: not linearly separable")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
