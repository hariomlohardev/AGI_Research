# Day 19 — Package Structure (`minisklearn`): coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only. No math to
derive today — this is a software-engineering day: move tested code, normalize
interfaces, expose a clean package. The package lives at `month_2/minisklearn/`
(month level, not inside this day's folder — Week 4 Days 2–6 all build on it).

> Where things live: your implementations stay where they are (Weeks 1–3 are
> the archive — never edit them for Week 4 reasons). `minisklearn/` gets
> **copies**, adapted to the uniform interface below. This day's tests live in
> this day's folder (`code/tests/test_package.py`) and import the package via
> a `sys.path` shim (same harness shape as prior days).

---

## Problem 1 — Move the models, normalize the interface

Fill the skeleton modules in `month_2/minisklearn/` (each has a MOVE MAP
comment naming the exact source — delete each map block as it lands):

| Module | Copy from | Expose as |
|---|---|---|
| `linear_model.py` | `week_3/day_1/code/linear_regression.py` (`LinearRegressionGD` + helpers) and `week_3/day_2/code/logistic_regression.py` (`LogisticRegressionGD` + helpers) | `LinearRegression`, `LogisticRegression` |
| `tree.py` | `week_3/day_4/code/decision_tree.py` (`DecisionTreeID3` + helpers — NOT day_5's older `helper.py` duplicate) | `DecisionTreeClassifier` |
| `ensemble.py` | `week_3/day_5/code/random_forest.py` (`RandomForestClassifier` + bootstrap helpers) | `RandomForestClassifier` (unchanged name) |
| `cluster.py` | `week_3/day_6/code/kmeans.py` (`KMeans` + helpers) | `KMeans` (unchanged name) |

Rules for the move (all tested):

- **Uniform interface.** Every supervised model: `.fit(X, y)` returns `self`,
  `.predict(X)` returns a list. Logistic keeps `.predict_proba(X)`.
  `KMeans` is the documented exception: `.fit(X)` (no `y` — there are no
  labels), `.predict(X)` assigns to nearest centroid. Say this asymmetry in
  `cluster.py`'s class docstring.
- **sklearn-style names.** `LinearRegressionGD` → `LinearRegression`,
  `LogisticRegressionGD` → `LogisticRegression`,
  `DecisionTreeID3` → `DecisionTreeClassifier`. Rename the classes or alias
  them (`DecisionTreeClassifier = DecisionTreeID3`) — your call, one comment
  saying which you chose and why.
- **Relative imports between siblings.** `ensemble.py` gets its tree via
  `from .tree import ...` — never a copied file, never `sys.path` hacks
  inside the package. (Tests use the path shim; the package itself must not.)
- **Docstrings on every public class and method** (one line each is enough —
  what it is, not how it works).
- Optional (ungraded, do it if it's quick): `svm.py` (`LinearSVMGD` from
  `week_3/day_3`) rides along as a sixth module. The plan's skeleton omits
  it — your call whether the package carries it.

## Problem 2 — `metrics.py` from scratch

New code, in `month_2/minisklearn/metrics.py` (signatures + `Raises` contracts
already stubbed). Binary `0`/`1` labels, positive class `1`:

```python
def confusion_matrix(y_true, y_pred) -> list[list[int]]: ...  # [[TN, FP], [FN, TP]]
def accuracy_score(y_true, y_pred) -> float: ...
def precision_score(y_true, y_pred) -> float: ...  # TP/(TP+FP), 0.0 if no predicted positives
def recall_score(y_true, y_pred) -> float: ...     # TP/(TP+FN), 0.0 if no actual positives
def f1_score(y_true, y_pred) -> float: ...         # 2PR/(P+R), 0.0 if P+R == 0
```

- Starting reference: `week_3/day_2/code/metrics.py` (trio + confusion
  matrix) — adapt it, don't blindly copy; its validation drifts between
  functions (see the Day-14 review) and `f1_score` is new.
- The zero-division guards (`0.0`) are documented conventions, not math.
- Hand check: `y_true=[1,0,1,1,0,0]`, `y_pred=[1,0,0,1,0,1]` →
  `cm=[[2,1],[1,2]]`, accuracy `4/6`, precision = recall = f1 = `2/3`.
- Every function raises `ValueError` on empty input, length mismatch, or any
  non-binary label.

## Problem 3 — `__init__.py`: the front door

Wire `month_2/minisklearn/__init__.py` with explicit re-exports so all of
this works from a *separate* script in a *different* directory:

```python
import minisklearn
minisklearn.linear_model.LogisticRegression()   # submodule path (the day's Done when)
from minisklearn import KMeans, f1_score        # top-level names
```

Top-level names required (tested): `LinearRegression`,
`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`,
`KMeans`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`,
`confusion_matrix`. Empty `__init__.py` vs wired: the practice questions make
you say why (Q2).

### Running tests

```bash
cd month_2/week_4/day_1 && python -m pytest code/tests/ -v
```

### Done when (from the source plan)

`import minisklearn` works cleanly from a separate script, and you can do
`model = minisklearn.linear_model.LogisticRegression()` and call
`.fit()`/`.predict()` on it.
