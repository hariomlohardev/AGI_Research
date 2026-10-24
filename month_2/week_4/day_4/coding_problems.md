# Day 22 — Compare Against Real `sklearn`: coding problems

Scale: **intermediate** (source plan as-is). Python with your existing
`minisklearn` package, Day-20/Day-21 helpers, and newly pinned
`scikit-learn==1.9.1`.

> The governing constraint is identity: same CSV → same cleaning →
> same features → same `test_ratio=0.2, seed=0` split → both libraries.
> If any side uses different rows, different features, or a different
> split, stop and fix the harness before interpreting the numbers.

---

## Problem 1 — Reload the exact Day 21 split (provided, not rewritten)

Use the provided `load_same_split(csv_path, test_ratio=0.2, seed=0)` in
`code/compare.py`:

- It calls Day 21’s `load_titanic_xy` and Day 20’s `stratified_split`
  where they already live — no copies, no reimplementation, no new
  imports from you.
- The path setup at the top of `compare.py` is already written for the
  same reason: one shim, reused, not something to redo every day.

Confirm the reloaded split still looks like Day 21:
   - 889 total rows.
   - Approximately 38% positive labels on each side.
   - 7 numeric features per row.

## Problem 2 — Same-ruler supervised comparison (`code/compare.py`)

Implement:

1. `score_classifier(model, X_train, y_train, X_test, y_test)`
   - Accept either a `minisklearn` classifier or an `sklearn` classifier.
   - Fit on the training split.
   - Predict the test split.
   - Score those predictions with `minisklearn.metrics`.
   - Return:
     ```python
     {
         "accuracy": ...,
         "precision": ...,
         "recall": ...,
         "f1": ...,
     }
     ```
   - `ValueError` if the model does not expose `.fit`/`.predict`, if
     either split is empty, or if labels and rows have mismatched lengths.
   - Using your own metrics for both sides is deliberate: it isolates
     differences in learned models from differences in metric definitions.

2. `accuracy_gap(mine, theirs)`
   - Return the absolute accuracy difference.
   - `ValueError` if either value is not finite.

3. `gap_verdict(gap, tolerance=0.03)`
   - Return `"match"` when `gap <= tolerance`.
   - Return `"investigate"` otherwise.
   - `ValueError` if `gap` is negative or non-finite, or if `tolerance`
     is non-positive or non-finite.
   - The default tolerance is a starting line, not proof. A gap below it
     still needs a plausible cause if the models are doing visibly
     different things.

## Problem 3 — Run all four pairs and write the honest paragraph

Use the helpers for the supervised models, then handle K-Means directly:

1. Run these supervised pairs on the exact same split:
   - Your logistic model, for example with the Day-21 settings
     `lr=0.001, epochs=300`, against
     `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
   - Your `DecisionTreeClassifier` against
     `sklearn.tree.DecisionTreeClassifier(random_state=0)`.
   - Your forest, for example
     `RandomForestClassifier(n_estimators=25, random_state=0)`, against
     `sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)`.
2. For K-Means:
   - Fit your `KMeans(n_clusters=2, random_state=0)` and
     `sklearn.cluster.KMeans(n_clusters=2, n_init=10, random_state=0)` on
     the same training rows.
   - Compare inertia:
     - yours with `minisklearn.cluster.inertia`;
     - theirs with the fitted model’s `.inertia_`.
   - Do not compare cluster IDs directly.
3. Produce:
   - a classifier table with accuracy, precision, recall, and F1 for both
     implementations;
   - the two inertia values;
   - the accuracy gaps and verdicts;
   - one fully investigated discrepancy;
   - a short paragraph saying where your code matched, where it diverged,
     and why.
4. Suggested K-Means investigation order:
   - initialization method;
   - number of restarts;
   - convergence tolerance and iteration limit;
   - effect of unscaled `Fare`;
   - only then, a possible centroid-update bug.

### Running tests

```bash
cd month_2/week_4/day_4 && py -m pytest code/tests/ -v
```

Tests cover the helper contracts, a logistic-regression integration run on
the real Titanic split, and an `sklearn` K-Means smoke test. The full
four-model table and investigation paragraph are manual work; tests check
the machinery, not your conclusions.

### Done when (from the source plan)

You can honestly say your implementations are within a small,
explainable margin of `sklearn` on real data — and you have written down
why for every case that did not match closely.
