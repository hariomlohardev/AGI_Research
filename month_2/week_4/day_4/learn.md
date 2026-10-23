# Day 22 — Compare Against Real `sklearn` (Week 4 Day 4, Month 2)

**Topic:** side-by-side validation of `minisklearn` against scikit-learn on the exact Titanic split
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> Validation day, not a learning day. The source plan explicitly says
> **“Watch first: none today.”** There are therefore no videos below by
> design. Do not go hunting for tutorial videos instead of running the
> comparison.

## Why this matters (from the plan)

This is the actual proof that your from-scratch implementations are
correct — not “it ran without crashing,” but “it gets essentially the
same answer as the industry-standard library.” A small, explainable gap
is a pass. An unexplained gap is not a pass yet: it may be defaults,
stopping rules, initialization, scaling, or a real bug.

## The validation protocol

1. **Reuse the exact Day 21 split.**
   - Rebuild `(X, y)` with Day 21’s `load_titanic_xy`.
   - Split with Day 20’s `stratified_split(X, y, test_ratio=0.2, seed=0)`.
   - Never change the seed, ratio, cleaning choice, or feature order and
     still call the result “the same comparison.”
2. **Train matching pairs.**
   - `LogisticRegression`
   - `DecisionTreeClassifier`
   - `RandomForestClassifier`
   - `KMeans`
3. **Use the same ruler.**
   - For the three classifiers, score both implementations with your own
     `minisklearn.metrics`: accuracy, precision, recall, and F1.
   - For K-Means there are no true labels in the algorithm, so compare
     **inertia** (within-cluster sum of squared distances), not accuracy.
     Cluster labels are arbitrary, so do not compare label `0` with label
     `0`.
4. **Investigate at least one discrepancy fully.**
   - Likely causes: regularization defaults, stopping criteria, random
     initialization, number of trees, K-Means initialization, unscaled
     features, or an actual implementation bug.
   - “The numbers are close” is not an investigation. Say what you
     changed, what moved, and what that implies.
5. **Write the honest paragraph.**
   - Where did your implementation match closely?
   - Where did it diverge?
   - What is your best-supported explanation?

## Defaults worth recording before you explain any gap

- `sklearn.linear_model.LogisticRegression` defaults to L2 regularization
  (`C=1.0`) and an iterative solver with `max_iter=100`. Your
  `minisklearn` logistic model may use little or no L2 and a fixed number
  of gradient steps. Different weights with similar accuracy can therefore
  mean a flat region/optimum, not necessarily a bug.
- `sklearn.tree.DecisionTreeClassifier` and
  `sklearn.ensemble.RandomForestClassifier` make different splitting,
  tie-breaking, bootstrap-sampling, and feature-subsampling choices than a
  compact ID3 implementation.
- `sklearn.cluster.KMeans` defaults to `k-means++` initialization and
  multiple restarts (`n_init=10`). Your implementation starts from
  distinct sampled data points and follows one Lloyd trajectory unless you
  add restarts yourself.
- Both K-Means implementations see the same unscaled Titanic features.
  Fare dominates Euclidean distance for both sides, so this is a fair
  implementation comparison even though unscaled K-Means would be a
  questionable modeling choice.

## Environment and dataset

- New pinned dependency: `scikit-learn==1.9.1`.
- The dataset remains the vendored Day-20 file:
  `month_2/week_4/day_2/code/data/titanic.csv`.
- Rebuild target: 889 rows and 7 numeric features, with approximately 38%
  positive labels.

## Practice questions (source plan)

1. If your model and `sklearn`’s converge to noticeably different weights but similar accuracy, what does that tell you about the loss surface?
2. Name one hyperparameter default in `sklearn`’s implementation that could explain a gap with your simpler version.
3. Why is this comparison step arguably the most important one this entire month, more than any single algorithm?

## Done when (from the source plan)

Your from-scratch implementations get within a small, explainable margin
of `sklearn`’s results on real data — and you have written down why for
every case that did not match closely.
