# Day 21 — Evaluate: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only for your
code — plus your own `minisklearn` package and your Day-20 helpers
(`month_2/week_4/day_2/code/`), imported, not copied.

> Building on Day 20: today's inputs are the four arrays from yesterday
> (`X_train/X_test/y_train/y_test`, 889 rows after the 2 Embarked-missing
> drops, 7 features). Reuse your `split.py` via a `sys.path` shim — the
> same pattern the test files already use — and rebuild `X, y` with
> `load_titanic_xy` below. If your rebuild disagrees with yesterday's
> arrays (shape, positive rate ~38%), stop: the bug is in the rebuild,
> and every number after it is suspect.

---

## Problem 1 — Derive the metrics by hand (paper, before code)

From the four confusion-matrix counts (TP, FP, TN, FN):

1. Write down the definitions of **accuracy**, **precision**, **recall**,
   and **F1** purely in terms of those counts.
2. Hand-work this example: `TN = 50, FP = 10, FN = 5, TP = 35`. Compute
   all four metrics. (Answers: accuracy `0.85`, precision `35/45 ≈ 0.778`,
   recall `35/40 = 0.875`, F1 `≈ 0.824`. If yours differ, re-derive —
   don't reach for code yet.)
3. Write one sentence for each: a model with high precision but low
   recall is telling you *what* about its mistakes? And the reverse?

## Problem 2 — Evaluation helpers (`code/evaluate.py`)

Small, generic, tested on synthetic fixtures:

1. `load_titanic_xy(csv_path)` — rebuild yesterday's pipeline from the
   raw CSV in one place: load rows, drop the 2 Embarked-missing rows,
   mean-impute Age, label-encode Sex/Embarked, assemble the same 7
   features (`Pclass,Sex,Age,SibSp,Parch,Fare,Embarked`) → `(X, y)`.
   `ValueError` on an empty/unusable file. (Same documented choices as
   Day 20 Problem 2 — if you changed a call, yesterday's arrays are the
   reference, not this function.)
2. `comparison_table(results)` — `results` is a list of
   `(name, accuracy, precision, recall, f1)` tuples → one formatted
   multi-line string: a header row naming all four metrics plus one row
   per model, floats at 3 decimals. `ValueError` on an empty list.
3. `misclassified_indices(y_true, y_pred, k=5)` — indices of the first
   `k` positions where prediction disagrees with truth (fewer than `k`
   mismatches → return what's there). `ValueError` on empty input or
   length mismatch.

## Problem 3 — Train two models, print the table, read the errors

1. Stratified-split your rebuilt `(X, y)` with **your own Day-20
   `split.py`** (`test_ratio=0.2, seed=0`) and assert the positive rate
   on each side sits near 38%.
2. Train at least 2 `minisklearn` models on the train side —
   `LogisticRegression` and `RandomForestClassifier` is the natural
   pair. **Day-13 lesson applies:** unscaled features (Fare reaches the
   hundreds) plus the default `lr=0.01` is the exact recipe that
   diverged on Day 13 — use a smaller `lr` (or standardize first) and
   confirm the loss actually falls before you trust any metric.
3. Score both on the test side with **your `minisklearn.metrics`**
   (`confusion_matrix`, `accuracy_score`, `precision_score`,
   `recall_score`, `f1_score`) — no reimplementation, that package is
   the point — and print the `comparison_table`.
4. Take your best model and manually inspect **5 misclassified test
   examples** (`misclassified_indices` finds them; print the raw
   feature rows + true label). Write one sentence on what pattern, if
   any, you notice among the errors. No pattern is also a finding —
   write that down instead of inventing one.

### Running tests

```bash
cd month_2/week_4/day_3 && py -m pytest code/tests/ -v
```

Tests cover synthetic fixtures for every helper, hand values for the
`minisklearn.metrics` functions (the `0.75 / 2/3 / 1.0 / 0.8` example),
CSV pins for the rebuild (889 rows, 7 numeric columns), and one
integration run (both models train, metrics land in sane ranges, the
table renders both names).

### Done when (from the source plan)

A clean printed comparison table of at least 2 models'
accuracy/precision/recall/F1 on the same real test set, computed
entirely with your own metric functions.
