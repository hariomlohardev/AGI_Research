# Day 20 — Apply to a Real Dataset: coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only
(`csv`, `math`, `random` — no pandas; the point is to feel every step).
No math to derive today — this is a data-wrangling day.

> The dataset is vendored at `code/data/titanic.csv` (Titanic passenger
> list, verified at generation time — never re-download blindly, this copy
> is what the tests pin):
> **891 rows**, columns
> `PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked`.
> Missing: **Age 177**, **Embarked 2** (Cabin is mostly missing — you will
> drop it, that's a decision, not an oversight). Target `Survived`:
> **342 survived / 549 not** (~38% positive — mildly imbalanced, which
> matters for Day 3's accuracy discussion). `Sex` is female/male,
> `Pclass` is 1/2/3.
> Days 3–6 build on the arrays you produce today — keep the pipeline
> re-runnable, not a one-off notebook scribble.

---

## Problem 1 — EDA (`code/eda.py`)

Load the CSV and look at it before touching anything:

1. `load_csv` → `(header, rows)` (raw strings, nothing parsed yet).
2. `missing_counts` → per-column count of missing entries. Confirm Age
   177 / Embarked 2 against the numbers above — if they don't match, your
   loader is wrong, not the data.
3. `class_balance` on the `Survived` column → `{0: 549, 1: 342}`. Write
   down what that imbalance will do to a naive "always predict 0"
   classifier (Day 3 will grade this).
4. `numeric_summary` (count/min/max/mean) for `Age` and `Fare` over the
   non-missing values. Eyeball the ranges — this is the same scale
   sensitivity Day 18's K-Means taught you, now in the wild.

## Problem 2 — Impute + encode (`code/encode.py`)

Build the helpers generically (they're tested on synthetic fixtures, not
just Titanic):

1. `to_float_column` — parse one column to `float | None` (`None` for
   missing; `ValueError` on genuine garbage like `"abc"` — missing is
   not garbage, conflate them and your mean is a lie).
2. `fill_mean` — replace the `None`s with the column mean; return the
   filled column *and* the mean used (Day 4's sklearn comparison needs to
   know exactly what you filled).
3. `label_encode` / `one_hot` — category → ints + the mapping/categories
   used (mappings are artefacts — unseen categories at predict time must
   raise, not silently map to 0).
4. **Your call, defended in one comment:** which features survive
   (`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked` is the sane default;
   `Name/Ticket/Cabin/PassengerId` need a reason if kept), and label vs
   one-hot for `Sex`/`Embarked`. Assemble the final fully-numeric `X`
   (`list[list[float]]`) and `y` (`list[int]`).

## Problem 3 — Stratified split by hand (`code/split.py`)

`stratified_split(X, y, test_ratio=0.2, seed=0)` — shuffle each class
separately with `random.Random(seed)`, then take the ratio from each.
No `sklearn`, no `random.shuffle` on the global RNG (seeded local RNG —
Day 18's determinism lesson). Assert in your own usage (not just tests):
train+test label counts sum back to 549/342, and each side's positive
rate is within a point or two of 38%.

### Running tests

```bash
cd month_2/week_4/day_2 && py -m pytest code/tests/ -v
```

Tests pin the real CSV's verified numbers (177/2/342) plus synthetic
fixtures for every helper — if a helper test fails on synthetic data,
the bug is yours; if only the CSV pins fail, your loader disagrees with
the file.

### Done when (from the source plan)

Clean, fully-numeric `X_train, X_test, y_train, y_test` with no missing
values remaining, ready to feed into your `minisklearn` models.
