# Day 17 — Random Forests: coding problems

Scale: **intermediate** (source plan as-is, built on Day 16's ID3). Python, stdlib only (`math` for `log2`/`sqrt`, `random` for bootstrap/feature sampling, `collections.Counter`). Plain Python lists (`X` is rows × columns of hashable categorical values), not numpy — same discipline as Days 13–16, now with an ensemble wrapper around a recursive splitter.

> Labels are arbitrary hashables (`"yes"`/`"no"`, `0`/`1`, …) — the forest never does arithmetic on labels, it only counts votes. No label-range validation. Data note: everything below reuses the small categorical tables from Day 16 (6-row hand example + 14-row Quinlan play-tennis table, public-domain teaching data, values hand-listed in `__main__`, no fetch) plus a tiny noisy variant you build by flipping one label. No external dataset — the point today is bagging + feature randomness + voting, not data wrangling. Week 4 Day 2 puts these models on a real dataset.

---

## Problem 1 — `bootstrap_indices`, `bootstrap_sample`

Create `code/random_forest.py` with from-scratch bootstrap helpers.

```python
def bootstrap_indices(n: int, random_state: int | None = None) -> list[int]:
    """Return `n` indices sampled with replacement from `range(n)`."""

def bootstrap_sample(
    X: Sequence[Sequence[Any]], y: Sequence[Any], random_state: int | None = None
) -> tuple[list[list[Any]], list[Any]]:
    """Return `(X_sample, y_sample)` of size `n = len(X)` drawn by `bootstrap_indices`."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises` contract (docstrings carry the contract — see the Day 10 process fix):

- `bootstrap_indices`: raise `ValueError` if `n < 1`. Deterministic when `random_state` is not `None` (same seed → same list); different seeds → different lists with high probability. Must use `random.Random(random_state)`, not the global `random` module state. Return length exactly `n`, every entry in `0 <= idx < n`. Duplicates are expected; the expected number of distinct indices is `≈ n·(1 − 1/e) ≈ 0.632·n` (show this for `n = 14` and for a large `n = 1000` in `__main__` — not asserted in tests, printed).
- `bootstrap_sample`: raise `ValueError` for empty `X`/`y`, `len(X) != len(y)`, ragged rows, or rows with zero features (same shape contract as Day 16's `information_gain`/`fit`). Preserve row↔label pairing: `X_sample[i]` comes from the same original row as `y_sample[i]`. A single-valued resample is allowed — it just means that bootstrap happened to miss some rows (that's OOB doing its job).
- Hand check first: for `n = 6` and `random_state=0`, the returned indices are reproducible across runs; for `n = 14` the mean OOB size across 1000 resamples with varying seeds is `≈ 5.1` (≈ 36.8% of 14).

## Problem 2 — `RandomForestClassifier` (bagged ID3 forest)

Same file (`code/random_forest.py`):

```python
class RandomForestClassifier:
    """Bagged ensemble of ID3 trees with per-split feature randomness."""

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Store hyper-parameters; the forest itself is built lazily in `fit`."""

    def fit(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> RandomForestClassifier:
        """Fit `n_estimators` ID3 trees on bootstrap resamples; return `self`."""

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Majority-vote across the `n_estimators` trees."""
```

Extra helpers (kept internal — not tested directly — but you may expose them for readability; the forest reuses them):

- `entropy`, `information_gain`, `majority_label` — same semantics and contracts as Day 16 (or import your Day-16 implementation verbatim — copy it into this file so the submission is self-contained). Pure input entropy `0.0`, 50/50 `1.0`, `information_gain` single-valued column `0.0`, tiebreak toward first-seen label.
- Per-tree `DecisionTreeID3` logic: recursive ID3 with `max_depth`/`min_samples_split`, no reuse of a feature along a path (a used feature is single-valued inside each child, so no path reuses it), unseen feature values fall back to that node's majority label, split ties break toward the lowest feature index.

- `RandomForestClassifier.__init__`: raise `ValueError` for `n_estimators < 1`, `max_depth` not `None` and `< 1`, `min_samples_split < 2`, `max_features` not in `{None, int >=1, "sqrt", "log2"}` (when `int`, require `1 <= max_features <= n_features` — but `n_features` is only known at `fit` time, so `__init__` only checks `>=1`; `fit` raises if `max_features > n_features`), or `bootstrap` not `bool` (be permissive — only the numeric/string contracts are strict). Document tie-breaking: majority vote ties break toward the first-seen label among the training labels (via `Counter` order), which makes predictions deterministic.
- `fit`: raise `ValueError` for empty data, length mismatch, ragged rows, or rows with zero features. Randomness contract: use a single `random.Random(random_state)` stream for both bootstrap index draws *and* per-split feature subsets, so `random_state=0` gives reproducible forests. When `bootstrap is False`, every tree sees the full training set (feature randomness still applies). Each tree is an ID3 tree built with the same `max_depth`/`min_samples_split`; at **each split** the tree considers only a random subset of the currently-available features of size `k`, where `k` is derived from `max_features`:
  - `None` → `k = n_features`
  - `int` → `k = that int` (clipped to `len(available_features)` at that node — never sample more than remain)
  - `"sqrt"` → `k = max(1, int(sqrt(n_features)))`
  - `"log2"` → `k = max(1, int(log2(n_features)))`
  The subset is drawn without replacement via `Random.sample` on the main RNG. This is the *per-split* randomness the videos call "feature bagging" — say so in the docstring.
- `predict`: raise `ValueError` if called before `fit` or on empty/ragged/width-mismatched input. Return one label per row, labels are hashables.

- Scope note, documented in the module docstring: categorical ID3 trees only — no midpoint threshold search for continuous features (CART) and no gain-ratio normalisation (C4.5). OOB error calculation is *not* implemented today (you just observe the OOB size) — Week 4 Day 3 revisits evaluation. `feature_importances_` and `predict_proba` are deliberately out of scope.

### Required `__main__` experiments

When you run `python code/random_forest.py` it must print all four (keep the numbers):

1. Bootstrap hand demo: `bootstrap_indices(14, random_state=0)` → print the 14 indices (show duplicates), plus the OOB set size for that resample; also print the *mean* OOB size across 500 resamples of `n=14` (should read ≈5.1) and the unique-count for one `n=1000` resample (≈632).
2. Single tree vs forest on tennis: fit a single `DecisionTreeID3`-equivalent (`RandomForestClassifier(n_estimators=1, max_features=None, bootstrap=False, random_state=0)`) vs a forest (`n_estimators=20, max_features="sqrt", random_state=0`) on the tennis table, print both train accuracies — the single deep tree hits `1.0` (memorised), the forest is also high but the point is stability across seeds (re-run with `random_state=1,2` and show less swing than single-tree).
3. `n_estimators` sweep (practice question 4): for `B in (1, 5, 20, 100)` fit `RandomForestClassifier(n_estimators=B, random_state=0)` on tennis (or on a tiny noisy-train / clean-test split you build by flipping one label), print `B` vs train accuracy *and* vs accuracy on a held-out 4-row test slice — accuracy should plateau by 20 (≈ diminishing returns).
4. Feature-randomness ablation: same data, `n_estimators=30, random_state=0`, compare `max_features=None` (bagging only) vs `max_features="sqrt"` — print both train and test accuracies and a one-line note that `sqrt` trees are less correlated (observe that predictions across trees vary more — you can print the disagreement rate).

### Running tests

```bash
cd month_2/week_3/day_5 && python -m pytest code/tests/ -v
```

### Done when (from the source plan)

Your random forest visibly outperforms — or at least matches with visibly lower variance across `random_state` — a single decision tree on the same test set, and you can explain why feature randomness matters beyond just row bagging.
