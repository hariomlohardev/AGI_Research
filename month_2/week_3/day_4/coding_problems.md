# Day 16 — Decision Trees (ID3 Algorithm): coding problems

Scale: **intermediate** (source plan as-is). Python, stdlib only in the tested
functions (`math` for `log2`, nothing else — not even `numpy`). Plain Python
lists (`X` is rows × columns of hashable categorical values), not numpy —
same discipline as Days 13–15, now with a recursive splitter instead of
gradient descent.

> Labels are arbitrary hashables today (`"yes"`/`"no"`, `0`/`1`, …) — trees
> never do arithmetic on labels, they only count them. There is deliberately
> no label-range `ValueError` (unlike the `{0, 1}` / `{-1, +1}` days); the
> contracts below validate shapes and indices instead.
>
> Data note: everything below uses small categorical tables embedded inline
> (a 6-row hand example plus the classic 14-row Quinlan play-tennis table —
> public-domain teaching data, values hand-listed in `__main__`, no fetch).
> No external dataset is fetched — the point today is gain arithmetic plus
> recursive ID3, not data wrangling. Week 4 Day 2 puts these models on a
> real dataset.

---

## Problem 1 — `entropy`, `information_gain`, `majority_label`

Create `code/decision_tree.py` with a from-scratch implementation.

```python
def entropy(labels: Sequence[Hashable]) -> float:
    """Return base-2 Shannon entropy (bits) of a label multiset."""

def information_gain(
    X: Sequence[Sequence[Any]], y: Sequence[Any], feature_idx: int
) -> float:
    """Return `H(y)` minus the size-weighted child entropies splitting by column `feature_idx`."""

def majority_label(labels: Sequence[Any]) -> Any:
    """Return the most common label; ties break toward the first-seen label."""
```

### Full validation contracts

Every public function/method docstring must include its complete `Raises`
contract (docstrings carry the contract — see the Day 10 process fix):

- `entropy`: raise `ValueError` for empty `labels`. Pure input gives
  exactly `0.0`; a 50/50 split gives exactly `1.0` (base 2, bits).
- `information_gain`: raise `ValueError` for empty `X`/`y`,
  `len(X) != len(y)`, ragged rows, rows with zero features, or
  `feature_idx` out of range (`0 <= feature_idx < n_features`). A
  single-valued column yields exactly `0.0` (no split is possible).
- `majority_label`: raise `ValueError` for empty input. The first-seen
  tiebreak is a documented convention, not a mathematical fact — say so
  in the docstring.
- Hand check first: the 6-row example from `learn.md` gives
  `information_gain(X, y, 0) == 1.0` and
  `information_gain(X, y, 1) ≈ 0.0817`; on tennis, Outlook `≈ 0.2467` >
  Humidity `≈ 0.1518` > Wind `≈ 0.0481` > Temperature `≈ 0.0292`, so the
  root split is Outlook.

## Problem 2 — recursive ID3 classifier + readable print

Same file (`code/decision_tree.py`):

```python
class DecisionTreeID3:
    """Recursive ID3 decision tree for categorical features."""

    def __init__(
        self, max_depth: int | None = None, min_samples_split: int = 2
    ) -> None:
        """Store stopping rules; the tree itself is built lazily in `fit`."""

    def fit(
        self, X: Sequence[Sequence[Any]], y: Sequence[Any]
    ) -> DecisionTreeID3:
        """Build the tree recursively; return `self` (sklearn-style)."""

    def predict(self, X: Sequence[Sequence[Any]]) -> list:
        """Return one label per row (raise if not fitted)."""

    def to_dict(self) -> dict:
        """Return the learned tree as nested dicts (raise if not fitted)."""

def format_tree(
    tree: dict, feature_names: Sequence[str] | None = None
) -> str:
    """Render a `to_dict` tree as an indented, human-readable string."""
```

- Tree nodes are plain dicts: a leaf is `{"label": <label>}`, an
  internal node is
  `{"feature": <idx>, "children": {<value>: <subtree>}, "majority": <label>}`.
- `DecisionTreeID3.__init__`: raise `ValueError` for `max_depth`
  smaller than 1 (when not `None`) or `min_samples_split` smaller than 2.
- `fit`: raise `ValueError` for empty data, length mismatch, ragged
  rows, or rows with zero features. Recursion stops at: pure node,
  `depth >= max_depth`, fewer than `min_samples_split` rows, or no
  multi-valued feature left (a used feature is single-valued inside each
  child, so no path reuses a feature). Split ties break toward the
  lowest feature index — deterministic, documented.
- `predict`: raise `ValueError` if called before `fit` or on
  empty/ragged/width-mismatched input. An **unseen** feature value falls
  back to that node's majority label (documented convention, same
  spirit as Day 14's `0.0`-on-empty-denominator rule).
- `to_dict`: raise `ValueError` if called before `fit`.
- `format_tree`: raise `ValueError` for an empty/non-dict tree. Without
  `feature_names`, splits render as `x[j] = value`.
- Scope note, documented in the module docstring: exact-value
  categorical splits only — no midpoint threshold search for continuous
  features (that is CART, not ID3) and no gain-ratio normalisation (that
  is C4.5). Both are named as the follow-ups, not implemented.

### Required `__main__` experiments

1. Hand example: print `H(S)` plus `IG(A)` / `IG(B)` for the 6-row
   dataset — must read `1.0`, `1.0`, `≈ 0.0817` (practice question 1's
   setup — keep the numbers).
2. Tennis tree: fit unlimited depth and print `format_tree` with
   `["Outlook", "Temperature", "Humidity", "Wind"]` — root must be
   Outlook, overcast branch a pure `yes` leaf.
3. Depth comparison (practice question 4): fit `max_depth=1` vs
   unlimited on tennis, print both trees plus both training accuracies
   (`≈ 0.714` vs `1.0`).
4. Unseen-value fallback: predict `["snow", "mild", "normal", "weak"]`
   and print the result (root majority `yes`).

### Running tests

```bash
cd month_2/week_3/day_4 && python -m pytest code/tests/ -v
```

### Done when (from the source plan)

Your ID3 implementation matches your hand-worked information gain
calculation on the small example, and it produces a sensible, readable
tree on real categorical data.
