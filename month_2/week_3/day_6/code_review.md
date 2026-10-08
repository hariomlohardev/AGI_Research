# Code review — Day 18: K-Means Clustering (Month 2, Week 3 Day 6)

Style/readability review only — tests pass 20/20, so nothing below is a correctness claim.

Attribution guard: three lines/blocks are known assistant-written during debugging
(`elbow_inertias` body `:354-357`, the `range(len(centroids))` fix `:74`, the
restored label-range check `:163-164`). They are reviewed for style but do not
count toward learner pattern counts. The test suite is generated code, not
learner code — notes on it sit at the end.

Sustained good (do not regress): excellent module-docstring scoping
(Lloyd-only, init convention, no k-means++, deferred-`matplotlib` note, `:1-15`);
`from __future__ import annotations` + `Sequence` hints on the public API; all
hot-path sums generator-form (`:40`, `:132`, `:203`); `self.cluster_centers_`
and `self.n_iters_` initialised in `__init__` (`:256-257`) — the
attribute-first-assigned-in-`fit` shape from Days 13/16/17 is fixed; no bare
sibling import; no dead helpers; four-experiment seeded `__main__` with deferred
`matplotlib` behind `try/except ImportError`.

## Findings

1. **Untyped `if_empty_raise`, empty-string default, spacing** (`:24-26`).
   `def if_empty_raise(data:list,message=""):` — unparameterized `data:list`,
   untyped `message`, no `-> None`, and `message=""` permits `ValueError("")`.
   Spacing off (`data:list`, `,message`, `len(data) == 0 :`). Name is settled
   (matches Days 16–17), keep it; fix the signature
   (`data: Sequence[...], message: str) -> None`), drop the `""` default,
   mark private with a leading underscore.
2. **Message wording/spelling** (`:38,104,110,113,194`).
   `"The lenght of a and b is different"`, `"lenths of X and labels doiffer"`
   (also `:158`), `"k cant be less than 1 "` (apostrophe + trailing space),
   `"lenght of rows is zero "` (trailing space, also `:282`, `:323`),
   four-voice empty-message drift, RST backticks leaked into runtime strings
   (`:113`, and `:164` which is assistant-restored wording, not a learner hit).
   Proofread messages; state the rule the docstring states.
3. **PEP 8 whitespace clusters** (`:24,57-58,76,109,115,197,202-203,205,207`
   + blank-line counts). Representative: `X , 'X is empty'`,
   `euclidean(row1 ,row2)`, `zip(a,b)`, `row1  =`, `if k < 1 :`,
   `dim =len(X[0])`, `list() , list() , list()`, 4 blank lines before
   `class KMeans`. One formatter pass clears all of it.
4. **`range(len(...))` indexing** (`:71`, `:126` — learner lines).
   `for i in range(n): row1 = X[i]` → `for row1 in X`;
   `members = [X[i] for i in range(len(X)) if labels[i] == j]` → `zip(X, labels)`.
   (`:74` is the known assistant fix; `for j in range(k)` and
   `for d in range(dim)` are index-legitimate, not hits.)
5. **Duplicated validation ×6** (`:57-68,101-122,155-164,188-194,276-285,317-326`).
   Ragged-row guard and empty guard copied across every sibling, plus `predict`
   re-validating input it then hands to `assign_clusters` (which validates
   again). Extract one `_validate_X(X) -> dim` helper; document the double-check
   if intentional.
6. **Vague accumulators** (`:70`, `:73`, `:167`).
   `out` (cluster indices → `assigned`), `temp_dist` (drop `temp_` → `dists`),
   `output = 0` (SSE → `total`). (`values` in `elbow_inertias` is same shape
   but assistant-written — not a learner hit.)
7. **`== None` returns** (`:314`). `if self.cluster_centers_ == None:` → `is None`.
   Days 16–17 had this fixed; it slipped back.
8. **Redundant parens** (`:202`, `:215`). `sum(col)/(n)`, `return (X_std, means, stds)`
   → `return X_std, means, stds`. Absent Days 16–17, back today.
9. **`list()` for empty lists** (`:197`). `= list(), list(), list()` → `= [], [], []`.
   Third occurrence (Days 14, 15) — now a Theme (see growth-notes).
10. **Garbled `recompute_centroids` docstring** (`:89-98`). The convention sentence
    dangles without a subject and the empty-cluster convention sits inside the
    `Raises:` block where no `ValueError` is raised for it. Move convention to the
    summary; keep `Raises:` strictly for cases that raise.
11. **Single-day idioms (no pattern claimed).** `distance = ...; return distance`
    (`:40-41`, return directly); `temp_dist` list built only for
    `.index(min(...))` (`:73-78`); `euclidean(...) ** 2` in `inertia` (`:170`,
    sqrt-then-square — sum squared diffs directly); divide-by-`1` zero-std dodge
    (`:204-205`, prefer `[0.0 for _ in col]` + comment); `X_std` reused for
    column-then-row orientation (`:197` vs `:213`, name them `cols_std`/`X_std`);
    single-letter `o, n` in the shift `max` (`:295`, `o` reads as zero);
    `enumerate`-then-index in `__main__` (`:391-392`, prefer `zip`);
    duplicated `matplotlib` import (`:387` vs `:415`).
12. **`predict` copies the `n_clusters > len(X)` guard** (`:325-326`), constraining
    prediction batch size by a training hyper-parameter — surprising next to
    `assign_clusters`, which takes any non-empty `X`. Drop it or comment why.

## Test-file note (generated code, not learner style)

`tests/test_kmeans.py:88` carries a leftover debugging comment (`# Label 2 with
k=2 is outside ... (fixed: ...)`). Worth deleting; the `sys.path` shim and
`_blobs`/`_make_blobs` duplication are the expected harness shape, not issues.
