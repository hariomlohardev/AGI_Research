# Roadmap — Week 3 Day 6: K-Means Clustering

**Current:** Month 2, Week 3, Day 6 (global Day 18) — *first unsupervised day: find structure with no labels*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (clustering intuition → from-scratch
   Lloyd loop, right before coding).
2. **Derive on paper:** the within-cluster SS objective, then the two-step
   argument (assignment step can't hurt given fixed centroids; mean-update
   can't hurt given fixed assignments — so the objective never increases),
   then the convergence-vs-global-optimum paragraph.
3. **Implement** `code/kmeans.py`:
   - `euclidean` / `assign_clusters` / `recompute_centroids` (empty cluster
     keeps its centroid) / `inertia` / `standardize`,
   - `KMeans` with `n_clusters`/`max_iters`/`tol`/`random_state`, `fit`
     (distinct-point init via `Random.sample`, Lloyd loop until max shift
     `<= tol`) / `predict`, plus `cluster_centers_` and `n_iters_`,
   - `elbow_inertias` for the k-sweep.
4. **Run the demos:** `python code/kmeans.py` (3-blob centroids + plot,
   init-sensitivity across seeds, elbow `1..8` with bend at 3, ×1000 stretch
   distortion + standardize rescue) — note the convention shift: float rows,
   integer labels, no `y` anywhere.
5. **Inspect the plots (recommended):** check the 3 clusters separate cleanly,
   the elbow bends at 3, and the stretched-axis run visibly follows feature 0.
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on the
   never-increases argument, local optima, Euclidean scaling, and the elbow —
   **plus the Week 3 review** (last day of the week: classical-ML `done_when`).

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `cluster.py` in embryo — the `KMeans` you write today *is*
`cluster.py` once renamed, behind the same `fit`/`predict` interface as the
supervised models (with `predict` meaning "assign to nearest centroid").
Unlike Days 13–17 there is no accuracy table for it — Week 4 evaluates it by
inertia and by eyeball on the same EDA plots from Day 2. The standardize-then-
cluster lesson (features beat algorithms, again after Day 15's radius rescue)
returns on Week 4 Day 2 when encoding/scaling decisions on real data matter
more than model choice. Keep `kmeans.py` — Week 4 imports it unchanged.

---

## Files in this day

- `learn.md` — Lloyd theory + verified videos
- `coding_problems.md` — build spec with complete contracts + hand checks
- `roadmap.md` — execution order and final-project tie
- `code/kmeans.py` — helpers + KMeans stub with runnable experiments
- `code/tests/` — pytest suite (distances + assignment + means + inertia + scaling + fit/convergence + elbow)
