# Day 18 — K-Means Clustering (Week 3 Day 6, Month 2)

**Topic:** unsupervised clustering via Lloyd's assign/update loop — within-cluster sum of squares, local optima, the elbow, and why Euclidean distance makes scaling matter
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> First unsupervised day: no labels anywhere. All data is tiny synthetic 2D blobs generated inline with `random.Random(seed).gauss` (fixed seeds → reproducible) — the point is the loop, local optima, and the elbow, not data wrangling. Code is stdlib-only (`math`, `random`); `matplotlib` appears only in `__main__` demos behind a deferred import (same pattern as Day 15).

## Why this matters

Days 13–17 all had a `y` to chase. Today there is none — you find structure with no teacher. K-Means is the simplest honest version of that: pick `k` centroids, assign each point to the nearest one, move each centroid to its points' mean, repeat until nothing moves. Two ideas from earlier weeks return with teeth: the mean is the optimal update (Week 2's MLE intuition — the mean minimises squared error, which is exactly why step 2 can't hurt the objective), and variance runs the metric (Week 1 Day 5 — stretch one feature ×1000 and Euclidean distance stops seeing the other one, which is why you standardize first). Week 4's `cluster.py` ships today's class behind the same `fit`/`predict` interface as the supervised models — but evaluated by inertia and by eyeball, never by accuracy.

## Watch (in this order)

1. **StatQuest: K-means clustering** — StatQuest with Josh Starmer — https://www.youtube.com/watch?v=4b5d3muPQmA
   Intuition-first pick: builds the core loop (pick `k` starting centroids, assign each point to the nearest centroid, recompute centroids as means, repeat until stable) from zero with StatQuest's signature visuals. Maps directly onto today's assignment/update/convergence implementation and gives the intuition behind *why* recomputing the mean is the right update. Watch before touching `assign_clusters`.

2. **Machine Learning for Everybody – Full Course** (K-Means sections only) — freeCodeCamp.org, taught by Kylie Ying — https://www.youtube.com/watch?v=i_LwzRVP7bg
   Implementation-leaning pick: shows K-means actually being coded and run in notebook style — use the video's chapters to jump straight to the "K-Means Clustering" and "K-Means and PCA Implementations" sections and skip the rest of the ~4-hour course. Models the *target behavior* (cluster assignments on 2D data) so you know what your finished loop should produce. Caveat: her implementation uses sklearn, not from-scratch code — the loop itself is yours to write. Watch right before coding.

(Both verified live at generation time via YouTube oEmbed. Neither video covers the elbow end-to-end or states the never-increases argument — both are specced by hand below, since they're today's math focus and your `__main__` experiment 3.)

## Math — work through by hand

1. **The objective: within-cluster sum of squares (inertia).** `J = Σ over rows of ||x − μ(label(x))||²` — each point pays its squared distance to its assigned centroid. Quota: write `J` from memory by `/done`, and evaluate it by hand on `X = [[0.0], [2.0]]`, `labels = [0, 0]`, `centroids = [[1.0]]` (exactly `2.0` — the same hand check your `inertia` must pass).

2. **The two-step argument (today's proof — derive on paper before coding).** Fix the centroids: reassigning each point to its *nearest* centroid can only lower (or keep) that point's term, so step 1 never increases `J`. Fix the assignments: moving each centroid to its points' *mean* minimises the sum of squared distances for that cluster (the mean is the least-squares minimiser — one-line calculus: `d/dμ Σ||x−μ||² = 0` gives `μ = mean`), so step 2 never increases `J` either. Chain them: `J` never increases across an iteration, is bounded below by 0, so the loop always converges. Quota: reproduce both halves in one sentence each by `/done`.

3. **Convergence is local, not global.** Monotone + bounded gives *a* fixed point, not the best one — `J` is non-convex in the assignments, so different initial centroids land in different local optima with different final inertias. That's why the spec fixes `random_state` for reproducibility *and* why practice Q1 makes you compare seeds 0/1/2: same data, different inits, visibly different inertias. More principled inits (k-means++) exist — named as the follow-up, not built today.

4. **Stopping rule + the empty-cluster convention (stated precisely because the videos hand-wave both).** Stop when the largest centroid shift (`euclidean` old vs new) is `<= tol`, or after `max_iters` iterations — whichever comes first; record the count in `n_iters_`. If a centroid loses all its points mid-loop, it keeps its previous position (documented convention, matching your `recompute_centroids` contract — the videos never specify this, so the spec does).

5. **The elbow is a judgment call, not a proof.** Fit `k = 1..8`, plot inertia vs `k`: inertia falls steeply while each new centroid captures a real blob, then flattens into diminishing returns once extra centroids just subdivide blobs. The "bend" at `k = 3` on today's data is where you'd stop — but on real data there is often no clean bend, and inertia keeps falling all the way to `k = n` (every point its own centroid, `J = 0`). Say by `/done` why a bigger `k` always *can* fit better yet shouldn't always win.

## Code — see `coding_problems.md`

`euclidean` → `assign_clusters` → `recompute_centroids` → `inertia` → `standardize` → `KMeans` (`n_clusters`/`max_iters`/`tol`/`random_state`, `fit`/`predict`, `cluster_centers_`/`n_iters_`) → `elbow_inertias`. Four `__main__` experiments: 3-blob fit + scatter, init-sensitivity seeds 0/1/2, elbow `1..8`, ×1000 stretch + standardize rescue.

## Practice questions (source plan)

1. Code check: fit `k=3` on *overlapping* blobs (same centres, std `2.0`) with `random_state=0,1,2` and compare the three inertias — why do they differ, and what does that tell you about what Lloyd's algorithm guarantees?
2. Explain in your own words why one full Lloyd iteration (assign, then update) can never increase the within-cluster sum of squares — and why that still doesn't promise the *global* optimum.
3. Code check: multiply feature 0 by 1000, refit, and describe what happens to the clusters — then `standardize` first and refit. Why did the stretched axis take over, and where have you seen "variance runs the metric" before?
4. Code check: run `elbow_inertias` for `k = 1..8`, plot it, and point to the bend — then say in one sentence why the elbow picks a model rather than proving one.
