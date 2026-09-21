# Day 17 — Random Forests (Week 3 Day 5, Month 2)

**Topic:** bootstrap aggregating (bagging) + feature randomness + majority-vote ensembles on top of yesterday's ID3 trees
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> All data today is small, categorical, and embedded inline — the same 6-row hand example and 14-row play-tennis table from Day 16, plus a tiny synthetic noisy version of it. No external dataset is fetched. Labels remain arbitrary hashables. Code is stdlib-only (`math`, `random`, `collections`).

## Why this matters

Day 16's unlimited tree memorised all 14 tennis rows (`1.0` train accuracy) — the classic low-bias, high-variance trap. Today fixes that by building a *crowd* of trees instead of one. Each tree sees a different bootstrap resample of the rows and, at each split, a different random subset of the features — then the forest votes. The trick is old, still wins on tabular data, and is the first honest ensemble you'll have built from scratch. Week 4's `ensemble.py` ships today's class behind the same `fit`/`predict` interface as Days 13–16, so `RandomForestClassifier` drops into the model comparison loop unchanged — and the `20`-vs-`1` tree comparison you run today is the prototype for the "single model vs ensemble" row in that final table. The OOB idea also returns on Week 4 Day 3 when you evaluate without an explicit hold-out.

## Watch (in this order)

1. **StatQuest: Random Forests Part 1 — Building, Using and Evaluating** — StatQuest with Josh Starmer — https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
   Intuition-first pick: how bootstrap sampling (sampling *with* replacement) and random feature subsets at *each* split decorrelate trees, why majority-vote reduces variance without adding much bias, and what Out-of-Bag samples are. Direct continuation of yesterday's ID3 — watch before touching `bootstrap_sample`.

2. **Random Forest in Python — Machine Learning From Scratch 10** — Patrick Loeber — https://www.youtube.com/watch?v=Oq1cKjR8hNo
   Implementation-leaning pick: part 10 of a respected *ML From Scratch* series. Codes a `RandomForest` class from scratch — loop `n_estimators`, draw bootstrap samples, fit an ID3-style tree with `max_depth`/`max_features` limits, store trees, `predict` by majority vote. Mirrors `coding_problems.md` almost one-to-one (your API is `n_estimators`, `max_features`, `max_depth`, `bootstrap`, `random_state`). Short; watch right before coding.

3. **Bootstrapping Main Ideas!!!** — StatQuest with Josh Starmer — https://www.youtube.com/watch?v=Xz0x-8-cgaQ
   Bootstrap-deep-dive (6 min, optional but recommended if `bootstrap_sample` / OOB feels fuzzy): why a resample of size `n` contains ~63% unique rows and where the ~37% OOB comes from. Makes the `(1 − 1/n)^n → 1/e` hand check trivial.

(All three verified live at generation time via YouTube oEmbed. If you want a longer lecture afterwards, search "Stanford CS229 bagging random forests" — first 30 min, same bagging+OOB story at lecture pace.)

## Math — work through by hand

1. **Bootstrap: where 37% comes from.** Sampling `n` rows *with* replacement from `n` rows: for any fixed original row, `P(miss in one draw) = 1 − 1/n`, so `P(never picked in n draws) = (1 − 1/n)^n → 1/e ≈ 0.368` as `n` grows. Hence a bootstrap sample contains ~63.2% distinct original rows and ~36.8% OOB rows. Quota: reproduce `1/e ≈ 0.368` from memory by `/done`, and state the expected OOB count for `n = 14` (≈5 rows).

2. **Feature randomness decorrelates.** Plain bagging resamples rows but always lets every tree consider *all* features at each split — if one feature (e.g. Outlook) dominates, every tree splits on it first and the trees stay correlated. Random Forests sample `m ≈ √p` features per split (or `log2 p`, or an integer you set via `max_features`): each tree is forced to try different splits, correlation drops, variance of the average drops roughly as `ρ·σ² + (1−ρ)·σ²/B` where `B = n_estimators` and `ρ` is mean tree-correlation. State in one sentence why `m ≪ p` hurts `ρ` more than it hurts bias.

3. **Variance down, bias ~flat.** A single deep ID3 tree has low bias (can memorise) but high variance (wiggles with the sample). Bagging averages `B` such trees: `Var(average) → ρ·σ²` as `B` grows, so more trees never *increase* variance and typically lower it; bias rises only slightly because each tree's bootstrap sample is ~ the same distribution as the original. Say without notes when you'd expect diminishing returns in `B` (practice question 4's plot).

4. **Majority vote, not average.** For classification, the forest predicts the most-voted label, not the mean of probabilities — ties (possible with even `B`) break by deterministic convention (documented tomorrow as first-seen). No derivation, just a convention to name.

## Code — see `coding_problems.md`

`bootstrap_indices` → `bootstrap_sample` → `RandomForestClassifier` (`fit`/`predict`, `n_estimators`/`max_depth`/`min_samples_split`/`max_features`/`bootstrap`/`random_state`, reuse of Day 16's `entropy`/`information_gain` inside each tree).

## Practice questions (source plan)

1. Why does bootstrap sampling (sampling with replacement) sometimes duplicate rows and sometimes leave rows out entirely, for the same original dataset?
2. Explain in your own words why averaging many high-variance, low-bias trees (via bagging) reduces overall variance without much increase in bias.
3. Why does Random Forest add *feature* randomness on top of *row* randomness (bagging alone) — what specific failure mode does this prevent?
4. Code check: run an experiment varying the number of trees (e.g. 1, 5, 20, 100) and plot accuracy vs. number of trees — at what point do you see diminishing returns?
