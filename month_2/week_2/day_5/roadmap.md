# Roadmap — Week 2 Day 5: Mutual Information

**Current:** Month 2, Week 2, Day 5 (global Day 11) — *how much does knowing one variable tell you about another*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` first (StatQuest MI, ~16 min, then Complexity Explorer, ~13 min) — ~30 min for the two core videos, plus Ben Lambert's ~8.5 min recap if the KL formulation still feels slippery. Focus on how the same quantity reads both as "uncertainty removed" and as "distance from independence".
2. **Math on paper (20-30 min, do not skip):**
   - Write both formulas from memory: `I(X;Y) = H(X) − H(X|Y)` and `I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))`.
   - Hand-compute: perfectly correlated binary pair → `1.0` bit; independent binary pair → `0.0`. Say out loud what each number means in "uncertainty removed" words.
   - Reason through why MI ≥ 0 needs no new proof (it's a KL), and why `I = 0` means independent while covariance 0 never did.
3. **Code — `mutual_information`** (`coding_problems.md` Problem 1): count the joint and marginals from the paired samples, the KL-shaped sum with `math.log`. Confirm the `1.0` / `0.0` hand values, symmetry, and non-negativity.
4. **Code — `rank_features_by_mi`** (Problem 2): one MI call per column, sorted best-first. Confirm the informative feature beats noise by a clear margin on seeded synthetic data.
5. **Experiments** (Problem 3): run the `__main__` demos — confirm the ranking print and the `Y = X²` pair (covariance `0.0` vs MI `≈ 2.28`).
6. **Practice questions** (in `learn.md`) on paper before step 5 — if you can't argue MI ≥ 0 from KL ≥ 0 without code, you don't know the math yet.
7. **Commit & run `/done`:** `python -m pytest code/tests/ -v` must pass before the quiz. `/done` will quiz you on the two hand values, the MI ≥ 0 argument, the independence-vs-uncorrelated distinction, and the `Y = X²` demo numbers.

---

## How this ties to the final project (`minisklearn`)

This day is the **feature-selection** day. Week 4 Day 2's EDA step decides which columns of a real dataset are worth modelling — `rank_features_by_mi` is the principled version of that decision, replacing correlation tables that go blind on nonlinear structure. The deeper tie is Week 3 Day 4: a decision tree's **information gain** `IG(S, feature) = H(S) − Σ (|Sᵥ|/|S|)·H(Sᵥ)` is mutual information between the feature and the label wearing different notation — today's function is nearly the split criterion verbatim, so keep it. (And per the source plan: MI is the conceptual ancestor of attention — "how much does this token tell you about that token" — file that away for Month 4.)

---

## Files in this day

- `learn.md` — what to learn + verified videos
- `coding_problems.md` — problem statements (intermediate scale)
- `code/mutual_information.py` — stub for `mutual_information`
- `code/feature_selection.py` — stub for `rank_features_by_mi` + runnable `__main__` experiments
- `code/tests/` — pytest suite (must pass before `/done`)
- `further_reading.md` — appended only if you run `/for-read` (opt-in)
