# Roadmap — Week 3 Day 5: Random Forests

**Current:** Month 2, Week 3, Day 5 (global Day 17) — *many jittered trees beat one memorising tree*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (StatQuest RF intuition → Patrick Loeber from-scratch RF build → optional StatQuest Bootstrapping 6-min for the `1/e` ounce).
2. **Derive on paper:** the `(1 − 1/n)^n → 1/e ≈ 0.368` OOB fraction (and the `≈5` OOB rows for `n = 14`), then the feature-randomness decorrelation argument (`m ≈ √p` per split) and the variance-vs-bias paragraph.
3. **Implement** `code/random_forest.py`:
   - `bootstrap_indices` / `bootstrap_sample` with `random.Random(random_state)` determinism,
   - a compact ID3 helper (reuse `entropy`/`information_gain`/`majority_label` from Day 16, now with `max_features` sampling per split),
   - `RandomForestClassifier` with `n_estimators`/`max_depth`/`min_samples_split`/`max_features`/`bootstrap`/`random_state`, `fit`/`predict`, and deterministic majority-vote tiebreak.
4. **Run the demos:** `python code/random_forest.py` (bootstrap duplicates + OOB count for `n = 14` and `n = 1000`, single-tree vs forest on tennis, `n_estimators` sweep 1/5/20/100, feature-randomness ablation `max_features=None` vs `"sqrt"`).
5. **Inspect the forest (recommended):** check that a bootstrap resample really has duplicates, that increasing `B` stabilises predictions, and that `sqrt` forests are less correlated than bag-only forests.
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on OOB `1/e`, variance reduction, and why feature randomness matters beyond row bagging.

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `ensemble.py` in embryo — the `RandomForestClassifier` you write today *is* `ensemble.py` once renamed, behind the same `fit`/`predict` interface as `linear_model.py` and `tree.py`. Day 16's `DecisionTreeID3` becomes a bagged crowd with one knob (`max_features`) added; the depth-vs-ensemble comparison you run today (single deep tree `1.0` train vs forest ~stable) is the direct precedent for the model-comparison table on Week 4 Day 3. Keep both `random_forest.py` and yesterday's `decision_tree.py` — Week 4 imports the forest and shows that an ensemble can match or edge a tuned tree on real, noisy tabular data.

---

## Files in this day

- `learn.md` — bagging RF theory + verified videos (StatQuest ×2 + ML-From-Scratch)
- `coding_problems.md` — build spec with complete contracts + hand checks
- `roadmap.md` — execution order and final-project tie
- `code/random_forest.py` — bootstrap helpers + RF classifier stub with runnable experiments
- `code/tests/` — pytest suite (bootstrap distribution + determinism + OOB, forest fit/predict + sweeps + validation)
