# Roadmap — Week 3 Day 4: Decision Trees (ID3 Algorithm)

**Current:** Month 2, Week 3, Day 4 (global Day 16) — *no gradients today: split by uncertainty reduction*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (StatQuest impurity/gain intuition →
   ML Recipes from-scratch build, right before coding).
2. **Derive on paper:** the IG formula, then the 6-row hand example
   (`H(S) = 1.0`, `IG(A) = 1.0`, `IG(B) ≈ 0.082`, ID3 picks A), then the
   pure-node gain-is-zero argument.
3. **Implement** `code/decision_tree.py` (`entropy`, `information_gain`,
   `majority_label` with first-seen tiebreak, `DecisionTreeID3` with
   `fit`/`predict`/`to_dict` + `max_depth`/`min_samples_split`, and
   `format_tree` for readable output).
4. **Run the demos:** `python code/decision_tree.py` (hand-example gains,
   full tennis tree print, `max_depth=1` ≈ 0.714 vs unlimited = 1.0
   train accuracy, unseen-value fallback) — note the label convention
   shift: arbitrary hashables, not `{0, 1}` / `{-1, +1}`.
5. **Inspect the tree (recommended):** read the printed tennis tree and
   check it against the textbook shape (root Outlook; sunny → Humidity;
   overcast → yes; rain → Wind).
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on
   the IG hand calculation, pure-node stopping, unlimited-depth risk,
   and the depth-1-vs-full comparison.

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `tree.py` in embryo — the `fit`/`predict` interface
matches Days 13–15 exactly, so a `DecisionTreeClassifier` drops into the
model loop unchanged, and tomorrow's `ensemble.py` (Random Forest) is a
bagged crowd of today's class with a feature-subset knob. The depth
experiment is also the first honest overfitting demo of Week 3: keep the
`0.714 vs 1.0` numbers — Week 4's evaluation step repeats the "training
accuracy is not the point" lesson on real data.

---

## Files in this day

- `learn.md` — IG/tree theory + verified videos
- `coding_problems.md` — build spec with complete contracts
- `roadmap.md` — execution order and final-project tie
- `code/decision_tree.py` — impurity helpers + ID3 classifier stub with runnable experiments
- `code/tests/` — pytest suite (gain hand values + tennis ordering + tree behaviour + depth demo)
