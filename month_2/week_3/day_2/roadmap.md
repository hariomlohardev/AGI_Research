# Roadmap — Week 3 Day 2: Logistic Regression

**Current:** Month 2, Week 3, Day 2 (global Day 14) — *a single-layer neural net in disguise*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the three videos** in `learn.md` (logistic regression →
   gradient descent step-by-step → 3Blue1Brown cost surfaces).
2. **Derive on paper:** sigmoid `σ'(z)`, then BCE-from-cross-entropy, then
   the `(p_i − y_i)` gradient with the sigmoid-derivative cancellation.
3. **Implement** `code/logistic_regression.py` (`sigmoid`,
   `binary_cross_entropy`, `bce_gradients`, `LogisticRegressionGD` with
   `fit`/`predict_proba`/`predict` + `l2` support).
4. **Implement** `code/metrics.py` (`confusion_matrix`, `accuracy_score`,
   `precision_score`, `recall_score`) — by hand, no libraries.
5. **Run the demos:** `python code/logistic_regression.py` (clean fit,
   threshold 0.5-vs-0.3 comparison, L2 shrinkage).
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on the
   sigmoid derivative, BCE↔cross-entropy, thresholds, and MSE-vs-BCE.

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `linear_model.py` second half: the `fit`/`predict_proba`/
`predict` interface and the L2 knob are exactly what `LogisticRegression`
ships with, and `metrics.py` is the embryo of Week 4's evaluation step
(confusion matrix, accuracy/precision/recall/F1). Day 13's GD loop reappears
with BCE instead of MSE — learn the loop once, reuse it all month.

---

## Files in this day

- `learn.md` — sigmoid/BCE/gradient theory + verified videos
- `coding_problems.md` — build spec with complete contracts
- `roadmap.md` — execution order and final-project tie
- `code/logistic_regression.py` — GD classifier stub + runnable experiments
- `code/metrics.py` — hand-rolled metrics stub
- `code/tests/` — pytest suite for both modules
