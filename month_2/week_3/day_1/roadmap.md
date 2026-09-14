# Roadmap — Week 3 Day 1: Linear Regression (Gradient Descent)

**Current:** Month 2, Week 3, Day 1 (global Day 13) — *the ancestor of every neural-net training loop*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the three StatQuest videos** in `learn.md` (linear regression →
   gradient descent → multiple regression).
2. **Derive on paper:** single-feature `∂MSE/∂w`, then the multi-feature form,
   then the L2-augmented gradient.
3. **Implement** `code/linear_regression.py` (`predict`, `mse_loss`,
   `mse_gradients`, `LinearRegressionGD` with `fit`/`predict` + `l2` support).
4. **Implement** `code/normal_equation.py` (Gaussian elimination from scratch)
   as the comparison target.
5. **Run the demos:** `python code/linear_regression.py` (clean fit, high-lr
   divergence, L2 shrinkage) and `python code/normal_equation.py`.
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on the
   gradient derivation, learning rates, and MAP↔L2.

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `linear_model.py` in embryo: the `fit`/`predict` interface,
the MSE objective, and the L2 knob are exactly what `LinearRegression` ships
with. Logistic Regression (Day 14) reuses the same GD loop with a different
loss — learn the loop once, reuse it all month.

---

## Files in this day

- `learn.md` — hypothesis/loss/gradient theory + verified videos
- `coding_problems.md` — build spec with complete contracts
- `roadmap.md` — execution order and final-project tie
- `code/linear_regression.py` — GD model stub + runnable experiments
- `code/normal_equation.py` — closed-form reference stub
- `code/tests/` — pytest suite for both modules
- `code_review.md` — written by `code-evaluator` during `/done`
