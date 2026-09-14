# Week 3 Day 1 — Linear Regression (Gradient Descent Version)

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 13**

---

## Why this matters

You likely already know the closed-form Normal Equation version. Today's
version — trained with gradient descent — is the direct ancestor of how every
neural network in this roadmap gets trained. The hypothesis is a line (or
hyperplane), the loss is MSE, and the optimizer walks downhill one gradient
step at a time. Nail this loop and Month 3's backprop is the same idea with
more layers.

## What to learn

### Math — work through by hand before coding

1. **Hypothesis (multiple features):** `ŷ = w·x + b`.
2. **MSE loss:** `MSE = (1/n) Σ (ŷ_i − y_i)²`.
3. **Gradients:** derive `∂MSE/∂w` and `∂MSE/∂b` by hand. Start with one
   feature (`ŷ = w·x + b`, scalar `w`), then generalize: each weight's
   gradient is the error-weighted average of its feature.
4. **L2 connection:** adding `l2 · Σw_j²` to the loss adds `2·l2·w_j` to the
   gradient — this is Week 2 Day 2's Gaussian-prior/MAP insight wearing a new
   hat. The bias `b` stays unregularised.

### Watch first (guided)

1. **Linear Regression, Clearly Explained!!!** — StatQuest with Josh Starmer
   [Watch](https://www.youtube.com/watch?v=nk2CQITm_eo)
   The hypothesis itself: fitting a line (intercept + slope — the
   single-feature version of `ŷ = w·x + b`), residuals, and least-squares
   intuition. Base for the MSE derivation.
2. **Gradient Descent, Step-by-Step** — StatQuest with Josh Starmer
   [Watch](https://www.youtube.com/watch?v=sDv4f4s2SB8)
   The optimizer refresher: differentiate the loss, step parameters downhill.
   Maps directly onto today's `∂MSE/∂w`, `∂MSE/∂b`, and the learning-rate
   update.
3. **Multiple Regression, Clearly Explained!!!** — StatQuest with Josh Starmer
   [Watch](https://www.youtube.com/watch?v=zITIFTsivN8)
   The multi-feature generalization: one equation, several predictors —
   exactly the `ŷ = w·x + b` form the code requires.

---

## Practice questions (paper first)

1. Derive `∂MSE/∂w` from scratch for the single-feature case, then generalize
   to multiple features.
2. Why might gradient descent converge to a slightly different answer than the
   closed-form Normal Equation on the same data?
3. What happens with too-high vs too-low learning rates? Describe the expected
   loss-curve shape for each, then verify with the `__main__` high-lr
   experiment.
4. Add L2 (weight decay) to the loss and gradient — connect it back to Week 2
   Day 2's MAP-as-regularization insight.
5. **Code check:** implement everything in `code/linear_regression.py` and
   `code/normal_equation.py`, run both `__main__` demos, and pass the pytest
   suite. GD weights should land close to the Normal Equation solution with a
   smooth decreasing loss curve.

## Done when

Your gradient-descent weights sit close to the Normal Equation solution and
the loss curve decreases smoothly. After this day, `/done` quizzes the
gradient derivation, learning-rate behaviour, and the MAP↔L2 connection.

## Further reading

Not bundled — opt-in only. Run `/for-read` for extras (e.g. Adam internals,
feature scaling, or the Normal Equation's numerical trade-offs).
