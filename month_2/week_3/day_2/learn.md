# Day 14 — Logistic Regression (Week 3 Day 2, Month 2)

**Topic:** binary logistic regression from scratch — sigmoid, binary
cross-entropy, gradient-descent training, hand-rolled metrics
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> All data today is synthetic and generated inline (seeded `random`) — see
> `coding_problems.md`. No external dataset is fetched.

## Why this matters

This is literally a single-layer neural network with a sigmoid activation
and cross-entropy loss. Day 13's GD loop returns with a new loss; Week 2
Day 4's cross-entropy becomes the thing you minimise; Week 4's
`linear_model.py` ships `LogisticRegression` next to `LinearRegression`.

## Watch (in this order)

1. **StatQuest: Logistic Regression** — StatQuest with Josh Starmer —
   https://www.youtube.com/watch?v=yIYKR4sgzI8
   The source-plan pick and the best conceptual on-ramp: builds the sigmoid
   from odds and log-odds with almost no jargon, so you understand *why* the
   output is a probability before you implement `sigmoid`. Pair it with the
   hand derivation below as the intuition check.
2. **Gradient Descent, Step-by-Step** — StatQuest with Josh Starmer —
   https://www.youtube.com/watch?v=sDv4f4s2SB8
   Same visual style, walks through the GD update loop concretely (step
   size, derivatives, iteration) — maps directly onto the training loop
   you'll code. Watch right before writing the weight-update code.
3. **Gradient descent, how neural networks learn | Deep Learning Chapter 2**
   — 3Blue1Brown — https://www.youtube.com/watch?v=IHZwWFHWa-w
   The geometric picture of cost surfaces and gradients — useful when
   debugging (recognising overshooting vs slow convergence from the loss
   curve). Frames a single neuron as essentially logistic regression: the
   bridge to Month 3.

(Runtimes omitted — not verifiable from here; the videos were confirmed
live via YouTube metadata at generation time. One gap the videos don't
cover: the BCE-as-maximum-likelihood derivation — that's Math §2 below,
worked by hand instead.)

## Math — work through by hand

1. **Sigmoid and its derivative.** Write `σ(z) = 1/(1+e^-z)`; explain why it
   squashes any real into `(0, 1)` (limits at `±∞`, midpoint `σ(0) = 0.5`).
   Derive `σ'(z) = σ(z)(1−σ(z))` from the formula — quota: from memory by
   `/done`.
2. **BCE is Week 2 Day 4's cross-entropy.** Write binary cross-entropy
   `-mean(y·log p + (1−y)·log(1−p))` and connect it to `H(P, Q)`: each sample
   is a 2-class distribution, the loss is its cross-entropy with the truth.
3. **BCE gradient via the chain rule.** Using `σ'(z)`, derive
   `∂BCE/∂w_j = (1/n)·Σ(p_i − y_i)·x_ij` — notice the error term has exactly
   the Day-13 shape with `p_i` in place of `ŷ_i`. The sigmoid derivative
   cancels; work the cancellation fully once.
4. **Thresholds move precision/recall.** Sketch why lowering the threshold
   from 0.5 to 0.3 can only grow the predicted-positive set, and what that
   does to each of precision and recall (practice question 4 previews
   Experiment 2).

## Code — see `coding_problems.md`

`sigmoid` → `binary_cross_entropy` → `bce_gradients` →
`LogisticRegressionGD` (`fit`/`predict_proba`/`predict` + `l2`) →
`metrics.py` (`confusion_matrix`, `accuracy`, `precision`, `recall`).

## Practice questions (source plan)

1. Derive `σ'(z) = σ(z)(1−σ(z))` from the sigmoid formula.
2. Why can't you use plain MSE loss for logistic regression the same way
   you did for linear regression? (Hint: think about whether the resulting
   loss surface stays convex.)
3. What does it mean, in probability terms, if your model outputs `0.5`
   for a given input?
4. Code check: change the classification threshold from 0.5 to 0.3 — how do
   precision and recall each change, and why does that direction of change
   make sense?
