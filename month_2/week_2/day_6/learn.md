# Week 2 Day 6 — Review + Consolidation

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 12**

---

## Why this matters

This week moved from **how models fit parameters** to **how we measure information**. Today is not another isolated implementation. You will retrieve the essential formulas from memory, then consolidate the week into one clean `week2_utils.py` file with small assertions at the bottom. The goal is load-bearing knowledge: you should be able to explain how Bayes, MLE, MAP, entropy, cross-entropy, KL divergence, and mutual information fit together without treating them as seven unrelated tricks.

## What to review

### Math — write these from memory before watching anything

1. **Bayes:** `P(theta | data) ∝ P(data | theta) P(theta)` — posterior is likelihood weighted by prior.
2. **Gaussian MLE:** `mu_hat = mean(x)` and `sigma_hat² = mean((x - mu_hat)²)` — the variance MLE divides by `n`, not `n - 1`.
3. **MAP:** `argmax [log P(data | theta) + log P(theta)]`. A Gaussian prior adds a quadratic penalty, so it is L2 regularization; `lambda = 1 / (2 * prior_sigma²)`.
4. **Entropy:** `H(X) = -sum P(x) log P(x)` — expected surprise; uniform distributions maximize uncertainty.
5. **Cross-entropy and KL:**
   - `H(P,Q) = -sum P(x) log Q(x)`
   - `D_KL(P || Q) = sum P(x) log(P(x)/Q(x))`
   - `H(P,Q) = H(P) + D_KL(P || Q)`
6. **Mutual information:**
   - `I(X;Y) = H(X) - H(X|Y)`
   - `I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))`
   - `I = 0` iff independent; covariance `0` only rules out linear co-movement.

### The connecting story

- **Bayes** updates beliefs using evidence.
- **MLE** chooses parameters that make observed data most likely.
- **MAP** does the same while adding a prior preference; regularization is the log-prior in disguise.
- **Entropy** measures uncertainty in one distribution.
- **Cross-entropy** measures the coding/loss cost when the model uses another distribution.
- **KL** isolates the extra cost caused by that mismatch.
- **MI** measures the mismatch between the true joint distribution and the independence assumption, or equivalently the uncertainty removed by observing one variable.

## Watch first (guided)

> These are review videos, not new material. Re-watch the one that addresses the concept you found least stable, then use the second as a synthesis pass.

Use these verified review videos in this order:

1. **Lecture 11 — Statistical Estimation** — Stanford Engineering Everywhere, Stephen Boyd
   [Watch the lecture](https://see.stanford.edu/Course/EE364A/86) · approximately 1 hr 17 min
   Review maximum-likelihood estimation as parameter fitting and optimization. This is the best first pass for the MLE half of the week and gives the baseline for understanding MAP as likelihood plus a prior.
2. **Bayesian Inference in Generative Models** — Luke Hewitt, MIT
   [Watch the lecture](https://youtu.be/PRY2NbOXbHk) · approximately 49 min 45 sec
   Reconnect priors, likelihoods, and posteriors to the Bayes/MAP relationship: MAP selects the highest-posterior parameter value, while MLE does not include a prior.
3. **Lecture 1: Overview: Information and Entropy** — MIT 6.02, George Verghese
   [Watch the lecture](https://learn.mit.edu/video/10677/lecture-1-overview-information-and-entropy?playlist=10676) · approximately 49 min 9 sec
   Reset the conceptual meaning of information, uncertainty, and entropy before consolidating cross-entropy, KL divergence, and mutual information.

---

## Practice questions (paper first)

1. In one paragraph, explain how Bayes, MLE, and MAP relate. Include the role of the prior and say when MAP becomes MLE.
2. In one paragraph, explain how entropy, cross-entropy, and KL divergence relate. State which term stays constant when optimizing a model distribution.
3. Derive the Gaussian MLE for `mu` in enough steps to show why the sample mean appears.
4. A flat prior is uniform over the parameter range. Why can it change no argmax? State the argument without using a limit such as `tau -> infinity`.
5. For `P = [0.5, 0.5]` and `Q = [0.9, 0.1]`, state the approximate values of `H(P)`, `H(P,Q)`, `D_KL(P || Q)`, and `D_KL(Q || P)` in bits.
6. Explain why `I(X;Y) = 0` means independence while `Cov(X,Y) = 0` does not. Use a nonlinear example such as `Y = X²`.
7. **Code check:** implement or restore every function in `code/week2_utils.py`, then run the assertions at the bottom and the pytest suite. The file should run cleanly from top to bottom.

## Done when

`week2_utils.py` runs cleanly with one short assertion for each consolidated function, and you can write the two summary paragraphs above without checking notes. After this day, `/done` also runs the Week 2 review against the week's `done_when` criterion.

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary material on information geometry, Bayesian model selection, or MI estimators for continuous variables.
