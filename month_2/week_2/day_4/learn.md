# Week 2 Day 4 — Cross-Entropy & KL Divergence

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 10**

---

## Why this matters

Yesterday entropy measured the surprise in the *true* distribution. Today asks the question every model actually faces: what does it cost when you use the *wrong* distribution? That cost is cross-entropy — and it is literally the loss function that logistic regression (Week 3 Day 2) and every neural net from Month 3 onward minimise. KL divergence is the same idea with entropy subtracted out: the *extra* bits you pay purely for being wrong. Learn these two formulas properly and the entire concept of "training = minimising loss" becomes a concrete statement instead of a slogan.

## What to learn today

### Math (on paper first, before code)

1. **Cross-entropy:**
   `H(P, Q) = -Σ P(x) log Q(x)` — expected surprise when the world follows `P` but you coded for `Q`. Hand-compute: `P = [0.5, 0.5]`, `Q = [0.9, 0.1]` → `≈ 1.737` bits, strictly *more* than yesterday's `H(P) = 1.0`. The extra `0.737` is the price of the wrong model.
2. **KL divergence:**
   `D_KL(P || Q) = Σ P(x) log(P(x) / Q(x)) = H(P, Q) − H(P)` — the extra bits *only*, with entropy subtracted out. Same pair: `D_KL(P || Q) ≈ 0.737`.
3. **Asymmetry — "distance" is the wrong mental model:**
   `D_KL(Q || P) ≈ 0.531 ≠ 0.737`. Swapping the arguments changes the answer, so KL is not a distance. Read `D_KL(P || Q)` as "how much does it cost to use `Q` when truth is `P`" — direction matters because truth and model play different roles.
4. **KL is never negative, and zero means identical:**
   `D_KL(P || P) = 0`, and Gibbs' inequality says it can't go below zero (intuition-level today: you can't do *better* than the true distribution at describing data it generated — no proof required). This is why minimising cross-entropy and minimising KL are the same optimisation: they differ by `H(P)`, which doesn't depend on your model.

### Code

- `cross_entropy(p, q, base=2.0)` and `kl_divergence(p, q, base=2.0)` — the one-line formulas from scratch (`math.log` only), with the `q(x) == 0 where p(x) > 0` case raising `ValueError`.
- `binary_log_loss(y_true, y_pred)` — mean cross-entropy over samples: the classifier's loss in miniature. The confident-wrong vs confident-correct pair (`3.322` vs `0.152`) is the whole reason this loss works.

See `coding_problems.md` for exact signatures and the three `__main__` experiments.

---

## Watch first (guided — at least two)

> Search YouTube for the exact phrases below if links don't open — don't rely on autoplay.

1. **But what is cross-entropy? | Compression is Intelligence Part 2 — 3Blue1Brown**
   - https://www.youtube.com/watch?v=GlYgs6v2YfU (~32 min)
   - The Part 2 you saved from yesterday. Rebuilds today's formulas from the compression framing you already know ("what does it cost to compress data when your model of the probabilities is wrong?"), asks what makes this loss function the best one, connects it to LLM pre-training loss, and has a dedicated **KL Divergence chapter (~31:35)**. Its chapters map almost 1:1 onto today's math bullets. Watch this one first.
2. **Neural Networks Part 6: Cross Entropy — StatQuest with Josh Starmer**
   - https://youtu.be/6ArSys5qHAU
   - Covers "cross-entropy is the loss classifiers minimise" head-on — Josh walks through cross-entropy as the loss for classification outputs in tiny steps with the math drawn out. Directly reinforces the logistic-regression/neural-net log-loss connection and pairs with implementing the loss in code.
3. **Mutual Information, Clearly Explained!!! — StatQuest with Josh Starmer** (reinforcement pick)
   - https://youtu.be/eJIp_mgVLwE
   - Mutual information *is* a KL divergence (`D_KL(P(X,Y) || P(X)P(Y))`), so this shows KL "in the wild" — how far reality sits from an independence assumption. Cements the "extra bits from using the wrong distribution" intuition and KL's non-negativity. Short; watch it if the asymmetry bullet still feels abstract after the first two.

---

## Practice questions (do on paper before coding)

1. Hand-compute `H(P, Q)` for `P = [0.5, 0.5]`, `Q = [0.9, 0.1]` (expect `≈ 1.737` bits). Why must the answer exceed `H(P) = 1.0`?
2. Using the identity `D_KL(P || Q) = H(P, Q) − H(P)`, compute both directions of KL for the pair above (`≈ 0.737` vs `≈ 0.531`). In one sentence, why does the direction change the answer?
3. A model predicts 0.9 for an event that doesn't happen. What is the log-loss (`≈ 3.322`), and why is it so much larger than the loss for predicting 0.9 when the event does happen (`≈ 0.152`)?
4. Code check: verify `cross_entropy(p, q) − H(p) == kl_divergence(p, q)` with your functions. Why does this identity mean minimising cross-entropy and minimising KL are the same optimisation?

---

## Done when

You can explain in one sentence each why cross-entropy is the loss classifiers minimise and why KL divergence being asymmetric means "distance" is the wrong mental model — and your functions produce the hand values above.

---

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers on Gibbs' inequality, maximum-likelihood-as-KL-minimisation, or cross-entropy in LLM training.
