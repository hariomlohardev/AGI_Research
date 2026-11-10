# Month 3 · Week 1 Day 5 — Manual backward pass, with independent verification

**Time:** 6–8 focused hours (split across study days if needed). **Goal:** derive and implement gradients through a small MLP with BatchNorm, and locate an intentionally introduced error by comparison with autograd.

## Recall first (15–20 min, no new reading)

1. Derive a linear-layer gradient: given `dL/dz` for `z = x @ Wᵀ + b`, write `dL/dW`, `dL/db`, `dL/dx` with shapes — and say where the bias sum comes from.
2. Write the tanh derivative in terms of its input `a` and in terms of its output `t = tanh(a)`. Which form does a backward pass use, and why?

## Read (60–90 min, then close the tabs and implement)

1. The Backprop Ninja lesson from [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — the lesson that manually derives gradients layer-by-layer through the normalized 2-layer net at tensor scale. Watch the scoped sections below; pause to derive each step yourself.
2. The exact scope in `month_3/docs/roadmap.md` §Day 5: a tiny floating-point input `X`, `Linear → training-mode BatchNorm → tanh → Linear → softmax cross-entropy`. Implement BatchNorm's forward explicitly with the stated population variance (divide by batch size) and `eps`, so both branches compute the same function.
3. Use double precision on CPU for these small numerical checks (`float64`, smoke shapes `B=4, D=3, H=5, C=3`).

> Video (guided mode, one verified lecture): [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) — Andrej Karpathy, ~1h55m. Verified this session via the creator's course index (lists it as the Part 4 lesson, "manually derives gradients layer-by-layer"), the watch page (no unavailable/private notice), and oEmbed channel attribution. Watch the manual-backprop core — linear backward, training-mode BatchNorm backward, tanh backward, mean-reduction cross-entropy gradient, each checked against autograd — not necessarily end to end; skim Part 3 recap you already know. Honest gap: it derives BatchNorm backward in *training* mode only and does not run the injected-error diagnosis — the plan's own material carries the eval-mode check (Day 4) and today's break/locate/repair drill.

## Derive and predict (on paper, before code)

1. Shapes plus gradients for mean cross-entropy/logits (where does the `1/B` enter?), the second linear layer, tanh, BatchNorm's scale/shift **and** its normalization path (`dx̂`, `dvar`, `dmean`, `dx`), then the first linear layer and `X`.
2. The bias rule: broadcasting `b` across the batch in the forward pass requires a sum over the batch in the backward pass — write why.
3. The coupling prediction: changing one example's input changes another example's input-gradient through training-mode BatchNorm — write which shared statistics carry that dependency.
4. First-mismatch prediction: if the BatchNorm backward treats mean/variance as constants, which gradient is the first wrong one walking backward from the loss — and why is everything above `xhat` still correct?

## Build and verify (in the notebook)

Library work (use as-is): tensor operations, `torch.autograd.grad`, softmax cross-entropy. Your work: the seeded params, the shared forward, the full manual backward, the autograd reference, the error table, the injected error and its localization, and every explanation.

1. **Params.** `init_params(...)`: seeded `float64` dict (`W1, b1, gamma, beta, W2, b2`), all requiring grad.
2. **Forward.** `forward_manual(...)`: one shared forward returning the scalar mean loss plus a cache of graph-connected intermediates (`z1, mu, var, xhat, zbn, a, logits`).
3. **Manual.** `manual_backward(...)`: tensor arithmetic only — no `backward`, no `autograd.grad` inside. Returns parameter grads plus every intermediate grad.
4. **Reference.** `reference_backward(...)`: `torch.autograd.grad` on the loss for params and selected intermediates (`retain_grad` before `backward` where needed; non-leaf `.grad` is not auto-populated).
5. **Compare.** `compare_grads(...)`: per-key maximum absolute error plus an allclose-style pass at `atol=1e-8, rtol=1e-6`.
6. **Break and locate.** `manual_backward_broken(...)` omits the mean/variance dependency (zeros for `dvar`/`dmean`); `find_first_mismatch(...)` walks the backward order and must report exactly `"dvar"` — then the repaired backward reports `None`.
7. **Probes.** An unbiased-variance (÷(B−1)) reference must *fail* the comparison (conventions must match); a one-row input perturbation must move another row's `dX`.

## Questions (answer in the notebook conclusion)

1. Where does division by batch size enter the mean-loss derivative?
2. Why does broadcasting a bias in the forward pass require a sum in its backward pass?
3. Why can changing one example affect another example through training-mode BatchNorm?
4. Why must the manual and autograd branches use the same variance definition and epsilon?

## Done when / evidence

The complete gradient table passes and the injected error is located and corrected. If incomplete, log the failing operation and schedule an extra study block — do not mark it passed because six hours elapsed. Restart the kernel and run the notebook in smoke mode before `/done`.
