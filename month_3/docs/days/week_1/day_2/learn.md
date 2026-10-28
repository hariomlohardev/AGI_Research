# Learn — Month 3 Week 1 Day 2: training pipeline

**Time:** 5–6 focused hours. **Goal:** build the reusable training skeleton used by the remaining months: fixed data, reproducible batches, a two-layer Module, separate train/evaluate paths, an honest baseline, and one bounded shuffle comparison.

## Recall (paper, before code)

- Explain gradient accumulation from Day 1 in one paragraph: what accumulates, where, and what clears it.
- Point at yesterday's `(B, D)` tensor and name the batch axis. Say what a "batch" becomes once a `DataLoader` enters the picture.

## Read

- The Module, Dataset/DataLoader, and optimization sections of [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). This is today's primary reading — follow it for API shape, not for study depth.
- **Video (verified 2026-10-26):** [PyTorch Tutorial 09 – Dataset and DataLoader – Batch Training](https://www.youtube.com/watch?v=PXOzkkB5eH0) — Patrick Loeber, ~15 min. A build-and-train walkthrough: custom `Dataset` (`__len__`/`__getitem__`) plus `DataLoader` batching, then an `nn.Module` trained with `MSELoss` and SGD. Verified via the creator's playlist feed, channel feed, watch page, and oEmbed. Honest gap: it does not focus on `model.train()`/`model.eval()` + `no_grad()` evaluation discipline — the plan's own material carries that part, and your `evaluate` stub enforces it.

## Derive (paper, before code)

For independent per-example gradient estimates with covariance Σ, the mean of B estimates has covariance Σ/B. Derive it (variances of a sum of independent variables, then the 1/B² from the mean). Then discuss, in writing: why do correlation between examples, sampling without replacement, and changed optimization dynamics complicate "double the batch, halve the variance"? This identity needs no CLT approximation — say why.

## Build and verify (details in `coding_problems.md`)

1. Fixed small regression data, reproducible 48/16 split, one tiny custom `Dataset`; show the `TensorDataset` equivalent; `num_workers=0`.
2. Two-layer `nn.Module` with separate `train_epoch` / `evaluate` (`train()` / `eval()` / `no_grad` in the right places).
3. Registered parameters inspected; one optimizer step moves weights; example-weighted epoch loss, never an unweighted mean of unequal batch means.
4. Constant predictor fitted on training targets as the required baseline (not `ln(K)`).
5. Sample-independence check on the BatchNorm-free MLP: loss from sample 0, gradients w.r.t. all batch inputs.
6. One bounded ordered-vs-shuffled comparison with identical inits and update budget; record either outcome.

## Questions (answer in the notebook, then verify)

1. What does `model(x)` do beyond a direct `forward(x)` call?
2. Why do validation batches normally avoid random augmentation and gradient recording?
3. Why are parameters shared across examples compatible with zero cross-sample input gradients?
4. Why is a larger batch not automatically a better use of a fixed compute budget?

## Done when / evidence

The complete loop runs, weights update, held-out MSE and baseline are logged, sample independence passes, and your shuffling observation is documented. The loop is saved as named function cells with inputs, outputs, and loss denominator explained. Restart the kernel and run the notebook in smoke mode before `/done`.
