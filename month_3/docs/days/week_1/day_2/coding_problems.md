# Coding problems — Month 3 Week 1 Day 2: training pipeline

Work in `training_loop.ipynb` (smoke mode, CPU, seed 11). Libraries do the
routine work (`torch` tensor ops, `nn.Module`, `Dataset`/`DataLoader`,
autograd, SGD); **you** supply the derivations, the loop semantics, and every
explanation. No downloads today — the regression data is generated in-code
with a pinned recipe. No full-scale training: tiny budgets only
(≤5 epochs in smoke). Implement the stubs; do not paste a complete solution
from anywhere.

## Recall (paper, before code)

- Explain gradient accumulation from Day 1 in one paragraph: what
  accumulates, where, and what clears it.
- Point at yesterday's `(B, D)` tensor and name the batch axis. Say what a
  "batch" will be once a `DataLoader` enters the picture.

## Derive (paper, before code)

For independent per-example gradient estimates with covariance Σ, the mean
of B estimates has covariance Σ/B. Derive it (variances of a sum of
independent variables, then the 1/B² from the mean). Then discuss, in
writing: why do correlation between examples, sampling without replacement,
and changed optimization dynamics complicate "double the batch, halve the
variance"? This identity needs no CLT approximation — say why.

## Problem 1 — Fixed data, reproducible split, one custom Dataset

Implement `make_regression_data(n=64, seed=11)` with the pinned recipe in
its docstring (`randn` from a seeded `Generator`, `y = X@[3,-2] + 0.5`
plus `0.1` noise). Implement `RegressionDataset` (`__len__` /
`__getitem__` returning one `(x_i, y_i)` pair). Implement
`build_loaders(X, y, batch_size=16, seed=11)`: first 48 rows train, last 16
validation; train wrapped in your custom Dataset with seeded shuffling, val
in a `TensorDataset` without shuffling; `num_workers=0` everywhere
(Windows/notebook default). Then, in a demo function, show the
`TensorDataset` equivalent of the training set and confirm it yields the
same first batch — after that, prefer the library wrapper whenever it fits.
- Check: split sizes, batch shapes `(16, 2)` / `(16,)`, `num_workers == 0`,
  same seed twice gives the same first batch.

## Problem 2 — Two-layer Module, separate train and evaluate

Implement `TwoLayerMLP` (`Linear(2,16)` → ReLU → `Linear(16,1)`) and the two
loop functions `train_epoch(model, loader, loss_fn, optimizer)` and
`evaluate(model, loader, loss_fn)`. `evaluate` sets eval mode, runs under
`no_grad`, and restores the mode it found. Both report example-weighted
loss (sum of `loss * batch_size` over `len(dataset)`), never an unweighted
mean of unequal batch means.
- Check: forward shape `(B, 1)`; `model.train()` / `model.eval()` flip
  `model.training`; `evaluate` leaves grads `None` on a fresh model and
  restores a training-mode model to training mode.

## Problem 3 — Parameters register, one step moves weights, loss is weighted

Inspect `list(model.parameters())` — four tensors, all with
`requires_grad=True`. In `run_smoke`, snapshot parameters, run one epoch,
and assert at least one tensor changed. Construct the two-batch case by
hand (batches of 16 and 8 with different means): show the naive mean of the
two batch-losses differs from the example-weighted epoch loss, and say
which one your `train_epoch` reports and why it is the honest denominator.
- Check: one optimizer step changes intended weights; weighted loss equals
  the full-set `MSELoss` on a single-batch loader exactly.

## Problem 4 — Beat the constant predictor

Implement `constant_baseline_mse(train_loader, val_loader)`: mean of all
training targets, MSE against all validation targets (collect under
`no_grad`; loaders are just target iterators here). Log held-out MSE next
to this baseline in your conclusion. The required regression baseline is a
constant fitted on training targets — not `ln(K)`, which belongs to a
different problem.
- Check: matches the manual mean-then-MSE computation (`isclose`).

## Problem 5 — Sample-independence check (no BatchNorm here)

Implement `sample_independence_check(model, batch)`: forward a full batch
whose inputs require grad, build the loss from sample 0 only, backward,
return per-row input-gradient norms. With no BatchNorm and no cross-sample
op, only row 0 may be nonzero — parameters are shared across examples, yet
input gradients stay per-sample. This is the check behind practice
question 3.
- Check: `norms[0] > 0`, all other rows exactly `0`.

## Problem 6 — One bounded ordered-vs-shuffled comparison

Implement `ordered_vs_shuffled_comparison(X, y, ...)`: sort by target, train
the same-seeded MLP for a few epochs on the ordered set (`shuffle=False`)
and again from the identical initialization on shuffled batches (seeded
generator), same update budget. Return both final train losses plus your
one-sentence observation. Record either outcome — shuffling need not win
visibly at this size, and a null result is a result.
- Check: both losses finite; observation non-empty; identical inits
  (re-run one arm and confirm the same number).

## Problem 7 — Evidence (conclusion cell + clean kernel)

Save in the conclusion: the variance-identity derivation reference, the
weighted-vs-naive loss numbers from Problem 3, held-out MSE vs baseline,
the independence result, and your shuffling observation. Save the loop as
named function cells for reuse with inputs/outputs/loss-denominator
explained. Restart the kernel and `Run All` in smoke mode before `/done`.
