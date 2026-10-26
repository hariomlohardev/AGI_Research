# Learn — Month 3 Week 1 Day 1: tensors, shape reasoning and autograd

**Time:** 5–6 focused hours. **Goal:** explain what every axis means and verify
gradients instead of treating backward as a black box.

## Recall first

Derive the chain rule for a three-function composition. Explain how a shared
node receiving contributions from two downstream branches passes gradient
upstream. If your Month 1 autograd engine handles only scalars, that is fine —
use it for the scalar comparison and do paper/PyTorch for vector expressions.

## Primary reading

- [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) —
  creation, shape/dtype/device, slicing, broadcasting, reductions. Read the
  sections you need, not the whole documentation site.
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) —
  graphs, leaf vs non-leaf, accumulation, gradient modes. Focus on: why
  intermediates require grad but keep no `.grad` by default; why repeated
  backward calls accumulate; what `retain_graph` actually retains.

## Video (one, verified, scoped)

[The spelled-out intro to neural networks and backpropagation: building
micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy,
2h25m. Verified this session via the creator's course index
([karpathy.ai/zero-to-hero](https://karpathy.ai/zero-to-hero.html)), the watch
page (no unavailable/private notice), and oEmbed channel attribution.

It is long because autograd mechanics is a genuinely complex topic — watch it
in scoped form: the scalar-Value engine construction, graph building, and the
backward pass with the accumulate-into-`.grad` plus `zero_grad` pattern (that
is the Day-1 payoff). Treat the final MLP-training demo as optional once the
accumulation pattern clicks. Tensor-API specifics (broadcasting rules, flag
semantics, `retain_graph`) stay with the reading above, not this video. No
second video — protect the implementation budget.

## Derive and predict (on paper, before code)

1. Scalar `z = x*y + x**2`: `dz/dx = y + 2x`, `dz/dy = x`. Check at `x=2, y=3`.
2. Shared node `z = x*x + 3*x`: both branches contribute, `dz/dx = 2x + 3`.
3. Shapes (`B=4, D=5, H=3`): `(B,D) @ (D,H)` → `(B,H)`; `+ bias (H,)` → `(B,H)`;
   `(B,D)` times mask `(B,1)` → `(B,D)`. Then construct one broadcast that runs
   but is not what you meant — and explain the rule that allowed it.

Full contracts are in `coding_problems.md`; the five stubs live in
`tensors_autograd.ipynb` (`CONFIG`: smoke mode, CPU, seed 11).

## Build and verify (in the notebook)

1. Floating-point vs integer-index tensors: shape, dtype, device, slicing,
   broadcasting, reductions with explicit dimensions.
2. Scalar expression on paper, Month 1 engine (where it runs), and PyTorch.
3. Leaf vs non-leaf `.grad`, accumulation across backward calls, `zero_()`.
   Fresh forward graph per step — that is the training-loop habit.
4. Reused-graph-without-retain error, with one paragraph on why `retain_graph`
   is not the normal fix.
5. Shape worksheet plus one deliberately wrong broadcast, corrected.

## Questions (answer, then verify)

1. Why do repeated uses of a tensor add gradient contributions?
2. Why can an intermediate require gradients while its `.grad` is not retained
   by default?
3. Why can broadcasting produce a mathematically valid result with the wrong
   intended shape?
4. What is the difference between clearing accumulated gradients and retaining
   a computation graph?

## Done when / evidence

Scalar gradients match; the shared-node case passes; the broadcast bug is
explained and corrected; accumulation/leaf/retain each have a demo plus a why.
Save the gradient comparison and shape table in the conclusion. Clean-kernel
smoke run passes before `/done`.

## Optional

Second derivative with `create_graph=True`; explain what extra graph must
exist. Only after the core checks pass.
