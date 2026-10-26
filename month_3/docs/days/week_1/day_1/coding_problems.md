# Coding problems — Month 3 Week 1 Day 1: tensors, shapes, autograd

Work in `tensors_autograd.ipynb` (smoke mode, CPU, seed 11). Libraries do the
routine work (`torch` tensor ops, autograd, broadcasting); **you** supply the
hand derivations, predicted shapes, and every explanation. No datasets, no
downloads, no training runs today. Implement the five stubs; do not paste a
complete solution from anywhere — derive first, then verify.

## Recall (paper, before code)

- Derive the chain rule for a three-function composition.
- Explain how a shared node receiving two downstream branches contributes to
  the upstream gradient.

## Derive and predict (paper, before code)

1. For scalar `z = x*y + x**2`, derive `dz/dx` and `dz/dy` by hand.
2. Reuse `x` in two branches (`z = x*x + 3*x`). Predict the summed
   contribution at `x = 2` before running anything.
3. Predict shapes for `(B, D) @ (D, H)` with `B=4, D=5, H=3`, adding a bias of
   shape `(H,)`, and elementwise multiplication of a `(B, D)` tensor by a mask
   of shape `(B, 1)`. Write each prediction down; the test checks it later.

## Problem 1 — Scalar gradients vs your hand derivation

Implement `scalar_grads(x, y)` for `z = x*y + x**2` using torch autograd.
- Input: Python floats `x`, `y`. Output: `(dz_dx, dz_dy)` as floats.
- Check: exact agreement with your hand derivation at `x=2, y=3`
  (`7.0` and `2.0`); test tolerance is exact float equality against a float64
  reference, so build the graph in a precision that survives it.
- Evidence: one-row comparison (hand vs autograd) saved in your conclusion.

## Problem 2 — Shared node sums its branches

Implement `shared_branch_grads(x)` for `z = x*x + 3*x`.
- Input: float `x`. Output: `dz_dx` as a float.
- Check: equals `2x + 3` (`7.0` at `x = 2`).
- Evidence: your pre-run prediction plus the verified value. If they differ,
  the prediction was wrong — say which branch you miscounted.

## Problem 3 — Accumulation, leaf vs non-leaf, graph reuse

Implement `accumulation_demo()` with `z = w**2` at `w = 3.0`: backward once,
backward again without zeroing, then `zero_()` and backward once more.
- Output: `(single, accumulated, cleared)` = `(6.0, 12.0, 6.0)`.
- Then, in two short demo cells of your own (reviewed at `/done`, not tested):
  1. Show an intermediate tensor with `requires_grad` whose `.grad` stays
     `None` after backward, and explain why it is not retained by default.
  2. Show the error from reusing a graph without retaining it, and explain in
     one paragraph why `retain_graph=True` is not the normal training-loop fix
     (fresh forward graph per step is).
- Budget: part of the 2-hour implementation block. If Month 1 left you a
  scalar autograd engine, compare the scalar case against it where it runs;
  extending that engine is not today's task.

## Problem 4 — Shape worksheet and one honest broadcast bug

Implement `shape_predictions()` returning `{"matmul": (4, 3), "bias_add": (4, 3),
"mask_mul": (4, 5)}` from your own predictions — write them before checking.
Implement `broadcast_error_example()` returning `{"intended", "actual", "why"}`:
a real case where PyTorch broadcasts validly but the result is not what you
meant (shapes must genuinely differ; `why` names the broadcast rule that
silently changed the meaning). Correct the bug beside it.

## Problem 5 — Evidence (no new code)

Save in the notebook conclusion: the gradient comparison (Problems 1–2), the
shape table with actuals, the accumulation triple, and one open question.
Clean-kernel `Run All` in smoke mode must pass before `/done`.

## Optional (only after core checks pass)

Second derivative with `create_graph=True` on the scalar case; explain what
extra graph must exist. Do not let this displace core shape practice.
