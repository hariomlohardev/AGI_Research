# Month 3 · Week 1 Day 5 — Coding problems: manual backward pass, with independent verification

Work in `month_3/notebooks/week_1/day_5/manual_backprop.ipynb`. Smoke mode only (`CONFIG["mode"] == "smoke"`, seed 11, CPU, `float64`). Use torch tensor operations for your manual branch; the reference branch alone may use autograd. Budgets are per problem and assume smoke sizes (`B=4, D=3, H=5, C=3` unless stated).

Shared forward (both branches compute the same function): `X → Linear → training-mode BatchNorm → tanh → Linear → mean softmax cross-entropy`. BatchNorm uses the stated population variance (divide by batch size `B`) and `eps`. Every stub below is your implementation; signatures carry type hints, docstrings state Takes/Returns, and every `ValueError` case is documented.

## Problem 1 — Parameters (one seeded constructor)

Implement `init_params(d_in: int = 3, hidden: int = 5, n_classes: int = 3, seed: int = 11) -> dict[str, torch.Tensor]`.

- In: widths and seed. Out: dict with `W1 (H,D)`, `b1 (H,)`, `gamma (H,)`, `beta (H,)`, `W2 (C,H)`, `b2 (C,)` — all `float64`, all `requires_grad=True`, small seeded values (`gamma` ones, `beta`/`b1`/`b2` zeros).
- `ValueError` when any width `< 1` or `seed < 0`.
- Check: shapes and dtypes match; same seed rebuilds identical tensors.

## Problem 2 — Shared forward (the function both branches differentiate)

Implement `forward_manual(X: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor], eps: float = 1e-5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`.

- In: input batch `X (B,D)`, class targets `y (B,)` (`long`), and the parameter dict. Out: `(loss, cache)` where `loss` is the scalar mean cross-entropy and `cache` holds graph-connected intermediates keyed `z1, mu, var, xhat, zbn, a, logits` (`mu`/`var` computed over dim 0 with population variance + `eps`).
- `ValueError` when `eps <= 0`, `X.dim() != 2`, batch size `< 1`, `y` shape/dtype is wrong, a params key is missing, or `X.shape[1]` mismatches `W1.shape[1]`.
- Check: loss is a finite scalar; cache shapes are `(B,H)` preactivations, `(H,)` statistics, `(B,C)` logits.

## Problem 3 — Manual backward (tensor arithmetic only)

Implement `manual_backward(X: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], eps: float = 1e-5) -> dict[str, torch.Tensor]`.

- In: the same inputs and targets plus the forward cache. Out: gradients keyed `dW1, db1, dgamma, dbeta, dW2, db2, dX` plus intermediate grads keyed `dlogits, da, dzbn, dxhat, dvar, dmean, dX` (single names per tensor — `dX` appears once).
- No `backward`, `autograd.grad`, or functional autodiff inside this function — plain tensor arithmetic from your paper derivation.
- `ValueError` when a cache key is missing or shapes disagree with `X`/params.
- Check: every gradient is finite with the expected shape; `db2` equals the upstream `dlogits` summed over the batch (bias-broadcast rule).

## Problem 4 — Autograd reference (the independent branch)

Implement `reference_backward(loss: torch.Tensor, params: dict[str, torch.Tensor], intermediates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`.

- In: the scalar loss, params, and the non-leaf intermediates to probe. Out: the same gradient keys as Problem 3, via `torch.autograd.grad` (retain graph as needed — ordinary non-leaf `.grad` is not populated automatically).
- `ValueError` when `loss.numel() != 1` or a requested key is missing.
- Check: keys match the manual dict exactly; all values finite.

## Problem 5 — Comparison (one error table)

Implement `compare_grads(manual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], atol: float = 1e-8, rtol: float = 1e-6) -> dict[str, dict[str, float | bool]]`.

- In: two gradient dicts and tolerances. Out: per key `{"max_abs_err": float, "passed": bool}` (allclose-style).
- `ValueError` when key sets differ or a tolerance is negative.
- Check: on the correct implementation every key passes at `atol=1e-8, rtol=1e-6` in `float64`.

## Problem 6 — Break it, locate it, repair it (the injected error)

Implement `manual_backward_broken(X, y, params, cache, eps=1e-5)` and `find_first_mismatch(broken: dict, reference: dict, order: list[str], atol=1e-8, rtol=1e-6) -> str | None`.

- The broken variant treats BatchNorm's batch mean/variance as constants: it returns zeros for `dvar`/`dmean` (documented injection — everything upstream of `xhat` stays correct, `dX`/`dW1`/`db1` go wrong downstream of the omission).
- `find_first_mismatch` walks `order` (backward order: `dlogits, da, dzbn, dgamma, dbeta, dxhat, dvar, dmean, dX, dW1, db1, dW2, db2`) and returns the first key failing the allclose check, or `None` when all pass.
- Check: the broken dict's first mismatch is exactly `"dvar"`; the repaired (Problem 3) dict returns `None`.

## Problem 7 — Evidence (gradient table, convention check, honest conclusion)

Run the smoke comparison, fill the gradient table (per-key `max_abs_err` + pass), and demonstrate two mechanism probes: (a) a reference built with unbiased variance (divide by `B−1`) **fails** the comparison — conventions and `eps` must match; (b) perturbing one input row changes another row's `dX` — training-mode BatchNorm couples examples. Fill the conclusion cell: the full error table, where the 1/B mean-loss factor enters, why the bias backward is a sum, what the injected error taught, and one sentence per item on what it does and does not prove.
