# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Month 3 · Week 1 Day 5 — Manual backward pass, with independent verification
#
# **Goal:** derive and code every gradient through `Linear → BatchNorm(train) → tanh → Linear → mean cross-entropy`, then check each one against a separate autograd reference and locate an injected error.
#
# ## Question and prediction
#
# *Question:* which intermediate gradient breaks first when the BatchNorm backward treats the batch mean/variance as constants, and why do the bias and mean-loss gradients take the exact forms they do?
#
# *Prediction (write yours before running):* the break should appear at or below the variance/mean path — upstream of `xhat` everything is shared arithmetic — while the bias gradient is a plain batch sum and the mean loss contributes one factor of `1/B` at the logits.
#
# *What would challenge it:* a first mismatch above `xhat` (wrong mental order), a bias gradient that is not the upstream sum, or a manual/reference table that passes under mismatched variance conventions.
#
# ## Shapes (smoke: B=4, D=3, H=5, C=3)
#
# | tensor | shape | notes |
# |---|---|---|
# | X | (B, D) | float64, requires grad in checks |
# | W1 / b1 | (H, D) / (H,) | |
# | z1 | (B, H) | `X @ W1.T + b1` |
# | mu / var | (H,) | batch stats, population variance + eps |
# | xhat / zbn / a | (B, H) | normalize → scale/shift → tanh |
# | gamma / beta | (H,) | |
# | W2 / b2 | (C, H) / (C,) | |
# | logits | (B, C) | `a @ W2.T + b2` |
# | loss | () | mean softmax cross-entropy |
#
# Library/manual boundary: tensor arithmetic is yours in the manual branch; autograd lives only in the reference branch. Both branches differentiate the same forward function.

# %%
import torch
import torch.nn.functional as F

CONFIG = {"mode": "smoke", "seed": 11}

B, D, H, C = 4, 3, 5, 3
ATOL, RTOL = 1e-8, 1e-6

BACKWARD_ORDER = ["dlogits", "da", "dzbn", "dgamma", "dbeta",
                  "dxhat", "dvar", "dmean", "dX", "dW1", "db1",
                  "dW2", "db2"]


# %% [markdown]
# ## Recall and derive (paper, before code)
#
# 1. Linear layer: given `dL/dz` for `z = x @ W.T + b`, write `dL/dW`, `dL/db`, `dL/dx` with shapes. Where does the bias sum come from?
# 2. `tanh`: write `d tanh(a)/da` in terms of the input `a` and in terms of the output `t = tanh(a)`. Which form does the backward use, and why?
# 3. Mean loss `L = (1/B) Σ lᵢ`: where exactly does the `1/B` enter `dL/dlogits`? What would change under a sum reduction?
# 4. BatchNorm (train): with `μ`, `σ²` over the batch, write the forward, then derive `dx̂`, `dσ²`, `dμ`, `dx` on paper. Circle the terms the injected error will delete.

# %%
def init_params(d_in: int = 3, hidden: int = 5, n_classes: int = 3,
                seed: int = 11) -> dict[str, torch.Tensor]:
    """Seeded float64 parameter dict for the smoke MLP.

    Takes: input width, hidden width, class count, seed.
    Returns: dict with W1 (H,D), b1 (H,), gamma (H,), beta (H,),
        W2 (C,H), b2 (C,) — float64, requires_grad=True.
    Raises: ValueError if any width < 1 or seed < 0.
    """
    if d_in < 1 or hidden < 1 or n_classes < 1:
        raise ValueError("widths must be >= 1")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    g = torch.Generator().manual_seed(seed)

    return {
        "W1" : torch.randn((H,D) , generator=g , dtype=torch.float64 , requires_grad=True),
        "b1" : torch.randn((H,) , generator=g, dtype=torch.float64 , requires_grad=True),
        "gamma" : torch.randn((H,) , generator=g, dtype=torch.float64 , requires_grad=True),
        "beta" : torch.randn((H,) , generator=g, dtype=torch.float64 , requires_grad=True),
        "W2" : torch.randn((C,H) , generator=g, dtype=torch.float64 , requires_grad=True),
        "b2" : torch.randn((C,) , generator=g, dtype=torch.float64 , requires_grad=True)
    }


# %%
def forward_manual(X: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor],
                   eps: float = 1e-5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shared forward: Linear -> train-BN -> tanh -> Linear -> mean CE.

    Takes: input batch X (B,D), class targets y (B,), and the parameter dict.
    Returns: (scalar mean cross-entropy loss, cache dict keyed
        z1, mu, var, xhat, zbn, a, logits) with graph-connected tensors.
    Raises: ValueError if eps <= 0, X is not 2-D, batch < 1, y shape/
        dtype is wrong, a params key is missing, or X features
        mismatch W1.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if X.dim() != 2:
        raise ValueError("X must be 2-D (B, D)")
    if X.shape[0] < 1:
        raise ValueError("batch size must be >= 1")
    if y.dim() != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be shape (B,) matching X")
    if y.dtype != torch.long:
        raise ValueError("y must be long dtype class indices")
    for k in ("W1", "b1", "gamma", "beta", "W2", "b2"):
        if k not in params:
            raise ValueError(f"params missing key: {k}")
    if X.shape[1] != params["W1"].shape[1]:
        raise ValueError("X features mismatch W1")

    W1, b1= params["W1"] , params["b1"]

    z1 = X @ W1.T + b1
    mu = z1.mean(dim=0)
    var = z1.var(dim=0 ,unbiased=False) + eps

    xhat = (z1 - mu) / var.sqrt()
    gamma, beta = params["gamma"] , params["beta"]
    zbn = gamma * xhat + beta

    a = torch.tanh(zbn)

    W2, b2 = params["W2"] , params["b2"]
    logits = a @ W2.T + b2
    loss = F.cross_entropy(logits , y)

    return (loss, {"z1": z1, "mu": mu, "var": var, "xhat": xhat, "zbn": zbn, "a": a, "logits": logits})



# %%
def manual_backward(X: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor],
                     cache: dict[str, torch.Tensor],
                     eps: float = 1e-5) -> dict[str, torch.Tensor]:
    """Hand-derived backward pass using tensor arithmetic only.

    Takes: inputs, targets, params, and the forward cache.
    Returns: gradients keyed dW1, db1, dgamma, dbeta, dW2, db2, dX
        plus intermediate grads dlogits, da, dzbn, dxhat, dvar, dmean.
    Raises: ValueError if a cache key is missing or shapes disagree.
        No backward / autograd.grad / functional autodiff in here.
    """
    for k in ("z1", "mu", "var", "xhat", "zbn", "a", "logits"):
        if k not in cache:
            raise ValueError(f"cache missing key: {k}")
    if cache["logits"].shape[0] != X.shape[0]:
        raise ValueError("cache batch disagrees with X")

    z1 = cache["z1"]
    mu = cache["mu"]
    var = cache["var"]
    xhat = cache["xhat"]
    zbn = cache["zbn"]
    a = cache["a"]
    logits = cache["logits"]

    B = X.shape[0]

    # 1. Cross entropy backward
    dlogits = (F.softmax(logits, dim=1) -
               F.one_hot(y, num_classes=logits.shape[1])) / B

    # 2. Second linear layer
    W2 = params["W2"]

    dW2 = dlogits.T @ a
    db2 = dlogits.sum(dim=0)
    da = dlogits @ W2

    # 3. Tanh backward
    dzbn = da * (1 - a**2)

    # 4. Batch normalization affine backward
    gamma = params["gamma"]

    dgamma = (dzbn * xhat).sum(dim=0)
    dbeta = dzbn.sum(dim=0)

    dxhat = dzbn * gamma

    # 5. Normalization backward
    # xhat = (z1 - mu) / sqrt(var)

    dvar = (dxhat * (z1 - mu) * -0.5 * var.pow(-1.5)).sum(dim=0)

    dmean = (dxhat * -1 / var.sqrt()).sum(dim=0)

    dz1 = (
        dxhat / var.sqrt()
        + dvar * 2 * (z1 - mu) / B
        + dmean / B
    )

    # 6. First linear layer
    W1 = params["W1"]

    dW1 = dz1.T @ X
    db1 = dz1.sum(dim=0)
    dX = dz1 @ W1

    return {
        "dW1": dW1,
        "db1": db1,
        "dgamma": dgamma,
        "dbeta": dbeta,
        "dW2": dW2,
        "db2": db2,
        "dX": dX,
        "dlogits": dlogits,
        "da": da,
        "dzbn": dzbn,
        "dxhat": dxhat,
        "dvar": dvar,
        "dmean": dmean,
    }


# %% [markdown]
#
# ## Shapes (smoke: B=4, D=3, H=5, C=3)
#
# | tensor | shape | notes |
# |---|---|---|
# | X | (B, D) | float64, requires grad in checks |
# | W1 / b1 | (H, D) / (H,) | |
# | z1 | (B, H) | `X @ W1.T + b1` |
# | mu / var | (H,) | batch stats, population variance + eps |
# | xhat / zbn / a | (B, H) | normalize → scale/shift → tanh |
# | gamma / beta | (H,) | |
# | W2 / b2 | (C, H) / (C,) | |
# | logits | (B, C) | `a @ W2.T + b2` |
# | loss | () | mean softmax cross-entropy |

# %%
def reference_backward(loss: torch.Tensor, params: dict[str, torch.Tensor],
                       intermediates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Independent autograd reference for the same gradients.

    Takes: scalar loss, params, and non-leaf intermediates to probe.
    Returns: gradient dict with exactly the Problem 3 keys, computed
        with torch.autograd.grad (non-leaf .grad is not auto-populated).
    Raises: ValueError if loss is not scalar or a key is missing.
    """
    if loss.numel() != 1:
        raise ValueError("loss must be scalar")

    param_keys = ("W1", "b1", "gamma", "beta", "W2", "b2")

    for k in param_keys:
        if k not in params:
            raise ValueError(f"params missing key: {k}")

    if not intermediates:
        raise ValueError("intermediates must not be empty")

    intermediate_keys = (
        "X", "logits", "a", "zbn", "xhat", "var", "mu"
    )

    for k in intermediate_keys:
        if k not in intermediates:
            raise ValueError(f"intermediates missing key: {k}")

    # Collect all tensors whose gradients we want.
    # NOTE: key "mean" (not "mu") so the output key is "dmean",
    # exactly matching manual_backward's Problem 3 keys.
    names = {
        "W1": params["W1"],
        "b1": params["b1"],
        "gamma": params["gamma"],
        "beta": params["beta"],
        "W2": params["W2"],
        "b2": params["b2"],
        "X": intermediates["X"],
        "logits": intermediates["logits"],
        "a": intermediates["a"],
        "zbn": intermediates["zbn"],
        "xhat": intermediates["xhat"],
        "var": intermediates["var"],
        "mean": intermediates["mu"],
    }

    keys = list(names.keys())
    tensors = [names[k] for k in keys]

    grads = torch.autograd.grad(
        loss,
        tensors,
        allow_unused=False
    )

    result = {}

    for k, grad in zip(keys, grads):
        result["d" + k] = grad

    return result


# %%
def compare_grads(manual: dict[str, torch.Tensor],
                  reference: dict[str, torch.Tensor],
                  atol: float = 1e-8,
                  rtol: float = 1e-6) -> dict[str, dict[str, float | bool]]:
    """Per-key allclose-style comparison of two gradient dicts.

    Takes: manual and reference gradient dicts, tolerances.
    Returns: per key {"max_abs_err": float, "passed": bool}.
    Raises: ValueError if key sets differ or a tolerance is negative.
    """
    if set(manual) != set(reference):
        raise ValueError("gradient key sets differ")

    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be >= 0")

    results = {}

    for key in manual:
        m = manual[key]
        r = reference[key]

        if m.shape != r.shape:
            raise ValueError(f"shape mismatch for {key}")

        max_abs_err = (m - r).abs().max().item()

        passed = torch.allclose(
            m, r,
            atol=atol,
            rtol=rtol
        )

        results[key] = {
            "max_abs_err": max_abs_err,
            "passed": passed
        }

    return results


# %%
def manual_backward_broken(X: torch.Tensor, y: torch.Tensor, params: dict[str, torch.Tensor],
                            cache: dict[str, torch.Tensor],
                            eps: float = 1e-5) -> dict[str, torch.Tensor]:

    if eps <= 0:
        raise ValueError("eps must be > 0")

    if X.dim() != 2 or X.shape[0] < 1:
        raise ValueError("X must be 2-D with batch size >= 1")

    for k in ("z1", "mu", "var", "xhat", "zbn", "a", "logits"):
        if k not in cache:
            raise ValueError(f"cache missing key: {k}")

    for k in ("W1", "b1", "gamma", "beta", "W2", "b2"):
        if k not in params:
            raise ValueError(f"params missing key: {k}")

    if cache["logits"].shape[0] != X.shape[0]:
        raise ValueError("cache batch disagrees with X")

    z1 = cache["z1"]
    mu = cache["mu"]
    var = cache["var"]
    xhat = cache["xhat"]
    a = cache["a"]
    logits = cache["logits"]

    B = X.shape[0]

    # 1. Cross entropy
    dlogits = (
        F.softmax(logits, dim=1)
        - F.one_hot(y, num_classes=logits.shape[1])
    ) / B

    # 2. Second linear layer
    W2 = params["W2"]

    dW2 = dlogits.T @ a
    db2 = dlogits.sum(dim=0)
    da = dlogits @ W2

    # 3. Tanh
    dzbn = da * (1 - a**2)

    # 4. BatchNorm affine
    gamma = params["gamma"]

    dgamma = (dzbn * xhat).sum(dim=0)
    dbeta = dzbn.sum(dim=0)

    dxhat = dzbn * gamma

    # 5. BROKEN: ignore mean and variance dependencies
    dvar = torch.zeros_like(var)
    dmean = torch.zeros_like(mu)

    dz1 = dxhat / var.sqrt()

    # 6. First linear layer
    W1 = params["W1"]

    dW1 = dz1.T @ X
    db1 = dz1.sum(dim=0)
    dX = dz1 @ W1

    return {
        "dW1": dW1,
        "db1": db1,
        "dgamma": dgamma,
        "dbeta": dbeta,
        "dW2": dW2,
        "db2": db2,
        "dX": dX,
        "dlogits": dlogits,
        "da": da,
        "dzbn": dzbn,
        "dxhat": dxhat,
        "dvar": dvar,
        "dmean": dmean,
    }



def find_first_mismatch(broken: dict[str, torch.Tensor],
                        reference: dict[str, torch.Tensor],
                        order: list[str], atol: float = 1e-8,
                        rtol: float = 1e-6) -> str | None:

    if not order:
        raise ValueError("order must not be empty")

    if atol < 0 or rtol < 0:
        raise ValueError("tolerances must be >= 0")

    for key in order:
        if key not in broken or key not in reference:
            raise ValueError(f"missing gradient key: {key}")

        if broken[key].shape != reference[key].shape:
            return key

        if not torch.allclose(
            broken[key],
            reference[key],
            atol=atol,
            rtol=rtol
        ):
            return key

    return None


# %% [markdown]
# ### Problem 7 — Verification and Conclusion
#
# The manual backward implementation was tested against an independent PyTorch autograd reference using float64 tensors and tolerances of `atol=1e-8` and `rtol=1e-6`.
#
# The full gradient comparison checks the gradients of all parameters, the input, and the intermediate tensors. The expected result is that every gradient passes the comparison.
#
# The division by batch size enters through the mean cross-entropy derivative at the logits. Bias gradients require summation because the same bias is broadcast across all examples, so their individual gradient contributions must be added together.
#
# Training-mode BatchNorm creates a dependency between examples because its mean and variance are calculated from the entire batch. Consequently, changing one input example can change the gradient of another example.
#
# Both implementations must use the same population variance definition (`unbiased=False`) and the same epsilon because changing either changes the normalization and its gradients.
#
# The injected-error experiment deliberately removes the mean and variance gradient dependencies. The first mismatch is expected to be `dvar`, showing that the comparison can identify the location of the introduced error. After repairing the backward implementation, no mismatches should remain.
#
# These experiments provide numerical evidence that the manual derivatives agree with autograd for the tested configuration. They do not prove correctness for every possible input shape or training configuration.

# %%
def run_smoke():
    """Smoke evidence for Day 5 (harness): gradient table, probes, verdict.

    Rebuilds the smoke net (seed 11), checks the full manual-vs-autograd
    table at atol=1e-8/rtol=1e-6, demonstrates the variance-convention
    probe (unbiased var differs) and the BN-coupling probe (one-row
    perturbation moves another row's dX), and asserts the injected-error
    drill (first mismatch exactly "dvar", repaired backward clean).
    Raises AssertionError on any failure.
    """
    params = init_params(D, H, C, seed=CONFIG["seed"])
    g = torch.Generator().manual_seed(CONFIG["seed"])
    X = torch.randn(B, D, dtype=torch.float64, generator=g) * 0.5
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    X.requires_grad_(True)

    loss, cache = forward_manual(X, y, params)
    assert loss.numel() == 1 and torch.isfinite(loss).item()

    manual = manual_backward(X, y, params, cache)
    ref = reference_backward(loss, params, {**cache, "X": X})
    table = compare_grads(manual, ref, atol=ATOL, rtol=RTOL)

    print("Per-key max abs err (manual vs autograd):")
    for k in BACKWARD_ORDER:
        row = table[k]
        print(f"  {k:10s} max_abs_err={row['max_abs_err']:.3e} "
              f"passed={row['passed']}")
    bad = [k for k, v in table.items() if not v["passed"]]
    assert not bad, f"mismatched grads: {bad}"

    # Probe (a): unbiased variance (divide by B-1) must differ.
    z1 = cache["z1"]
    var_unbiased = z1.var(dim=0, unbiased=True)
    assert not torch.allclose(var_unbiased, cache["var"],
                              atol=ATOL, rtol=RTOL)
    print("Convention probe: unbiased variance differs "
          f"(max abs diff "
          f"{(var_unbiased - cache['var']).abs().max().item():.3e}) "
          "-- conventions must match.")

    # Probe (b): perturbing row 1 must move row 0's dX.
    X2 = X.detach().clone()
    X2[1] = X2[1] + 0.5
    X2.requires_grad_(True)
    loss2, cache2 = forward_manual(X2, y, params)
    manual2 = manual_backward(X2, y, params, cache2)
    assert not torch.allclose(manual["dX"][0], manual2["dX"][0],
                              atol=1e-12, rtol=0.0)
    print("Coupling probe: row-1 perturbation moved row-0 dX "
          f"(max abs diff "
          f"{(manual['dX'][0] - manual2['dX'][0]).abs().max().item():.3e}).")

    # Injected-error drill.
    broken = manual_backward_broken(X, y, params, cache)
    first = find_first_mismatch(broken, ref, BACKWARD_ORDER,
                                atol=ATOL, rtol=RTOL)
    assert first == "dvar", f"expected dvar, got {first}"
    assert find_first_mismatch(manual, ref, BACKWARD_ORDER,
                               atol=ATOL, rtol=RTOL) is None
    print('Error drill: broken first mismatch "dvar", repaired clean.')

    print("run_smoke OK: full table passes, probes behave, drill located.")


if __name__ == "__main__":
    run_smoke()

# %%
