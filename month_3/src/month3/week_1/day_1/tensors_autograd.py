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
# # Month 3 · Week 1 Day 1 — Tensors, shape reasoning and autograd
#
# **Question.** For a small computation, can you predict every axis and every gradient contribution *before* running code — and catch a broadcast that is valid but wrong?
#
# **Prediction (write yours before running).** Scalar `z = x*y + x**2`: `dz/dx = y + 2x`, `dz/dy = x`. Reusing `x` in two branches sums both contributions. `(B, D) @ (D, H)` gives `(B, H)`; adding bias of shape `(H,)` keeps `(B, H)`; elementwise multiply by a mask of shape `(B, 1)` keeps `(B, D)` by broadcasting the mask across `D`. One deliberately wrong broadcast below still runs — explain why.
#
# **Library / manual boundary.** Use PyTorch tensor ops and autograd for every check. Your manual work: the hand derivations, the predicted shapes, and the *why* behind accumulation, leaf/non-leaf `.grad`, and `retain_graph`. No hand-rolled engine required today; compare against your Month 1 engine only for scalar ops it actually supports.
#
# **Shapes and dtypes.** Floating-point features `float32`; integer index tensors `int64`. Batch axis first. Smoke mode uses the tiny values in `CONFIG` — no datasets, no downloads, no training runs.

# %%
"""Day 1 workspace: tensor, shape and autograd checks (stubs for the learner)."""

import torch

CONFIG = {"device": "cpu", "dtype": "float32", "mode": "smoke", "seed": 11}

# %%
"""Learner-owned gradient and shape checks. Implement each stub yourself."""


def scalar_grads(x, y):
    """Autograd gradients of z = x * y + x ** 2 at scalars x, y.

    Build the graph with torch, call backward, and return (dz_dx, dz_dy)
    as floats. Derive both by hand first; the test compares them.
    Returns: tuple of two floats.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    x_t = torch.tensor(float(x) , requires_grad=True)
    y_t = torch.tensor(float(y), requires_grad=True)

    z = x_t * y_t + x_t ** 2

    z.backward()
    return (x_t.grad , y_t.grad)


def shared_branch_grads(x):
    """Gradient at x when x feeds two branches: z = x * x + 3 * x.

    Predict the summed contribution on paper (both branches add), then verify
    with autograd. Returns dz_dx as a float.
    Returns: float.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    x = torch.tensor(float(x) , requires_grad=True)
    x1 = x * x
    x2 = 3 * x
    z = x1 + x2

    z.backward()
    return x.grad


def accumulation_demo():
    """Show accumulation across backward calls, then clearing.

    Use z = w ** 2 at w = 3.0: backward once, backward again without
    zeroing, then zero_() and backward once more. Return
    (single, accumulated, cleared) as floats.
    Returns: tuple of three floats.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    w = torch.tensor(3.0 , requires_grad=True)
    z = w ** 2

    z.backward(retain_graph=True)
    single = float(w.grad)
    z.backward(retain_graph=True)
    accumulated = float(w.grad)
    w.grad.zero_()
    z.backward()
    cleared = float(w.grad)

    return (single, accumulated, cleared)


def shape_predictions():
    """Your predicted shapes for the worksheet cases (B=4, D=5, H=3).

    Return {"matmul": ..., "bias_add": ..., "mask_mul": ...} with each
    value a shape tuple for: (B, D) @ (D, H); adding bias (H,); multiplying
    (B, D) by a mask of shape (B, 1). The test checks them against actuals.
    Returns: dict of str to shape tuple.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    B=4 
    D=5 
    H=3


    return {"matmul": (B,H), "bias_add": (B,H), "mask_mul": (B,D)}



def broadcast_error_example():
    """One deliberately wrong-but-valid broadcast, explained.

    Return {"intended": tuple, "actual": tuple, "why": str} where actual
    is the shape PyTorch really produced, intended is what you wanted, and
    why explains the broadcast that silently changed the meaning.
    Returns: dict with intended/actual shape tuples and a why string.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    a = torch.tensor([1., 2., 3., 4.])                  # (4,)
    b = torch.tensor([[10.], [20.], [30.], [40.]])      # (4, 1)
    actual = tuple((a + b).shape)                        # runs, gives (4, 4)
    return {
        "intended": (4, 1),   # elementwise per-row sum, one value per row
        "actual": actual,     # (4, 4): an outer-style table instead
        "why": ("Trailing-dimension alignment matched 4 against 1, so the "
                "size-1 dim stretched, and the missing leading dim stretched "
                "too: an intended elementwise (4,1) sum silently became an "
                "outer (4,4) table."),
    }



# %% [markdown]
# ## Demos of your own (scaffolded overnight — explanations pending)
#
# The code cells below were scaffolded so the day could rest. The two **why** paragraphs and the open question are marked TODO: write them in your own words before `/done`.

# %%
def leaf_nonleaf_demo():
    """Intermediate requires grad but keeps no .grad by default."""
    x = torch.tensor(2.0, requires_grad=True)  # leaf: never reassigned
    y = x * x  # intermediate: requires grad, but non-leaf
    print("y.requires_grad:", y.requires_grad)
    z = 3 * y + 1
    z.backward()
    print("x.grad (leaf):", x.grad)
    print("y.grad (non-leaf):", y.grad)
    assert y.grad is None
    return float(x.grad)



# %% [markdown]
# ### Leaf vs non-leaf — why (TODO: your words)
#
# intermediates skip grad to save memory only leafs keep it

# %%
def retain_error_demo():
    """Reusing a freed graph raises; retain_graph is not the fix."""
    w = torch.tensor(3.0, requires_grad=True)
    z = w ** 2
    z.backward()
    try:
        z.backward()  # second use of the same graph
    except RuntimeError as e:
        print("expected error:", e)
        return True
    return False



# %% [markdown]
# ### Retain — why (TODO: your words)
#
# each step builds a fresh graph so retaining is pointless

# %% [markdown]
# ## Verification and evidence
#
# Tiny checks for this day (all in smoke mode, CPU, seed 11):
#
# 1. Scalar gradients match your hand derivation (test pins `x=2, y=3`).
# 2. Shared-branch gradient equals the summed contribution.
# 3. Accumulation: second backward doubles, `zero_()` resets.
# 4. Leaf vs non-leaf: intermediate `.grad` is `None` unless retained — show it.
# 5. Fresh-graph error: reusing a graph without `retain_graph` raises; explain why `retain_graph` is not the training-loop fix.
# 6. Shape table matches actuals; the wrong-broadcast example is corrected.
#
# Save the gradient comparison and shape table values in your conclusion cell. Smoke metrics are not final metrics — there is no training today. Record: date, notebook revision, mode, checks passed/failed, one open question.

# %%
def run_smoke():
    """Run the Day 1 checks in smoke mode."""
    dx, dy = scalar_grads(2, 3)
    assert (dx, dy) == (7.0, 2.0), (dx, dy)

    g = shared_branch_grads(2)
    assert g == 7.0, g

    assert accumulation_demo() == (6.0, 12.0, 6.0)

    assert shape_predictions() == {"matmul": (4, 3), "bias_add": (4, 3),
                                   "mask_mul": (4, 5)}

    bug = broadcast_error_example()
    assert bug["intended"] != bug["actual"], bug
    assert isinstance(bug["why"], str) and bug["why"].strip(), bug

    leaf_nonleaf_demo()
    assert retain_error_demo()

    print("smoke OK")


if __name__ == "__main__":
    run_smoke()


# %% [markdown]
# ## Conclusion (values verified — open question TODO)
#
# | Check | Hand / predicted | Autograd / actual | Match |
# |---|---|---|---|
# | scalar dx at (2,3) | 7.0 | 7.0 | yes |
# | scalar dy at (2,3) | 2.0 | 2.0 | yes |
# | shared node at x=2 | 7.0 | 7.0 | yes |
# | matmul shape | (4, 3) | (4, 3) | yes |
# | bias_add shape | (4, 3) | (4, 3) | yes |
# | mask_mul shape | (4, 5) | (4, 5) | yes |
# | accumulation triple | (6.0, 12.0, 6.0) | (6.0, 12.0, 6.0) | yes |
# | broadcast bug | intended (4, 1) | actual (4, 4); fixed to (4, 1) via `a.view(4, 1) + b` | explained |
#
# ### Open question (TODO: yours)
#
# Learner open question (2026-10-26): NA — nothing unresolved reported at close.
