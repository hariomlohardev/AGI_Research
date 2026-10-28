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
# # Month 3 · Week 1 Day 2 — Module, Dataset, DataLoader and a complete training loop
#
# **Question.** Can you assemble the reusable training skeleton: fixed data, reproducible batches, a two-layer Module, separate train/evaluate paths, an honest baseline, and one bounded shuffle comparison?
#
# **Prediction (write yours before running).** Say what you expect for: the train/val split sizes; the training batch shape; whether one optimizer step must change weights; whether example-weighted epoch loss equals the full-set `MSELoss`; and whether shuffling must help. Keep it short.
#
# **Library / learner boundary.** Use `torch` tensor ops, `nn.Module`, `Dataset`/`DataLoader`, autograd, and SGD for routine work. Your work: the split semantics, loop/mode discipline, loss denominator, baseline choice, independence reasoning, and every explanation.
#
# **Shapes, dtypes, budgets.** Features `float32` with batch axis first; targets `float32`; index tensors `int64` if you use them. Smoke mode only: `n=64`, train 48 / val 16, batches of 16, at most 5 epochs. No downloads and no full-scale training.

# %%
"""Day 2 workspace: Module/Dataset/DataLoader training-loop checks (stubs for the learner)."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

CONFIG = {"device": "cpu", "dtype": "float32", "mode": "smoke", "seed": 11}


# %%
def make_regression_data(n=64, seed=11):
    """Generate the fixed small regression dataset.

    Pinned recipe: use ``torch.Generator().manual_seed(seed)``; draw
    ``X = torch.randn(n, 2, generator=g)`` in ``float32``; set
    ``y = X @ [3.0, -2.0] + 0.5 + 0.1 * torch.randn(n, generator=g)``.
    Return ``(X, y)`` with shapes ``(n, 2)`` and ``(n,)``.
    Returns: tuple of two float32 tensors.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    g = torch.Generator().manual_seed(11)
    X = torch.randn(64, 2, generator=g).to(torch.float32)
    y = X @ torch.tensor([3.0, -2.0]) + 0.5 + 0.1 * torch.randn(64, generator=g)
    y  = y.to(torch.float32)

    return (X , y)



# %%
class RegressionDataset(Dataset):
    """Tiny custom dataset for one ``(x_i, y_i)`` pair per index.

    Store the given ``X`` and ``y``; ``__len__`` returns the row count and
    ``__getitem__`` returns ``(X[idx], y[idx])``. This exists to teach
    indexing; prefer ``TensorDataset`` for ordinary tensor pairs afterwards.
    """

    def __init__(self, X, y):
        """Store paired features and targets."""
        self.X = X
        self.y = y

    def __len__(self):
        """Return the number of rows."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return one ``(x_i, y_i)`` pair."""
        return (self.X[idx] , self.y[idx])



# %%
def build_loaders(X, y, batch_size=16, seed=11):
    """Build reproducible train/val loaders for the fixed ``n=64`` dataset.

    Use rows ``0:48`` for training and ``48:64`` for validation. Wrap train in
    ``RegressionDataset`` with seeded shuffling (``torch.Generator`` seeded by
    ``seed``); wrap val in ``TensorDataset`` without shuffling. Use
    ``num_workers=0`` for Windows/notebook safety.
    Returns: ``(train_loader, val_loader)``.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    train_X = X[0:48]
    train_y = y[0:48]
    test_X = X[48:64]
    test_y = y[48:64]
    g = torch.Generator().manual_seed(seed)
    train_dataset = RegressionDataset(train_X , train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=g, num_workers=0)

    val_dataset = RegressionDataset(test_X , test_y)
    val_loader = DataLoader(val_dataset, batch_size=16, generator=g, num_workers=0)

    return (train_loader, val_loader)

    



# %%
class TwoLayerMLP(nn.Module):
    """Two-layer MLP: ``Linear(2, 16)`` → ReLU → ``Linear(16, 1)``.

    ``forward`` returns shape ``(B, 1)``. Keep this BatchNorm-free so the
    sample-independence check is meaningful.
    """

    def __init__(self, d_in=2, d_hidden=16):
        """Create the two linear maps and ReLU."""
        super().__init__()
        self.flattern = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(d_in,d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden,1)
        )

    def forward(self, x):
        """Map ``(B, d_in)`` inputs to ``(B, 1)`` outputs."""
        flat = self.flattern(x)
        return self.stack(x)



# %%
def train_epoch(model, loader, loss_fn, optimizer):
    """Run one training epoch and report example-weighted loss.

    Set train mode; for each batch, zero grads, forward, squeeze ``(B, 1)``
    logits to ``(B,)`` before the loss (never let ``(B, 1)`` vs ``(B,)``
    broadcast), backward, and step. Accumulate ``loss * batch_size`` and
    divide by ``len(loader.dataset)``.
    Returns: float epoch loss.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    size = len(loader.dataset)
    total = 0.0
    model.train()
    for batch, (xb, yb) in enumerate(loader):
        pred = model(xb)
        pred = pred.squeeze(-1)
        loss = loss_fn(pred , yb)

        #back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.shape[0]

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(xb)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return total / len(loader.dataset)


    

# %%
def evaluate(model:nn.Module, loader, loss_fn):
    """Evaluate without gradients and restore the incoming mode.

    Remember ``model.training``, set eval mode, run under ``no_grad`` with
    the same example-weighted denominator as training, then restore the
    remembered mode. Return the loss as a float.
    Returns: float validation-style loss.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    was_training = model.training
    size = len(loader.dataset)
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, (xb, yb) in enumerate(loader):
            pred = model(xb)
            pred = pred.squeeze(-1)
            loss = loss_fn(pred , yb)

            total += loss * xb.shape[0]

        model.train(was_training)

        return total / size


# %%
def constant_baseline_mse(train_loader, val_loader):
    """Constant predictor fitted on training targets, scored on validation.

    Collect targets under ``no_grad`` (loaders are only target iterators
    here), take the training-target mean, and return its MSE against all
    validation targets as a float.
    Returns: float baseline MSE.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    with torch.no_grad():
        targets = []
        for _, yb in train_loader:
            targets.append(yb)
        train_targets = torch.cat(targets)
        mean = train_targets.mean()

        test_targets = []
        for _, yb in val_loader:
            test_targets.append(yb)

        val_targets = torch.cat(test_targets)

        return ((val_targets - mean) ** 2).mean().item()
    


# %%
def sample_independence_check(model, batch):
    """Check that one sample's loss leaves other input rows untouched.

    Forward a full batch whose inputs require grad, build MSE from sample 0
    only, backward, and return per-row input-gradient norms. With no
    BatchNorm or other cross-sample op, only row 0 may be nonzero even though
    parameters are shared.
    Returns: tuple of floats, one gradient norm per batch row.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    model.eval()
    xb, yb = batch
    xb = xb.detach().clone().requires_grad_(True)
    pred = model(xb).squeeze(-1)
    loss = F.mse_loss(pred[0:1], yb[0:1])
    model.zero_grad()
    loss.backward()
    norms = xb.grad.norm(dim=1)
    return tuple(float(v) for v in norms)




# %%
def ordered_vs_shuffled_comparison(X, y, batch_size=16, epochs=5, lr=0.1, seed=11):
    """Compare target-sorted training against shuffled-batch training.

    Sort by target, train the same-seeded MLP for the same update budget once
    with ``shuffle=False`` and once with seeded shuffling, and return both
    final train losses plus your one-sentence observation. Record either
    outcome; a null result is still a result.
    Returns: ``{"ordered": float, "shuffled": float, "observation": str}``.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    order = torch.argsort(y)
    Xs, ys = X[order], y[order]
    ordered_loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=False, num_workers=0)
    shuffled_loader = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), num_workers=0)
    loss_fn = nn.MSELoss()

    torch.manual_seed(seed)
    model = TwoLayerMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch(model, ordered_loader, loss_fn, optimizer)
    loss_ord = evaluate(model, ordered_loader, loss_fn)

    torch.manual_seed(seed)
    model = TwoLayerMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_epoch(model, shuffled_loader, loss_fn, optimizer)
    loss_shuffled = evaluate(model, shuffled_loader, loss_fn)

    return  {"ordered": loss_ord, "shuffled": loss_shuffled, "observation": "the suffeled tensor gives an lover loss than the ordered one thats what i observed by running this"}



# %% [markdown]
# ## Verification and evidence
#
# Tiny checks for this day (all smoke mode, CPU, seed 11):
#
# 1. Data recipe is fixed: same seed gives the same `(X, y)`; split is 48/16.
# 2. Batches have shapes `(16, 2)` / `(16,)`; loaders use `num_workers=0`.
# 3. Forward returns `(B, 1)`; `train()`/`eval()` flip `model.training`.
# 4. One epoch moves weights; epoch loss is example-weighted, not a naive mean of unequal batch means.
# 5. `evaluate` runs under `no_grad`, leaves fresh grads `None`, and restores the incoming mode.
# 6. Held-out MSE is logged next to the training-target constant baseline.
# 7. Sample independence passes: only row 0 has nonzero input-gradient norm.
# 8. Ordered-vs-shuffled uses identical inits and the same update budget; document either outcome.
#
# Save the loss/baseline/independence/shuffle table in your conclusion. Restart the kernel and `Run All` in smoke mode before `/done`.

# %%
def run_smoke():
    """Run the Day 2 checks in smoke mode."""
    torch.manual_seed(CONFIG["seed"])
    X, y = make_regression_data()
    assert X.shape == (64, 2) and y.shape == (64,), (X.shape, y.shape)

    train_loader, val_loader = build_loaders(X, y)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 2) and yb.shape == (16,), (xb.shape, yb.shape)
    assert train_loader.num_workers == 0 and val_loader.num_workers == 0

    model = TwoLayerMLP()
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    epoch_loss = train_epoch(model, train_loader, loss_fn, opt)
    assert math.isfinite(epoch_loss), epoch_loss
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    held_out = evaluate(model, val_loader, loss_fn)
    assert math.isfinite(held_out), held_out
    baseline = constant_baseline_mse(train_loader, val_loader)
    assert math.isfinite(baseline), baseline

    torch.manual_seed(CONFIG["seed"])
    probe = TwoLayerMLP()
    norms = sample_independence_check(probe, (xb, yb))
    assert len(norms) == xb.shape[0], norms
    assert norms[0] > 0, norms
    assert all(v == 0 for v in norms[1:]), norms

    comparison = ordered_vs_shuffled_comparison(X, y)
    assert math.isfinite(comparison["ordered"]), comparison
    assert math.isfinite(comparison["shuffled"]), comparison
    assert isinstance(comparison["observation"], str) and comparison["observation"].strip(), comparison

    print("smoke OK")


if __name__ == "__main__":
    run_smoke()


# %% [markdown]
# ## Conclusion (values verified 2026-10-28)
#
# | Check | Result | Notes |
# |---|---|---|
# | split / batch shapes | 48 train / 16 val; batch `(16, 2)` / `(16,)` | asserted in `run_smoke` |
# | one-step weight change | snapshot assertion passed — at least one parameter tensor changed after one epoch | `run_smoke` compares before/after clones |
# | weighted vs naive batch-mean loss | honest 4.0 vs naive 5.0 on the 16-and-8 hand example | naive mean treats both batches equally, ignoring that the first holds twice the examples |
# | held-out MSE vs constant baseline | model 5.0789 vs training-mean constant 21.2285 | model beats the baseline; baseline is the training-target mean |
# | sample independence | row-0 norm 12.746, rows 1–15 exactly 0.0 | only sample 0's inputs carry gradient |
# | ordered vs shuffled | ordered 4.4861 vs shuffled 0.0511, same init, same 5-epoch budget | fair comparison (sorted data, fresh re-seeded model per arm); learner observation (2026-10-28): the suffeled tensor gives an lover loss than the ordered one thats what i observed by running this |
#
# ### Open question
#
# Nothing added at close — learner said "close" with no open question recorded (2026-10-28).
