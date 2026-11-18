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
# # Day 10 — A CNN baseline and the tensor workshop
#
# **You implement exactly two things:** the `SmallCNN` wrap cell (fixed stack, layers + forward)
# and the `split_heads` / `merge_heads` wrap cell (reshape → permute → contiguous).
# Every other code cell ships complete and is **read-only**.
# MNIST comes from your Day-8 module (same 55k/5k contract, test still sealed);
# the training loops below do **not** flatten — a CNN needs its images.
#
# **Colab round-trip:** work in Colab, download the finished `.ipynb` back over this repo copy,
# say `handoff done`. Predict the CNN-vs-MLP direction *before* running (conclusion cell).

# %%
TOL = 1e-9
CONFIG = {"mode": "smoke", "seed": 11}
EPOCHS = 5
N_HEADS = 4
D_MODEL = 32
MEMO_N = 64
MEMO_STEPS = 300
SMOKE_MEMO_STEPS = 200
MLP_ACC = 0.9738
MLP_PARAMS = 101770

# %%
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# %% [markdown]
# ## Step 0 — the two conv blocks on paper (Day-9 formula, both axes)
#
# `step_conv_sizes` applies `floor((n + 2p − k)/s) + 1` to each block: conv keeps 28,
# pool halves to 14; conv keeps 14, pool halves to 7. `step_flatten` shows the head input.

# %%
def conv_out(n, k, stride, pad):
    return (n + 2 * pad - k) // stride + 1


def step_conv_sizes():
    """Poke 0 (complete): spatial chain 28 -> 28 -> 14 -> 14 -> 14 -> 7."""
    a = conv_out(28, 3, 1, 1)
    b = (a - 2) // 2 + 1
    c = conv_out(b, 3, 1, 1)
    d = (c - 2) // 2 + 1
    print(f"conv1: 28 -> {a}; pool: {a} -> {b}; conv2: {b} -> {c}; pool: {c} -> {d}")
    return (a, b, c, d)


def step_flatten():
    """Poke 1 (complete): head input is 16 channels of 7x7."""
    n = 16 * 7 * 7
    print(f"flatten: (N,16,7,7) -> (N,{n})")
    return n


# %% [markdown]
# ## Wrap 1 — your turn: `SmallCNN`
#
# Fixed stack — Conv(1→8, 3×3, s1, p1) → ReLU → MaxPool(2) → Conv(8→16, 3×3, s1, p1)
# → ReLU → MaxPool(2) → Flatten → Linear(784→64) → ReLU → Linear(64→10).
# `forward` takes `(N,1,28,28)` → `(N,10)`; raise `ValueError` on non-4D input.

# %%
class SmallCNN(nn.Module):
    """Small conv stack + head (your code). Fixed contract, see learn.md."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1 , 8 , 3 , 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8,16,3,1,1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784 , 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64 , 10)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("need (N,1,28,28)")
        z = self.pool1(self.relu1(self.conv1(x)))
        z = self.pool2(self.relu2(self.conv2(z)))
        z = self.flat(z)
        z = self.relu3(self.fc1(z))
        return self.fc2(z)



# %%

# %% [markdown]
# ## Harness: data contract, image loops, block printer, comparison (all complete)
#
# MNIST loading below is a vendored copy of your Day-8 module's contract
# (ToTensor + Normalize(0.1307, 0.3081), seeded 55k/5k split, test sealed) —
# vendored, not imported, so this notebook runs standalone in Colab.
# `run_epoch_img` / `eval_img` are Day-8-style loops **without** flattening.

# %%
def get_mnist(root="./data"):
    """MNIST, Day-8 contract vendored (ToTensor + Normalize, test sealed)."""
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return train_full, test_set


def make_mnist_splits(root="./data", val_n=5000, batch=64, seed=11):
    """Seeded 55k/5k split, test sealed (Day-8 contract, vendored)."""
    train_full, test_set = get_mnist(root)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g).tolist()
    train_set = torch.utils.data.Subset(train_full, perm[val_n:])
    val_set = torch.utils.data.Subset(train_full, perm[:val_n])
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch, shuffle=sh,
                                   generator=g if sh else None, num_workers=0)
    info = {"train_n": len(train_set), "val_n": len(val_set),
            "test_n": len(test_set), "seed": seed}
    return mk(train_set, True), mk(val_set, False), mk(test_set, False), info


def run_epoch_img(model, loader, opt, loss_fn):
    """One training epoch on image batches (harness, complete)."""
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
        n += len(xb)
    return tot / n


def eval_img(model, loader, loss_fn):
    """Eval pass: mean loss + accuracy on image batches (harness)."""
    model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            tot += loss_fn(logits, yb).item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)
    return tot / n, correct / n


def block_shapes(model, x):
    """Print per-block shapes in children() order; return the list."""
    shapes, cur = [], x
    with torch.no_grad():
        for i, blk in enumerate(model.children()):
            cur = blk(cur)
            shapes.append(tuple(cur.shape))
            print(f"block {i:2d} {type(blk).__name__:12s} -> {tuple(cur.shape)}")
    return shapes


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def tiny_memorize(model, xb, yb, steps, lr=0.1):
    """Fit one fixed tiny batch; return (steps_used, final_acc)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for s in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (model(xb).argmax(1) == yb).float().mean().item()
        if acc >= 1.0:
            return s, acc
    print(f"memorize: reached {acc:.3f} after {steps} steps")
    return steps, acc


def compare_row(cnn_acc, cnn_params, cnn_secs):
    """Your CNN vs the Day-8 MLP record (quoted, never re-run)."""
    print(f"MLP : acc {MLP_ACC:.4f} params {MLP_PARAMS} time n/a (Day 8 record)")
    print(f"CNN : acc {cnn_acc:.4f} params {cnn_params} time {cnn_secs:.1f}s")
    print(f"delta acc {cnn_acc - MLP_ACC:+.4f}")


# %% [markdown]
# ## MNIST chain (eyeball): splits, memorization, bounded run
#
# `_mnist_chain` loads the Day-8 contract, memorizes 64 samples (plumbing check,
# not a generalization claim), trains `EPOCHS` bounded epochs on SGD, and prints
# the comparison row. Set your wall-time cap before running; stop at the cap
# and compare a common completed-epoch budget if interrupted.

# %%
def _mnist_chain():
    """Eyeball chain (complete): data, memorize, bounded train, compare."""
    tr, va, te, info = make_mnist_splits(root="./data")
    print(f"train {info["train_n"]} val {info["val_n"]} test {info["test_n"]} (sealed)")
    model = SmallCNN()
    xb0, yb0 = next(iter(tr))
    xb0, yb0 = xb0[:MEMO_N], yb0[:MEMO_N]
    s, a = tiny_memorize(model, xb0, yb0, MEMO_STEPS)
    print(f"memorize-64: {a:.3f} in {s} steps")
    model = SmallCNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for e in range(EPOCHS):
        tl = run_epoch_img(model, tr, opt, loss_fn)
        vl, va_acc = eval_img(model, va, loss_fn)
        print(f"epoch {e}: train {tl:.4f} val {vl:.4f} acc {va_acc:.4f}")
    secs = time.time() - t0
    compare_row(va_acc, count_params(model), secs)
    return va_acc


# %% [markdown]
# ## Conclusion (fill in — your own sentences, terms defined in one line each)
#
# 1. Prediction (written BEFORE the run): which wins, CNN or MLP, and why.
# ans... the CNN should win because od its structure and things we used in this one and how its processing images and all of those things i think it will work better than simple MLP (just comonsense thing that if this doesn;t work better why we learn it )
# 2. Verdict: val acc / params / wall time + what the gap does NOT prove (plan Q4).
# ans... MLP : acc 0.9738 params 101770 time n/a (Day 8 record)CNN : acc 0.9820 params 52138 time 148.9s delta acc +0.0082 so this is what this that came and the gap is small in bettwen botch that might not prove but has potential
#
# 3. Pooling: one case where it discards information the classifier needed (plan Q3).
#
# Pooling discards info when two different patches share the same max. E.g. a thin stroke (3 next to zeros) and a thick blob (all 3s) both pool to 3 — the thickness clue is gone. If the digit pair (like 1 vs 7) differs exactly there, the classifier can't recover it no matter how good the fc layers are, because the data was deleted upstream.
#
#
# 4. Wrong-axis bug: why the dim-1 softmax shape looks fine but means nothing.
# shape fine, columns sum to 1, rows meaningless — and rows are all the model ever reads. That's why it "looks fine but means nothing."
#
# 5. QKᵀ: which axis holds candidate keys (plan Q2) + your verified shape.
# The last axis (S) holds the candidate keys.
# 6. Shape vs layout — the trap poke built a reshape-only "split" with the perfect shape (B,H,T,Dh) and completely wrong values inside. Shape checks the box sizes; it says nothing about which number landed in which box. Only torch.equal(merge_heads(split_heads(x)), x) — every value back in its exact place — proves the move was honest.
#
# 7. Limitation — one honest paragraph covering three things: the run was capped at 5 epochs (a bigger budget could change the ranking); CPU wall times are noisy (background apps shift them, so small gaps mean nothing); and the MLP row is quoted from Day 8, not re-run — same code, same split, but not a side-by-side race on the same machine state.

# %% [markdown]
# ## Tensor half — the trap (complete)
#
# `trap_reshape` builds `(B,T,D) = (2,5,32)` and splits it two ways: reshape-only
# (right shape, wrong values) vs reshape → permute → contiguous (right both).
# Run it before touching the wrap cell.

# %%
def trap_reshape():
    """Poke 2 (complete): reshape-only split has the shape but not the values."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(2, 5, 32, generator=g)
    bad = x.reshape(2, 5, 4, 8)
    good = x.reshape(2, 5, 4, 8).permute(0, 2, 1, 3).contiguous()
    print("reshape-only:", tuple(bad.shape), "permute-route:", tuple(good.shape))
    back_bad = bad.permute(0, 2, 1, 3).reshape(2, 5, 32)
    back_good = good.permute(0, 2, 1, 3).reshape(2, 5, 32)
    print("reshape-only round-trips:", bool(torch.equal(back_bad, x)))
    print("permute-route round-trips:", bool(torch.equal(back_good, x)))
    idx = tuple(torch.nonzero(back_bad != x)[0].tolist())
    print(f"first mismatch at {idx}: bad {back_bad[idx]:.4f} vs true {x[idx]:.4f}")


# %% [markdown]
# ## Wrap 2 — your turn: `split_heads` / `merge_heads`
#
# `(B,T,D)` → `(B,H,T,Dh)` with `Dh = D // H`; `merge_heads` inverts **exactly**
# (`torch.equal`). Raise `ValueError` when `D % n_heads != 0`. Follow the poke's route.

# %%
from einops import rearrange

def split_heads(x, n_heads):
    """(B,T,D) -> (B,H,T,Dh). Raise ValueError if D % n_heads != 0."""
    _, _, D = x.shape
    if D % n_heads != 0:
      raise ValueError()
    return rearrange(x, "b t (h dh) -> b h t dh", h=n_heads)



def merge_heads(x):
    """(B,H,T,Dh) -> (B,T,D). Exact inverse of split_heads."""
    return rearrange(x , "B H T Dh  -> B T (H  Dh) ")


# %% [markdown]
# ## QKᵀ, softmax rows, wrong axis (predict then verify)
#
# Fill `qkt_pred` (plain tuple literal) with your predicted shape of
# `Q @ K.transpose(-2,-1)` for `Q: (B,H,T,Dh)`, `K: (B,H,S,Dh)` — then `check_qkt`.
# `wrong_axis_demo` soft-maxes dim 1: shape looks fine, values mean nothing.

# %%
qkt_pred = (2, 4, 5, 6)


def check_qkt():
    """Compare your predicted QK^T shape against the real matmul."""
    B, H, T, S, Dh = 2, 4, 5, 6, 8
    g = torch.Generator().manual_seed(11)
    Q = torch.randn(B, H, T, Dh, generator=g)
    K = torch.randn(B, H, S, Dh, generator=g)
    real = tuple((Q @ K.transpose(-2, -1)).shape)
    if not qkt_pred:
        print("qkt_pred is empty — predict the 4-tuple first.")
        return None
    print(f"predicted {tuple(qkt_pred)} vs real {real} "
          + ("MATCH" if tuple(qkt_pred) == real else "MISMATCH"))
    return real


def check_softmax_rows():
    """Each row over the last axis sums to one (harness)."""
    g = torch.Generator().manual_seed(11)
    a = torch.softmax(torch.randn(2, 4, 5, 8, generator=g), dim=-1)
    dev = float((a.sum(-1) - 1.0).abs().max())
    print(f"worst row-sum deviation: {dev:.3e}")
    return dev


def wrong_axis_demo():
    """Softmax over dim 1: plausible shape, meaningless values."""
    g = torch.Generator().manual_seed(11)
    scores = torch.randn(2, 4, 5, 8, generator=g)
    w = torch.softmax(scores, dim=1)
    row = w.sum(-1)
    print(f"output {tuple(w.shape)}; sums over last axis: min {row.min():.3f} max {row.max():.3f} (not 1)")
    print("axis 1 mixes head slots; the distribution is over nothing meaningful.")


# %%
def run_smoke():
    """Evidence chain, synthetic only (no download). Stubs fail loudly."""
    g = torch.Generator().manual_seed(11)
    m = SmallCNN()
    x = torch.randn(2, 1, 28, 28, generator=g)
    out = m(x)
    assert tuple(out.shape) == (2, 10) and torch.isfinite(out).all()
    m.zero_grad()
    nn.CrossEntropyLoss()(out, torch.tensor([3, 7])).backward()
    assert all(p.grad is not None for p in m.parameters())
    shapes = block_shapes(m, x)
    for want in ((2, 8, 28, 28), (2, 8, 14, 14), (2, 16, 14, 14), (2, 16, 7, 7)):
        assert want in shapes, f"missing block shape {want}"
    assert shapes[-1] == (2, 10)
    assert count_params(m) == 52138
    try:
        m(torch.randn(2, 28, 28))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-4D forward")
    t = torch.randn(2, 5, 32, generator=g)
    s = split_heads(t, 4)
    assert tuple(s.shape) == (2, 4, 5, 8)
    assert torch.equal(merge_heads(s), t)
    assert torch.equal(merge_heads(split_heads(t, 2)), t)
    for bad in (lambda: split_heads(torch.randn(2, 5, 30), 4),
                lambda: split_heads(torch.randn(2, 5), 4)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    xs = torch.randn(16, 1, 28, 28, generator=g)
    ys = torch.randint(0, 10, (16,), generator=g)
    used, acc = tiny_memorize(SmallCNN(), xs, ys, SMOKE_MEMO_STEPS)
    assert acc >= 1.0, f"smoke memorize failed: {acc}"
    print(f"smoke memorize-16: 1.000 in {used} steps")
    print("smoke: all green")


# %%
def _demo():
    """Eyeball chain: pokes, shapes, MNIST, trap, QKT, softmax, wrong axis."""
    step_conv_sizes()
    step_flatten()
    block_shapes(SmallCNN(), torch.randn(2, 1, 28, 28))
    _mnist_chain()
    trap_reshape()
    check_qkt()
    check_softmax_rows()
    wrong_axis_demo()


if __name__ == "__main__":
    _demo()
    run_smoke()

# %%
