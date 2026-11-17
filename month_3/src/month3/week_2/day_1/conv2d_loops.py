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
# # Day 9 — Convolution mechanics and numerical correctness
#
# **You implement exactly one thing:** the `corr2d` wrap cell (explicit loops, stride, symmetric zero padding).
# Every other code cell ships complete and is **read-only** — if one breaks, re-download this file.
# Read the three step pokes (`step_pad`, `step_window`, `step_dot`) first: one padded array, one window,
# one dot product. Your wrap cell generalizes them into loops.
#
# **Colab round-trip:** work in Colab for speed. When finished, download the `.ipynb` back over this repo copy
# and say `handoff done` — sync/tests run only after that. Scratch work goes in wrapped cells or a separate
# scratch notebook, never as top-level statements (they break the import-safety gate).

# %%
TOL = 1e-9
CONFIG = {"mode": "smoke", "seed": 11}

# %%
import numpy as np
import torch
import torch.nn as nn


# %% [markdown]
# ## Step 0 — shapes before loops
#
# `out_hw` applies the plan formula `floor((n + 2p − k)/s) + 1` per axis (dilation 1).
# Check it against the three fixtures by hand, then run the driver cell to see them printed.

# %%
def out_hw(n, k, stride, pad):
    """Output length along one axis for dilation-1 cross-correlation."""
    return (n + 2 * pad - k) // stride + 1


def step_shapes():
    """Poke 0 (complete): formula applied to the three check cases."""
    print("hand   4x4 k=2 s=1 p=0 ->", out_hw(4, 2, 1, 0))
    print("padded 3x3 k=2 s=1 p=1 ->", out_hw(3, 2, 1, 1))
    print("stride 5x5 k=2 s=2 p=0 ->", out_hw(5, 2, 2, 0))


# %% [markdown]
# ## Step 1 — symmetric zero padding
#
# `step_pad` pads the 4×4 hand input with a 1-pixel zero border. Note the shape change 4→6;
# the wrap cell must produce this exact padded array before looping.

# %%
def step_pad():
    """Poke 1 (complete): one padded array, shapes printed."""
    x = np.asarray(HAND_X, dtype=np.float64)
    padded = np.pad(x, 1, mode="constant")
    print("in:", x.shape, "padded:", padded.shape)
    print(padded)
    return padded


# %% [markdown]
# ## Step 2 — one window
#
# `step_window` extracts the window at output position (i=1, j=0) from the *unpadded* hand input.
# One window is a `k × k` view; the full operation repeats this at every legal position.

# %%
def step_window():
    """Poke 2 (complete): one extracted window, shape and values printed."""
    x = np.asarray(HAND_X, dtype=np.float64)
    i, j, k = 1, 0, 2
    window = x[i:i + k, j:j + k]
    print("window at (1, 0):", window.shape)
    print(window)
    return window


# %% [markdown]
# ## Step 3 — one dot product
#
# `step_dot` dots that window with the kernel. This scalar **is** `output[1, 0]` — the wrap cell
# writes one such scalar per legal position.

# %%
def step_dot():
    """Poke 3 (complete): one window-kernel dot product."""
    window = step_window()
    w = np.asarray(HAND_W, dtype=np.float64)
    value = float((window * w).sum())
    print("dot =", value, "-> output[1, 0]")
    return value


# %% [markdown]
# ## Wrap — your turn: `corr2d`
#
# Generalize the three pokes: pad once, loop over legal positions with stride `s`, dot each window.
# Inputs arrive as float64 — accumulate in float64 so the 1e-9 tolerance holds.

# %% id="d9-12"
def corr2d(x, w, stride=1, pad=0):
    """Single-channel 2D cross-correlation (no kernel flip), explicit loops.

    Output shape follows floor((n + 2p - k)/s) + 1 per axis.
    Raise ValueError for: non-2D x or w, stride < 1, pad < 0,
    or a kernel larger than the padded input."""

    xa = np.asarray(x, dtype=np.float64)
    wa = np.asarray(w, dtype=np.float64)
    if xa.ndim != 2 or wa.ndim != 2:
        raise ValueError("need 2D grids")

    if stride < 1 or pad < 0:
      raise ValueError("stride >= 1 and pad >= 0 required")

    padded = np.pad(xa, pad, mode="constant")  # xa is 4x4, pad=1 -> 6x6

    if wa.shape[0] > padded.shape[0] or wa.shape[1] > padded.shape[1]:
        raise ValueError("kernel larger than padded input")

    kh = wa.shape[0]   # 3 for the test's kernel (tall side)
    kw = wa.shape[1]   # 2 for the test's kernel (wide side)


    out_h = (padded.shape[0] - kh) // stride + 1   # rows of the answer
    out_w = (padded.shape[1] - kw) // stride + 1   # columns of the answer


    out = np.zeros((out_h , out_w) , dtype= np.float64)

    for i in range(out_h):
      for j in range(out_w):
        window = padded[i * stride :i * stride + kh , j * stride: j * stride + kw]
        out[i,j] = float((window * wa).sum())

    return out


# %% [markdown]
# ## Reference, fixtures and comparisons (all complete)
#
# `corr2d_torch` is the independent reference: float64 `Conv2d`, identical weights, no bias.
# `compare_case` prints the max-abs-diff table row. Fill `hand_expected` with your paper-computed 3×3
# (plain nested lists only — keep it a literal) and run `check_hand`.

# %%
HAND_X = [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]]
HAND_W = [[1, 2], [0, -1]]
PAD_X = [[1, 0, 2], [3, 1, 0], [0, 2, 1]]
PAD_W = [[1, -1], [0, 2]]
STRIDE_X = [[2, 0, 1, 3, 1], [1, 2, 0, 1, 0], [0, 1, 3, 2, 1], [2, 0, 1, 1, 3], [1, 3, 0, 2, 1]]
STRIDE_W = [[1, 0], [-1, 1]]
EDGE_K = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
BLUR_K = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
hand_expected = []


# %%
def as_f64(a):
    return np.asarray(a, dtype=np.float64)


def corr2d_torch(x, w, stride=1, pad=0):
    """Independent float64 Conv2d reference: identical weights, no bias."""
    xa = as_f64(x)
    wa = as_f64(w)
    conv = nn.Conv2d(1, 1, kernel_size=wa.shape, stride=stride,
                     padding=pad, bias=False, dtype=torch.float64)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(wa).unsqueeze(0).unsqueeze(0))
        out = conv(torch.from_numpy(xa).unsqueeze(0).unsqueeze(0))
    return out.squeeze(0).squeeze(0).numpy()


def compare_case(name, x, w, stride, pad):
    """Run learner corr2d vs the torch reference; print the table row."""
    got = np.asarray(corr2d(x, w, stride, pad), dtype=np.float64)
    ref = corr2d_torch(x, w, stride, pad)
    diff = float(np.abs(got - ref).max())
    print(f"{name}: got {got.shape} ref {ref.shape} max-abs-diff {diff:.3e} "
          f"tol {TOL:.0e} " + ("PASS" if diff <= TOL else "FAIL"))
    return diff


def check_hand():
    """Compare your paper-computed hand_expected against the reference."""
    if not hand_expected:
        print("hand_expected is empty — type your paper-computed 3x3 first.")
        return None
    got = as_f64(hand_expected)
    ref = corr2d_torch(HAND_X, HAND_W)
    if got.shape != ref.shape:
        print(f"shape {got.shape} != reference {ref.shape} — recheck the formula.")
        return None
    diff = float(np.abs(got - ref).max())
    print(f"hand check: max-abs-diff {diff:.3e} " + ("MATCH" if diff <= TOL else "MISMATCH"))
    return diff


# %% [markdown]
# ## Images, channels, flip convention (all complete)
#
# Edge and blur kernels run over a synthetic deterministic grayscale image (`make_gray`: gradient +
# squares + diagonal — labeled synthetic, not a photo). `report_ranges` prints input/output min-max:
# raw outputs are not viewable until shifted/scaled. `channel_demo` shows where input channels get summed;
# `flip_demo` shows the exact disagreement a flipped kernel causes.

# %%
def make_gray():
    """Deterministic synthetic grayscale image (gradient + squares + diagonal)."""
    g = np.tile(np.linspace(0, 255, 24), (24, 1))
    g[6:14, 6:14] = 255.0
    g[14:20, 2:8] = 0.0
    for i in range(24):
        g[i, i] = 200.0
    return g


def report_ranges(name, arr):
    a = as_f64(arr)
    print(f"{name}: min {a.min():.2f} max {a.max():.2f}")
    return float(a.min()), float(a.max())


def count_conv_params(cin, cout, kh, kw, bias=True):
    """Cout x (Cin x kh x kw + 1) with bias, without the +1 otherwise."""
    return int(cout * (cin * kh * kw + (1 if bias else 0)))


def channel_demo():
    """Show input-channel summation: out[0,0,0] rebuilt by hand from channels."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 3, 5, 5, dtype=torch.float64, generator=g)
    w = torch.randn(2, 3, 2, 2, dtype=torch.float64, generator=g)
    out = nn.functional.conv2d(x, w)
    manual = sum((x[0, c, 0:2, 0:2] * w[0, c]).sum() for c in range(3))
    print("in", tuple(x.shape), "-> out", tuple(out.shape))
    print(f"out[0,0,0,0]={out[0, 0, 0, 0]:.6f} sum-over-channels={manual:.6f}")
    return float(out[0, 0, 0, 0]), float(manual)


def flip_demo():
    """Disagreement caused by flipping the kernel: correct vs flipped reference."""
    ref = corr2d_torch(HAND_X, HAND_W)
    flipped = corr2d_torch(HAND_X, np.flip(HAND_W))
    diff = float(np.abs(ref - flipped).max())
    print(f"unflipped-vs-flipped max-abs-diff {diff:.3e} (nonzero <=> convention matters)")
    return diff


# %% [markdown]
# ## Conclusion (fill in — your own sentences)
#
# 1. Kernel-flip disagreement: why, in one line.
# 2. Output-formula derivation: from the last legal window position.
# 3. Padding: what boundary information changes.
# 4. Parameter sharing: which part beats a fully connected map.
# 5. The three max-abs-diff numbers + display-scaling rule + one limitation
#    (e.g. dilation unsupported, float64-only evidence).

# %%
def run_smoke():
    """Evidence chain: parity, shapes, guards, param count. Stub fails loudly."""
    assert compare_case("hand  ", HAND_X, HAND_W, 1, 0) <= TOL
    assert compare_case("padded", PAD_X, PAD_W, 1, 1) <= TOL
    assert compare_case("stride", STRIDE_X, STRIDE_W, 2, 0) <= TOL
    assert tuple(np.shape(corr2d(PAD_X, PAD_W, 1, 1))) == (4, 4)
    assert tuple(np.shape(corr2d(STRIDE_X, STRIDE_W, 2, 0))) == (2, 2)
    assert tuple(np.shape(corr2d(HAND_X, HAND_W, 2, 0))) == (2, 2)
    for bad in (lambda: corr2d([[1]], [[1, 2], [3, 4]]),
                lambda: corr2d([1, 2, 3], HAND_W),
                lambda: corr2d(HAND_X, [1, 2, 3]),
                lambda: corr2d(HAND_X, HAND_W, stride=0),
                lambda: corr2d(HAND_X, HAND_W, pad=-1)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    assert count_conv_params(3, 8, 3, 3) == 224
    print("smoke: all green")


# %%
def _demo():
    """Eyeball chain: pokes, comparisons, hand check, images, channels, flip."""
    step_shapes()
    step_pad()
    step_window()
    step_dot()
    compare_case("hand  ", HAND_X, HAND_W, 1, 0)
    compare_case("padded", PAD_X, PAD_W, 1, 1)
    compare_case("stride", STRIDE_X, STRIDE_W, 2, 0)
    check_hand()
    gray = make_gray()
    report_ranges("input ", gray)
    report_ranges("edge  ", corr2d(gray, EDGE_K))
    report_ranges("blur  ", corr2d(gray, as_f64(BLUR_K) / 9.0))
    channel_demo()
    flip_demo()
    print("params(3->8, 3x3, bias) =", count_conv_params(3, 8, 3, 3))


if __name__ == "__main__":
    _demo()
    run_smoke()

# %%
