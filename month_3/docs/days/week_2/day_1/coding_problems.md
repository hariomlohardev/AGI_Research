# Month 3 · Week 2 Day 1 — Coding problems: loop-based cross-correlation

**Harness ships complete; you implement only the mechanism.** Harness cells are read-only — if one breaks, re-download the day file rather than repairing the crime scene. Scratch work goes in wrapped cells or a separate scratch notebook, never as top-level statements (they break the import-safety gate).

Harness catalog (complete, do not modify): `CONFIG`, fixtures (`HAND_X`, `HAND_W`, `PAD_X`, `PAD_W`, `STRIDE_X`, `STRIDE_W`, `EDGE_K`, `BLUR_K`, `GRAY_IMG`), `corr2d_torch`, `compare_case`, `check_hand`, `step_pad`, `step_window`, `step_dot`, `report_ranges`, `count_conv_params`, `channel_demo`, `flip_demo`, `run_smoke`. Tolerance `TOL = 1e-9` (float64 both sides).

## Problem 1 — `corr2d` (the only mechanism; step cells then wrap)

Read `step_pad`, `step_window`, `step_dot` first — one padded array, one window, one dot product, shapes printed. Then fill the wrap cell:

```python
def corr2d(x, w, stride=1, pad=0):
    """Single-channel 2D cross-correlation (no kernel flip), explicit loops."""
    raise NotImplementedError("Combine the three steps here")
```

Contract: `x` and `w` 2D arrays; output shape follows `floor((n + 2p − k)/s) + 1` per axis. Raise `ValueError` for: non-2D `x` or `w`, `stride < 1`, `pad < 0`, or a kernel larger than the padded input. No dilation support required.

## Problem 2 — three numerical comparisons (eyeball + gates)

Run `compare_case` on (a) the hand-worked 4×4 / 2×2, (b) the padded case, (c) the stride-2 case. Record the max-abs-diff table in the conclusion cell. Separately, fill `hand_expected` (plain nested lists — keep it a literal) with your paper-computed 3×3 output and run `check_hand`.pytest gates the `corr2d`-vs-torch parity and the shape formula; your hand arithmetic is notebook evidence, not a gate.

## Problem 3 — edge and blur on a grayscale image (eyeball)

Run the edge kernel and the blur kernel over `GRAY_IMG` (synthetic, deterministic — labeled as such). Record input/output ranges via `report_ranges` and note the display-scaling rule: raw outputs are **not** directly viewable as images (negatives, out-of-range) until shifted/scaled. Label the before/after trio in the conclusion. No visual winner is required — ranges plus scaling reasoning are the evidence.

## Problem 4 — channels and parameters (write-up)

Predict the parameter count for a `Cin=3, Cout=8, 3×3, bias` layer by hand, then check with `count_conv_params`. Run `channel_demo` and explain in one sentence where the input-channel summation happens. This explanation lives in the conclusion cell.

## Problem 5 (optional) — diagonal-edge kernel

Design a kernel that fires on diagonal edges, test it on a synthetic diagonal pattern and a non-diagonal pattern. A failed design with a diagnosis counts; there is no required visual winner. Do not expand this into a second project.

## Conclusion (fill in)

Answer the four plan questions in your own words, quote the three max-abs-diff numbers, state the display-scaling rule, and record one limitation (e.g. dilation unsupported, float64-only evidence). Conclusions stay in your own sentences — pasted skeletons don't count as synthesis.
