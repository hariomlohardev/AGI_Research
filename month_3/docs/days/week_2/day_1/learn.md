# Month 3 · Week 2 Day 1 — Learn: convolution mechanics and numerical correctness

**Goal:** connect sliding-window arithmetic, output shapes and parameter sharing — then prove a loop-based implementation matches PyTorch numerically.

## The one convention that causes all the confusion

Mathematical convolution flips the kernel before sliding it. Every CNN library — PyTorch included — implements **cross-correlation**: slide without flipping. Two correctly written implementations disagree exactly when one flips and the other doesn't (plan Q1). Name yours: this day implements **cross-correlation**, matching `torch.nn.Conv2d`.

## Output size from the last legal window

For one spatial dimension with input `n`, kernel `k`, stride `s`, symmetric padding `p`, dilation `d=1`: window starts sit at `0, s, 2s, …`. The last legal start is `n + 2p − k` (the window's right edge must stay inside the padded input). Counting starts `0 … n + 2p − k` in steps of `s` gives `floor((n + 2p − k)/s) + 1`, i.e. the plan formula with `d = 1`. The notebook's first poke cell walks this on the three check cases before any loop runs.

## Channels and parameter sharing

One output value sums over **all input channels**: each output channel owns `Cin` kernels of size `kh × kw`, plus one bias. Parameters: `Cout × (Cin × kh × kw + 1)`. Sharing is what makes this cheap next to a fully connected map: the same `kh × kw` weights are reused at every spatial position instead of learning a separate weight per position (plan Q4). The notebook demos the channel-summation with a real multi-channel `Conv2d`; counting is a predict-then-check cell.

## The apparatus (read it, don't rebuild it)

Everything below ships complete in `conv2d_loops.ipynb`:

- Fixtures: the hand-worked 4×4 / 2×2 case, one padded case, one stride-2 case (seeded, fixed).
- `corr2d_torch` — float64 PyTorch reference, identical weights, no bias.
- `compare_case` — runs your function against the reference and prints the max-abs-diff table.
- `hand_expected` — a plain nested list **you** fill with hand-computed values; a checker compares it to the reference.
- `step_pad` / `step_window` / `step_dot` — complete poke functions: one padded array, one extracted window, one dot product, with shapes printed. Generalize them in the wrap cell.
- Edge kernel, blur kernel, a synthetic grayscale image, and a range reporter (input/output min-max plus display-scaling note).
- `count_conv_params`, a multi-channel torch demo, and a flip-vs-no-flip disagreement demo.
- `run_smoke` — the whole evidence chain on the fixtures: three parity comparisons within `TOL = 1e-9`, output-shape checks, `ValueError` guards.

Your mechanism: the three step pokes read as-is; the final `corr2d` wrap cell is yours — explicit loops, configurable stride, symmetric zero padding.

## Verified video (scoped sections)

- [How Convolutional Neural Networks work — Brandon Rohrer](https://www.youtube.com/watch?v=FmpDIaiMIeA)
  (~24 min — inside the 10–40 min budget).
  Verified 2026-11-14 via the watch page title, the oEmbed author record
  (`Brandon Rohrer`, channel `@BrandonRohrer`), the channel page header, and the
  creator's own companion page (`brohrer.github.io/how_convolutional_neural_networks_work.html`),
  which embeds this exact video with 17 timestamped sections.
- **Watch closely** from the start through ~9:20: toy X-vs-O task, sliding multiply-and-sum
  scoring at every position (cross-correlation — exactly what you loop-implement), match maps,
  then shrinking by keeping the local peak (pooling/subsampling intuition).
- **Skim** from ~11:30 on (stacking stages, voting layer, training) — good context, not needed today.
- **Honest gap:** no zero padding, no output-size formula, no multi-channel/multi-filter arithmetic,
  no kernel-flip convention, no parameter counting. All of that comes from the notebook's
  verification against `torch.nn.Conv2d`, not the video.

## Plain-words glossary (every new term on this day)

- **Kernel / weights (`w`)** — the small stamp of numbers you slide (hand case: 2×2). "Kernel" is the shape-word, "weights" the ML-word; same object.
- **Input (`x`)** — the big grid you stamp over (hand case: 4×4; later: the grayscale picture).
- **Window** — the `k × k` patch currently under the stamp. It moves; the kernel is the stamp, the window is the paper under it.
- **Dot product** — multiply stamp-numbers × paper-numbers one by one, add them up. One window → one output cell (`step_dot`: 2.0 at position (1, 0)).
- **Stride (`s`)** — how many cells the stamp jumps between positions. Stride 2 = skip one every time = smaller output.
- **Pad (`p`)** — rings of zeros glued around all four sides before sliding. `pad=1` on 4×4 → 6×6. Lets border pixels get stamped; grows the output.
- **Output shape (`out_h`, `out_w`)** — rows and columns of the answer grid, from `floor((n + 2p − k)/s) + 1` per axis. Hand case: 3 and 3.
- **`k` / `kh` / `kw`** — kernel side length. Square stamps need one number; rectangular ones (3×2 test) need height (`kh = wa.shape[0]`) and width (`kw = wa.shape[1]`) separately.
- **Cross-correlation** — slide-and-multiply *without* flipping the stamp. What you implemented; what PyTorch does.
- **Convolution (math sense)** — same, but the stamp is rotated 180° first. The only difference from cross-correlation; the source of plan Q1.
- **Channel** — one layer of an image. Grayscale = 1 layer (your loops); RGB = 3 stacked layers (summed in `channel_demo`).
- **Bias** — one extra number added after a convolution. Not used today (reference runs bias-free).
- **Fixture** — a fixed preset input (the 4×4 hand grid, the padded case, the stride case) so every check runs on the same numbers.
- **Tolerance (`TOL = 1e-9`)** — how close your numbers must be to the reference: equal up to the 9th decimal. Tiny rounding differences are fine.
- **float64** — numbers stored with ~15 decimal digits. The fixtures arrive in this form; don't round anything yourself.
- **Ranges** — just min and max of an image's numbers. "Report ranges" = print smallest and biggest (raw outputs aren't viewable until shifted/scaled).
- **Harness** — everything shipped complete (reference, checks, demos). Read-only; your job is the mechanism.
- **Mechanism** — the one thing you write: the `corr2d` loops.
- **Poke** — a small finished demo cell (`step_pad`, `step_window`, `step_dot`). Read it, learn the move, generalize it.
- **Smoke (`run_smoke`)** — the final button: runs all checks at once, prints green or red.

## Done when

Three numerical comparisons pass within the recorded tolerance, output dimensions match the formula on every case, before/after images are labeled with ranges, and two explained patterns (channels, flip convention) are written down. `run_smoke` passes on a fresh kernel. The deliberate loop exercise ends here — remaining CNN work uses `nn.Conv2d`.
