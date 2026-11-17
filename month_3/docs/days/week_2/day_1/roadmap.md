# Month 3 · Week 2 Day 1 — Roadmap: convolution mechanics and numerical correctness

**Time:** 5–6 hours (plan budget). **Compute:** CPU only, seconds-scale runs — no GPU, no long training. Rest stays protected; this day needs none of it.

## Order of work

1. **Recall first (paper, 15 min):** explain training target vs validation decision vs final test result in one line each. Predict the shape of a batch of 32 RGB 32×32 images as `(N, C, H, W)` with numbers.
2. **Read:** this folder's `learn.md`, then the [PyTorch Conv2d definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) for the exact op and output-size convention. Watch the verified video's scoped sections.
3. **Derive (paper):** the 1-D output formula from the last legal window position; hand-compute the 4×4 / 2×2 case and type it into `hand_expected`.
4. **Build:** read the three step pokes, fill the `corr2d` wrap cell, run the three comparisons, run edge/blur, predict-then-check the parameter count.
5. **Conclude:** fill the conclusion cell (four plan questions in your own words + numbers + limitation), run the full notebook top-to-bottom, download the `.ipynb` back over the repo copy, say "handoff done".

## Blocks (with breaks — full-day schedule, not a sprint)

- Block A (~90 min): recall + reading + video + 1-D derivation.
- Break. Block B (~2 h): hand-worked example + step pokes + `corr2d` wrap + three comparisons green.
- Break. Block C (~90 min): edge/blur inspection + channels/params + conclusion + clean top-to-bottom run.
- Buffer (~60 min): stuck time, write-up polish, handoff.

## Done when (plan evidence)

Three numerical comparisons pass within `TOL = 1e-9`, output dimensions match the formula on every case, before/after images are labeled with ranges, and the notebook holds the hand-worked check plus two explained patterns. `run_smoke` passes on a fresh kernel. Dilation support and the diagonal kernel are explicitly optional.

## Plan practice questions (adapted for `/done`)

1. Why can two correctly written implementations disagree if only one flips the kernel?
2. How do you derive the output formula from the last legal window position?
3. What boundary information changes when zero padding is introduced?
4. Which part of parameter sharing makes convolution cheaper than a fully connected map on an image?

## Smoke evidence contract

`run_smoke` (fresh kernel, `CONFIG["mode"] == "smoke"`): `corr2d` vs float64 torch reference on all three fixtures (max abs diff ≤ 1e-9), output-shape formula on three configs, `ValueError` on all five bad-input cases, `count_conv_params(3, 8, 3, 3) == 224`. No downloads, no training, no GPU. Save the smoke log locally before clearing outputs.
