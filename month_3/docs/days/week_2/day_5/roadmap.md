# Roadmap — Month 3 Week 2 Day 5: VGG-style blocks and the limits of architecture comparisons

## 1. What to learn today

- What a VGG-style block is: two or three small 3×3 conv layers stacked, with a ReLU after each one, instead of one big 5×5 or 7×7 layer.
- What receptive field means: how much of the photo a single unit "sees".
- Why two 3×3 layers see the same 5×5 patch as one 5×5 layer, but use fewer weights (when the widths match).
- What "controlled comparison" means: change only the model, keep everything else fixed.

New words, defined where they first appear:

- **VGG** = a family of nets from 2014 that won by stacking many small layers instead of a few big ones. The idea travels; the giant sizes don't.
- **Receptive field** = the patch of the original photo that feeds into one unit. Example: a unit in the first 3×3 layer sees a 3×3 patch. A unit in the *second* 3×3 layer sees a 5×5 patch of the photo, because each of its 3×3 inputs already saw 3×3.
- **Block** = a small group of layers that repeats. Example: conv → ReLU → conv → ReLU → pool is one block.
- **Controlled comparison** = a race where only one thing differs. Same photos, same split, same seed, same epochs, same learning rate. Only the model changes.
- **Contextual number** = an old result you quote for background, without claiming it raced under the same rules. Example: Day 11's 15-epoch runs are context next to your 10-epoch runs, not a head-to-head.

## 2. Why learn this

Big kernels are expensive. A 5×5 filter has 25 weights per channel pair; two 3×3 filters cover the same 5×5 view with 18. VGG's bet: small layers, stacked deep, with a bend (ReLU) between each pair, can learn richer shapes than one wide shallow layer for the same view. This stacking idea shows up everywhere later — ResNet blocks, modern conv nets, even parts of transformers borrow the "small repeated unit" habit.

## 3. Where it helps in the journey to AI / AGI research

Research is mostly controlled comparisons. "My idea beats the baseline" means nothing until the budgets match. Today you run the same 10-epoch budget a third time and write down exactly what is controlled and what is not. That table — controlled vs contextual — is the skill. It transfers to every paper you will ever read or write.

## 4. About the day's code

You build a second compact net for the same 32×32 photos. Same loaders, same split, same seed, same augmentation, same optimizer settings as Day 12 — the shared budget you already wrote down. Only the model body changes: stacked 3×3 conv layers with ReLU between them, in one or two pool blocks. Same head size family (small), same training loop shape, same full-order confusion matrix, same best-checkpoint discipline.

## 5. Mission = things we must GET FROM the code / result

- A VGG-style net that runs on 32×32 (no copied 224×224 geometry).
- A receptive-field count per stage, worked on paper (3×3 → 5×5 → 7×7 at stride 1), plus a counted parameter total.
- One bounded run on the shared Day-12 budget (10 epochs, seed 42, aug ON, SGD lr 0.01, momentum 0.9, wd 5e-4).
- A three-way comparison table: simple CNN vs AlexNet-inspired vs VGG-style, with train loss, check loss, check acc, params, and time — each row labeled controlled or contextual.
- An honest note: what the three rows prove and what they don't. Nobody needs VGG to win.

## 6. Understand-before-code = things to know BEFORE coding

- Read the VGG paper, sections 1–2 (about an hour): https://arxiv.org/abs/1409.1556 — focus on *why* they stack 3×3 layers (same view, fewer weights, more bends), not on their giant ImageNet sizes.
- Receptive-field math at stride 1: each new 3×3 layer adds 2 to the view. One layer sees 3. Two see 5. Three see 7. Work it on paper with a sketch before you code.
- Parameter math: two 3×3 layers at equal width C hold 2 × C × C × 9 = 18C² weights (bias aside). One 5×5 holds 25C². Example at C = 16: 18 × 256 = 4,608 vs 25 × 256 = 6,400. Small example, same lesson.
- The width warning: the 18-vs-25 claim needs equal widths in and out. Your Day-12 net widened 18 → 36. When your widths differ, calculate your own real numbers — no blanket claims.
- Equal view is not equal power. Two 3×3s with a ReLU between them can draw bendier boundaries than one flat 5×5. That extra ReLU is the function-class difference (source question 1) — say it in your own words in the results.
- Controlled vs contextual: Day 11's 15-epoch, lr-0.05 runs never matched your budget. They are context rows. Only Day 12 vs Day 13 share a budget, and only if you don't touch it.

## 7. Extras

- Optional, paper only: sketch a plain-versus-residual depth experiment on paper (source plan's optional). Execution is a second full training project — not required, don't start it.
- If one truly relevant recent breakthrough paper on small-kernel stacking exists, it may go in `further_reading.md`. Never invent a link; empty is fine.

## 8. Success matrix = what success means for that day

| Goal | Good looks like |
|---|---|
| Learn the stacking idea | Can say why 2 × 3×3 replaces 5×5, and when the param claim breaks |
| Receptive field | Paper count matches the code's shapes (5 after two, 7 after three) |
| Params | Counted total, widths-honest, no copied 224 geometry |
| Fair race | Same budget as Day 12, controlled/contextual labels on every row |
| Honest note | Small gaps called inconclusive; names what would settle it |
| Speed | Run finishes in the same time family as Day 12 (~9 min) |

## 9. Restrictions = what you CAN use and what you CANNOT use

CAN:
- Your Day-12 pipeline, loop, and eval code, unchanged (same split, seed, stats, sealed test set).
- Small 3×3 conv stacks with ReLU, max-pool 2×2, a modest head, dropout 0.5 in the head.
- The shared budget exactly as written: 10 epochs, seed 42, aug ON, SGD lr 0.01, momentum 0.9, wd 5e-4.

CANNOT:
- Change the budget to chase a winner (no extra epochs, no new LR).
- Copy VGG-16/19 widths or the 224×224 geometry.
- Reuse Day 11's unmatched numbers as a head-to-head claim (context label only).
- Use the test set for any picking decision.
- Claim an architecture wins from one short run or one seed.
- Copy a full tutorial solution — docs are open, final code is yours.
