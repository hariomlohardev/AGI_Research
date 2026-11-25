# Roadmap — Month 3 Week 2 Day 6: ResNet-20, a documented recipe and an honest checkpoint

## 1. What to learn today

- What a residual block is: the output is the input plus a small correction, y = x + F(x).
- What a skip (shortcut) path is: a wire that carries x past the correction layers and adds it back.
- How to build the CIFAR ResNet-20: one 3×3 conv first, then three stages (16, 32, 64 channels) with three two-conv blocks each, then global average pooling, then a 10-class head.
- How to downsample between stages, and the paper's option A: shrink the photo with stride, pad the extra channels with zeros — not a learned layer.
- Why the paper's numbers (about 0.27M params, 91.25% accuracy) are a reference, not your pass mark.

New words, defined where they first appear:

- **Residual block** = a block that learns a correction F(x) and adds the input back: y = x + F(x). Example: if the block's two convs output zeros, the block passes x through unchanged.
- **Shortcut (skip)** = the wire carrying x around the correction layers to the addition point. Example: in code, `out = x + self.correction(x)`.
- **Option A** = the paper's CIFAR shortcut for stage changes: subsample the space with stride 2 and fill new channels with zeros. No learned weights in the shortcut.
- **Downsampling** = shrinking the photo inside the net (here stride-2 conv plus stride-2 shortcut), while doubling channels 16 → 32 → 64.
- **Global average pooling** = squeezing each final 8×8 channel map down to one number by averaging. Example: 64 maps become 64 numbers for the head.
- **Budgeted adaptation** = a shorter training schedule you declare honestly as yours. It is not the same experiment as the paper's 64k-step protocol.
- **Chance baseline** = the score of guessing: 10% accuracy, loss near ln(10) ≈ 2.30. Your run must beat this clearly or be diagnosed, not celebrated.

## 2. Why learn this

Plain deep nets stalled: adding layers stopped helping and sometimes hurt. The skip connection was the fix that unlocked depth — the input gets a direct ride forward, and the gradient gets a direct ride back. Every modern vision net, and much of the transformer world, inherits this trick. Today you build the original version with your own hands.

## 3. Where it helps in the journey to AI / AGI research

ResNet is the first architecture where the *wiring* is the idea, not the layer sizes. Reading it teaches you to ask "what path does the signal take?" instead of "how many params?" — the question behind residuals, attention shortcuts, and every deep model you will train. Plus: this checkpoint may become your Month 7 distillation teacher, so its recipe card must be honest enough for future-you to reuse.

## 4. About the day's code

You write a BasicBlock (two 3×3 convs + shortcut) and stack it into the 6n+2 recipe with n = 3: 6×3+2 = 20 layers. Same CIFAR loaders and sealed test set as all week. Same training-loop shape you already own. New discipline: a recipe card (every setting written down), a timing pilot before the real run, a hard 2 GPU-hour cap, and a checkpoint that reloads exactly on a fixed validation batch.

## 5. Mission = things we must GET FROM the code / result

- A BasicBlock where both branches have identical shapes at the addition — checked, not assumed.
- A full CIFAR ResNet-20 (stages 16/32/64, three blocks each, option A or a labeled projection).
- A shape table plus a counted parameter total near 0.27M.
- Tiny-batch memorization BEFORE the real run. If it cannot memorize, it is broken — repair first.
- A timing pilot plus a wall-time estimate and a max duration decided before starting.
- One bounded run (≤ 2 GPU-hours, one corrected restart only for a diagnosed issue).
- A reload check: saved checkpoint reproduces predictions on a fixed validation batch.
- Curves against the chance baseline, validation metrics, and a status label: "validated development checkpoint" or "candidate teacher; needs additional training/assessment".

## 6. Understand-before-code = things to know BEFORE coding

- Read the ResNet paper, Section 4.2 (about 30 min): https://arxiv.org/pdf/1512.03385 — the CIFAR architecture and the reference protocol only. Skip the ImageNet sections.
- The Jacobian line: for y = x + F(x), J = I + J_F. The identity I is a direct path — it helps, but cancellation is possible, so it does not promise nonzero gradients.
- The ReLU warning: the original block puts a ReLU after the addition, so the full block derivative includes that activation mask. Never write "skips guarantee gradients never vanish."
- The counting line: 6n+2 with n = 3 gives 1 first conv + 3 stages × 3 blocks × 2 convs = 19 conv layers + head = 20.
- The recipe line (paper reference): SGD, momentum 0.9, weight decay 1e-4, batch 128, LR 0.1 with drops at 32k and 48k steps, 64k-step endpoint, no dropout. Any shorter schedule is your adaptation — record every deviation.
- The still-learning vs broken line: still-learning = loss falling, above chance, curves moving the right way, just slow. Probably-broken = stuck at chance, NaN, shapes wrong, or the tiny batch won't memorize.
- Recall first (write it down): the Jacobian of a linear map, plus why a depth comparison must control more than the dataset name.

## 7. Extras

- Optional, paper only: sketch a plain-vs-residual depth study on paper (two depths, matched widths, equal budgets, at most four short runs). Execution is optional — do not start a second training project.
- If one truly relevant recent breakthrough paper on residual learning exists, it may go in `further_reading.md`. Never invent a link; empty is fine.

## 8. Success matrix = what success means for that day

| Goal | Good looks like |
|---|---|
| Residual idea | Can say what the identity path gives and why "guaranteed gradients" is too strong |
| Correct wiring | Branch shapes match at every addition; option A or a labeled projection at stage changes |
| Params | Counted total near 0.27M, no ImageNet ResNet-18 swap |
| Tiny fit | Memorizes a tiny batch before the real run — non-negotiable |
| Budget discipline | Timing pilot, wall estimate, max duration set first; ≤ 2 GPU-hours total |
| Honest checkpoint | Reloads exactly on a fixed val batch; recipe card complete; status label attached |
| Reading the run | Beats chance clearly or carries an explicit diagnosis, not a shrug |
| Speed | Pilot + run fit the cap; unfinished schedule saved as resumable, never silently compressed |

## 9. Restrictions = what you CAN use and what you CANNOT use

CAN:
- Your week's pipeline, loop, and eval code (same split, seed discipline, sealed test set).
- Standard torch layers, SGD with momentum, BatchNorm as the paper uses it.
- A shorter schedule than the paper's — declared on the recipe card as your adaptation.
- One corrected restart inside the cap, only for a diagnosed issue.

CANNOT:
- Swap in a stock ImageNet ResNet-18 — different architecture, doesn't count.
- Use dropout silently — the paper recipe has none; any addition is a labeled deviation.
- Call a compressed schedule a replication of the paper protocol.
- Pick anything on the test set; select by validation metrics only.
- Claim "research-grade" from architecture or accuracy alone.
- Train past the 2 GPU-hour cap or start undeclared extra runs.
- Copy a full tutorial solution — docs are open, final code is yours.
