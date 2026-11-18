# Month 3 · Week 2 Day 2 — Learn: a small CNN + the tensor workshop

**Goal:** train a small vision model with a fixed routine — and get comfortable moving tensor axes around without breaking the data. Half real model, half warm-up for Month 4.

## Why two halves

**Half A** takes yesterday's sizing math and builds a working classifier: two conv blocks plus a small head, trained on the same MNIST setup as Week 1 (55k train / 5k val, same normalization). You compare it against your Day-8 MLP — but this is a description, not a contest. Nobody needs the CNN to win, and nobody needs fewer weights. What you must write down: validation accuracy, parameter count, and measured wall-clock time.

**Half B** is the one Month 4 will charge you for. Attention code lives or dies on layout ops — split, permute, matmul, softmax-axis — where the shape can look right while the data is wrong. Today's rule, learned on toy tensors: **a correct shape is necessary but not enough. Only exact value-recovery proves a rearrangement worked.**

## New words (used below, defined here)

- **pool (max-pooling):** shrink a picture by keeping only the loudest number in each small patch. 2×2 pool turns 4 numbers into 1 (their max). It makes the model care less about tiny shifts — but it does *not* make shifting invisible. Stride, borders, and the final layers still leak position. Your conclusion will name one case where pooling throws away something the classifier needed (plan Q3), not a slogan.
- **fc (fully connected):** a layer where every output looks at every input. If 64 outputs each read 784 inputs, that's 64 × 785 weights (the +1 is one bias each). You count these by hand today.
- **head:** one slice of the model's width that gets processed on its own. 4 heads = split the data into 4 thinner streams.
- **split / merge heads:** cut one wide tensor into H thin streams (split), process them, glue them back (merge). Merge must give back *exactly* the starting values — checked with `torch.equal`, not eyeballed.
- **permute:** reorder a tensor's axes (e.g. move the head axis from position 2 to position 1). `reshape` alone *cannot* do this — it only re-reads the same memory in a new shape. The notebook's trap poke shows a reshape-only "split" with the right shape and wrong values.
- **contiguous:** after a permute, the data sits scattered in memory; `contiguous()` packs it back into one dense block, because some later ops need dense layout.
- **QKᵀ (Q times K-transposed):** one attention score per (query position, key position) pair. Its shape is (B, H, T, S) — batch, heads, query spots, key spots. You predict this shape *before* running the checker.
- **softmax rows:** softmax turns one row of scores into weights that sum to 1. It must run over the *last* axis. Softmax over the wrong axis still returns numbers that sum to something — and means nothing. The notebook shows this failure live.
- **memorize vs generalize:** `tiny_memorize` fits 64 samples to prove the plumbing learns. That proves the code works — nothing about new data.
- **bounded run:** a capped, timed training run (this day: 5 epochs on MNIST), so the experiment ends on schedule instead of eating your evening.
- **comparison verdict:** your written call on CNN-vs-MLP (accuracy, params, wall time side by side), predicted *before* you run, recorded after.

## The apparatus (read it, don't rebuild it)

- `SmallCNN` contract (fixed, so every block is checkable): Conv(1→8, 3×3) → ReLU → MaxPool(2) → Conv(8→16, 3×3) → ReLU → MaxPool(2) → Flatten → fc(784→64) → ReLU → fc(64→10). Picture sizes shrink 28→28→14→14→14→7. 52,138 weights total — printed by `block_shapes`.
- MNIST follows the Day-8 contract (55k/5k split, same normalization, test still sealed), copied into this notebook so it runs standalone in Colab — no imports from other days.
- `tiny_memorize` — fits a 64-sample subset (≤300 steps), prints the curve. Proves the plumbing learns, nothing about generalization.
- `compare_row` — your CNN's val accuracy / params / wall time next to the Day-8 MLP record (0.9738, 101,770 params). Predict the direction *before* running.
- Tensor pokes: `trap_reshape` (right shape, wrong values), `check_qkt` (your predicted `(B,H,T,S)` vs the real matmul), `check_softmax_rows`, `wrong_axis_demo` (softmax on the wrong axis: sums to something, means nothing).
- `run_smoke` — everything checkable on **fake data only** (no download): forward contract, block shapes, 52,138 params, split/merge exact recovery, shape guards. MNIST runs are eyeball evidence in `_demo`, not gates.

Your mechanisms: the `SmallCNN` class (layers + forward) and the `split_heads` / `merge_heads` pair. Step pokes first, wrap cells second — same rhythm as yesterday.

## Verified video (the tensor half only)

- [Reshape, Permute, Squeeze, Unsqueeze made simple using einops — Kapil Sachdeva (The Gems)](https://www.youtube.com/watch?v=xGy75Pjsqzo)
  (~11 min — inside the 10–40 min budget).
  Verified 2026-11-17 via oEmbed (exact title + author), the watch page, a
  text scrape (chapters, description with Colab notebook), and the official
  einops.rocks docs page, which links it as their intro video — the endorsement
  that matters, not the channel size.
- **Skip** 0:00–2:22 (preamble) unless you want the flavor.
- **Watch** 2:22–9:36, especially 7:21 ("Advanced reshaping") — axis moves shown
  through einops `rearrange`, which maps directly onto the split/merge idea.
- **Skim** 9:36–end (framework compat, NN layers, conclusion).
- **Honest gap:** the video teaches rearrange *thinking* (which sidesteps stride bugs),
  not the manual reshape→permute→contiguous mechanics — the trap poke, the exact-recovery
  check, the (B,H,T,Dh) layout, the QKᵀ shape, and the wrong-axis softmax bug all
  come from the notebook, not the video. Half A (small CNN) is doc-covered; no video needed.

## Done when

A trained CNN with recorded val accuracy/params/wall time and a written comparison verdict; the split/merge round-trip recovers values exactly; the QKᵀ shape prediction and the wrong-axis explanation are written down; `run_smoke` passes on a fresh kernel. MLP numbers stay quoted from Day 8 — never re-run for a rematch.
