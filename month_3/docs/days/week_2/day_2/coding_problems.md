# Month 3 · Week 2 Day 2 — Coding problems: small CNN + head-splitting

**Harness ships complete; you implement only the two mechanisms.** Harness cells are read-only. Colab round-trip as usual: download back, say "handoff done". Scratch stays wrapped.

Harness catalog (complete): `CONFIG` (`EPOCHS=5`, `N_HEADS=4`, `D_MODEL=32`, memorize caps), MNIST via your Day-8 module (`get_mnist`, `make_mnist_splits` — same contract, test sealed), `run_epoch_img`, `eval_img`, `tiny_memorize`, `compare_row`, `block_shapes`, `trap_reshape`, `check_qkt`, `check_softmax_rows`, `wrong_axis_demo`, `run_smoke`. MLP record row: val acc **0.9738**, **101,770** params (quoted from Day 8, never re-run).

## Problem 1 — `SmallCNN` (mechanism 1; pokes then wrap)

Read `step_conv_sizes` (Day-9 formula on both conv blocks: 28→28, 14→14) and `step_flatten` (16×7×7 = 784) first. Then fill the wrap cell: the fixed stack Conv(1→8,3,s1,p1) → ReLU → MaxPool(2) → Conv(8→16,3,s1,p1) → ReLU → MaxPool(2) → Flatten → Linear(784→64) → ReLU → Linear(64→10).

Contract: `forward(x)` takes `(N,1,28,28)` → returns `(N,10)`; raise `ValueError` on non-4D input. Use `nn.Module` normally (this is assembly — autograd and optimizers do their jobs). `block_shapes` runs your model through `children()` order and prints every block's shape; the smoke test asserts the spatial chain plus 52,138 params.

## Problem 2 — bounded run + honest comparison (eyeball)

Run `tiny_memorize` on a 64-sample subset (proves the plumbing learns; says nothing about generalizing), then the bounded `EPOCHS=5` run with SGD, recording val accuracy and wall time. `compare_row` prints your CNN next to the MLP record. Write the verdict in the conclusion: which won, by how much, under what budget — and what that does *not* prove (plan Q4). No winner is required.

## Problem 3 — `split_heads` / `merge_heads` (mechanism 2; trap then wrap)

Run `trap_reshape` first: a reshape-only "split" with the right shape and wrong values. Then fill the wrap cell:

```python
def split_heads(x, n_heads): ...
def merge_heads(x): ...
```

Contract: `(B,T,D)` → `(B,H,T,Dh)` with `Dh = D // H`; `merge_heads` inverts exactly (`torch.equal`, not allclose). Raise `ValueError` when `D % n_heads != 0`. The correct route is reshape → permute → contiguous (the poke shows why each step is there); the tests gate exact recovery, output shapes, and the guard.

## Problem 4 — QKᵀ, softmax rows, wrong axis (predict then verify)

Fill `qkt_pred` (plain tuple literal) with your predicted shape of `Q @ K.transpose(-2,-1)` for `Q: (B,H,T,Dh)`, `K: (B,H,S,Dh)`; run `check_qkt`. Run `check_softmax_rows` (each row sums to 1 over the last axis) and `wrong_axis_demo` (softmax over dim 1: the shape still "looks fine" — explain in the conclusion why the values are meaningless). These are notebook evidence, not pytest gates.

## Conclusion (fill in, your own sentences + glossary-worthy terms defined)

Answer the four plan questions, quote CNN val acc / params / wall time and the comparison verdict, name one pooling-information-loss case, explain the wrong-axis bug, and record one limitation (e.g. 5-epoch budget, CPU timing noise). New terms you use must be defined in one line each — yesterday's glossary rule now applies to your writing too.
