# Month 3 · Week 2 Day 2 — Roadmap: CNN baseline + tensor workshop

**Time:** 5–6 hours (plan budget). **Compute:** CPU only. CNN bounded at `EPOCHS=5` on 55k MNIST plus a 64-sample memorization check; set a wall-time cap before the bounded run and stop there — an interrupted run compares a common completed epoch budget, same discipline as later CIFAR days.

## Order of work

1. **Recall first (paper, 15 min):** derive one conv output shape (28→28→14 chain) and count the 52,138 weights without opening code. State the augmentation/evaluation split rule in one line (it returns on Day 3).
2. **Read:** this folder's `learn.md`, then the scoped video sections. Predict the comparison direction (CNN vs MLP 0.9738) *before* running — write it in the conclusion cell first.
3. **Build A:** step pokes → `SmallCNN` wrap → `block_shapes` green → memorization check → bounded run → `compare_row`.
4. **Build B:** `trap_reshape` → split/merge wrap → exact-recovery green → QKᵀ prediction → softmax checks → wrong-axis write-up.
5. **Conclude:** fill the conclusion (four plan questions in your own words, numbers, verdict, limitation, defined terms), full top-to-bottom run, download back, "handoff done".

## Blocks (with breaks)

- Block A (~90 min): recall + reading + video + Build A through `block_shapes`.
- Break. Block B (~2 h): memorization + bounded run + comparison verdict.
- Break. Block C (~90 min): tensor workshop + conclusion + clean run.
- Buffer (~60 min): stuck time, write-up, handoff.

## Done when (plan evidence)

CNN run documented (val acc, params, wall time); split/merge round-trip exact; QKᵀ prediction verified; wrong-axis bug explained in writing; shape worksheet values (the four plan answers) on record. `run_smoke` passes on a fresh kernel with **no download**. Neither a CNN accuracy win nor fewer parameters is a pass condition.

## Plan practice questions (adapted for `/done`)

1. Why is a correct output shape necessary but insufficient for correct data layout?
2. Which axis represents candidate keys in the QKᵀ exercise?
3. Give an example where pooling discards information needed by a classifier.
4. What can you conclude if the CNN has fewer parameters but worse validation accuracy under today's budget?

## Smoke evidence contract

`run_smoke` (fresh kernel, synthetic data only): `SmallCNN` forward `(2,1,28,28)→(2,10)` finite; grads reach every parameter; block chain 28→28→14→14→14→7 with 52,138 params; split/merge exact recovery on `(2,5,32)→(2,4,5,8)`; `ValueError` on non-4D forward, non-3D split input, and `D % H != 0`; 16-sample synthetic memorization hits 1.000 within 200 steps. Save the log locally before clearing outputs.
