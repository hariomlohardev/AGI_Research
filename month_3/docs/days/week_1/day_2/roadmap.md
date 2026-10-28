# Roadmap — Month 3 Week 1 Day 2: training pipeline

**Time:** 5–6 focused hours. **Goal:** build the reusable skeleton used by the remaining months.

## Order of the day

| Block | Time | Work |
|---|---|---|
| Retrieval | 15–20 min | Accumulation recap; batch axis in Day-1 shapes |
| Reading + derivation | 60–90 min | Learn-the-Basics sections; Σ/B identity + discussion |
| Implementation | ~2 hours | Problems 1–3 in `coding_problems.md` (data, module, loop) |
| Checks + experiment | ~90 min | Problems 4–6 (baseline, independence, shuffle comparison) |
| Notes + explanation | 30–45 min | Conclusion, reusable-loop notes, open question |

Take breaks between blocks.

## Done when

- The complete loop runs: weights update, held-out MSE and baseline logged.
- Sample independence passes; your shuffling observation is documented either way.
- The loop is saved as named function cells with inputs/outputs/denominator explained.
- Clean-kernel smoke run passes before `/done`.

## Practice questions (answer in the notebook, then verify)

1. What does `model(x)` do beyond a direct `forward(x)` call?
2. Why do validation batches normally avoid random augmentation and gradient recording?
3. Why are parameters shared across examples compatible with zero cross-sample input gradients?
4. Why is a larger batch not automatically a better use of a fixed compute budget?

## Evidence for `/done`

`training_loop.ipynb` (smoke), its `.py` mirror, `test_training_loop.py`
results, and your conclusion with the loss/baseline/independence/shuffle
table. Smoke budgets only — ≤5 epochs, tiny data, no datasets downloaded.
