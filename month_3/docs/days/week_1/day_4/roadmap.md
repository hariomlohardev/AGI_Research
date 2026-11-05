# Month 3 · Week 1 Day 4 — Roadmap: Dropout, BatchNorm and useful histograms

**Time:** 6–7 hours. **Mode:** guided. **Tags:** `coding-heavy`, `experiment-discipline`.

## Order of work (with breaks between blocks)

| # | Block | Minutes | Output |
|---|---|---|---|
| 1 | Recall: held-out error + why lower training loss need not mean better | 15–20 | Short paragraph in the notebook |
| 2 | Read: Dropout + BatchNorm1d docs; Karpathy course activation/gradient section | 60–90 | Closed tabs; scaling + statistics questions answered |
| 3 | Verified video (linked in `learn.md`) | ~120 (lecture; watch once, pause to take notes) | MLP init + activation/gradient diagnostics watched |
| 4 | Implement: Problems 1–3 (data, builder, training loop) | 60–90 | Deterministic history on a fixed split |
| 5 | Implement: Problems 4–5 (four-way table, hidden stats) | 60–90 | Isolation table with finite metrics |
| 6 | Implement: Problems 6–7 (saturation, single-sample BN) | 45–60 | Saturation gap + train-error/eval-rescue recorded |
| 7 | Evidence + conclusion (Problem 8), kernel restart, Run All in smoke mode | 45–60 | Labeled histograms; conclusion table; `run_smoke` passes |

## Done when

- Four MLP arms train on the same fixed split under the identical update budget; the table records either outcome honestly.
- Hidden-state statistics and gradient norms are tabulated per layer for the reasonable-init and saturated MLPs.
- The `(1, C)` BatchNorm failure is recorded verbatim and the eval-mode rescue succeeds on the same input.
- The conclusion explains what each number and histogram does and does not prove; no winner is required.

## Practice questions (also the `/done` quiz pool)

1. Which comparison isolates dropout's effect, and which isolates BatchNorm's?
2. Why might regularization fail to improve validation performance on this dataset?
3. What do activation histograms reveal that a final accuracy number does not?
4. Why does `model.eval()` matter even inside a `no_grad` context?

## Smoke evidence (seed 11, CPU)

`run_smoke` asserts: preprocessing contract (shapes, dtypes, train-fitted mean ≈ 0); deterministic rebuilds and histories; four-way table completes with finite metrics; saturated-init fraction clearly exceeds reasonable-init; the BN single-sample train error is captured and eval succeeds; histogram/gradient lists line up with the architecture. Fresh-kernel execution must stop at no stub — every stub implemented before `/done`.
