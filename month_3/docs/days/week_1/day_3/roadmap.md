# Month 3 · Week 1 Day 3 — Roadmap: GPU setup, honest timing and checkpoint recovery

**Time:** 4–5 hours. **Mode:** guided. **Tags:** `coding-heavy`, `experiment-discipline`.

## Order of work (with breaks between blocks)

| # | Block | Minutes | Output |
|---|---|---|---|
| 1 | Recall: regression baseline + eval-mode vs no-grad | 15–20 | Two short paragraphs in the notebook |
| 2 | Read: CUDA semantics + checkpoint reference + §3.3 contract; check Colab/Kaggle limits | 60–90 | Closed tabs; device question answered |
| 3 | Verified video (linked in `learn.md`) | 10–40 | Timing/checkpoint demo watched once |
| 4 | Implement: Problems 1–3 (device, benchmark, transfer) | 60–90 | Timing table with medians |
| 5 | Implement: Problems 4–5 (checkpoint, parity, resume) | 60–90 | Save → rebuild → reload → resume checks green |
| 6 | Evidence + conclusion (Problem 6), kernel restart, Run All in smoke mode | 30–45 | Conclusion table; `run_smoke` passes |

## Done when

- `describe_device` reports versions with a working CPU fallback.
- Three-size matmul medians are tabled; compute-only and transfer-inclusive timings are separated; agreement tolerance is stated.
- A checkpoint saved to durable storage reloads into a config-rebuilt model with ~0 parity difference on a fixed batch.
- A controlled epoch-boundary resume reproduces the uninterrupted update (parameter comparison, not loss alone).
- The conclusion explains what each number does and does not prove.

## Practice questions (also the `/done` quiz pool)

1. Why is unsynchronized GPU wall-clock timing misleading?
2. When can transfer and launch overhead dominate the arithmetic?
3. What optimizer information would be lost by saving weights alone?
4. Why does a similar loss after restart not prove the exact same training trajectory?

## Smoke evidence (seed 11, CPU)

`run_smoke` asserts: device-contract keys; finite positive medians at small sizes; finite transfer timings; reload parity ~0; deterministic full-batch SGD; epoch-boundary resume parameter match. Fresh-kernel execution must stop at no stub — every stub implemented before `/done`.
