# Month 3 · Week 1 Day 5 — Roadmap: manual backward pass, with independent verification

**Time:** 6–8 hours (split across study days if needed). **Mode:** guided. **Tags:** `math-heavy`, `coding-heavy`.

## Order of work (with breaks between blocks)

| # | Block | Minutes | Output |
|---|---|---|---|
| 1 | Recall: linear-layer gradient + tanh derivative (input and output forms) | 15–20 | Paper derivation in the notebook |
| 2 | Read: Backprop Ninja sections; roadmap §Day 5 scope (population variance + eps contract) | 60–90 | Closed tabs; per-op gradient derivations on paper |
| 3 | Verified video (linked in `learn.md`) | scoped sections only | Chain-rule-through-graph mechanics watched |
| 4 | Implement: Problems 1–2 (params, shared forward) | 45–60 | Finite scalar loss, shaped cache |
| 5 | Implement: Problems 3–4 (manual + reference backward) | 90–120 | Two gradient dicts, same keys |
| 6 | Implement: Problems 5–6 (compare table, break/locate/repair) | 60–90 | Full pass table; first mismatch `"dvar"`; repair returns `None` |
| 7 | Evidence + conclusion (Problem 7), kernel restart, Run All in smoke mode | 45–60 | Error table; convention + coupling probes; `run_smoke` passes |

## Done when

- The complete gradient table passes at `atol=1e-8, rtol=1e-6` in `float64` on the smoke shapes.
- The injected mean/variance omission is located (`"dvar"` first) and the repaired backward returns no mismatch.
- The conclusion explains the 1/B factor, the bias sum, the convention requirement, and the coupling probe — each with what it does and does not prove.

## Practice questions (also the `/done` quiz pool)

1. Where does division by batch size enter the mean-loss derivative?
2. Why does broadcasting a bias in the forward pass require a sum in its backward pass?
3. Why can changing one example affect another example through training-mode BatchNorm?
4. Why must the manual and autograd branches use the same variance definition and epsilon?

## Smoke evidence (seed 11, CPU, float64)

`run_smoke` asserts: params rebuild identically from the seed; forward loss finite with shaped cache; manual and reference dicts share keys and pass allclose on every key; bias grad equals summed upstream; mean-loss grads equal sum-loss grads divided by `B`; unbiased-variance reference fails the comparison; perturbing one input row moves another row's `dX`; broken variant first-mismatches at `"dvar"` while the repaired backward mismatches nowhere. Fresh-kernel execution must stop at no stub — every stub implemented before `/done`.
