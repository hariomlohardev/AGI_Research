# Roadmap — Month 3 Week 1 Day 1: tensors, shape reasoning and autograd

**Time:** 5–6 focused hours. **Goal:** explain what every axis means and verify
gradients instead of treating backward as a black box.

## Order of the day

| Block | Time | Work |
|---|---|---|
| Retrieval | 15–20 min | Chain rule for a three-function composition; shared-node contributions |
| Reading + derivation | 60–90 min | Tensor tutorial sections you need; hand derivations and shape predictions |
| Implementation | ~2 hours | Problems 1–4 in `coding_problems.md` |
| Checks + experiment | ~90 min | Tiny checks, accumulation/leaf/retain demos, broadcast correction |
| Notes + explanation | 30–45 min | Gradient comparison, shape table, conclusion, open question |

Take breaks between blocks.

## Done when

- Scalar gradients match your hand derivation; the shared-node case passes.
- You can explain and correct the broadcasting bug (valid code, wrong shape).
- Accumulation, leaf/non-leaf `.grad`, and the retain-graph error are each
  demonstrated with a one-paragraph why.
- Gradient comparison and shape table are saved in the notebook; clean-kernel
  smoke run passes. Record achieved agreement — do not chase extra precision.

## Practice questions (answer in the notebook, then verify)

1. Why do repeated uses of a tensor add gradient contributions?
2. Why can an intermediate require gradients while its `.grad` is not retained
   by default?
3. Why can broadcasting produce a mathematically valid result with the wrong
   intended shape?
4. What is the difference between clearing accumulated gradients and retaining
   a computation graph?

## Evidence for `/done`

`tensors_autograd.ipynb` (smoke), its `.py` mirror, `test_tensors_autograd.py`
results, and your conclusion with the comparison tables. Smoke mode only — no
training exists today, so there is nothing full-scale to report.
