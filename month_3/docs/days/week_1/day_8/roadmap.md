# Month 3 · Week 1 Day 8 — Roadmap: independent MNIST classifier

**Time:** 4–5 hours (split across study days if needed). **Mode:** guided. **Tags:** `vision-basics`, `evaluation`, `reproducibility`.

## Order of work (with breaks between blocks)

| # | Block | Minutes | Output |
|---|---|---|---|
| 1 | Recall: training loop skeleton + eval branch from memory; Day 7's one-question-per-check habit | 15–20 | Spoken skeleton, no notes |
| 2 | Read: learn.md (scope, budget, sealed-test discipline) + scoped video sections | 30–45 | Closed tabs; predicted confusion pairs (4/9? 3/5?) |
| 3 | Data: inspect 20–30 examples, class counts; write 2–3 confusion hypotheses | 20–30 | Hypotheses quoted in conclusion |
| 4 | Implement: Problems 1–2 (classifier + budgeted train-select) | 60–75 | Forward shape holds; best-val checkpoint selected |
| 5 | Evaluate: confusion matrix, ten mistakes, Problem 3 + two error patterns | 45–60 | Matrix printed; patterns written, not just named |
| 6 | Reproduce: save bundle, kernel restart, rebuild-from-config, reload, agreement 1.0 | 30–45 | Reload verdict quoted |
| 7 | Evidence + README conclusion (Problem 5), Run All in Colab, handoff, sync, test | 30–45 | `run_smoke` green locally; full numbers quoted |

## Budgets (hard caps — no chasing)

- Full run: one hidden layer baseline, ≤ 10 epochs, batch 64, CPU-or-Colab-GPU. One baseline before any regularizer.
- Smoke/tests: synthetic fixture only (no download, seconds on CPU).
- Accuracy is recorded, never chased: no extra epochs, no width search, no regularizer added to beat a number.

## Done when

- A reloadable model exists (config + preprocessing + class mapping + weights, rebuilt from config alone with 1.0 prediction agreement).
- Validation metrics, a confusion matrix, ten inspected mistakes, and two explained error patterns are written down.
- A short README conclusion answers the four plan questions and records the achieved accuracy honestly.

## Practice questions (also the `/done` quiz pool)

1. What did data inspection change about your expectations?
2. Why is guessing a human error rate from 30 images not a valid benchmark?
3. What information besides weights is required to classify a new image correctly?
4. Which debugging step did you run independently, and which still required notes?

## Smoke evidence (seed 11, CPU, synthetic fixture)

`run_smoke` asserts: classifier forward shape `(N, 10)` float32 finite; `train_select` returns the argmin-val state with finite losses; `top_confusions` matches hand values on the fixture; save→rebuild→reload gives bitwise-identical predictions on fixed examples. Fresh-kernel execution must stop at no stub — every stub implemented before `/done`.
