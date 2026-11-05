# Month 3 · Week 1 Day 4 — Dropout, BatchNorm and useful histograms

**Time:** 6–7 hours. **Goal:** distinguish regularization, normalization and initialization effects — and learn to see them in histograms before trusting a single accuracy number.

## Recall first (15–20 min, no new reading)

1. Explain held-out error in one paragraph: what data it is measured on, why that data must stay out of every training and selection decision, and why a lower training loss need not mean a better model.
2. From Day 2, restate the two halves of inference discipline: what `model.eval()` changes and what `torch.no_grad()` changes — today's Question 4 will test whether that distinction survived contact with Dropout and BatchNorm.

## Read (60–90 min, then close the tabs and implement)

1. [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html). Read the training-vs-evaluation behaviour: inverted scaling during training, identity at evaluation. Work the preservation claim on paper: with mask `m ~ Bernoulli(1-p)`, `E[m·x/(1-p)] = x` — and note exactly what it does *not* preserve (variance, downstream nonlinear expectations).
2. [BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html). Read the statistics and mode sections: batch statistics in training mode vs running statistics in evaluation mode, and which input shapes are supported.
3. The activation/gradient-statistics section of [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (the makemore MLP lessons): how preactivation distributions evolve with depth, what healthy vs saturated histograms look like, and why gradient norms are worth tabulating per layer.
4. The cross-cutting comparison rule in `month_3/docs/roadmap.md` §2 (fresh state per run: recorded seed at the start of each independent run, no carried-over momentum, gradients, or BatchNorm statistics between arms).

> Tooling note: the plan suggests pandas inspection and a Month 2 dataset. This day's working environment has torch 2.14.1+cpu, scikit-learn 1.9.1 and matplotlib 3.10.5 — pandas is not installed, so the scaffold uses a seeded synthetic tabular fixture plus `sklearn` train-fitted `StandardScaler` preprocessing instead. The preprocessing contract (fit on train, apply to both) is the requirement being exercised, not the dataset brand. Re-running the four-way table on the Month 2 data is a legitimate optional extra once the smoke checks pass.

> Video (guided mode, one verified lecture): [Karpathy — Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) (~2 h, watch once with pauses). Builds the MLP whose internal activation and gradient distributions you will diagnose today — initialization, preactivation statistics, saturation behaviour. Optional follow-up (same series, also verified): [Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — the BatchNorm half of the day; watch it only if the reading budget allows, since Part 2 alone covers the histogram core.

## Build and verify (notebook `dropout_batchnorm.ipynb`, smoke mode, seed 11)

Library work (use as-is): `make_classification` data generation, `StandardScaler` preprocessing, `Linear`/`BatchNorm1d`/`Dropout`/`Tanh`, cross-entropy loss, full-batch SGD, `matplotlib` figures.
Your work: the preprocessing contract, the four-arm isolation design, hidden-state statistics, the saturation experiment, the single-sample BatchNorm failure and rescue, and every explanation.

1. **Data.** Implement `make_tabular_data(...)`: fixed-seed classes, a fixed train/val split, standardization fitted on train only. Prove the fit-on-train half by checking the train mean, not by asserting it.
2. **Variants.** Implement `build_mlp(...)`: one builder producing baseline, dropout-only, BatchNorm-only and both. Same arguments → same initialization; each comparison arm is rebuilt fresh, never continued from another arm's trained weights.
3. **Budget.** Implement `train_classifier(...)`: full-batch SGD for a fixed epoch count, recording per-epoch train loss and validation accuracy. Deterministic given the same seeded rebuild.
4. **Isolate.** Implement `run_four_way(...)`: all four arms, same width, same data, same budget. Record either outcome — dropout-only vs baseline isolates dropout; batchnorm-only vs baseline isolates BatchNorm; no arm is required to win.
5. **See.** Implement `hidden_stats(...)` (per-`Tanh` preactivation mean/std, activation spread, saturated fraction) and `grad_norms(...)` (one L2 norm per parameter). Tabulate them for the reasonable-init MLP and again after deliberate saturation.
6. **Saturate.** Implement `scaled_saturation(...)`: multiply a four-layer tanh MLP's weights by `scale` and report the final layer's saturated fraction. Large-scale weights must clearly exceed scale-1.0 saturation.
7. **Break and rescue.** Implement `bn_single_sample(...)`: capture the training-mode `(1, C)` error message verbatim, then run the same input in evaluation mode with established running statistics and confirm the shape.

## Questions (answer in the notebook conclusion)

1. Which comparison isolates dropout's effect, and which isolates BatchNorm's?
2. Why might regularization fail to improve validation performance on this dataset?
3. What do activation histograms reveal that a final accuracy number does not?
4. Why does `model.eval()` matter even inside a `no-grad` context?

## Done when / evidence

The four-run table (final train loss + val accuracy per arm), labeled histograms (loss curves, reasonable-vs-saturated preactivations, final metrics), the verbatim BN single-sample error plus eval rescue, and an honest conclusion where each item carries one sentence on what it does and does not prove. Restart the kernel and run the notebook in smoke mode before `/done`.
