# Month 3 · Week 1 Day 7 — Learn: diagnose a real training run

**Goal:** use evidence to distinguish implementation bugs from ordinary
optimization/generalization behaviour — one honest ordinary run, two injected
bugs, each closed with a written bug report.

## The debugging mindset for today

A training run that looks wrong is usually one of three things: the optimizer
never moved the weights, the data says something different from what you think
it says, or nothing is wrong and the curve is just an ordinary curve. Today
you practice telling those apart with checks that each answer exactly one
question — not with vibes, and not with a bigger model.

Two rules carry over from the whole week. First, **baseline before
regularizers**: the ordinary run uses a plain MLP and plain SGD, no dropout,
no BatchNorm, no weight decay added "just in case". Second, **describe before
you diagnose**: write down what the ordinary curve actually does (fast fall,
flat stretches, validation tracking train or not) before you ever call
anything a bug.

## The apparatus (read it, don't rebuild it)

Everything below ships complete in `diagnose_run.ipynb`:

- `make_data` — seeded tabular binary task, 320 train / 80 val, batch 32,
  deterministic shuffle, `num_workers=0`.
- `build_model` — small MLP: Linear(8→16), tanh, Linear(16→2).
- `run_training` — full loop returning a per-epoch log: `train_loss`,
  `val_loss`, `train_acc`, `val_acc`, `grad_norm`, `lr`. Train metrics are
  measured in train mode, validation metrics in eval mode — the mode is part
  of the evidence, because a metric measured in the wrong mode is a logging
  bug wearing a model's clothes.
- `snapshot_params` / `branch_no_update` / `branch_repaired` — Bug branch A:
  forward + backward run, optimizer step omitted, then the repaired twin.
- `branch_misaligned` — Bug branch B: inputs untouched, label column
  permuted, trained with Adam so the memorize-vs-chance signature is visible
  inside a short-epoch budget (SGD barely moves on shuffled targets here; the
  ordinary SGD run is the comparison, not a matched-optimizer control).
- `memorization_demo` — fixed 16-sample batch, shuffled labels, trained to a
  1.000 fit with Adam. It succeeds. That success is the lesson.

Your mechanism is the diagnosis layer: exact-equality frozen check, indexed
spot pairs, mechanical curve summary — plus the two written bug reports and
the conclusion.

## The two signatures to learn

**Omitted update.** Losses read flat across steps *and* parameter snapshots
are bitwise identical before and after. Flat losses alone prove nothing — a
logging mistake also prints flat lines. The snapshot comparison is the signal
that separates "no learning" from "bad logging": if the weights are exactly
unchanged, no update ran; if they changed, the update ran and the problem is
elsewhere (data, loss, or the log itself).

**Misaligned labels.** Training accuracy climbs toward memorization while
validation accuracy sits near chance — and indexed spot pairs show inputs
paired with wrong targets. The loop never complains: shapes match, dtypes
match, loss falls. This is the bug class that shape checks and tiny-batch
memorization checks both miss, which is why the spot-pair inspection exists.

## Why tiny-batch memorization is necessary but not sufficient

A sufficiently expressive model fits any fixed tiny batch, even with shuffled
labels — the demo proves it at 1.000. So a memorize-a-batch check passing
tells you the model *can* learn; it says nothing about whether the full
training targets are the ones you intended. Branch B is the counterexample you
can point at: it would pass a tiny-batch check (same model, same capacity)
while training on lies. Capacity checks and alignment checks answer different
questions, and today you run both.

## Verified video (scoped sections)

- [FSDL 2022 Lecture 3 — Troubleshooting & Testing](https://www.youtube.com/watch?v=RLemHNAO5Lw)
  (Charles Frye, ~42 min) — training-curve reading and bug-vs-noise judgment.
  Verified 2026-11-10 via the course channel index, the course lecture index,
  and the watch page.
- **Honest gap:** the lecture is framework-general. It does not cover the
  PyTorch-specific failure modes this day drills (a missing `zero_grad` /
  missing optimizer step, train/eval mode flips, indexed alignment checks) —
  the notebook carries those, so treat the video as curve-reading background,
  not as the procedure.

## Done when

One readable ordinary-run log described without claiming a bug per
fluctuation, two short bug reports (symptom, evidence, cause, fix,
verification), and a conclusion that answers the four plan questions plus the
memorization sentence. `run_smoke` passes on a fresh kernel.
