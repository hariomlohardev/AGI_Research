# Month 3 · Week 1 Day 7 — Coding problems: diagnose a real training run

Work in `month_3/notebooks/week_1/day_7/diagnose_run.ipynb`. Smoke mode only (`CONFIG["mode"] == "smoke"`, seed 11, CPU). Budgets assume smoke sizes (`n_train=320, n_val=80, d=8, hidden=16`, ordinary run 15 epochs, misaligned branch 30 short epochs, frozen branch 3 steps).

The whole training apparatus ships complete — read it, don't rewrite it: `make_data` (seeded tabular binary task + train/val split), `build_model` (small MLP), `run_training` (full loop returning a per-epoch log of train/val loss, train/val accuracy, gradient norm, learning rate, and the mode each metric was measured in), `snapshot_params` (cloned param dicts), `branch_no_update` (trains N steps with the optimizer step omitted, returns before/after snapshots + loss trace), `branch_misaligned` (shuffles the label alignment, short run, returns log + the permuted pairs), `memorization_demo` (fixed tiny batch, shuffled labels, trains to memorization), `run_smoke` (all evidence in one command).

Your work is the diagnosis layer: three small functions plus two written bug reports and the ordinary-run description.

## Problem 1 — Frozen-update check

Implement `params_unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool`.

- In: two parameter snapshot dicts. Out: `True` iff every tensor is bitwise identical (`torch.equal`, not allclose — an untouched parameter is *exactly* unchanged, and the strict contract is the point).
- `ValueError` when key sets differ.
- Check: identical snapshots → `True`; a single-element perturbation → `False`.

## Problem 2 — Alignment spot check

Implement `alignment_report(X: torch.Tensor, y: torch.Tensor, indices: list[int]) -> list[tuple[int, list[float], int]]`.

- In: features `(N, d)`, labels `(N,)`, row indices. Out: `(row index, feature row as a plain list, label)` per index, in the requested order.
- `ValueError` on length mismatch or an out-of-range index.
- Check: exact pairs on the fixture.

## Problem 3 — Curve summary (extraction only)

Implement `curve_summary(log: dict[str, list[float]]) -> dict`.

- In: the per-epoch log from `run_training`. Out: `epochs`, `first_train_loss`, `last_train_loss`, `best_val_loss`, `best_val_epoch`, `final_gap` (last val loss minus last train loss), `val_rising` (mean of last 3 val losses > mean of first 3), `train_falling` (last train loss < first train loss).
- Pure mechanical extraction — the interpretation belongs in your bug reports, not in this function.
- Check: hand values on a tiny synthetic log; a rising-val + falling-train fixture sets both flags.

## Problem 4 — Two bug reports (markdown, your words)

- **Branch A (update omitted):** symptom, evidence (`params_unchanged` verdict + snapshot norms + flat loss trace), cause, fix, verification (re-run *with* the step: losses move, check returns `False`).
- **Branch B (labels misaligned):** symptom (train memorizes, validation near chance), evidence (spot pairs + curve summary), cause, fix, verification.
- **One honest sentence:** why the memorization demo proves a memorize-a-tiny-batch check alone cannot catch every data bug.

## Problem 5 — Evidence (log, reports, conclusion)

Run the smoke comparison: the ordinary run finishes with a finite, well-shaped log; the frozen branch reads unchanged and a perturbed copy reads changed; spot pairs are exact; the summary matches hand values on the fixture. Fill the conclusion cell: a plain description of the ordinary run's learning curve (no bug claimed per fluctuation), answers to the four plan questions, and one sentence per check on what it does and does not prove.
