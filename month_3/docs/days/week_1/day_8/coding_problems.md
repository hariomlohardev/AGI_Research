# Month 3 · Week 1 Day 8 — Coding problems: independent MNIST classifier

Work in `month_3/notebooks/week_1/day_8/mnist_classifier.ipynb`. Smoke mode only (`CONFIG["mode"] == "smoke"`, seed 11, CPU). Smoke uses a tiny synthetic fixture (no download); the full MNIST run happens in Colab (`CONFIG["mode"] == "full"`), then the finished `.ipynb` comes back over the repo copy.

The whole apparatus ships complete — read it, don't rewrite it: `get_mnist` (torchvision download + ToTensor, `./data` ignored), `make_splits` (55k/5k train/val, official test sealed unopened, persisted indices, deterministic val pipeline), `inspect_samples` (20–30 examples, labels, class counts), `run_epoch` / `eval_loss_acc` (one train epoch, one eval pass), `confusion_matrix` (raw counts — reading it is yours), `save_bundle` / `load_bundle` (config + preprocessing + class mapping + state_dict), `reload_compare` (rebuild from config, load weights, compare predictions on fixed val examples), `run_smoke` (all evidence in one command).

Your work is the assembly layer: three small functions plus the written error notes, README conclusion, and the reload verdict.

## Problem 1 — Assemble the classifier

Implement `build_classifier(input_dim: int = 784, hidden: int = 128, num_classes: int = 10) -> nn.Module`.

- In: flattened input size, hidden width, class count. Out: an `nn.Module` mapping `(N, input_dim)` float32 to `(N, num_classes)` logits — flatten is the caller's job (MNIST arrives `(N, 1, 28, 28)`; the harness flattens before calling), yours is the MLP.
- Baseline first: one hidden layer, your activation choice recorded. No dropout/BatchNorm unless you can explain the choice in the conclusion cell.
- Check: forward shape `(N, 10)`, float32, finite outputs on the fixture.

## Problem 2 — Train under budget, keep the best

Implement `train_select(model, train_loader, val_loader, epochs: int, lr: float) -> tuple[dict, dict]`.

- In: model, loaders, a preset epoch budget, lr. Out: `(best_state, log)` — `best_state` is the `state_dict` snapshot from the epoch with the lowest validation loss, `log` holds per-epoch `train_loss`/`val_loss` lists.
- Selection is by validation, never by train. The test set stays sealed — it is not loaded, not peeked at, not passed anywhere.
- Check: on the synthetic fixture the returned `best_state` loads cleanly and matches the argmin-val epoch; losses finite.

## Problem 3 — Read the confusion matrix (extraction only)

Implement `top_confusions(cm: torch.Tensor, k: int = 3) -> list[tuple[int, int, int]]`.

- In: a `(10, 10)` confusion matrix (rows true, columns predicted). Out: the top-`k` off-diagonal cells as `(true, predicted, count)` sorted by count descending.
- Pure mechanical extraction — which pairs confuse and why is your written error analysis, not this function.
- Check: exact triples on a hand-built matrix fixture.

## Problem 4 — Two error patterns + reload verdict (markdown, your words)

- **Ten mistakes:** inspect ten misclassified validation examples (the harness prints them). Explain at least two error patterns (e.g. 4/9, 3/5/8, thin-vs-thick strokes) as hypotheses from the pixels — not a measured human error rate.
- **Reload verdict:** restart the kernel, reconstruct the model from the saved config alone, `load_state_dict`, and confirm identical predictions on the same fixed validation examples. Quote the agreement fraction (must be 1.0 — bitwise, like Day 7's frozen check).
- **README conclusion:** what the model is, preprocessing + class mapping, validation accuracy (record it, don't chase it), the two error patterns, and one line on what the sealed test set is for in Week 4.

## Problem 5 — Evidence (log, matrix, notes, verdict)

Run the full comparison in Colab: baseline trains under budget, best-val checkpoint selected, validation confusion matrix printed, ten mistakes inspected, reload agreement 1.0. Fill the conclusion cell: data-inspection hypotheses vs outcome, answers to the four plan questions, and one sentence per check on what it does and does not prove. `run_smoke` passes on a fresh kernel (synthetic fixture — the full MNIST numbers live in your Colab run, quoted into the conclusion).
