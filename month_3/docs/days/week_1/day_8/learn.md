# Month 3 · Week 1 Day 8 — Learn: independent MNIST classifier

**Goal:** assemble a complete small classifier from your own utilities — and prove it reloads bit-for-bit.

## The shape of today

This is an assembly day, not a derivation day. The pieces already exist in your hands — data loading, the training loop, eval discipline, snapshot comparison, confusion reading from Day 7. Today they combine into one reloadable artifact: a config, a preprocessing recipe, a class mapping, and weights that reproduce predictions exactly after a kernel restart.

Three disciplines carry the day, all lived before:

1. **Baseline before regularizers.** One hidden layer, plain SGD, ≤ 10 epochs. If you want dropout or BatchNorm, the conclusion cell must carry the explanation — otherwise it doesn't go in.
2. **Selection by validation, never by train.** The best checkpoint is the lowest-val-loss epoch. The test set stays sealed — not loaded, not peeked at, not passed anywhere. It exists for Week 4.
3. **Describe before you diagnose.** The inspection hypotheses (which digit pairs will confuse?) get written *before* the matrix is printed. A hypothesis the matrix contradicts is information; a pattern named after the fact is storytelling.

## The apparatus (read it, don't rebuild it)

Everything below ships complete in `mnist_classifier.ipynb`:

- `get_mnist` / `make_mnist_splits` — torchvision MNIST, `ToTensor` + `Normalize(0.1307, 0.3081)`, deterministic 55k/5k split (seed 11, indices persisted), official test set loaded but returned sealed.
- `inspect_samples` — 24 examples with labels and pixel stats plus class counts. Hypotheses, not measurements.
- `run_epoch` / `eval_loss_acc` — one train epoch, one eval pass. Both flatten image batches; your module always sees flat vectors.
- `confusion_matrix` — raw counts, rows true / columns predicted.
- `save_bundle` / `load_bundle` / `reload_compare` — config + preprocessing string + class list + weights; agreement fraction against predictions recorded before the restart.
- `print_mistakes` — the first ten mistakes as (index, true, pred). The pixels are yours to look at.
- `run_smoke` — the whole evidence chain on the synthetic fixture (no download): forward contract, best-val selection, reader hand values, reload agreement 1.0.

Your mechanism: `build_classifier` (one-hidden-layer MLP on flat input), `train_select` (budget + argmin-val snapshot), `top_confusions` (top-k off-diagonal reader) — plus the ten-mistake error notes, the README conclusion, and the reload verdict.

## The reload proof, precisely

After the full run: the bundle (`./day8_bundle.pt`) and the pre-restart validation predictions (`./day8_val_preds.pt`) sit on disk. Restart the kernel. Rebuild the model from the saved config *alone* — no notebook state, no re-running training. Load the weights. Compare predictions on the same fixed validation order (deterministic loader, same seed). Agreement must read **1.0** — bitwise, the Day 7 strictness. Anything less names its missing piece: config (wrong architecture), preprocessing (shifted inputs), class mapping (relabeled outputs), or weights.

## Verified video (scoped sections)

- [Pytorch Neural Network example — Aladdin Persson](https://youtu.be/Jy4wM2X21u0)
  (companion: `pytorch_simple_fullynet.py` in the author's Machine-Learning-Collection repo —
  MNIST via torchvision, 784→50→10 ReLU MLP, 3-epoch training loop, accuracy eval).
  Verified 2026-11-12 via the repo README listing, the companion script contents, and the
  watch page. Runtime could not be verified in-session — budget it as one focused sitting.
  Watch the data-loading, model-definition, training-loop, and evaluation portions; skip setup recap.
- **Honest gap:** the video evaluates directly on the test split with overall accuracy only —
  no train/val split with validation-based selection, no checkpoint save/reload-verify, no
  confusion matrix or mistake inspection. All three come from the notebook, not the video.

## Done when

A reloadable bundle (config + preprocessing + classes + weights) rebuilt from config alone with 1.0 prediction agreement; validation metrics, a confusion matrix, ten inspected mistakes, and two explained error patterns written down; a short README conclusion answering the four plan questions and recording the achieved accuracy honestly. `run_smoke` passes on a fresh kernel.
