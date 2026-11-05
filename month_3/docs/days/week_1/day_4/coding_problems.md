# Month 3 · Week 1 Day 4 — Coding problems: Dropout, BatchNorm and diagnostic histograms

Work in `month_3/notebooks/week_1/day_4/dropout_batchnorm.ipynb`. Smoke mode only (`CONFIG["mode"] == "smoke"`, seed 11, CPU). Use `torch` layers/losses/optimizers, `sklearn` data generation/scaling and `matplotlib` figures as library work; every stub below is your implementation. Budgets are per problem and assume smoke sizes.

## Problem 1 — Train-fitted tabular data (preprocessing contract)

Implement `make_tabular_data(n_train=400, n_val=100, n_features=12, n_informative=6, seed=11)`.

- In: split sizes, feature counts, seed. Out: a dict with `X_train`, `y_train`, `X_val`, `y_val` as torch tensors (`float32` features, `long` labels).
- Generate the classes with a fixed seed, split into train/val, fit the standardization on the train split only, and apply it to both splits.
- `ValueError` when any size/count is `< 1`, when `n_informative > n_features`, or when `n_val >= n_train`.
- Check: shapes and dtypes match; the standardized train features have mean ≈ 0 (proving the scaler was fit on train, not on the pooled data).

## Problem 2 — Four MLP variants (one builder, four configurations)

Implement `build_mlp(input_dim, hidden=(32, 32), p_dropout=0.0, use_batchnorm=False, seed=11)`.

- In: input width, hidden widths, dropout probability, a BatchNorm on/off flag, seed. Out: an `nn.Sequential` MLP: per hidden layer `Linear → [BatchNorm1d] → Tanh → [Dropout]`, then a final `Linear` to one logit per class.
- Seed the weight initialization so the same arguments always build the same model.
- `ValueError` when `input_dim < 1`, `hidden` is empty, any width `< 1`, `p_dropout` is outside `[0, 1)`, or `seed < 0`.
- Check: output shape is `(B, n_classes)`; with `p_dropout=0` and `use_batchnorm=False` two builds from the same seed have identical parameters.

## Problem 3 — Fixed-budget training (deterministic history)

Implement `train_classifier(model, X_train, y_train, X_val, y_val, lr=0.05, epochs=20)`.

- In: a model plus fixed train/val tensors and an update budget. Out: a dict `{"train_loss": [...], "val_acc": [...]}` with one entry per epoch (full-batch SGD, no shuffling, cross-entropy loss).
- `ValueError` when `lr <= 0` or `epochs < 1`.
- Check: history lengths equal `epochs`; losses are finite; repeating the run from an identically seeded rebuild gives the identical history.

## Problem 4 — The four-way isolation table (regularization vs normalization)

Implement `run_four_way(X_train, y_train, X_val, y_val, hidden=(32, 32), lr=0.05, epochs=20, seed=11)`.

- In: the fixed data, shared width, shared update budget, seed. Out: a dict keyed `baseline`, `dropout`, `batchnorm`, `both`, each holding `{"train_loss": float, "val_acc": float}` (final-epoch values).
- Each arm gets a freshly seeded model (consistent initialization policy — never continue one arm's trained weights into another) and the identical budget.
- Check: all four arms complete with finite metrics. No arm is required to win — either outcome is recorded honestly.

## Problem 5 — What histograms see (hidden-state statistics + gradient norms)

Implement `hidden_stats(model, X)` and `grad_norms(model, X, y)`.

- `hidden_stats`: in — a model and an input batch; out — one dict per `Tanh` layer with `pre_mean`, `pre_std` (preactivation), `act_std`, and `frac_saturated` (fraction of activations with `|a| > 0.95`). Runs in evaluation mode under `no_grad` and restores the model's prior mode.
- `grad_norms`: in — a model, inputs, class targets; out — a list with one L2 gradient norm per parameter (single cross-entropy backward from zeroed grads; grads are zeroed again before returning).
- Check: lists line up with the model's `Tanh`/parameter counts; all values finite; on a deliberately saturated network the final layer's `frac_saturated` is near 1.

## Problem 6 — Saturation by construction (initialization effect)

Implement `scaled_saturation(input_dim=12, hidden=(16, 16, 16, 16), scale=4.0, n=256, seed=11)`.

- In: a four-layer tanh MLP shape, a weight-scale factor, a batch size, a seed. Out: the final hidden layer's `frac_saturated` after multiplying every weight (not bias) by `scale`.
- `ValueError` when `scale <= 0` or `n < 1`.
- Check: `scaled_saturation(scale=4.0)` is clearly larger than `scaled_saturation(scale=1.0)` on the same seed — large initial weights saturate tanh, small ones do not.

## Problem 7 — Batch size 1 and BatchNorm1d (the exact failure + rescue)

Implement `bn_single_sample(num_features=8, seed=11)`.

- In: channel count and seed. Out: a dict with `train_error` (the message from the training-mode forward on a `(1, C)` input — it must raise) and `eval_output_shape` (the shape from the evaluation-mode forward on the same input after establishing running statistics, which must succeed).
- Check: `train_error` is a non-empty string; `eval_output_shape == [1, num_features]`.

## Problem 8 — Evidence (four-way table, labeled histograms, honest conclusion)

Run `run_four_way` at smoke budget, plot (a) train-loss curves, (b) preactivation histograms for the reasonable-init vs saturated MLP, (c) final train-loss/val-accuracy bars. Fill the conclusion cell: the four-run table, what each histogram shows, the train/eval-mode explanation — each with one sentence saying what it does and does not prove. A smaller generalization gap is an observation, not a requirement.
