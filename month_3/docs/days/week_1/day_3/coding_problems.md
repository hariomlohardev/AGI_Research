# Month 3 · Week 1 Day 3 — Coding problems: honest timing and checkpoint recovery

Work in `month_3/notebooks/week_1/day_3/timing_checkpoint.ipynb`. Smoke mode only (`CONFIG["mode"] == "smoke"`, seed 11, CPU). Use `torch` tensor ops, `state_dict` handling and SGD as library work; every stub below is your implementation. Budgets are per problem and assume smoke sizes.

## Problem 1 — Device report with a CPU fallback (recall + setup)

Implement `describe_device()`.

- Return a dict with exactly these keys: `torch_version` (str), `cuda_available` (bool), `device` (str: `"cuda:0"` when CUDA is available else `"cpu"`), `name` (GPU name or `None`), `total_memory_GB` (float or `None`), `capability` (e.g. `"8.6"` or `None`).
- Consistency rule: when `cuda_available` is `False`, `device` is `"cpu"` and the three GPU fields are `None`.
- Check: keys and consistency hold on this machine; record the versions in your conclusion.

## Problem 2 — Synchronized matmul benchmark (three sizes, medians)

Implement `time_matmul(n=512, device="cpu", repeats=10, warmup=3, seed=11)`.

- Build two fixed `(n, n)` `float32` matrices from a generator seeded by `seed`.
- Run `warmup` untimed iterations, then time `repeats` iterations: synchronize before and after each timed iteration when the device is CUDA; report the median in milliseconds as a float.
- `ValueError` when `n < 1`, `repeats < 1`, `warmup < 0`, or the device string is not `"cpu"` / `"cuda"`.
- Check: finite positive medians at three sizes (e.g. 128, 512, 1024); table them. No GPU win is required at any size.

## Problem 3 — Compute-only vs transfer-inclusive timing (isolation + agreement)

Implement `transfer_vs_compute(n=512, device="cpu", repeats=5, seed=11)` and `max_abs_diff(a, b)`.

- `transfer_vs_compute` returns `{"compute_ms": float, "transfer_ms": float}`: compute-only uses pre-placed tensors; transfer-inclusive moves inputs CPU → device and the result back per repetition, synchronized when CUDA. Same `ValueError` rules as Problem 2.
- `max_abs_diff` returns the largest absolute element-wise difference as a float; `ValueError` when shapes differ.
- Check: both timings finite and non-negative; CPU and GPU results (when a GPU exists) agree within your stated tolerance; write the tolerance in the conclusion.

## Problem 4 — Checkpoint save and rebuild-from-config reload (recovery)

Implement `save_checkpoint(model, optimizer, epoch, config, path)` and `load_checkpoint(path, model, optimizer=None)`.

- `save_checkpoint` stores `model_state`, `optimizer_state`, `epoch` and `config` plus the torch version in one file; returns the path as a string. `ValueError` when `epoch < 0` or `config` is not a dict.
- `load_checkpoint` loads the states into the given objects in place and returns `{"epoch": ..., "config": ...}`. `FileNotFoundError` when the path is missing; `ValueError` when a required key is absent.
- Check: save a `TinyMLP` checkpoint to a temporary directory (durable-storage stand-in), rebuild model + optimizer fresh from config, reload, and confirm `load_checkpoint` returns your epoch and config.

## Problem 5 — Reload parity and a controlled resume (the actual claim)

Implement `prediction_max_diff(model_a, model_b, xb)` and `sgd_steps_on_batch(model, xb, yb, lr=0.01, steps=5)`.

- `prediction_max_diff` runs both models in evaluation mode under `no_grad` (restoring their prior modes), returning the max absolute output difference as a float.
- `sgd_steps_on_batch` runs full-batch MSE gradient descent for `steps` updates (zero grads each step) and returns the final loss as a float. `ValueError` when `steps < 1` or `lr <= 0`.
- Check: parity difference after reload is ~0 on a fixed batch; then the controlled resume at an epoch boundary — train K steps, save, continue M steps vs rebuild + reload + continue the same M steps — and compare the final parameters, not just the loss.

## Problem 6 — Evidence (conclusion table, no new code)

Fill the conclusion cell: device report, the three-size timing table, compute vs transfer numbers, agreement tolerance, parity difference, and the resume parameter comparison — each with one sentence saying what it does and does not prove. Single-run numbers are observations, not significance claims.
