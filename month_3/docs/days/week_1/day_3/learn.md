# Month 3 · Week 1 Day 3 — GPU setup, honest timing and checkpoint recovery

**Time:** 4–5 hours. **Goal:** know when a GPU helps, and recover work after a runtime reset.

## Recall first (15–20 min, no new reading)

1. Explain the Day 2 regression baseline in one paragraph: what constant predictor you fitted, on which targets, and what it means when the trained model beats it.
2. Explain the difference between evaluation mode and disabling gradients: what `model.eval()` changes, what `torch.no_grad()` changes, and why inference needs both.

## Read (60–90 min, then close the tabs and implement)

1. [CUDA notes — timing and synchronization semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html). Read the semantics section: kernels launch asynchronously, so wall-clock time around a GPU call measures launch, not work, unless you synchronize.
2. [Checkpoint reference — saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). Read the `state_dict` sections: what a checkpoint stores for the model and the optimizer, and why you rebuild the objects from config before loading.
3. The checkpoint contract in `month_3/docs/roadmap.md` §3.3: a resumable checkpoint stores model + optimizer state, epoch/global step, scheduler state if used, RNG state for a controlled continuation, and architecture config alongside the weights.
4. Check your own Colab/Kaggle GPU limits in your account ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). Free GPU availability, model and runtime duration are not guaranteed — that is why this day keeps a CPU fallback and a backup environment.

> Video (guided mode, one verified practical demo): [Patrick Loeber — PyTorch Tutorial 17, Saving and Loading Models](https://www.youtube.com/watch?v=9L9jEOwRrCg) (~18 min). Same series as Day 2's video; covers the `torch.save` checkpoint-dict pattern plus `torch.load` / `load_state_dict` into a rebuilt model — the save → rebuild → reload workflow. It does not cover GPU timing discipline, so the timing half stays with the CUDA-notes reading above.

## Build and verify (notebook `timing_checkpoint.ipynb`, smoke mode, seed 11)

Library work (use as-is): tensor creation, matmul, `torch.save`/`torch.load`, `state_dict` handling, SGD.
Your work: synchronized timing, transfer isolation, the checkpoint save/load contract, the parity and resume comparisons, and every explanation.

1. **Device.** Implement `describe_device()`: torch version, CUDA availability, active device string, GPU name / memory / capability or `None` on a CPU-only machine. CPU fallback is required, not a failure.
2. **Benchmark.** Implement `time_matmul(n, device, repeats, warmup, seed)`: fixed seeded matrices, warmup iterations, `torch.cuda.synchronize()` around GPU wall-clock timing, median of repeats in milliseconds. Benchmark three manageable sizes at the same dtype.
3. **Isolate transfer.** Implement `transfer_vs_compute(n, device, repeats, seed)`: compute-only timing (tensors pre-placed) next to transfer-inclusive timing (CPU → device → compute → CPU per repetition). Confirm numerical agreement within a stated tolerance via `max_abs_diff`. Do not require a GPU win at any size.
4. **Checkpoint.** Implement `save_checkpoint` / `load_checkpoint` using the §3.3 contract (model + optimizer state, epoch, config). Copy the checkpoint to durable storage; simulate the runtime restart by rebuilding the model and optimizer from configuration in a fresh object and reloading.
5. **Parity + resume.** On a fixed batch, compare predictions before/after reload in evaluation mode (`prediction_max_diff`). For resume, start at an epoch boundary: compare a controlled next update with and without interruption, restoring optimizer state. A similar loss after restart does not prove the identical trajectory — compare the update itself.

## Questions (answer in the notebook conclusion)

1. Why is unsynchronized GPU wall-clock timing misleading?
2. When can transfer and launch overhead dominate the arithmetic?
3. What optimizer information would be lost by saving weights alone?
4. Why does a similar loss after restart not prove the exact same training trajectory?

## Done when / evidence

A correctly measured timing table (three sizes, medians, compute vs transfer, agreement tolerance stated) and a successful durable save → rebuild → reload → controlled-resume check. Explain remaining numerical differences instead of claiming bitwise identity. Restart the kernel and run the notebook in smoke mode before `/done`.
