# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Month 3 · Week 1 Day 3 — GPU setup, honest timing and checkpoint recovery
#
# **Question.** Can you time honestly (synchronized, transfer-isolated) and recover exactly (rebuild from config, reload, resume)?
#
# **Prediction (write yours before running).** Say whether you expect a GPU win at small sizes and why; what `state_dict` alone cannot restore; and whether equal loss after restart proves an identical trajectory.
#
# **Library / learner boundary.** Use `torch` tensor ops, matmul, `torch.save`/`torch.load`, `state_dict` handling and plain SGD updates for routine work. Your work: synchronization discipline, transfer isolation, the checkpoint contract, parity/resume comparisons, and every explanation.
#
# **Shapes, dtypes, budgets.** Square `float32` matrices, batch axis first, `TinyMLP` maps `(B, 4)` to `(B, 1)`. Smoke mode only: small matmul sizes, a handful of repeats, tiny full-batch SGD. No downloads and no full-scale training.

# %%
"""Day 3 workspace: GPU timing and checkpoint recovery (stubs for the learner)."""

import math
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

CONFIG = {"device": "cpu", "dtype": "float32", "mode": "smoke", "seed": 11}


# %%
def describe_device():
    """Report the compute device and software versions.

    Return a dict with exactly these keys: ``torch_version`` (str),
    ``cuda_available`` (bool), ``device`` (str: ``"cuda:0"`` when CUDA is
    available else ``"cpu"``), ``name`` (GPU name or None),
    ``total_memory_GB`` (float or None) and ``capability`` (like ``"8.6"``
    or None). When CUDA is unavailable the three GPU fields are None;
    a CPU-only machine is a valid result, not a failure.
    Returns: dict device report.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    is_cuda = torch.cuda.is_available()

    if is_cuda:
      device = torch.device('cuda')


      device_name = torch.cuda.get_device_name(device)

      total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

      major, minor = torch.cuda.get_device_capability()
      capability = f"{major}.{minor}"



    return {
        "torch_version" : torch.__version__,
        "cuda_available" : True if is_cuda else False,
        "device" : "cuda:0" if is_cuda else "cpu",
        "name" : device_name if is_cuda else None,
        "total_memory_GB" : total_memory_gb if is_cuda else None,
        "capability" : capability if is_cuda else None
    }


# %%
def time_matmul(n=512, device="cpu", repeats=10, warmup=3, seed=11):
    """Time one ``(n, n)`` float32 matmul; report median milliseconds.

    Build two fixed matrices from a generator seeded by ``seed``, run
    ``warmup`` untimed iterations, then time ``repeats`` iterations with
    ``time.perf_counter``. Synchronize before and after each timed
    iteration when the device is CUDA; GPU kernels launch asynchronously,
    so unsynchronized wall time only measures the launch.
    Returns: float median milliseconds.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``n < 1``, ``repeats < 1``, ``warmup < 0``, or
        ``device`` is not ``"cpu"`` or ``"cuda"``.
    """
    if n < 1:
      raise ValueError("n can't be less than 1")

    if repeats < 1:
      raise ValueError("repeats can't be less than 1")

    if warmup < 0:
      raise ValueError("warmup can't be less than 0")

    if device not in {"cpu", "cuda"}:
      raise ValueError("device can only be ``cuda`` or ``cpu``")

    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(n, n, generator=g, device=device)
    b = torch.randn(n, n, generator=g, device=device)

    for _ in range(warmup):
      a @ b



    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()   # flush anything queued before — clean start
        start = time.perf_counter()
        c = a @ b
        if device == "cuda":
            torch.cuda.synchronize()   # wait for THIS multiply to actually finish
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


# %%
def transfer_vs_compute(n=512, device="cpu", repeats=5, seed=11):
    """Separate compute-only timing from transfer-inclusive timing.

    Compute-only uses tensors already placed on ``device``; the inclusive
    arm moves inputs CPU -> device and the result back per repetition
    (synchronized when CUDA). Same fixed seeded matrices for both arms.
    Returns: ``{"compute_ms": float, "transfer_ms": float}`` medians.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``n < 1``, ``repeats < 1``, or ``device`` is not
        ``"cpu"`` or ``"cuda"``.
    """

    compute_ms = time_matmul(n=n, device=device, repeats=repeats, seed=seed)

    g = torch.Generator().manual_seed(seed)
    a_cpu = torch.randn(n, n, generator=g)   # no device= → stays on CPU
    b_cpu = torch.randn(n, n, generator=g)

    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        a_dev = a_cpu.to(device)      # RAM -> device
        b_dev = b_cpu.to(device)      # RAM -> device
        c_dev = a_dev @ b_dev         # compute
        c_cpu = c_dev.to("cpu")       # device -> RAM
        if device == "cuda":
            torch.cuda.synchronize()  # everything above, actually finished
        times.append((time.perf_counter() - start) * 1000)



    return {"compute_ms": compute_ms, "transfer_ms": statistics.median(times)}
    

# %%
def max_abs_diff(a, b):
    """Return the largest absolute element-wise difference as a float.

    Used to state numerical agreement between CPU and GPU results (or
    before/after reload) instead of claiming bitwise identity.
    Returns: float maximum absolute difference.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if the shapes differ.
    """
    
    if a.shape != b.shape:
        raise ValueError("a and b both tensors shapes should be same")
    return (a - b).abs().max().item()



# %%
class TinyMLP(nn.Module):
    """Tiny regressor used as the checkpoint vehicle: Linear(4, 16) -> ReLU -> Linear(16, 1).

    Given fully implemented: the mechanism under study today is timing
    and checkpointing, not architecture. ``forward`` maps ``(B, 4)`` to
    ``(B, 1)``.
    """

    def __init__(self):
        """Create the two linear maps and ReLU."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        """Map ``(B, 4)`` inputs to ``(B, 1)`` outputs."""
        return self.net(x)



# %%
def sgd_steps_on_batch(model:TinyMLP, xb, yb, lr=0.01, steps=5):
    """Run plain full-batch MSE gradient descent for ``steps`` updates.

    Zero grads, forward, MSE against ``yb``, backward, then update each
    parameter by ``lr`` times its gradient under ``no_grad`` (manual
    update, no momentum: the point is a deterministic trajectory, not a
    fast optimizer). Squeeze ``(B, 1)`` outputs to ``(B,)`` before the
    loss; never let ``(B, 1)`` vs ``(B,)`` broadcast.
    Returns: float final loss.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``steps < 1`` or ``lr <= 0``.
    """
    if steps < 1:
      raise ValueError("steps can't be less than 1")

    if lr <= 0:
      raise ValueError("learning rate(lr) can't be less than or equal to 0")
    

    for _ in range(steps):
      
      model.zero_grad()
      pred = model(xb).squeeze(-1)
      loss =  F.mse_loss(pred, yb.squeeze(-1))
      loss.backward()

      with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad

    return loss.item()


# %%
def save_checkpoint(model:TinyMLP, optimizer, epoch, config, path):
    """Save one resumable checkpoint file.

    Store ``model_state``, ``optimizer_state``, ``epoch`` and ``config``
    plus the torch version in a single file. Arguments are validated before
    anything is written. The caller copies the file to durable storage;
    recovery rebuilds fresh objects from ``config``.
    Returns: str path saved to.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``epoch < 0`` or ``config`` is not a dict.
    """
    if epoch < 0:
      raise ValueError("epoch can't be less than 0")
    
    if not isinstance(config,dict):
      raise ValueError("config should be an dict and nothing else")
    
    # Save model
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "torch_version": str(torch.__version__),
    }

    torch.save(checkpoint, path)
    return path


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint states into the given objects, in place.

    Rebuilt-from-config objects only: never deserialize a whole pickled
    model. Restores the optimizer too when one is passed.
    Returns: ``{"epoch": int, "config": dict}``.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: FileNotFoundError: if ``path`` does not exist.
    Raises: ValueError: if a required key is absent from the file.
    """
    file_path = Path(path)

    
    if not file_path.exists():
      raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, weights_only=True)
    for key in ("epoch", "model_state", "optimizer_state", "config", "torch_version"):
        if key not in ckpt:
            raise ValueError(f"checkpoint missing required key: {key}")

    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None: 
      optimizer.load_state_dict(ckpt["optimizer_state"])

    return {"epoch": ckpt["epoch"], "config": ckpt["config"]}


# %%
def prediction_max_diff(model_a, model_b, xb):
    """Compare fixed-batch predictions of two models in evaluation mode.

    Remember each model's mode, run both under ``no_grad`` in eval mode,
    restore the remembered modes, and return the max absolute output
    difference. Used for before/after-reload parity.
    Returns: float maximum absolute difference.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """
    
    was_a = model_a.training
    was_b = model_b.training
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(xb)
        out_b = model_b(xb)
    model_a.train(was_a)
    model_b.train(was_b)
    return max_abs_diff(out_a, out_b)



# %% [markdown]
# ## Verification and evidence
#
# Tiny checks for this day (all smoke mode, CPU, seed 11):
#
# 1. Device report has all six keys and is consistent (CPU fallback valid).
# 2. Matmul medians are finite and positive at three sizes; GPU arms (when run) synchronize around wall time.
# 3. Transfer-inclusive timing is separated from compute-only; CPU/GPU agreement is stated with a tolerance.
# 4. Checkpoint round trip restores epoch and config into config-rebuilt objects, including optimizer state.
# 5. Reload parity on a fixed batch is ~0 in evaluation mode.
# 6. Full-batch SGD is deterministic for a fixed seed; the epoch-boundary resume reproduces the uninterrupted update (parameter comparison, not loss alone).
#
# Save the timing table and the parity/resume numbers in your conclusion. Restart the kernel and `Run All` in smoke mode before `/done`.

# %%
def run_smoke():
    """Run the Day 3 checks in smoke mode."""
    info = describe_device()
    assert set(info) == {"torch_version", "cuda_available", "device",
                          "name", "total_memory_GB", "capability"}, info
    assert isinstance(info["cuda_available"], bool)
    if info["cuda_available"]:
        assert info["device"] == "cuda:0", info
    else:
        assert info["device"] == "cpu" and info["name"] is None, info

    for n in (128, 512, 1024):
        med = time_matmul(n=n, device="cpu", repeats=3, warmup=1,
                            seed=CONFIG["seed"])
        assert math.isfinite(med) and med > 0, (n, med)
    timings = transfer_vs_compute(n=256, device="cpu", repeats=3,
                                    seed=CONFIG["seed"])
    assert math.isfinite(timings["compute_ms"]), timings
    assert math.isfinite(timings["transfer_ms"]), timings
    assert timings["compute_ms"] >= 0 and timings["transfer_ms"] >= 0

    g = torch.Generator().manual_seed(CONFIG["seed"])
    xb = torch.randn(8, 4, generator=g)
    yb = xb.sum(dim=1, keepdim=True) + 0.1 * torch.randn(8, 1, generator=g)

    torch.manual_seed(CONFIG["seed"])
    model = TinyMLP()
    loss_a = sgd_steps_on_batch(model, xb, yb, lr=0.01, steps=3)
    assert math.isfinite(loss_a), loss_a

    torch.manual_seed(CONFIG["seed"])
    twin = TinyMLP()
    loss_b = sgd_steps_on_batch(twin, xb, yb, lr=0.01, steps=3)
    assert loss_a == loss_b, (loss_a, loss_b)

    torch.manual_seed(CONFIG["seed"])
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = str(Path(tmp) / "smoke.pt")
        saved = save_checkpoint(model, opt, 3,
                                {"seed": CONFIG["seed"], "lr": 0.01}, ckpt)
        assert saved == ckpt and Path(ckpt).exists(), saved

        torch.manual_seed(CONFIG["seed"])
        fresh = TinyMLP()
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.01, momentum=0.9)
        restored = load_checkpoint(ckpt, fresh, fresh_opt)
        assert restored["epoch"] == 3, restored
        assert restored["config"]["seed"] == CONFIG["seed"], restored
        assert prediction_max_diff(model, fresh, xb) < 1e-9

        sgd_steps_on_batch(model, xb, yb, lr=0.01, steps=2)
        params_c = [p.detach().clone() for p in model.parameters()]
        sgd_steps_on_batch(fresh, xb, yb, lr=0.01, steps=2)
        diffs = [max_abs_diff(a, b) for a, b in zip(params_c, fresh.parameters())]
        assert all(d < 1e-9 for d in diffs), diffs

    print("smoke OK")


if __name__ == "__main__":
    run_smoke()


# %% [markdown]
# ## Conclusion (measured numbers; two environments)
#
# | Check | Result | What it does / does not prove |
# |---|---|---|
# | device report | Colab GPU: `{'torch_version': '2.11.0+cu130', 'cuda_available': True, 'device': 'cuda:0', 'name': 'Tesla T4', 'total_memory_GB': 14.56317138671875, 'capability': '7.5'}`. Local CPU: `{'torch_version': '2.14.1+cpu', 'cuda_available': False, 'device': 'cpu', 'name': None, 'total_memory_GB': None, 'capability': None}` | it gives the report of device like torch_version, cuda avilablility etc things |
# | matmul medians (3 sizes) | Colab reported one size: `2.1442494999064365` ms. Local CPU smoke, repeats=3, warmup=1, seed=11: `{128: 0.03029999788850546, 512: 0.7168000156525522, 1024: 5.031200009398162}` ms | it gives usthe GPU things and its time stats of processing things and also for the cpu how much it takes time in miliseconds and it gives us this so we could estimate whihc one be better for us |
# | compute vs transfer | Colab reported: `{'compute_ms': 3.1329899998127075, 'transfer_ms': 2.042554999661661}`. Local CPU smoke, n=256, repeats=3, seed=11: `{'compute_ms': 0.12019998393952847, 'transfer_ms': 0.10460001067258418}` | this fuction shows us how much compute is on the cpu and if we trransfer it to an gpu or just copy to cpu then how much time it will take in ms |
# | agreement tolerance | no numeric CPU/GPU agreement value was supplied; smoke parity/resume thresholds used `< 1e-9` | it gices the tolerance by this fuction max_abs_diff |
# | reload parity (fixed batch) | `0.0` on Colab and locally; smoke OK | TODO — your sentence: what this does / does not prove, in your own words |
# | controlled resume (params) | `[0.0, 0.0, 0.0, 0.0]` on Colab and locally; smoke OK | TODO — your sentence: what this does / does not prove, in your own words |
#
# ### Open question
#
# TODO — one thing that still puzzles you, in your own words.

# %% [markdown]
#
