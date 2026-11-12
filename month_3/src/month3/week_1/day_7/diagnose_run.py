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
#     name: python3
# ---

# %% [markdown]
# # Month 3 · Week 1 Day 7 — Diagnose a real training run
#
# **Goal:** use evidence to tell implementation bugs apart from ordinary optimization/generalization behaviour — one honest ordinary run plus two injected bugs, each closed with a written bug report.
#
# ## Question and prediction
#
# *Question:* which logged signal tells "the optimizer never moved the weights" apart from "the data is broken", and which checks stay silent on each bug?
#
# *Prediction (write yours before running):* the missing update shows flat losses **and** bitwise-identical snapshots; misaligned labels show train accuracy climbing toward memorization while validation sits near chance, with spot-checked pairs looking wrong. The tiny-batch memorization demo will succeed — and that success is exactly why it can't catch every data bug.
#
# *What would challenge it:* snapshots that differ with no optimizer step (nondeterminism sneaking in), a misaligned branch whose validation is *not* near chance, or a curve summary whose flags disagree with the printed log.
#
# ## Shapes (smoke: n_train=320, n_val=80, d=8, hidden=16, 2 classes)
#
# | tensor | shape | notes |
# |---|---|---|
# | X_train / y_train | (320, 8) / (320,) | float32 features, long labels |
# | X_val / y_val | (80, 8) / (80,) | held-out split, same recipe |
# | logits | (B, 2) | small MLP, cross-entropy |
# | log["train_loss"] | (epochs,) | per-epoch means, plain floats |
#
# Learner/harness boundary: the training apparatus below is complete — data, model, loop, both bug branches, the memorization demo, the smoke runner. Your mechanism: three small diagnostic functions plus the written bug reports. Probes are for your eyes, not pytest gates.

# %%
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG = {"mode": "smoke", "seed": 11}

N_TRAIN, N_VAL, D, HIDDEN, N_CLASSES = 320, 80, 8, 16, 2
EPOCHS, LR = 15, 0.05


# %% [markdown]
# ## Recall (no new reading yet)
#
# 1. From Day 5: why do two training examples share gradient information through training-mode BatchNorm — which statistics carry the dependency, and why does it vanish in eval mode?
# 2. From Day 4: pick two of your histogram plots and explain each one's shape without reading its caption — what would the same plot look like if the layer it measures were frozen?

# %%
def make_data(n_train: int = N_TRAIN, n_val: int = N_VAL,
              d: int = D, seed: int = CONFIG["seed"]):
    """Seeded tabular binary task + train/val split (harness, complete).

    Takes: split sizes, width, seed. Returns: (train_loader,
        val_loader, X_train, y_train, X_val, y_val) — float32
        features, long labels, batch 32, train shuffled
        deterministically, num_workers=0.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val, d, generator=g)
    y = ((X[:, 0] + 0.5 * X[:, 1]
          + 0.1 * torch.randn(n_train + n_val, generator=g)) > 0).long()
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=32, shuffle=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=64, shuffle=False, num_workers=0)
    return train_loader, val_loader, X_train, y_train, X_val, y_val


def _poke_data():
    train_loader, val_loader, X_train, y_train, X_val, y_val = make_data()
    print("train:", tuple(X_train.shape), y_train.shape, "| val:",
          tuple(X_val.shape), y_val.shape)
    print("train balance:", y_train.float().mean().item(),
          "| val balance:", y_val.float().mean().item())


if __name__ == "__main__":
    _poke_data()


# %%
def build_model(seed: int = CONFIG["seed"]) -> nn.Module:
    """Small MLP classifier (harness, complete)."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D, HIDDEN), nn.Tanh(),
                         nn.Linear(HIDDEN, N_CLASSES))


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Cloned param dict for before/after comparison (harness)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_training(model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, epochs: int = EPOCHS,
                 lr: float = LR, optimizer: str = "sgd") -> dict[str, list]:
    """Full train loop with per-epoch logging (harness, complete).

    Returns: log with train_loss, val_loss, train_acc, val_acc,
        grad_norm (mean per-sample batch grad norm while
        training, 0.0 for eval rows) and lr, each one float
        per epoch. Train rows measured in train mode, val
        rows in eval mode. optimizer is "sgd" or "adam".
    """
    loss_fn = nn.CrossEntropyLoss()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    log: dict[str, list] = {"train_loss": [], "val_loss": [],
                            "train_acc": [], "val_acc": [],
                            "grad_norm": [], "lr": []}
    for _ in range(epochs):
        model.train()
        tot, correct, n, gn = 0.0, 0, 0, 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            gn += sum(p.grad.norm().item() ** 2
                      for p in model.parameters()) ** 0.5 * len(xb)
            opt.step()
            tot += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)
        log["train_loss"].append(tot / n)
        log["train_acc"].append(correct / n)
        log["grad_norm"].append(gn / n)
        model.eval()
        vtot, vcorrect, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                vtot += loss_fn(logits, yb).item() * len(xb)
                vcorrect += (logits.argmax(1) == yb).sum().item()
                vn += len(xb)
        log["val_loss"].append(vtot / vn)
        log["val_acc"].append(vcorrect / vn)
        log["lr"].append(lr)
    return log


def _poke_model():
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print("params:", n_params)


if __name__ == "__main__":
    _poke_model()


# %% [markdown]
# ## Ordinary run (describe, don't diagnose)
#
# Run it, read the printed table, and write 3–4 honest sentences in the conclusion cell: where loss falls fast vs flat, what validation does relative to train, what the gradient norm does. No bug claimed per fluctuation — an ordinary curve is the reference everything else is compared against.

# %%
def _ordinary_run():
    tr, va, _, _, _, _ = make_data()
    m = build_model()
    log = run_training(m, tr, va)
    print(f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} "
          f"{'train_acc':>9} {'val_acc':>8} {'grad_norm':>9}")
    for e in range(len(log["train_loss"])):
        print(f"{e:>5} {log['train_loss'][e]:>10.4f} "
              f"{log['val_loss'][e]:>9.4f} "
              f"{log['train_acc'][e]:>9.3f} "
              f"{log['val_acc'][e]:>8.3f} "
              f"{log['grad_norm'][e]:>9.4f}")


if __name__ == "__main__":
    _ordinary_run()


# %% [markdown]
# ## Step 1 — Frozen-update check
#
# If the optimizer step never runs, no parameter can change — not approximately, *exactly*. So the right contract here is bitwise equality (`torch.equal` per tensor), not allclose: allclose would also pass, but it answers a question nobody asked and hides the point. Your function takes two snapshot dicts and returns a plain `bool`.

# %%
def params_unchanged(before: dict[str, torch.Tensor],
                     after: dict[str, torch.Tensor]) -> bool:
    """True iff every param tensor is bitwise identical (your code).

    Takes: two snapshot dicts from snapshot_params.
    Returns: plain bool — True when nothing moved.
    Raises: ValueError if the key sets differ.
    """
    if set(before) != set(after):
        raise ValueError(...)
    return bool(all(torch.equal(before[k], after[k]) for k in before))



# %%
def branch_no_update(steps: int = 3, lr: float = LR):
    """Bug branch A: forward + backward, optimizer step omitted.

    Harness, complete. Returns (before, after, losses) — the
    snapshots must read unchanged, the losses must read flat.
    """
    m = build_model()
    tr, _, _, _, _, _ = make_data()
    loss_fn = nn.CrossEntropyLoss()
    xb, yb = next(iter(tr))
    before = snapshot_params(m)
    losses = []
    m.train()
    for _ in range(steps):
        m.zero_grad()
        loss = loss_fn(m(xb), yb)
        loss.backward()
        losses.append(loss.item())
    return before, snapshot_params(m), losses


def branch_repaired(steps: int = 3, lr: float = LR):
    """Same as branch A but WITH the optimizer step (harness).

    Returns (before, after): the snapshots must read changed.
    """
    m = build_model()
    tr, _, _, _, _, _ = make_data()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    xb, yb = next(iter(tr))
    before = snapshot_params(m)
    m.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(m(xb), yb)
        loss.backward()
        opt.step()
    return before, snapshot_params(m)


def _branch_a():
    frozen_before, frozen_after, frozen_losses = branch_no_update()
    print("frozen losses:", [f"{v:.4f}" for v in frozen_losses])
    assert params_unchanged(frozen_before, frozen_after) is True
    fixed_before, fixed_after = branch_repaired()
    assert params_unchanged(fixed_before, fixed_after) is False
    print("branch A verdicts hold: frozen True, repaired False")


if __name__ == "__main__":
    _branch_a()


# %% [markdown]
# ## Bug report A (fill in — symptom, evidence, cause, fix, verification)
#
# 1. **Symptom:** what did the branch-A loss trace look like, and what did the printed table show?
# the losses are frozen and shows this ['0.7590', '0.7590', '0.7590']
# 2. **Evidence:** which check did you run, on what, and what did it return?
# params_unchanged(frozen_before, frozen_after) returned True — the snapshots are bitwise identical, not just close. And that's the point of the strict contract: flat losses alone could also be a logging bug, but exactly-unchanged weights mean no update ran, period.
#
# 3. **Cause:** which single missing line produces exactly this evidence?
# its the opt.step() we cacluate the grads but didn't do the optimizer step so thats why it leaves us with this
# 4. **Fix + verification:** what changed, and which two asserts now confirm the repair?
# Fix: add the optimizer step (opt.step() after backward(), with zero_grad() at the top — that's what branch_repaired does). Verification: params_unchanged returns True on the frozen pair and False on the repaired pair — both asserts in _branch_a pass, and the repaired losses actually move instead of printing 0.7590 three times.

# %% [markdown]
# ## Step 2 — Alignment spot check
#
# A shuffled label column trains fine — the loop never complains — but the pairs are lies. No statistic needed: pull indexed `(row, label)` pairs and look at them. Your function returns plain data (index, feature list, label) in the requested order so the caller, not the function, does the judging.

# %%
def alignment_report(X: torch.Tensor, y: torch.Tensor,
                     indices: list[int]) -> list[tuple[int, list[float], int]]:
    """Indexed (row, label) pairs for eyeball inspection (your code).

    Takes: features (N, d), labels (N,), row indices.
    Returns: [(index, feature row as plain list, label)] in order.
    Raises: ValueError on length mismatch or out-of-range index.
    """
    if len(X) != len(y):
      raise ValueError("the length is mismatch")

    output = []
    for i in indices:
      if not 0 <= i < len(X):
       raise ValueError(f"index out of range: {i}")

      output.append((i, X[i].tolist(), int(y[i].item())))

    return output


# %%
def branch_misaligned(epochs: int = 30, lr: float = 0.05,
                      perm_seed: int = 12):
    """Bug branch B: labels permuted, inputs untouched (harness).

    Trains against wrong targets with Adam — Adam is chosen
        deliberately here because SGD barely moves on shuffled
        targets inside a short-epoch budget, and the branch
        must exhibit the memorize-vs-chance signature. The
        ordinary run keeps SGD; the comparison to beat is its
        printed curve, not the optimizer name.
    Returns (log, X_train, y_shuffled): watch train accuracy
        climb toward memorization while validation sits near
        chance.
    """
    g = torch.Generator().manual_seed(perm_seed)
    tr, va, Xtr, ytr, _, _ = make_data()
    perm = torch.randperm(len(ytr), generator=g)
    y_bad = ytr[perm]
    bad_loader = DataLoader(TensorDataset(Xtr, y_bad),
                            batch_size=32, shuffle=True,
                            generator=g, num_workers=0)
    m = build_model()
    return run_training(m, bad_loader, va, epochs=epochs,
                        lr=lr, optimizer="adam"), Xtr, y_bad


def _branch_b():
    bad_log, bad_X, bad_y = branch_misaligned()
    print("misaligned: final train_acc "
          f"{bad_log['train_acc'][-1]:.3f}, val_acc {bad_log['val_acc'][-1]:.3f}")
    SPOT = [0, 1, 2, 3, 4, 5]
    for i, row, lab in alignment_report(bad_X, bad_y, SPOT):
        print(i, [f"{v:.2f}" for v in row[:4]], "... label", lab)


if __name__ == "__main__":
    _branch_b()


# %% [markdown]
# ## Bug report B (fill in — symptom, evidence, cause, fix, verification)
#
#
# 1. Symptom. Branch B ends at train_acc 0.884 against val_acc 0.537 — a 0.347 gap. The gap alone isn't the diagnosis; the levels are: train climbs toward memorization while validation sits pinned at chance (≈0.5 on this binary task). Ordinary overfitting leaves validation above chance; chance-level validation with climbing train means nothing generalizable was learned. The spot pairs (rows 0–5) print inputs next to labels the shuffle detached from them — e.g. row 2 [0.25, 1.57, -1.64, -1.59, …] → label 1 — showing the inspection method, row by row.
#
# 2. Evidence. Three quoted facts: (a) the printed spot rows 0–5 with their labels; (b) the 0.884/0.537 train/val split; (c) curve_summary on the branch-B log fires train_falling=True (train loss descends — the optimizer works) and val_rising=True (val loss climbs while train falls). Learning happens; generalization doesn't.
#
# 3. Cause. A single operation: y_bad = ytr[perm] where perm = torch.randperm(len(ytr)). Inputs untouched, only the label column permuted. The loop stays silent because everything it checks still holds — shapes (320, 8)/(320,), long dtype, well-defined cross-entropy, falling loss. Shuffling preserves every marginal statistic and destroys only the pairing, and no shape, dtype, or loss check inspects pairings. That invisibility is exactly why the spot-pair check exists.
#
# 4. Fix + verification. Train against the true ytr instead of y_bad (i.e., the ordinary run — the fix is deleting the permutation, not adding machinery). Confirmation: ordinary run ends train_acc 0.941 / val_acc 0.938, gap 0.003 instead of 0.347, validation tracking train instead of sitting at chance. The signature collapses back to a healthy curve.
#
# Then one honest sentence: the demo below memorizes a fixed tiny batch with shuffled labels — so why can't a memorize-a-batch check catch every data bug?

# %%
def memorization_demo(X_all: torch.Tensor, y_all: torch.Tensor,
                      samples: int = 16, epochs: int = 300,
                      lr: float = 0.05):
    """Fixed tiny batch, shuffled labels, trained to fit (harness).

    Prints final train accuracy: a sufficiently expressive
        model memorizes even random targets. The check that
        passes here and still misses branch B is the lesson.
    """
    g = torch.Generator().manual_seed(99)
    Xd = X_all[:samples]
    yd = y_all[:samples][torch.randperm(samples, generator=g)]
    m = build_model()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(m(Xd), yd)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (m(Xd).argmax(1) == yd).float().mean().item()
    print(f"tiny-batch shuffled-label fit: train_acc {acc:.3f}")
    return acc


def _run_demo():
    _, _, Xtr, ytr, _, _ = make_data()
    return memorization_demo(Xtr, ytr)


if __name__ == "__main__":
    _run_demo()


# %% [markdown]
# ## Step 3 — Curve summary (extraction only, no verdicts)
#
# Pull the mechanical features out of a log dict: how many epochs, first/last train loss, best validation loss and its (0-based) epoch, the final train-val gap, and two trend flags — `val_rising` (mean of the last up-to-3 val losses above the mean of the first up-to-3) and `train_falling` (last train loss below the first). Needs at least 3 epochs. Interpretation stays in your bug reports.

# %%
def curve_summary(log):
  if "train_loss" not in log or "val_loss" not in log:
    raise ValueError("log must contain train_loss and val_loss")

  tl = log["train_loss"]
  vl = log["val_loss"]

  if len(tl) < 3 or len(vl) < 3:
    raise ValueError("need at least 3 epochs")

  epochs = len(tl)
  first_train_loss = tl[0]
  last_train_loss = tl[-1]
  best_val_loss = min(vl)
  best_val_epoch = vl.index(best_val_loss)
  final_gap = vl[-1] - tl[-1]
  val_rising = (sum(vl[-3:]) / 3.0) > (sum(vl[:3]) / 3.0)
  train_falling = tl[-1] < tl[0]

  return {
      "epochs": epochs,
      "first_train_loss": first_train_loss,
      "last_train_loss": last_train_loss,
      "best_val_loss": best_val_loss,
      "best_val_epoch": best_val_epoch,
      "final_gap": final_gap,
      "val_rising": val_rising,
      "train_falling": train_falling,
  }



# %%
def run_smoke():
    """Smoke evidence for Day 7 (harness, complete).

    Short ordinary run finishes finite and well-shaped; the
        frozen branch reads unchanged and a perturbed copy
        reads changed; the misaligned branch memorizes train
        while validation sits near chance; spot pairs are exact
        on the fixture; the summary matches hand values with
        both trend flags set. Raises AssertionError on failure.
    """
    tr_loader, va_loader, Xtr, ytr, Xva, yva = make_data()
    assert Xtr.shape == (N_TRAIN, D) and Xva.shape == (N_VAL, D)
    m = build_model()
    log = run_training(m, tr_loader, va_loader, epochs=5)
    for k in ("train_loss", "val_loss", "train_acc", "val_acc",
              "grad_norm", "lr"):
        assert len(log[k]) == 5 and all(math.isfinite(v) for v in log[k]), k
    print("ordinary 5-epoch run: finite, shaped "
          f"(train {log['train_loss'][0]:.3f} -> {log['train_loss'][-1]:.3f}, "
          f"val {log['val_loss'][0]:.3f} -> {log['val_loss'][-1]:.3f})")

    before, after, _ = branch_no_update()
    assert params_unchanged(before, after) is True
    moved = {k: v.clone() for k, v in before.items()}
    first = next(iter(moved))
    flat = moved[first].clone()
    flat.flatten()[0] += 1.0
    moved[first] = flat
    assert params_unchanged(before, moved) is False
    print("frozen check: frozen True, perturbed False")

    bad_log, _, _ = branch_misaligned()
    assert bad_log["train_acc"][-1] > 0.85, bad_log["train_acc"][-1]
    assert bad_log["val_acc"][-1] < 0.65, bad_log["val_acc"][-1]
    print(f"misaligned signature: train_acc "
          f"{bad_log['train_acc'][-1]:.3f} vs val_acc "
          f"{bad_log['val_acc'][-1]:.3f}")

    Xf = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    yf = torch.tensor([0, 1])
    assert alignment_report(Xf, yf, [1, 0]) == [(1, [3.0, 4.0], 1),
                                                (0, [1.0, 2.0], 0)]
    print("spot pairs: exact on fixture")

    toy = {"train_loss": [0.9, 0.7, 0.6, 0.55],
           "val_loss": [0.85, 0.8, 0.82, 0.9]}
    s = curve_summary(toy)
    assert s["epochs"] == 4
    assert math.isclose(s["first_train_loss"], 0.9)
    assert math.isclose(s["last_train_loss"], 0.55)
    assert math.isclose(s["best_val_loss"], 0.8)
    assert s["best_val_epoch"] == 1
    assert math.isclose(s["final_gap"], 0.35)
    assert s["val_rising"] is True and s["train_falling"] is True
    print("curve summary: hand values match, both flags set")

    print("run_smoke OK: log sane, frozen logic exact, "
          "branch-B signature holds, pairs exact, summary exact")


if __name__ == "__main__":
    run_smoke()

# %% [markdown]
# ## Conclusion (fill in)
#
# 1. **Ordinary run:** 3–4 honest sentences on the printed curve (fast fall vs flat stretches, validation vs train, grad norm) — no bug claimed per fluctuation.
# ans = the train loss fall from 0.7010 to 0.2157 and the validation loss fall from the 0.6182 to 0.2041 and the starting differrance beternn the train loss and val loss is 0.083  which till end of the last epoch be 0.012 much lower and so ass the train acc diff in start is -0.116 but as we come to the finish final epoch that remains to 0.003 as so the grad norms too drop from 0.4755 to 0.2949
# 2. Plan questions:
#
# (a) Bitwise snapshot equality. Flat losses are ambiguous — a broken logger prints flat lines too. params_unchanged returning True means no update ran (no learning); weights changed + flat log means the update runs and the lie is in the logging.
#
# (b) First confirm it's real, not noise: multiple epochs of sustained rise, not one wobble. Then split the suspects — read the curve summary flags, check whether validation was measured in eval mode, inspect spot pairs for label errors, and only after the data checks pass consider genuine overfitting (where you'd compare against a regularized rerun under the same budget, not just declare it).
#
# (c) Because the rest of the network can compensate — gradients still flow through unfrozen layers, which can adjust around the frozen one, especially in an overparameterized MLP on an easy task. Freezing removes capacity and adaptability in one layer; it doesn't stop learning unless that layer was the only path (it never is in a dense net). Also, "fail" is vague: frozen layers slow or skew training far more often than they flatline it.
#
# (d) Indexed spot pairs (alignment_report on rows 0–5). The tiny-batch check passed at 1.000 on this exact architecture — capacity was never the problem. Only looking at actual row↔label pairings exposes a shuffled label column, because shapes, dtypes, and loss all stay valid under permutation.
#
# 3. Limits (one sentence each):
#
# - Frozen equality: proves whether an update ran, but says nothing about whether a ran update was correct (wrong loss, wrong data, and bad hyperparameters all move weights just fine).
# - Spot pairs: expose pairing lies in the rows you inspect, but prove nothing about the rows you didn't look at — six clean rows don't certify 320.
# - Curve summary: flags trend shapes (rising val, falling train) mechanically, but can't tell a data bug from genuine overfitting — the same flags fire for both, so the verdict always needs the pair-level evidence beside it.
#

# %%
