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
# # Month 3 · Week 1 Day 8 — Independent MNIST classifier
#
# **Goal:** assemble a complete small classifier from your own utilities — inspect, split, train with a budget, select by validation, read the mistakes, and prove the saved bundle reloads bit-for-bit.
#
# ## Question and prediction
#
# *Question:* what besides the weights must be saved for a reload to reproduce predictions exactly — and which of your checks would catch each missing piece?
#
# *Prediction (write yours before running):* the reload reproduces predictions bitwise (agreement 1.0) only when config, preprocessing, and class mapping travel with the weights; the confusion matrix will show digit-pair confusions (4/9, 3/5/8 guesses) that the tiny inspection set can only hypothesize, not measure.
#
# *What would challenge it:* reload agreement below 1.0 (a missing bundle piece), a matrix whose top confusions contradict the inspection hypotheses with no pixel story, or a best-val epoch that disagrees with the printed log.
#
# ## Shapes (smoke: synthetic 4-class, d=64; full: MNIST 28×28 → 784, 10 classes)
#
# | tensor | shape | notes |
# |---|---|---|
# | MNIST image batch | (N, 1, 28, 28) | uint8 → ToTensor float32 in [0, 1] |
# | model input | (N, 784) | harness flattens before calling your MLP |
# | logits | (N, 10) | your `build_classifier` output |
# | confusion matrix | (10, 10) | rows true, columns predicted |
#
# Learner/harness boundary: data loading, splits, epoch helpers, the confusion counter, the save/load bundle, and the smoke runner are complete. Your mechanism: the classifier, the budgeted train-select, the confusion reader — plus the error notes, README conclusion, and reload verdict. Probes are for your eyes, not pytest gates.

# %%
import math
import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

CONFIG = {"mode": "full", "seed": 11}

SYN_N_TRAIN, SYN_N_VAL, SYN_D, SYN_CLASSES = 512, 128, 64, 4
FULL_HIDDEN, FULL_EPOCHS, FULL_LR, FULL_BATCH = 128, 10, 0.05, 64
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# %% [markdown]
# ## Recall (no new reading yet)
#
# 1. From Day 2: write the training-loop skeleton (batches, zero_grad, backward, step, eval branch) from memory — where does the mode flip, and who restores it?
# 2. From Day 7: which check answers "did it update?", which answers "are the pairings true?", and why can a capacity check never answer the second?

# %%
def make_synthetic(n_train: int = SYN_N_TRAIN,
                   n_val: int = SYN_N_VAL, d: int = SYN_D,
                   classes: int = SYN_CLASSES,
                   seed: int = CONFIG["seed"]):
    """Seeded synthetic fixture: no download (harness, complete).

    Takes: split sizes, width, class count, seed. Returns:
        (train_loader, val_loader, X_train, y_train, X_val,
        y_val) — float32 features, long labels, batch 64,
        deterministic shuffle, num_workers=0.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, classes, generator=g)
    X = torch.randn(n_train + n_val, d, generator=g)
    y = (X @ W).argmax(1)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=64, shuffle=True,
                              generator=g, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=256, shuffle=False,
                            num_workers=0)
    return train_loader, val_loader, X_train, y_train, X_val, y_val


def _poke_data():
    tr, va, Xtr, ytr, Xva, yva = make_synthetic()
    print("synthetic train:", tuple(Xtr.shape), "| val:",
          tuple(Xva.shape))
    print("train balance:", [(ytr == c).float().mean().item()
                             for c in range(SYN_CLASSES)])


if __name__ == "__main__":
    _poke_data()


# %%
def get_mnist(root: str = "./data"):
    """Torchvision MNIST with the recorded pipeline (harness).

    ToTensor + Normalize(0.1307, 0.3081), same transform for
        train/val/test — the val pipeline is deterministic by
        construction (no augmentation anywhere). Downloads
        once into root (ignored by Git).
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    train_full = datasets.MNIST(root, train=True, download=True,
                                transform=tfm)
    test_set = datasets.MNIST(root, train=False, download=True,
                              transform=tfm)
    return train_full, test_set


def make_mnist_splits(root: str = "./data", val_n: int = 5000,
                      batch: int = FULL_BATCH,
                      seed: int = CONFIG["seed"]):
    """55k/5k split, test sealed (harness, complete).

    Randomly splits official train 55k/5k with a seeded
        generator; the official test set is loaded but only
        returned — never evaluated here (reserved Week 4).
        Split indices persist to root/split_indices.pt.
    Returns (train_loader, val_loader, test_loader, info).
    """
    train_full, test_set = get_mnist(root)
    g = torch.Generator().manual_seed(seed)
    n = len(train_full)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[val_n:], perm[:val_n]
    torch.save({"train_idx": train_idx, "val_idx": val_idx},
               os.path.join(root, "split_indices.pt"))
    train_set = torch.utils.data.Subset(train_full, train_idx)
    val_set = torch.utils.data.Subset(train_full, val_idx)
    mk = lambda ds, sh: DataLoader(
        ds, batch_size=batch, shuffle=sh,
        generator=g if sh else None, num_workers=0)
    info = {"train_n": len(train_set), "val_n": len(val_set),
            "test_n": len(test_set), "seed": seed}
    return mk(train_set, True), mk(val_set, False), mk(test_set, False), info


def inspect_samples(loader, k: int = 24):
    """Print k images as label + pixel stats (harness).

    Prints (index, label, mean, std, min, max) per sample
        plus class counts over the inspected slice. Pixels
        tell confusion hypotheses, not measurements.
    """
    from collections import Counter

    xs, ys, n = [], [], 0
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
        n += len(xb)
        if n >= k:
            break
    X = torch.cat(xs)[:k]
    y = torch.cat(ys)[:k]
    for i in range(len(X)):
        xi = X[i]
        print(f"{i:>3} label {int(y[i])} mean {xi.mean():+.3f} "
              f"std {xi.std():.3f} min {xi.min():+.3f} "
              f"max {xi.max():+.3f}")
    print("counts:", dict(sorted(Counter(int(v) for v in y).items())))


def _full_inspect():
    tr, va, _, info = make_mnist_splits()
    print(info)
    inspect_samples(tr)


if __name__ == "__main__":
    if CONFIG["mode"] == "full":
        _full_inspect()


# %% [markdown]
# ## Step 1 — Assemble the classifier
#
# One hidden layer, your activation, no dropout/BatchNorm unless you can defend it in the conclusion. The harness always flattens `(N, 1, 28, 28)` to `(N, 784)` before calling — your module takes flat vectors and returns logits. Small cell now, wrap later: check the forward shape on the synthetic fixture first.

# %%
def build_classifier(input_dim: int = 784,
                     hidden: int = FULL_HIDDEN,
                     num_classes: int = 10) -> nn.Module:
    """Small MLP classifier (your code).

    Takes: flat input size, hidden width, class count.
    Returns: nn.Module mapping (N, input_dim) float32 to
        (N, num_classes) logits. Baseline: one hidden
        layer + your activation.
    Raises: ValueError if hidden < 1.
    """
    if hidden < 1:
      raise ValueError("hidden can't be less than 1")

    return nn.Sequential(
        nn.Linear(input_dim , hidden),
        nn.Tanh(),
        nn.Linear(hidden , num_classes)
    )


# %%
def _classifier_demo():
    _, _, Xtr, _, _, _ = make_synthetic()
    m = build_classifier(input_dim=SYN_D, hidden=32,
                         num_classes=SYN_CLASSES)
    m.eval()
    with torch.no_grad():
        out = m(Xtr[:16])
    assert out.shape == (16, SYN_CLASSES) and out.dtype == torch.float32
    assert torch.isfinite(out).all()
    print("forward:", tuple(out.shape), out.dtype,
          "finite: True")


if __name__ == "__main__":
    _classifier_demo()


# %% [markdown]
# ## Step 2 — Train under budget, keep the best
#
# The epoch helpers below are complete. Your function runs the budget, tracks validation, and keeps the `state_dict` from the lowest-val-loss epoch — selection by validation, never by train. The test loader is sealed: it never enters this function.

# %%
def run_epoch(model: nn.Module, loader: DataLoader,
              opt: torch.optim.Optimizer,
              loss_fn: nn.Module) -> float:
    """One training epoch (harness, complete)."""
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.flatten(1) if xb.dim() > 2 else xb
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
        n += len(xb)
    return tot / n


def eval_loss_acc(model: nn.Module, loader: DataLoader,
                  loss_fn: nn.Module) -> tuple[float, float]:
    """Eval pass: mean loss + accuracy (harness, complete)."""
    model.eval()
    tot, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.flatten(1) if xb.dim() > 2 else xb
            logits = model(xb)
            tot += loss_fn(logits, yb).item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)
    return tot / n, correct / n


# %%
def train_select(model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, epochs: int,
                 lr: float) -> tuple[dict, dict]:
    """Budgeted train, best-val checkpoint (your code).

    Takes: model, loaders, epoch budget, lr (SGD).
    Returns: (best_state, log) — best_state is the
        state_dict snapshot from the lowest-val-loss
        epoch; log holds train_loss/val_loss lists.
    Raises: ValueError if epochs < 1.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    train_loss =[]
    val_loss = []

    best_state = {}
    best_val = float("inf")

    opt = torch.optim.SGD(model.parameters() , lr)
    loss_fn = nn.CrossEntropyLoss()


    for i in range(epochs):
      epo_loss = run_epoch(model , train_loader ,opt , loss_fn)
      train_loss.append(epo_loss)
      eval_loss , eval_acc = eval_loss_acc(model , val_loader ,loss_fn)
      val_loss.append(eval_loss)
      if eval_loss < best_val:
        best_val = eval_loss
        best_state = {
            k: v.clone()
            for k, v in model.state_dict().items()
        }


    log = {
        "train_loss": train_loss,
        "val_loss":   val_loss
    }
    return (best_state ,log)


# %%
def _train_demo():
    tr, va, _, _, _, _ = make_synthetic()
    m = build_classifier(input_dim=SYN_D, hidden=32,
                         num_classes=SYN_CLASSES)
    best, log = train_select(m, tr, va, epochs=4, lr=0.05)
    print("train:", [f"{v:.3f}" for v in log["train_loss"]])
    print("val:  ", [f"{v:.3f}" for v in log["val_loss"]])
    print("best epoch:", min(range(len(log["val_loss"])),
                            key=lambda e: log["val_loss"][e]))


if __name__ == "__main__":
    _train_demo()


# %% [markdown]
# ## Step 3 — Read the confusion matrix
#
# The counter below is complete; reading it is yours. Rows are true labels, columns are predictions — the diagonal is agreement, everything else is a named confusion. Your function pulls the top-`k` off-diagonal cells; the *why* behind each pair belongs in your error notes, not in the function.

# %%
def confusion_matrix(model: nn.Module,
                     loader: DataLoader) -> torch.Tensor:
    """Count matrix, rows true / cols pred (harness, complete)."""
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.flatten(1) if xb.dim() > 2 else xb
            all_p.append(model(xb).argmax(1))
            all_t.append(yb)
    t = torch.cat(all_t)
    p = torch.cat(all_p)
    k = int(t.max().item()) + 1
    cm = torch.zeros(k, k, dtype=torch.long)
    for ti, pi in zip(t.tolist(), p.tolist()):
        cm[ti, pi] += 1
    return cm


# %%
def top_confusions(cm: torch.Tensor,
                   k: int = 3) -> list[tuple[int, int, int]]:
    """Top-k off-diagonal cells (your code).

    Takes: square count matrix (rows true, cols pred).
    Returns: [(true, predicted, count)] sorted by count
        descending. Mechanical extraction — the story
        behind each pair is your error analysis.
    Raises: ValueError if k < 1 or cm is not square.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if cm.dim() != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square")
    results = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:
                results.append((i, j, cm[i, j].item()))

    results.sort(key=lambda x: x[2], reverse=True)

    return results[:k]


# %%
def _confusion_demo():
    cm = torch.tensor([[5, 1, 0],
                       [0, 4, 3],
                       [2, 0, 3]])
    for true, pred, cnt in top_confusions(cm, k=2):
        print(f"true {true} -> pred {pred}: {cnt}")


if __name__ == "__main__":
    _confusion_demo()


# %% [markdown]
# ## Save, restart, reload — the reproducibility proof
#
# The bundle format is complete: config + preprocessing string + class list + weights. Your job in Colab: after the full run, restart the kernel, rebuild the model from the saved config *alone* (no notebook state), load the weights, and confirm bitwise-identical predictions on the same fixed validation examples. Agreement must read 1.0 — same strictness as Day 7's frozen check, same reason.

# %%
def save_bundle(path: str, config: dict,
                preprocessing: str, classes: list[int],
                state: dict) -> None:
    """Persist config + pipeline + weights (harness)."""
    torch.save({"config": config, "preprocessing": preprocessing,
                "classes": classes, "state": state}, path)


def load_bundle(path: str) -> dict:
    """Load a bundle (harness). Keys: config,
    preprocessing, classes, state."""
    return torch.load(path, weights_only=True)


def reload_compare(build_fn, bundle_path: str,
                   X: torch.Tensor,
                   expected: torch.Tensor) -> float:
    """Rebuild-from-config agreement (harness, complete).

    Takes: zero-arg model factory, bundle path, fixed
        input batch, and the predictions recorded before
        the restart. Loads the bundle, builds a fresh
        model from nothing but the factory, loads weights,
        and returns the fraction of predictions identical
        to expected. 1.0 means bitwise reproduction.
    """
    bundle = load_bundle(bundle_path)
    fresh = build_fn()
    fresh.load_state_dict(bundle["state"])
    fresh.eval()
    Xf = X.flatten(1) if X.dim() > 2 else X
    with torch.no_grad():
        got = fresh(Xf).argmax(1)
    return (got == expected).float().mean().item()


# %%
def _reload_demo():
    _, va, _, _, Xva, _ = make_synthetic()
    m = build_classifier(input_dim=SYN_D, hidden=32,
                         num_classes=SYN_CLASSES)
    m.eval()
    with torch.no_grad():
        expected = m(Xva[:64]).argmax(1)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        save_bundle(path, {"hidden": 32},
                    "synthetic/identity", list(range(SYN_CLASSES)),
                    m.state_dict())
        agree = reload_compare(
            lambda: build_classifier(
                input_dim=SYN_D, hidden=32,
                num_classes=SYN_CLASSES), path, Xva[:64],
            expected)
    finally:
        os.unlink(path)
    print(f"reload agreement: {agree:.4f}")
    assert agree == 1.0


if __name__ == "__main__":
    _reload_demo()


# %% [markdown]
# ## Full run in Colab (mode == "full" only)
#
# Flip `CONFIG["mode"]` to `"full"` and Run All: inspect the data, train the baseline under budget, print the matrix and ten mistakes, save the bundle. Then **restart the kernel**, rebuild from the saved config alone, reload, and verify — the two cells after this one are your manual reload script. Quote every number into the conclusion.

# %%
def print_mistakes(model: nn.Module, loader: DataLoader,
                   k: int = 10) -> list[tuple[int, int, int]]:
    """First-k mistakes as (index, true, pred) (harness).

    Prints each with its flat index into the loader
        order. Looking at the pixels is your job — this
        only points at where to look.
    """
    model.eval()
    out, idx = [], 0
    with torch.no_grad():
        for xb, yb in loader:
            xf = xb.flatten(1) if xb.dim() > 2 else xb
            pred = model(xf).argmax(1)
            for t, p in zip(yb.tolist(), pred.tolist()):
                if t != p:
                    out.append((idx, t, p))
                    if len(out) >= k:
                        for i, t2, p2 in out:
                            print(f"idx {i}: true {t2} "
                                  f"-> pred {p2}")
                        return out
                idx += 1
    for i, t2, p2 in out:
        print(f"idx {i}: true {t2} -> pred {p2}")
    return out


def _full_train():
    tr, va, _, info = make_mnist_splits()
    print(info)
    inspect_samples(tr)
    m = build_classifier()
    best, log = train_select(m, tr, va, epochs=FULL_EPOCHS,
                             lr=FULL_LR)
    m.load_state_dict(best)
    vl, vacc = eval_loss_acc(m, va, nn.CrossEntropyLoss())
    print(f"best val: loss {vl:.4f} acc {vacc:.4f}")
    cfg = {"input_dim": 784, "hidden": FULL_HIDDEN,
           "num_classes": 10, "epochs": FULL_EPOCHS,
           "lr": FULL_LR, "batch": FULL_BATCH,
           "seed": CONFIG["seed"],
           "preprocessing": "ToTensor+Normalize(0.1307,0.3081)",
           "classes": list(range(10))}
    save_bundle("./day8_bundle.pt", cfg, cfg["preprocessing"],
                cfg["classes"], best)
    m.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in va:
            preds.append(m(xb.flatten(1)).argmax(1))
    torch.save(torch.cat(preds), "./day8_val_preds.pt")
    print("saved ./day8_bundle.pt + ./day8_val_preds.pt "
          "(local only, never commit)")


def _full_evaluate():
    _, va, _, _ = make_mnist_splits()
    bundle = load_bundle("./day8_bundle.pt")
    expected = torch.load("./day8_val_preds.pt",
                          weights_only=True)
    print("config:", bundle["config"])
    m = build_classifier(**{k: bundle["config"][k]
                             for k in ("input_dim", "hidden",
                                       "num_classes")})
    m.load_state_dict(bundle["state"])
    cm = confusion_matrix(m, va)
    print("val errors:", int(cm.sum() - cm.diag().sum()),
          "of", int(cm.sum()))
    for true, pred, cnt in top_confusions(cm, k=5):
        print(f"true {true} -> pred {pred}: {cnt}")
    print_mistakes(m, va)
    first_xb = next(iter(va))[0]
    agree = reload_compare(
        lambda: build_classifier(
            **{k2: bundle["config"][k2]
               for k2 in ("input_dim", "hidden",
                         "num_classes")}),
        "./day8_bundle.pt", first_xb,
        expected[:len(first_xb)])
    print(f"reload agreement: {agree:.4f}")
    assert agree == 1.0, "reload must reproduce predictions bitwise"


if __name__ == "__main__":
    if CONFIG["mode"] == "full":
        _full_train()
        _full_evaluate()


# %% [markdown]
# ## Conclusion — README (fill in)
#
# 1. Data. Your 24-sample slice was missing digit 8 entirely (counts jump 7→9) — that alone proves the point: 24 examples can't cover 10 classes, so inspection gives hypotheses, never measurements. If you hypothesized 4/9 or 3/5 before the matrix, say so and check: your matrix says 9→7 (9 cases), 5→3 (8), 4→9 (6) — the 9/4/7 loop-digit family dominates, 3/5 didn't make top-5. A contradicted hypothesis is information; write which of yours survived.
#
# 2. Model + run. One hidden layer (128, tanh), 10 epochs SGD lr 0.05, best-val checkpoint: val loss 0.0900, acc 0.9738. Recorded, not chased — no extra epochs to beat it.
#
# 3. Two error patterns (pixel stories, from your tens):
# - Loop digits (9→7 ×9, 4→9 ×6): 9, 4, 7 share straight-stroke + loop/cross fragments; a thin or broken loop turns a 9 into a 7's diagonal in the model's eyes. Check idx 502 (9→7) pixels yourself.
# - Curved-top digits (5→3 ×8, 8→3 ×6, 3→7 ×4): 3/5/8 share the double-curve silhouette; when the middle notch is faint, they collapse into each other. Idx 250 (5→3) and 305 (8→2) are your specimens.
# - Why no human error rate: 30 images is an anecdote, not a sample — no confidence interval, rater bias, and your mistakes are model-specific (a human never confuses idx 343's 3→2). Benchmarks need hundreds of rated images and an adjudication rule.
#
# 4. Reload verdict. Agreement 1.0000 after rebuild-from-config. Required besides weights: the config (architecture), the preprocessing string (normalize constants — shift these and every pixel moves), the class list (label meaning). Each missing piece has a catcher: wrong config → load/shape error, wrong preprocessing → accuracy collapse with no error, wrong classes → silently relabeled outputs.
#
# 5. Limits, one sentence each:
# - Forward-shape check: proves plumbing (dims, dtype, finiteness), says nothing about learned quality — a random init passes it.
# - Best-val selection: proves the checkpoint generalizes best within this run, says nothing about whether the run itself was good (seed, budget, data fixed).
# - Top-k reader: proves which pairs confuse most, says nothing about why — the pixel story is yours, not the function's.
# - Reload agreement: proves the bundle reproduces exactly, says nothing about correctness — a perfectly reloaded bad model is still bad.
#

# %%
def run_smoke():
    """Smoke evidence for Day 8 (harness, complete).

    Synthetic fixture only: classifier forward shape,
        budgeted best-val selection, confusion reader hand
        values, save/rebuild/reload bitwise agreement.
        Raises AssertionError on any failure.
    """
    tr, va, Xtr, _, Xva, _ = make_synthetic()
    m = build_classifier(input_dim=SYN_D, hidden=32,
                         num_classes=SYN_CLASSES)
    m.eval()
    with torch.no_grad():
        out = m(Xtr[:8])
    assert out.shape == (8, SYN_CLASSES)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    print("forward: shape, dtype, finite hold")

    m2 = build_classifier(input_dim=SYN_D, hidden=32,
                          num_classes=SYN_CLASSES)
    best, log = train_select(m2, tr, va, epochs=4, lr=0.05)
    assert len(log["train_loss"]) == 4 and len(log["val_loss"]) == 4
    assert all(math.isfinite(v) for v in
               log["train_loss"] + log["val_loss"])
    argmin = min(range(4), key=lambda e: log["val_loss"][e])
    m2.load_state_dict(best)
    l_best, _ = eval_loss_acc(m2, va, nn.CrossEntropyLoss())
    assert math.isclose(l_best, log["val_loss"][argmin],
                        rel_tol=1e-6)
    print(f"train_select: best epoch {argmin}, losses finite")

    toy = torch.tensor([[5, 1, 0],
                        [0, 4, 3],
                        [2, 0, 3]])
    assert top_confusions(toy, k=2) == [(1, 2, 3), (2, 0, 2)]
    print("top_confusions: hand values match")

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        save_bundle(path, {"hidden": 32}, "synthetic/identity",
                    list(range(SYN_CLASSES)), m.state_dict())
        m.eval()
        with torch.no_grad():
            expected = m(Xva[:64]).argmax(1)
        agree = reload_compare(
            lambda: build_classifier(
                input_dim=SYN_D, hidden=32,
                num_classes=SYN_CLASSES), path, Xva[:64],
            expected)
    finally:
        os.unlink(path)
    assert agree == 1.0
    print("reload: agreement 1.0, bitwise")

    print("run_smoke OK: forward exact, selection exact, "
          "reader exact, reload exact")


if __name__ == "__main__":
    run_smoke()

# %%
