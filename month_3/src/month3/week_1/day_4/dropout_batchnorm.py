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
# # Month 3 · Week 1 Day 4 — Dropout, BatchNorm and useful histograms
#
# **Question.** Can you isolate regularization from normalization from initialization — and see each one in a histogram instead of a single accuracy number?
#
# **Prediction (write yours before running).** Say which of the four arms you expect to win validation accuracy and why; what a saturated tanh histogram looks like versus a healthy one; and why `model.eval()` should matter even when gradients are already disabled.
#
# **Library / learner boundary.** Use `make_classification`, `StandardScaler`, `Linear`/`BatchNorm1d`/`Dropout`/`Tanh`, cross-entropy, full-batch SGD and `matplotlib` figures as routine work. Your work: the preprocessing contract, the four-arm isolation design, hidden-state statistics, the saturation experiment, the single-sample BatchNorm failure and rescue, and every explanation.
#
# **Shapes, dtypes, budgets.** Tabular `float32` features, `long` class labels, batch axis first. Smoke mode only: a few hundred rows, two-to-four hidden layers, tens of epochs full-batch. The four short runs are the budget; no search.

# %%
"""Day 4 workspace: Dropout, BatchNorm and diagnostic histograms (stubs for the learner)."""

import math

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONFIG = {"mode": "smoke", "seed": 11}


# %%
def make_tabular_data(n_train=400, n_val=100, n_features=12, n_informative=6, n_classes=3, seed=11):
    """Build a fixed train/val tabular classification fixture with train-fitted scaling.

    Takes split sizes, feature counts and a seed; returns a dict with ``X_train``,
    ``y_train``, ``X_val``, ``y_val`` as torch tensors (``float32`` features shaped
    ``(n, n_features)``, ``long`` labels shaped ``(n,)``). Classes are generated
    from ``seed``; standardization is fit on the train split only and applied to
    both splits.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if any size/count is < 1, ``n_informative > n_features``
        or ``n_val >= n_train``.
    """
    if n_train < 1:
        raise ValueError("n_train must be at least 1")

    if n_val < 1:
        raise ValueError("n_val must be at least 1")

    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    if n_classes < 1:
        raise ValueError("n_classes must be at least 1")

    if n_informative < 1:
        raise ValueError("n_informative must be at least 1")

    if n_informative > n_features:
        raise ValueError("n_informative cannot exceed n_features")

    if n_val >= n_train:
        raise ValueError("n_val must be smaller than n_train")

    X_np, y_np = make_classification(
        n_samples= n_train+n_val,        # n_train + n_val: generate the pool first, split after
        n_features=n_features,
        n_informative=n_informative,      # 6 columns carry signal, 6 are pure noise
        n_classes=n_classes,
        random_state=seed,    # same seed → same table every run
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_np, y_np,
        test_size=n_val,        # n_val as an exact count; the rest (400) stays train
        random_state=seed,
        stratify=y_np,        # keeps the 0/1/2 proportions equal in both splits
    )

    scaler = StandardScaler().fit(X_tr)   # mean/std learned from TRAIN only
    X_tr_s = scaler.transform(X_tr)       # train gets centered to mean ≈ 0
    X_va_s = scaler.transform(X_va)       # val uses train's ruler, never its own

    return {
        "X_train" : torch.tensor(X_tr_s, dtype=torch.float32),
        "y_train" : torch.tensor(y_tr , dtype=torch.int64),
        "X_val" : torch.tensor(X_va_s , dtype=torch.float32),
        "y_val" : torch.tensor(y_va , dtype=torch.int64)
    }


# %%
def build_mlp(input_dim, hidden=(32, 32), n_classes=3, p_dropout=0.0, use_batchnorm=False, seed=11):
    """Build one MLP variant: Linear -> [BatchNorm1d] -> Tanh -> [Dropout] per layer.

    Takes the input width, hidden widths, class count, dropout probability, a
    BatchNorm on/off flag and a seed; returns an ``nn.Sequential`` mapping
    ``(B, input_dim)`` to ``(B, n_classes)`` logits. Identical arguments always
    build identical initial weights (seeded initialization).
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``input_dim < 1``, ``hidden`` is empty, any width < 1,
        ``p_dropout`` is outside ``[0, 1)`` or ``seed < 0``.
    """

    if input_dim < 1:
      raise ValueError("input_dim can't be less than 1")

    if not hidden or len(hidden) == 0:
      raise ValueError("hidden is empty")

    for h in hidden:              # the "any width < 1" rule: check each layer
      if h < 1:
          raise ValueError(f" the width of {h} hidden layer is less than 1 ")

    if p_dropout < 0 or p_dropout >= 1:
      raise ValueError("p_dropout is outside [0, 1)")

    if seed < 0:
      raise ValueError("seed can't be less than 0")

    torch.manual_seed(seed)

    lyrs = []
    crinp = input_dim
    for h in hidden:
      lyrs.append(nn.Linear(crinp,h))
      if use_batchnorm == True:
        lyrs.append(nn.BatchNorm1d(h))
      lyrs.append(nn.Tanh())
      if p_dropout > 0:
        lyrs.append(nn.Dropout(p_dropout))
      crinp = h

    lyrs.append(nn.Linear(crinp , n_classes))

    return nn.Sequential(*lyrs)


# %%

# %%
def train_classifier(model:nn.Module, X_train, y_train, X_val, y_val, lr=0.05, epochs=20):
    """Train with fixed-budget full-batch SGD; return the per-epoch history.

    Takes a model, fixed train/val tensors and an update budget; returns
    ``{"train_loss": [...], "val_acc": [...]}`` with one entry per epoch.
    Full-batch SGD, no shuffling, cross-entropy loss: deterministic for a given
    seeded rebuild. The recorded train loss is dropout-active; validation uses
    evaluation mode under ``no_grad``.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``lr <= 0`` or ``epochs < 1``.
    """
    if lr <= 0:
      raise ValueError("learning rate (lr) can't be less than or equal to 0")

    if epochs < 1:
      raise ValueError("epochs can't be less than 1")


    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for _ in range(epochs):

      ## train
      model.train()
      model.zero_grad()
      pred_y = model(X_train)
      loss = criterion(pred_y,y_train)

      # backward
      loss.backward()

      optimizer.step()                          # apply the gradients — the line that actually learns
      history["train_loss"].append(loss.item()) # plain float, not a tensor


      ## val loss
      with torch.no_grad():
        model.eval()
        logits = model(X_val)
        pred = logits.argmax(dim=1)                    # winning class per row
        acc = (pred == y_val).float().mean().item()    # fraction correct
        history["val_acc"].append(acc)



    return history


# %%
def run_four_way(X_train, y_train, X_val, y_val, input_dim, hidden=(32, 32), n_classes=3, p=0.5, lr=0.05, epochs=20, seed=11):
    """Train baseline / dropout / batchnorm / both arms under one shared budget.

    Takes the fixed data, shared width, dropout probability, budget and seed;
    returns ``{name: {"train_loss": float, "val_acc": float}}`` with final-epoch
    values for each of the four arms. Each arm is rebuilt fresh from ``seed``
    (consistent initialization, no trained weights carried across arms) and the
    global generator is seeded once so dropout masks are reproducible.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: for the same bad budgets ``train_classifier`` rejects.
    """
    torch.manual_seed(seed)

    output = {}
    model_baseline = build_mlp(hidden=hidden ,input_dim=input_dim , n_classes=n_classes ,p_dropout=0 , seed=seed ,use_batchnorm=False)
    model_dropout = build_mlp(hidden=hidden ,input_dim=input_dim , n_classes=n_classes ,p_dropout=p , seed=seed ,use_batchnorm=False)
    model_batchnorm = build_mlp(hidden=hidden ,input_dim=input_dim , n_classes=n_classes ,p_dropout=0 , seed=seed ,use_batchnorm=True)
    model_both = build_mlp(hidden=hidden ,input_dim=input_dim , n_classes=n_classes ,p_dropout=p , seed=seed ,use_batchnorm=True)
    h = train_classifier(model_baseline , X_train, y_train , X_val ,y_val , lr , epochs)
    output["baseline"] ={"train_loss": h["train_loss"][-1], "val_acc": h["val_acc"][-1]}
    h = train_classifier(model_dropout , X_train, y_train , X_val ,y_val , lr , epochs)
    output["dropout"] = {"train_loss": h["train_loss"][-1], "val_acc": h["val_acc"][-1]}
    h = train_classifier(model_batchnorm , X_train, y_train , X_val ,y_val , lr , epochs)
    output["batchnorm"] ={"train_loss": h["train_loss"][-1], "val_acc": h["val_acc"][-1]}
    h = train_classifier(model_both , X_train, y_train , X_val ,y_val , lr , epochs)
    output["both"] = {"train_loss": h["train_loss"][-1], "val_acc": h["val_acc"][-1]}

    return output


# %%
def hidden_stats(model, X):
    """Summarize each Tanh layer's input distribution and saturation level.

    Takes a model and an input batch; returns one dict per ``nn.Tanh`` layer
    with ``pre_mean``, ``pre_std`` (preactivation), ``act_std`` and
    ``frac_saturated`` (fraction of activations with ``|a| > 0.95``). Runs in
    evaluation mode under ``no_grad`` and restores the model's prior mode.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    captured = []    # (pre, act) per Tanh layer, in order
    handles = []
    for m in model.modules():              # walk every submodule
        if isinstance(m, nn.Tanh):         # only the Tanh layers
            handles.append(m.register_forward_hook(
                lambda mod, inp, out: captured.append((inp[0].detach(), out.detach()))))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(X)
    model.train(was_training)
    for h in handles:
        h.remove()

    result = []
    for pre, act in captured:
        result.append({
            "pre_mean": pre.mean().item(),                          # average input value
            "pre_std": pre.std().item(),                            # spread of input values
            "act_std": act.std().item(),                            # spread after squashing
            "frac_saturated": (act.abs() > 0.95).float().mean().item(),  # share stuck at extremes
        })
    return result

def grad_norms(model, X, y):
    """Report one L2 gradient norm per parameter from a single backward pass.

    Takes a model, inputs and class targets; returns a list of floats aligned
    with ``model.parameters()``. Runs in evaluation mode (so Dropout is the
    identity and BatchNorm uses running statistics), zeroes grads first, runs
    one cross-entropy backward, records the norms, then zeroes grads again so
    the model is left clean. Restores the model's prior mode.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    """

    model.zero_grad()
    model.eval()
    out = model(X)
    loss = F.cross_entropy(out, y)
    loss.backward()
    norms = [p.grad.norm().item() for p in model.parameters()]
    model.zero_grad()

    return norms


# %%
def scaled_saturation(input_dim=12, hidden=(16, 16, 16, 16), scale=4.0, n=256, seed=11):
    """Measure final-layer tanh saturation after scaling a fresh MLP's weights.

    Takes a four-layer tanh MLP shape, a weight-scale factor, a batch size and
    a seed; returns the final hidden layer's ``frac_saturated`` (fraction with
    ``|a| > 0.95``) after multiplying every weight matrix (not biases) by
    ``scale``. Same seed, larger scale must clearly saturate more.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``scale <= 0`` or ``n < 1``.
    """
    if scale <= 0 :
      raise ValueError("scale cant be negative or zero")

    if n < 1:
      raise ValueError("n cannot be less than 1")

    model = build_mlp(input_dim,hidden ,seed = seed)

    with torch.no_grad():
      for p in model.parameters():
          if p.dim() == 2:
              p.mul_(scale)

    torch.manual_seed(seed)
    Xb = torch.randn(n, input_dim)

    return hidden_stats(model, Xb)[-1]["frac_saturated"]



def bn_single_sample(num_features=8, seed=11):
    """Capture the (1, C) BatchNorm1d training failure, then rescue it in eval.

    Takes a channel count and a seed; returns ``{"train_error": str,
    "eval_output_shape": [...]}``. The training-mode forward on a ``(1, C)``
    input must raise (the message is recorded verbatim); the evaluation-mode
    forward on the same input must succeed once running statistics exist.
    Raises: NotImplementedError: until you implement the assigned mechanism.
    Raises: ValueError: if ``num_features < 1``.
    """

    if num_features < 1:
      raise ValueError("num_features cannot be less than 1")

    torch.manual_seed(seed)
    bn = nn.BatchNorm1d(num_features)
    bn.train()
    x = torch.ones(1, num_features)

    try:
        bn(x)
        train_error = "NO ERROR RAISED"
    except ValueError as e:
        train_error = str(e)

    for _ in range(10):
        bn(torch.randn(32, num_features))

    bn.eval()
    out = bn(x)
    return {"train_error": train_error, "eval_output_shape": list(out.shape)}



# %%

def evidence_figures():
    data = make_tabular_data()
    figs = []
    arms = {
        "baseline": {"p_dropout": 0.0, "use_batchnorm": False},
        "dropout": {"p_dropout": 0.5, "use_batchnorm": False},
        "batchnorm": {"p_dropout": 0.0, "use_batchnorm": True},
        "both": {"p_dropout": 0.5, "use_batchnorm": True},
    }
    histories = {}
    for name, kw in arms.items():
        torch.manual_seed(CONFIG["seed"])
        histories[name] = train_classifier(
            build_mlp(12, seed=CONFIG["seed"], **kw),
            data["X_train"], data["y_train"], data["X_val"], data["y_val"])

    # (a) loss curves
    fig, ax = plt.subplots()
    for name, h in histories.items():
        ax.plot(h["train_loss"], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.legend()
    figs.append(fig)

    # (b) reasonable vs saturated final-layer preactivations
    def final_pre(model, xb):
        pres = []
        handles = [m.register_forward_hook(
            lambda mod, inp, out: pres.append(inp[0].detach()))
            for m in model.modules() if isinstance(m, nn.Tanh)]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(xb)
        model.train(was_training)
        for h in handles:
            h.remove()
        return pres[-1].flatten().tolist()

    torch.manual_seed(CONFIG["seed"])
    Xb = torch.randn(256, 12)
    reasonable = build_mlp(12, hidden=(16, 16, 16, 16), seed=CONFIG["seed"])
    saturated = build_mlp(12, hidden=(16, 16, 16, 16), seed=CONFIG["seed"])
    with torch.no_grad():
        for p in saturated.parameters():
            if p.dim() == 2:
                p.mul_(4.0)
    fig, ax = plt.subplots()
    ax.hist(final_pre(reasonable, Xb), bins=30, alpha=0.6, label="reasonable init")
    ax.hist(final_pre(saturated, Xb), bins=30, alpha=0.6, label="scaled x4")
    ax.set_xlabel("final-layer preactivation")
    ax.set_ylabel("count")
    ax.legend()
    figs.append(fig)

    # (c) final bars, read off the same histories (same seed ⇒ same as run_four_way)
    names = list(arms)
    fig, ax = plt.subplots()
    ax.bar(names, [histories[n]["train_loss"][-1] for n in names])
    ax.set_ylabel("final train loss")
    figs.append(fig)
    fig, ax = plt.subplots()
    ax.bar(names, [histories[n]["val_acc"][-1] for n in names])
    ax.set_ylabel("final val accuracy")
    figs.append(fig)

    return figs


# %% [markdown]
# ## Verification and evidence
#
# Tiny checks for this day (all smoke mode, CPU, seed 11):
#
# 1. Preprocessing contract: shapes/dtypes match and the scaler was fit on train (train mean ≈ 0).
# 2. Deterministic rebuilds and histories: same seed twice gives identical parameters and identical training curves.
# 3. Four arms complete under one shared budget with finite metrics; no arm is required to win.
# 4. Saturation by construction: scaled-up weights clearly saturate more than scale 1.0; histograms line up with the architecture.
# 5. The `(1, C)` BatchNorm training error is captured verbatim and the eval-mode rescue returns the right shape.
# 6. Dropout is stochastic in training mode and the identity in evaluation mode.
#
# Save the four-run table and the histogram readings in your conclusion. Restart the kernel and `Run All` in smoke mode before `/done`.

# %%
def run_smoke():
    """Run the Day 4 checks in smoke mode."""
    data = make_tabular_data()
    assert data["X_train"].shape == (400, 12), data["X_train"].shape
    assert data["X_train"].dtype == torch.float32
    assert data["y_train"].dtype == torch.int64
    assert data["X_val"].shape == (100, 12), data["X_val"].shape
    assert data["X_train"].mean().abs().item() < 0.1, data["X_train"].mean()

    m1 = build_mlp(12)
    m2 = build_mlp(12)
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    assert m1(torch.randn(5, 12)).shape == (5, 3)

    torch.manual_seed(CONFIG["seed"])
    h1 = train_classifier(build_mlp(12, seed=CONFIG["seed"]), data["X_train"],
                            data["y_train"], data["X_val"], data["y_val"])
    assert len(h1["train_loss"]) == 20 and len(h1["val_acc"]) == 20, h1.keys()
    assert all(math.isfinite(v) for v in h1["train_loss"]), h1["train_loss"]
    torch.manual_seed(CONFIG["seed"])
    h2 = train_classifier(build_mlp(12, seed=CONFIG["seed"]), data["X_train"],
                            data["y_train"], data["X_val"], data["y_val"])
    assert h1["train_loss"] == h2["train_loss"], "histories must be deterministic"

    table = run_four_way(data["X_train"], data["y_train"], data["X_val"],
                          data["y_val"], 12, seed=CONFIG["seed"])
    assert set(table) == {"baseline", "dropout", "batchnorm", "both"}, table.keys()
    print(table)
    for name, res in table.items():
        assert math.isfinite(res["train_loss"]) and 0.0 <= res["val_acc"] <= 1.0, (name, res)

    stats = hidden_stats(build_mlp(12, seed=CONFIG["seed"]), data["X_val"][:64])
    assert len(stats) == 2 and all(math.isfinite(s["pre_mean"]) for s in stats), stats
    assert all(0.0 <= s["frac_saturated"] <= 1.0 for s in stats), stats
    norms = grad_norms(build_mlp(12, seed=CONFIG["seed"]), data["X_val"][:64],
                         data["y_val"][:64])
    assert len(norms) == 6 and all(n >= 0.0 and math.isfinite(n) for n in norms), norms

    sat_big = scaled_saturation(seed=CONFIG["seed"])
    sat_one = scaled_saturation(scale=1.0, seed=CONFIG["seed"])
    assert sat_big > sat_one + 0.2, (sat_big, sat_one)

    bn = bn_single_sample(seed=CONFIG["seed"])
    assert isinstance(bn["train_error"], str) and bn["train_error"], bn
    assert bn["eval_output_shape"] == [1, 8], bn

    torch.manual_seed(CONFIG["seed"])
    drop = build_mlp(12, p_dropout=0.5, seed=CONFIG["seed"])
    xb = data["X_val"][:32]
    drop.eval()
    assert torch.equal(drop(xb), drop(xb)), "eval mode must disable dropout"

    print("smoke OK")


if __name__ == "__main__":
    run_smoke()
    evidence_figures()

# %% [markdown]
# ## Conclusion (measured numbers; smoke budget)
#
# | Check | Result | What it does / does not prove |
# |---|---|---|
# | four-way table | baseline 1.026 / 0.54; dropout 1.047 / 0.52; batchnorm 0.788 / 0.71; both 0.985 / 0.61. | Baseline vs dropout isolates regularization; baseline vs batchnorm isolates normalization. Dropout val 0.52 is below baseline val 0.54, so dropout was worse here, not better. BatchNorm val 0.71 vs baseline val 0.54. |
# | loss curves | Curves show how those four types of models perform compared to each other. | The graphs show how accuracy changes over the time of epochs, which the final numbers alone do not tell us. |
# | reasonable vs saturated histograms | Reasonable init gives a zero-centered bell curve; scaled x4 gives a bimodal shape at the outer extremes. | x4 scaling amplifies inputs, pushing outputs into tanh flat tails. This shows weight scale drives saturation. |
# | BN single-sample | 'train_error': 'Expected more than 1 value per channel when training, got input size torch.Size([1, 8])', 'eval_output_shape': [1, 8] | Train mode failed on (1, C) because variance needs more than one sample; eval worked because it uses pre-calculated running statistics, so no new batch statistics are needed. |
#
# ### Open question
#
# how to choose which one is better because of the variation in the train and eval
