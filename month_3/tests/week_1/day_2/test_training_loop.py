"""Independent checks for Month 3 Week 1 Day 2: the reusable training loop.

Acceptance gate for the learner's notebook stubs. Reference tests pin the
fixed data recipe, example-weighted loss math, and baseline math; learner
tests call the stub functions and fail with NotImplementedError until the day
is genuinely implemented. A failing learner test is unfinished work, never a
pass.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from month3.week_1.day_2.training_loop import (
    CONFIG,
    RegressionDataset,
    TwoLayerMLP,
    build_loaders,
    constant_baseline_mse,
    evaluate,
    make_regression_data,
    ordered_vs_shuffled_comparison,
    sample_independence_check,
    train_epoch,
)

SEED = 11


def _reference_data(n=64, seed=SEED):
    """Independent copy of the pinned fixed-data recipe."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g, dtype=torch.float32)
    y = X @ torch.tensor([3.0, -2.0]) + 0.5 + 0.1 * torch.randn(n, generator=g)
    return X.float(), y.float()


def test_config_is_smoke_on_cpu():
    assert CONFIG["mode"] == "smoke"
    assert CONFIG["device"] == "cpu"
    assert CONFIG["seed"] == SEED


def test_reference_data_recipe_is_fixed():
    X, y = _reference_data()
    assert X.shape == (64, 2)
    assert y.shape == (64,)
    assert X.dtype == torch.float32 and y.dtype == torch.float32
    assert torch.isfinite(X).all() and torch.isfinite(y).all()
    X2, y2 = _reference_data()
    assert torch.equal(X, X2) and torch.equal(y, y2)


def test_reference_weighted_loss_math():
    # Unequal batch means: naive mean of batch losses differs from the
    # example-weighted epoch loss, which equals the full-set MSE.
    a = torch.tensor([0.0, 2.0])
    b = torch.tensor([10.0])
    naive = (a.mean() + b.mean()) / 2
    weighted = (a.sum() + b.sum()) / (a.numel() + b.numel())
    full = torch.cat([a, b]).mean()
    assert naive.item() != weighted.item()
    assert weighted.item() == full.item()


def test_reference_constant_baseline_math():
    train_targets = torch.tensor([1.0, 2.0, 3.0])
    val_targets = torch.tensor([1.5, 2.5])
    mean = train_targets.mean()
    expected = ((val_targets - mean) ** 2).mean().item()
    assert expected == 0.25


def test_make_regression_data_matches_reference():
    X, y = make_regression_data()
    expected_X, expected_y = _reference_data()
    assert torch.equal(X, expected_X)
    assert torch.equal(y, expected_y)


def test_custom_dataset_indexes_pairs():
    X, y = _reference_data()
    dataset = RegressionDataset(X[:48], y[:48])
    assert len(dataset) == 48
    x0, y0 = dataset[0]
    assert x0.shape == (2,) and y0.shape == ()
    assert torch.equal(x0, X[0]) and torch.equal(y0, y[0])


def test_loaders_split_batch_and_reproduce():
    X, y = _reference_data()
    train_loader, val_loader = build_loaders(X, y)
    assert len(train_loader.dataset) == 48
    assert len(val_loader.dataset) == 16
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0

    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 2) and yb.shape == (16,)
    again_xb, again_yb = next(iter(build_loaders(X, y)[0]))
    assert torch.equal(xb, again_xb) and torch.equal(yb, again_yb)

    seen = torch.cat([batch_y for _, batch_y in train_loader])
    assert torch.equal(torch.sort(seen).values, torch.sort(y[:48]).values)
    val_xb, val_yb = next(iter(val_loader))
    assert torch.equal(val_xb, X[48:]) and torch.equal(val_yb, y[48:])


def test_mlp_registers_parameters_and_modes():
    torch.manual_seed(SEED)
    model = TwoLayerMLP()
    params = list(model.parameters())
    assert len(params) == 4
    assert all(p.requires_grad for p in params)

    torch.manual_seed(SEED)
    probe = TwoLayerMLP()
    xb = torch.randn(16, 2)
    assert probe(xb).shape == (16, 1)
    probe.eval()
    assert probe.training is False
    probe.train()
    assert probe.training is True


def test_train_epoch_reports_weighted_loss_and_learns():
    X, y = _reference_data()
    single = DataLoader(TensorDataset(X[:16], y[:16]), batch_size=16)
    torch.manual_seed(SEED)
    model = TwoLayerMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        expected = loss_fn(model(X[:16]).reshape(-1), y[:16]).item()
    assert train_epoch(model, single, loss_fn, optimizer) == expected

    torch.manual_seed(SEED)
    learner = TwoLayerMLP()
    optimizer = torch.optim.SGD(learner.parameters(), lr=0.1)
    train_loader, _ = build_loaders(X, y)
    before = evaluate(learner, train_loader, loss_fn)
    for _ in range(5):
        train_epoch(learner, train_loader, loss_fn, optimizer)
    after = evaluate(learner, train_loader, loss_fn)
    assert after < before


def test_evaluate_is_no_grad_and_restores_mode():
    X, y = _reference_data()
    _, val_loader = build_loaders(X, y)
    torch.manual_seed(SEED)
    model = TwoLayerMLP()
    loss_fn = nn.MSELoss()

    got = evaluate(model, val_loader, loss_fn)
    with torch.no_grad():
        expected = loss_fn(model(X[48:]).reshape(-1), y[48:]).item()
    assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-8)
    assert all(p.grad is None for p in model.parameters())

    model.train()
    evaluate(model, val_loader, loss_fn)
    assert model.training is True


def test_constant_baseline_matches_manual():
    X, y = _reference_data()
    train_loader, val_loader = build_loaders(X, y)
    got = constant_baseline_mse(train_loader, val_loader)
    mean = y[:48].mean()
    expected = ((y[48:] - mean) ** 2).mean().item()
    assert math.isfinite(got)
    assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-8)


def test_sample_independence_only_touches_row_zero():
    X, y = _reference_data()
    train_loader, _ = build_loaders(X, y)
    xb, yb = next(iter(train_loader))
    torch.manual_seed(SEED)
    model = TwoLayerMLP()
    norms = sample_independence_check(model, (xb, yb))
    assert len(norms) == xb.shape[0]
    assert norms[0] > 0
    assert all(v == 0 for v in norms[1:])


def test_ordered_vs_shuffled_is_bounded_and_reproducible():
    X, y = _reference_data()
    first = ordered_vs_shuffled_comparison(X, y)
    second = ordered_vs_shuffled_comparison(X, y)
    assert math.isfinite(first["ordered"])
    assert math.isfinite(first["shuffled"])
    assert isinstance(first["observation"], str) and first["observation"].strip()
    assert first["ordered"] == second["ordered"]
    assert first["shuffled"] == second["shuffled"]
