"""Independent checks for Month 3 Week 1 Day 4: Dropout, BatchNorm, histograms.

Reference tests (R) pass on stubs: they verify torch mechanics without calling
learner code. Learner-gate tests (G) call the exported module and fail with
NotImplementedError until the day is implemented.
"""

import math

import pytest
import torch
from torch import nn

from month3.week_1.day_4 import dropout_batchnorm as db


# --- Reference behavior (no learner code) ---


def test_R1_inverted_dropout_preserves_mean():
    torch.manual_seed(0)
    x = torch.full((20000,), 2.0)
    drop = nn.Dropout(p=0.5)
    drop.train()
    assert abs(drop(x).mean().item() - 2.0) < 0.1


def test_R2_batchnorm_single_sample_raises_inline():
    bn = nn.BatchNorm1d(8)
    bn.train()
    with pytest.raises(ValueError):
        bn(torch.ones(1, 8))


def test_R3_tanh_saturation_fraction_math():
    torch.manual_seed(0)
    big = torch.tanh(20.0 * torch.randn(5000))
    assert (big.abs() > 0.95).float().mean().item() > 0.9


def test_R4_fullbatch_determinism_inline():
    def run():
        torch.manual_seed(11)
        m = nn.Linear(12, 3)
        xb = torch.randn(64, 12)
        yb = torch.randint(0, 3, (64,))
        return nn.functional.cross_entropy(m(xb), yb).item()

    assert run() == run()


# --- Learner gates (fail with NotImplementedError on stubs) ---


def test_G1_data_contract():
    data = db.make_tabular_data()
    assert data["X_train"].shape == (400, 12)
    assert data["X_train"].dtype == torch.float32
    assert data["y_train"].dtype == torch.int64
    assert data["X_val"].shape == (100, 12)
    assert data["X_train"].mean().abs().item() < 0.1


def test_G2_data_rejects_bad_args():
    with pytest.raises(ValueError):
        db.make_tabular_data(n_train=0)
    with pytest.raises(ValueError):
        db.make_tabular_data(n_informative=99, n_features=4)
    with pytest.raises(ValueError):
        db.make_tabular_data(n_train=50, n_val=50)


def test_G3_builder_shapes_and_rebuilds():
    m = db.build_mlp(12)
    assert m(torch.randn(5, 12)).shape == (5, 3)
    m2 = db.build_mlp(12)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    with pytest.raises(ValueError):
        db.build_mlp(12, hidden=())
    with pytest.raises(ValueError):
        db.build_mlp(12, p_dropout=1.0)


def test_G4_training_history_and_determinism():
    data = db.make_tabular_data()
    kwargs = dict(X_train=data["X_train"], y_train=data["y_train"],
                  X_val=data["X_val"], y_val=data["y_val"])
    h1 = db.train_classifier(db.build_mlp(12, seed=11), **kwargs)
    h2 = db.train_classifier(db.build_mlp(12, seed=11), **kwargs)
    assert len(h1["train_loss"]) == 20 and len(h1["val_acc"]) == 20
    assert all(math.isfinite(v) for v in h1["train_loss"])
    assert h1["train_loss"] == h2["train_loss"]
    with pytest.raises(ValueError):
        db.train_classifier(db.build_mlp(12), epochs=0, **kwargs)


def test_G5_four_way_table():
    data = db.make_tabular_data()
    table = db.run_four_way(data["X_train"], data["y_train"], data["X_val"],
                            data["y_val"], 12, seed=11)
    assert set(table) == {"baseline", "dropout", "batchnorm", "both"}
    for res in table.values():
        assert math.isfinite(res["train_loss"])
        assert 0.0 <= res["val_acc"] <= 1.0


def test_G6_hidden_stats_and_grad_norms():
    data = db.make_tabular_data()
    model = db.build_mlp(12, seed=11)
    stats = db.hidden_stats(model, data["X_val"][:64])
    assert len(stats) == 2
    assert all(set(s) == {"pre_mean", "pre_std", "act_std", "frac_saturated"} for s in stats)
    norms = db.grad_norms(model, data["X_val"][:64], data["y_val"][:64])
    assert len(norms) == 6
    assert all(n >= 0.0 and math.isfinite(n) for n in norms)


def test_G7_saturation_gap():
    big = db.scaled_saturation(seed=11)
    small = db.scaled_saturation(scale=1.0, seed=11)
    assert big > small + 0.2
    with pytest.raises(ValueError):
        db.scaled_saturation(scale=0.0)


def test_G8_bn_single_sample_rescue():
    res = db.bn_single_sample(seed=11)
    assert isinstance(res["train_error"], str) and res["train_error"]
    assert res["eval_output_shape"] == [1, 8]
