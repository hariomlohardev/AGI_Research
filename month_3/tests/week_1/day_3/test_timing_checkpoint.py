"""Independent checks for Month 3 Week 1 Day 3: GPU timing and checkpoints.

Reference tests (R) pass on stubs: they verify torch mechanics and reporting
math without calling learner code. Learner-gate tests (G) call the exported
module and fail with NotImplementedError until the day is implemented.
"""

import statistics

import pytest
import torch
from torch import nn

from month3.week_1.day_3 import timing_checkpoint as tc


# --- Reference behavior (no learner code) ---


def test_R1_cuda_flag_is_bool():
    assert isinstance(torch.cuda.is_available(), bool)


def test_R2_median_reporting_math():
    assert statistics.median([3.0, 1.0, 2.0]) == 2.0


def test_R3_state_dict_roundtrip_inline(tmp_path):
    torch.manual_seed(11)
    src = nn.Linear(4, 1)
    path = str(tmp_path / "ref.pt")
    torch.save(src.state_dict(), path)
    dst = nn.Linear(4, 1)
    with torch.no_grad():
        for p in dst.parameters():
            p.zero_()
    dst.load_state_dict(torch.load(path, weights_only=True))
    for a, b in zip(src.parameters(), dst.parameters()):
        assert torch.equal(a, b)


def test_R4_optimizer_state_holds_momentum():
    torch.manual_seed(11)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    xb = torch.randn(8, 4)
    yb = torch.randn(8, 1)
    opt.zero_grad()
    nn.functional.mse_loss(model(xb), yb).backward()
    opt.step()
    assert len(opt.state_dict()["state"]) > 0
    assert set(model.state_dict()) == {"weight", "bias"}


# --- Learner gates ---


def test_G1_describe_device_contract():
    info = tc.describe_device()
    assert set(info) == {"torch_version", "cuda_available", "device",
                         "name", "total_memory_GB", "capability"}
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["torch_version"], str) and info["torch_version"]
    if info["cuda_available"]:
        assert info["device"] == "cuda:0"
        assert isinstance(info["name"], str) and info["name"]
    else:
        assert info["device"] == "cpu"
        assert info["name"] is None
        assert info["total_memory_GB"] is None
        assert info["capability"] is None


def test_G2_time_matmul_finite_positive():
    med = tc.time_matmul(n=64, device="cpu", repeats=2, warmup=1, seed=11)
    assert isinstance(med, float)
    assert med > 0


def test_G3_transfer_vs_compute_keys_finite():
    out = tc.transfer_vs_compute(n=64, device="cpu", repeats=2, seed=11)
    assert set(out) == {"compute_ms", "transfer_ms"}
    assert out["compute_ms"] >= 0 and out["transfer_ms"] >= 0


def test_G4_max_abs_diff_value():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0, 2.5], [3.0, 4.0]])
    assert tc.max_abs_diff(a, b) == pytest.approx(0.5)
    assert tc.max_abs_diff(a, a) == 0.0


def _fixed_batch():
    g = torch.Generator().manual_seed(11)
    xb = torch.randn(8, 4, generator=g)
    yb = xb.sum(dim=1, keepdim=True) + 0.1 * torch.randn(8, 1, generator=g)
    return xb, yb


def test_G5_sgd_deterministic_for_fixed_seed():
    xb, yb = _fixed_batch()
    torch.manual_seed(11)
    m1 = tc.TinyMLP()
    loss1 = tc.sgd_steps_on_batch(m1, xb, yb, lr=0.01, steps=3)
    torch.manual_seed(11)
    m2 = tc.TinyMLP()
    loss2 = tc.sgd_steps_on_batch(m2, xb, yb, lr=0.01, steps=3)
    assert loss1 == loss2
    assert torch.isfinite(torch.tensor(loss1))


def test_G6_error_contracts(tmp_path):
    with pytest.raises(ValueError):
        tc.time_matmul(n=0)
    with pytest.raises(ValueError):
        tc.transfer_vs_compute(n=0)
    with pytest.raises(ValueError):
        tc.max_abs_diff(torch.zeros(2), torch.zeros(3))
    xb, yb = _fixed_batch()
    torch.manual_seed(11)
    with pytest.raises(ValueError):
        tc.sgd_steps_on_batch(tc.TinyMLP(), xb, yb, steps=0)
    with pytest.raises(ValueError):
        tc.save_checkpoint(tc.TinyMLP(), None, -1, {"seed": 11}, "x.pt")
    with pytest.raises(ValueError):
        tc.save_checkpoint(tc.TinyMLP(), None, 0, [("seed", 11)], "x.pt")
    with pytest.raises(FileNotFoundError):
        tc.load_checkpoint(str(tmp_path / "missing.pt"), tc.TinyMLP())
    bad = tmp_path / "bad.pt"
    torch.save({"not": "a checkpoint"}, str(bad))
    with pytest.raises(ValueError):
        tc.load_checkpoint(str(bad), tc.TinyMLP())


def test_G7_checkpoint_roundtrip_restores_all_state(tmp_path):
    xb, yb = _fixed_batch()
    torch.manual_seed(11)
    model = tc.TinyMLP()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    opt.zero_grad()
    nn.functional.mse_loss(model(xb).squeeze(-1), yb.squeeze(-1)).backward()
    opt.step()
    path = str(tmp_path / "day3.pt")
    assert tc.save_checkpoint(model, opt, 3, {"seed": 11, "lr": 0.01}, path) == path

    torch.manual_seed(11)
    fresh = tc.TinyMLP()
    fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.01, momentum=0.9)
    restored = tc.load_checkpoint(path, fresh, fresh_opt)
    assert restored["epoch"] == 3
    assert restored["config"] == {"seed": 11, "lr": 0.01}
    assert tc.prediction_max_diff(model, fresh, xb) == 0.0
    old_state = opt.state_dict()["state"]
    new_state = fresh_opt.state_dict()["state"]
    assert set(new_state) == set(old_state)
    for k in old_state:
        assert torch.equal(new_state[k]["momentum_buffer"],
                           old_state[k]["momentum_buffer"])
