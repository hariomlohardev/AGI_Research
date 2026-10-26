"""Independent checks for Month 3 Week 1 Day 1: tensors, shapes, autograd.

Acceptance gate for the learner's notebook stubs. Reference tests pin the
expected values independently; learner tests call the stub functions and fail
with NotImplementedError until the day is genuinely implemented. A failing
learner test is unfinished work, never a pass.
"""

import torch

from month3.week_1.day_1.tensors_autograd import (
    CONFIG,
    accumulation_demo,
    broadcast_error_example,
    scalar_grads,
    shape_predictions,
    shared_branch_grads,
)

X, Y = 2.0, 3.0
B, D, H = 4, 5, 3


def _reference_scalar(x, y):
    """Independent autograd reference for z = x*y + x**2 (float64)."""
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    z = xt * yt + xt ** 2
    z.backward()
    return xt.grad.item(), yt.grad.item()


def test_config_is_smoke_on_cpu():
    assert CONFIG["mode"] == "smoke"
    assert CONFIG["device"] == "cpu"


def test_reference_scalar_hand_values():
    # Hand derivation: dz/dx = y + 2x = 7, dz/dy = x = 2.
    dx, dy = _reference_scalar(X, Y)
    assert dx == Y + 2 * X
    assert dy == X


def test_reference_shared_branch_sums_contributions():
    # z = x*x + 3*x, dz/dx = 2x + 3 = 7 at x = 2.
    xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    (xt * xt + 3 * xt).backward()
    assert xt.grad.item() == 2 * X + 3


def test_reference_accumulation_semantics():
    w = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    (w ** 2).backward()
    single = w.grad.item()
    (w ** 2).backward()
    accumulated = w.grad.item()
    assert single == 6.0
    assert accumulated == 12.0
    w.grad.zero_()
    (w ** 2).backward()
    assert w.grad.item() == 6.0


def test_reference_matmul_bias_mask_shapes():
    a = torch.zeros(B, D)
    assert (a @ torch.zeros(D, H)).shape == (B, H)
    assert ((a @ torch.zeros(D, H)) + torch.zeros(H)).shape == (B, H)
    assert (a * torch.ones(B, 1)).shape == (B, D)


def test_scalar_grads_matches_reference():
    assert scalar_grads(X, Y) == _reference_scalar(X, Y)


def test_shared_branch_grads_matches_reference():
    assert shared_branch_grads(X) == 2 * X + 3


def test_accumulation_demo_protocol():
    assert accumulation_demo() == (6.0, 12.0, 6.0)


def test_shape_predictions_match_actuals():
    assert shape_predictions() == {
        "matmul": (B, H),
        "bias_add": (B, H),
        "mask_mul": (B, D),
    }


def test_broadcast_error_is_genuine_mismatch():
    report = broadcast_error_example()
    assert tuple(report["actual"]) != tuple(report["intended"])
    assert report["why"].strip() != ""
