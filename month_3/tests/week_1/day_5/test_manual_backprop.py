"""Day 5 tests: manual backward vs autograd reference (smoke shapes, float64).

Reference checks (pass on the scaffold): input-validation contracts.
Learner gates (raise NotImplementedError until implemented): gradient parity,
mechanism invariants, and the injected-error diagnosis.
"""
import pytest
import torch
import torch.nn.functional as F

from month3.week_1.day_5.manual_backprop import (
    BACKWARD_ORDER,
    compare_grads,
    find_first_mismatch,
    forward_manual,
    init_params,
    manual_backward,
    manual_backward_broken,
    reference_backward,
)

torch.manual_seed(11)

B, D, H, C = 4, 3, 5, 3
ATOL, RTOL = 1e-8, 1e-6


def _inputs():
    g = torch.Generator().manual_seed(11)
    X = torch.randn(B, D, dtype=torch.float64, generator=g) * 0.5
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    return X, y


# ---------- Reference checks: validation contracts (pass on scaffold) ----------

def test_forward_rejects_nonpositive_eps():
    params = {"W1": torch.zeros(1, 1, dtype=torch.float64)}
    X = torch.zeros(2, 1, dtype=torch.float64)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        forward_manual(X, y, params, eps=0.0)


def test_forward_rejects_feature_mismatch():
    params = {k: torch.zeros(1, dtype=torch.float64)
              for k in ("b1", "gamma", "beta", "b2")}
    params.update({"W1": torch.zeros(H, D + 1, dtype=torch.float64),
                   "W2": torch.zeros(C, H, dtype=torch.float64)})
    X = torch.zeros(B, D, dtype=torch.float64)
    y = torch.zeros(B, dtype=torch.long)
    with pytest.raises(ValueError):
        forward_manual(X, y, params)


def test_manual_backward_rejects_missing_cache_key():
    params = {"W1": torch.zeros(1, 1, dtype=torch.float64)}
    X = torch.zeros(2, 1, dtype=torch.float64)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        manual_backward(X, y, params, cache={"logits": X})


def test_compare_grads_rejects_key_mismatch():
    a = {"dW1": torch.zeros(2, dtype=torch.float64)}
    b = {"dW2": torch.zeros(2, dtype=torch.float64)}
    with pytest.raises(ValueError):
        compare_grads(a, b)


# ---------- Learner gates (fail until implemented) ----------

def test_forward_loss_finite_and_cache_shaped():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    assert loss.numel() == 1 and torch.isfinite(loss).item()
    assert cache["z1"].shape == (B, H)
    assert cache["mu"].shape == (H,) and cache["var"].shape == (H,)
    assert cache["logits"].shape == (B, C)


def test_full_gradient_parity():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    ref = reference_backward(loss, params, {**cache, "X": X})
    table = compare_grads(manual, ref, atol=ATOL, rtol=RTOL)
    bad = [k for k, v in table.items() if not v["passed"]]
    assert not bad, f"mismatched grads: {bad}"


def test_bias_grad_is_batch_sum():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    assert torch.allclose(manual["db2"],
                          manual["dlogits"].sum(dim=0),
                          atol=ATOL, rtol=RTOL)


def test_mean_loss_carries_one_over_B():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    ref_sum = reference_backward(loss * B, params, {**cache, "X": X})
    for k in manual:
        assert torch.allclose(ref_sum[k], manual[k] * B,
                              atol=1e-7, rtol=1e-5), k


def test_broken_variant_first_mismatch_is_dvar():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    broken = manual_backward_broken(X, y, params, cache)
    ref = reference_backward(loss, params, {**cache, "X": X})
    assert find_first_mismatch(broken, ref, BACKWARD_ORDER,
                               atol=ATOL, rtol=RTOL) == "dvar"


def test_repaired_backward_mismatches_nowhere():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    ref = reference_backward(loss, params, {**cache, "X": X})
    assert find_first_mismatch(manual, ref, BACKWARD_ORDER,
                               atol=ATOL, rtol=RTOL) is None


def test_variance_convention_mismatch_detected():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    z1 = cache["z1"]
    var_unbiased = z1.var(dim=0, unbiased=True, keepdim=True)
    assert not torch.allclose(var_unbiased.squeeze(0), cache["var"],
                              atol=ATOL, rtol=RTOL)
    keys = set(manual) - {"dW2", "db2", "dlogits"}
    assert keys, "expected BN-affected keys to exist"


def test_training_bn_couples_examples():
    params = init_params(D, H, C, seed=11)
    X, y = _inputs()
    X.requires_grad_(True)
    loss, cache = forward_manual(X, y, params)
    manual = manual_backward(X, y, params, cache)
    X2 = X.detach().clone()
    X2[1] = X2[1] + 0.5
    X2.requires_grad_(True)
    loss2, cache2 = forward_manual(X2, y, params)
    manual2 = manual_backward(X2, y, params, cache2)
    assert not torch.allclose(manual["dX"][0], manual2["dX"][0],
                              atol=1e-12, rtol=0.0)
