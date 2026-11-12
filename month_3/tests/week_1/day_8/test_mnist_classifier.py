"""Day 8 tests: assembly mechanism only (harness ships complete).

Learner gates (raise NotImplementedError until Problems 1-3 are
implemented): classifier forward contract, budgeted best-val
selection, and the confusion-matrix reader. Probes (inspection
hypotheses, mistake patterns, reload verdict) are notebook eyeball
cells, not gates.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from month3.week_1.day_8.mnist_classifier import (
    build_classifier,
    eval_loss_acc,
    top_confusions,
    train_select,
)


def _tiny_loaders():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, 8, generator=g)
    y = (X.sum(1) > 0).long()
    tr = DataLoader(TensorDataset(X[:48], y[:48]), batch_size=16,
                    shuffle=True, generator=g, num_workers=0)
    va = DataLoader(TensorDataset(X[48:], y[48:]), batch_size=16,
                    shuffle=False, num_workers=0)
    return tr, va


def test_build_classifier_forward_contract():
    m = build_classifier(input_dim=8, hidden=16, num_classes=2)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(5, 8))
    assert out.shape == (5, 2)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_build_classifier_rejects_bad_hidden():
    with pytest.raises(ValueError):
        build_classifier(input_dim=8, hidden=0, num_classes=2)


def test_train_select_keeps_argmin_val_state():
    tr, va = _tiny_loaders()
    m = build_classifier(input_dim=8, hidden=16, num_classes=2)
    best, log = train_select(m, tr, va, epochs=3, lr=0.05)
    assert len(log["train_loss"]) == 3 and len(log["val_loss"]) == 3
    assert all(math.isfinite(v)
               for v in log["train_loss"] + log["val_loss"])
    argmin = min(range(3), key=lambda e: log["val_loss"][e])
    m.load_state_dict(best)
    l_best, _ = eval_loss_acc(m, va, nn.CrossEntropyLoss())
    assert math.isclose(l_best, log["val_loss"][argmin], rel_tol=1e-6)


def test_train_select_rejects_bad_budget():
    tr, va = _tiny_loaders()
    m = build_classifier(input_dim=8, hidden=16, num_classes=2)
    with pytest.raises(ValueError):
        train_select(m, tr, va, epochs=0, lr=0.05)


def test_top_confusions_matches_hand_values():
    cm = torch.tensor([[5, 1, 0],
                       [0, 4, 3],
                       [2, 0, 3]])
    assert top_confusions(cm, k=2) == [(1, 2, 3), (2, 0, 2)]


def test_top_confusions_rejects_bad_args():
    cm = torch.zeros(3, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        top_confusions(cm, k=0)
    with pytest.raises(ValueError):
        top_confusions(torch.zeros(2, 3, dtype=torch.long), k=1)
