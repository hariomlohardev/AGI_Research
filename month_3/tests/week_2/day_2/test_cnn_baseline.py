"""Day 10 tests: SmallCNN + split/merge mechanisms only (harness ships complete).

Learner gates (raise NotImplementedError until Problems 1/3 are wrapped):
forward contract, gradient flow, block shapes, parameter count, head-split
exact recovery and guards. Training curves, the MLP comparison verdict, the
QKT prediction and the wrong-axis explanation are notebook eyeball cells.
"""
import torch
import torch.nn as nn

import pytest

from month3.week_2.day_2.cnn_baseline import (
    SmallCNN,
    block_shapes,
    count_params,
    merge_heads,
    split_heads,
)


def test_smallcnn_forward_contract():
    m = SmallCNN().eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        out = m(torch.randn(2, 1, 28, 28, generator=g))
    assert tuple(out.shape) == (2, 10)
    assert torch.isfinite(out).all()


def test_smallcnn_forward_rejects_non_4d():
    m = SmallCNN().eval()
    with torch.no_grad(), pytest.raises(ValueError):
        m(torch.randn(2, 28, 28))


def test_smallcnn_grads_reach_all_params():
    m = SmallCNN()
    m.zero_grad()
    out = m(torch.randn(2, 1, 28, 28))
    nn.CrossEntropyLoss()(out, torch.tensor([3, 7])).backward()
    assert all(p.grad is not None for p in m.parameters())


def test_smallcnn_block_shapes_and_params():
    m = SmallCNN().eval()
    g = torch.Generator().manual_seed(0)
    shapes = block_shapes(m, torch.randn(2, 1, 28, 28, generator=g))
    for want in ((2, 8, 28, 28), (2, 8, 14, 14),
                 (2, 16, 14, 14), (2, 16, 7, 7)):
        assert want in shapes
    assert shapes[-1] == (2, 10)
    assert count_params(m) == 52138


def test_split_merge_exact_round_trip():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 32, generator=g)
    s = split_heads(x, 4)
    assert tuple(s.shape) == (2, 4, 5, 8)
    assert torch.equal(merge_heads(s), x)
    assert torch.equal(merge_heads(split_heads(x, 2)), x)


def test_split_heads_rejects_bad_inputs():
    with pytest.raises(ValueError):
        split_heads(torch.randn(2, 5, 30), 4)
    with pytest.raises(ValueError):
        split_heads(torch.randn(2, 5), 4)
