"""Day 9 tests: corr2d mechanism only (harness ships complete).

Learner gates (raise NotImplementedError until Problem 1 is wrapped):
parity against the float64 torch reference on the three plan cases plus one
seeded random config (anti-hardcoding), output-shape formula, and the five
ValueError guards. Image ranges, channel summation and the flip convention
are notebook eyeball cells, not gates.
"""
import numpy as np
import pytest
import torch

from month3.week_2.day_1.conv2d_loops import (
    HAND_W,
    HAND_X,
    PAD_W,
    PAD_X,
    STRIDE_W,
    STRIDE_X,
    corr2d,
    corr2d_torch,
)

TOL = 1e-9


def test_corr2d_matches_hand_case():
    got = np.asarray(corr2d(HAND_X, HAND_W), dtype=np.float64)
    assert got.shape == (3, 3)
    assert np.abs(got - corr2d_torch(HAND_X, HAND_W)).max() <= TOL


def test_corr2d_matches_padded_case():
    got = np.asarray(corr2d(PAD_X, PAD_W, stride=1, pad=1), dtype=np.float64)
    assert got.shape == (4, 4)
    assert np.abs(got - corr2d_torch(PAD_X, PAD_W, stride=1, pad=1)).max() <= TOL


def test_corr2d_matches_stride2_case():
    got = np.asarray(corr2d(STRIDE_X, STRIDE_W, stride=2), dtype=np.float64)
    assert got.shape == (2, 2)
    assert np.abs(got - corr2d_torch(STRIDE_X, STRIDE_W, stride=2)).max() <= TOL


def test_corr2d_random_parity():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(6, 7, generator=g).numpy()
    w = torch.randn(3, 2, generator=g).numpy()
    for stride, pad in ((1, 0), (2, 1), (3, 0)):
        got = np.asarray(corr2d(x, w, stride=stride, pad=pad), dtype=np.float64)
        ref = corr2d_torch(x, w, stride=stride, pad=pad)
        assert got.shape == ref.shape == ((6 + 2 * pad - 3) // stride + 1,
                                          (7 + 2 * pad - 2) // stride + 1)
        assert np.abs(got - ref).max() <= TOL


def test_corr2d_output_shape_formula():
    z5 = np.zeros((5, 5))
    assert np.shape(corr2d(z5, np.zeros((3, 3)), stride=2, pad=1)) == (3, 3)
    assert np.shape(corr2d(np.zeros((4, 6)), np.zeros((2, 2)))) == (3, 5)
    assert np.shape(corr2d(np.zeros((3, 3)), np.zeros((2, 2)), pad=1)) == (4, 4)


def test_corr2d_rejects_bad_args():
    with pytest.raises(ValueError):
        corr2d([[1]], [[1, 2], [3, 4]])  # kernel > padded input
    with pytest.raises(ValueError):
        corr2d([1, 2, 3], HAND_W)  # x not 2D
    with pytest.raises(ValueError):
        corr2d(HAND_X, [1, 2, 3])  # w not 2D
    with pytest.raises(ValueError):
        corr2d(HAND_X, HAND_W, stride=0)
    with pytest.raises(ValueError):
        corr2d(HAND_X, HAND_W, pad=-1)
