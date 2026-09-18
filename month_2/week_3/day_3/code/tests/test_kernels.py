"""Tests for polynomial_kernel and add_radius_feature."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from svm import add_radius_feature, polynomial_kernel


def test_polynomial_kernel_hand_values():
    # dot([1, 2], [3, 4]) = 11; (11 + 1)^2 = 144.
    assert polynomial_kernel([1.0, 2.0], [3.0, 4.0]) == pytest.approx(144.0)
    # Pure square, no offset: dot = 5, 5^2 = 25.
    assert polynomial_kernel([1.0, 2.0], [1.0, 2.0], coef0=0.0) == pytest.approx(25.0)
    # Degree 1 with zero offset is the plain dot product.
    assert polynomial_kernel([1.0, 2.0], [3.0, 4.0], degree=1, coef0=0.0) == pytest.approx(11.0)


def test_polynomial_kernel_similarity_ordering():
    # Neighbours score higher than far points — the similarity intuition
    # the kernel trick relies on.
    a, neighbour, far = [1.0, 0.0], [0.9, 0.1], [-2.5, 0.0]
    assert polynomial_kernel(a, neighbour) > polynomial_kernel(a, far)


def test_polynomial_kernel_rejects_bad_input():
    with pytest.raises(ValueError):
        polynomial_kernel([], [1.0])
    with pytest.raises(ValueError):
        polynomial_kernel([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        polynomial_kernel([1.0], [1.0], degree=0)


def test_add_radius_feature_hand_values():
    out = add_radius_feature([[3.0, 4.0], [0.0, 0.0]])
    assert out[0] == pytest.approx([3.0, 4.0, 25.0])
    assert out[1] == pytest.approx([0.0, 0.0, 0.0])


def test_add_radius_feature_separates_radii():
    inner = add_radius_feature([[0.5, 0.0]])
    outer = add_radius_feature([[3.0, 0.0]])
    assert inner[0][2] < 1.0
    assert outer[0][2] > 6.0


def test_add_radius_feature_rejects_bad_input():
    with pytest.raises(ValueError):
        add_radius_feature([])
    with pytest.raises(ValueError):
        add_radius_feature([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        add_radius_feature([[1.0, 2.0, 3.0]])
