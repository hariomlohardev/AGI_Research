"""Day 7 tests: diagnostic mechanism only (harness ships complete).

Learner gates (raise NotImplementedError until Problems 1-3 are
implemented): frozen-update equality, alignment spot pairs, and the
curve-summary extraction incl. trend flags. Probes (bug-branch
behavior, memorization demo) are notebook eyeball cells, not gates.
"""
import math

import pytest
import torch

from month3.week_1.day_7.diagnose_run import (
    alignment_report,
    curve_summary,
    params_unchanged,
)


def test_params_unchanged_detects_frozen_and_moved():
    before = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    same = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
    assert params_unchanged(before, same) is True
    moved = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5000001])}
    assert params_unchanged(before, moved) is False


def test_params_unchanged_rejects_key_mismatch():
    with pytest.raises(ValueError):
        params_unchanged({"w": torch.zeros(1)}, {"b": torch.zeros(1)})


def test_alignment_report_returns_indexed_pairs():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([0, 1, 0])
    assert alignment_report(X, y, [2, 0]) == [(2, [5.0, 6.0], 0),
                                              (0, [1.0, 2.0], 0)]


def test_alignment_report_rejects_bad_index():
    X = torch.zeros(2, 2)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        alignment_report(X, y, [2])


def test_curve_summary_matches_hand_values():
    log = {"train_loss": [0.9, 0.7, 0.6, 0.55],
           "val_loss": [0.85, 0.8, 0.82, 0.9]}
    s = curve_summary(log)
    assert s["epochs"] == 4
    assert math.isclose(s["first_train_loss"], 0.9)
    assert math.isclose(s["last_train_loss"], 0.55)
    assert math.isclose(s["best_val_loss"], 0.8)
    assert s["best_val_epoch"] == 1
    assert math.isclose(s["final_gap"], 0.35)
    assert s["val_rising"] is True
    assert s["train_falling"] is True
