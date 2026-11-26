"""Placeholder for Month 3 Week 3 Day 1 (student-owned).

The synced mirror is a review copy, not an importable library:
importing it would run the student's training. So this test only
checks the mirror exists and parses as Python. All real tests are
written by the student inside the notebook.
"""

import ast
from pathlib import Path

MIRROR = Path(__file__).resolve().parents[3] / "src" / "month3" / "week_3" / "day_1" / "rnn_cell_gradients.py"


def test_mirror_exists_and_parses():
    assert MIRROR.exists(), f"missing synced mirror: {MIRROR}"
    ast.parse(MIRROR.read_text(encoding="utf-8"))
