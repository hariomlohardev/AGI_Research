"""Day 12 placeholder: student writes all real tests inside the notebook.

The mirror is a review copy, not an importable library — importing it would
run the learner's training. This file only checks the synced mirror exists
and parses as Python.
"""
import ast
from pathlib import Path

MIRROR = (Path(__file__).resolve().parents[3] / "src" / "month3"
          / "week_2" / "day_4" / "alexnet_cifar.py")


def test_mirror_exists_and_parses():
    assert MIRROR.exists(), "run python scripts/notebooks.py sync --month 3 --day 2 4 first"
    ast.parse(MIRROR.read_text(encoding="utf-8"))
