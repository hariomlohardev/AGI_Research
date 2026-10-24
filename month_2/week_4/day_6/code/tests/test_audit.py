"""Tests for Day 24's dependency audit (synthetic fixtures + real-tree pins).

Unit tests run against `tmp_path` fixtures so they pin the contracts, not
the repo. Two integration tests pin facts about this repo that must stay
true for the push to be honest: the package tree is stdlib-only, and every
root requirements entry is pinned.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from audit_requirements import pinned_requirements, third_party_imports

DAY6 = pathlib.Path(__file__).resolve().parents[2]
WEEK4 = DAY6.parent
MONTH2 = WEEK4.parent
ROOT = MONTH2.parent

PKG_SRC = MONTH2 / "minisklearn" / "src"
PKG_REQ = MONTH2 / "minisklearn" / "requirements.txt"
ROOT_REQ = ROOT / "requirements.txt"


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_stdlib_and_relative_imports_excluded(tmp_path):
    src = tmp_path / "pkg"
    src.mkdir()
    _write(src / "a.py", "import math\nimport os\nfrom . import sib\nfrom .mod import thing\n")
    _write(src / "b.py", "from __future__ import annotations\nfrom collections import Counter\n")
    assert third_party_imports(src) == set()


def test_third_party_names_collapse_to_top_level(tmp_path):
    src = tmp_path / "pkg"
    src.mkdir()
    _write(src / "a.py", "import numpy\nfrom sklearn.cluster import KMeans\n")
    _write(src / "sub.py", "import matplotlib.pyplot as plt\n")
    assert third_party_imports(src) == {"numpy", "sklearn", "matplotlib"}


def test_rejects_non_directory(tmp_path):
    with pytest.raises(ValueError):
        third_party_imports(tmp_path / "missing")
    with pytest.raises(ValueError):
        third_party_imports(_write(tmp_path / "f.py", "import os\n"))


def test_pinned_entries_with_extras_and_markers(tmp_path):
    req = _write(
        tmp_path / "r.txt",
        "# comment\n\nrequests[security]==2.31.0\n"
        'numpy==2.2.6; python_version >= "3.10"\n',
    )
    assert pinned_requirements(req) == {"requests": "2.31.0", "numpy": "2.2.6"}


def test_unpinned_or_missing_requirements_rejected(tmp_path):
    for bad in ("numpy\n", "numpy>=1.0\n", "numpy~=2.0\n", "-e .\n"):
        req = _write(tmp_path / "r.txt", bad)
        with pytest.raises(ValueError):
            pinned_requirements(req)
    with pytest.raises(ValueError):
        pinned_requirements(tmp_path / "missing.txt")


def test_empty_requirements_file_is_valid(tmp_path):
    req = _write(tmp_path / "r.txt", "# nothing needed\n\n")
    assert pinned_requirements(req) == {}


def test_package_tree_is_stdlib_only():
    # Push-day fact 1: month_2/minisklearn/src/ needs no third-party
    # packages — the plan's "nothing more should be required" holds.
    assert third_party_imports(PKG_SRC) == set()
    assert pinned_requirements(PKG_REQ) == {}


def test_root_requirements_fully_pinned():
    # Push-day fact 2: every root requirements entry is pinned, and the
    # Day-22 validation deps are present.
    reqs = pinned_requirements(ROOT_REQ)
    assert set(reqs) >= {"numpy", "scikit-learn"}
