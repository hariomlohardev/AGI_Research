"""Day 24 — dependency audit for push day (see Problem 1).

Two helpers that turn "requirements are accurate" from an assertion into a
check: which third-party packages a source tree actually imports, and
whether a requirements file pins every entry.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def third_party_imports(root: str | Path) -> set[str]:
    """Collect top-level third-party import names under `root`.

    Walks `root` for `.py` files, parses each with `ast`, and returns the
    set of top-level names that are neither stdlib nor relative imports
    (e.g. `from sklearn.cluster import KMeans` contributes `sklearn`).

    Raises:
        ValueError: if `root` is not an existing directory.
    """
    raise NotImplementedError


def pinned_requirements(path: str | Path) -> dict[str, str]:
    """Parse a requirements file into `{package: version}`.

    Every non-blank, non-comment line must be exactly
    `name[extras]==version` with an optional `; markers` tail. An empty
    (or all-comment) file is valid and returns `{}`.

    Raises:
        ValueError: if the file does not exist, or on any entry that is
            not `==`-pinned.
    """
    raise NotImplementedError
