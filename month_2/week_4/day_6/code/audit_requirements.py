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
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"not an existing directory: {root!r}")

    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for path in sorted(root_path.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in stdlib:
                        found.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue  # relative import — intra-package, not third-party
                module = (node.module or "").split(".")[0]
                if module and module not in stdlib:
                    found.add(module)
    return found


def pinned_requirements(path: str | Path) -> dict[str, str]:
    """Parse a requirements file into `{package: version}`.

    Every non-blank, non-comment line must be exactly
    `name[extras]==version` with an optional `; markers` tail. An empty
    (or all-comment) file is valid and returns `{}`.

    Raises:
        ValueError: if the file does not exist, or on any entry that is
            not `==`-pinned.
    """
    req_path = Path(path)
    if not req_path.is_file():
        raise ValueError(f"requirements file not found: {path!r}")

    pinned: dict[str, str] = {}
    for lineno, raw in enumerate(req_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, version = line.partition("==")
        name = name.strip()
        version = version.split(";")[0].strip()
        if not sep or not name or not version or " " in version or "=" in version:
            raise ValueError(f"{path}:{lineno}: not an ==-pinned entry: {raw!r}")
        base = name.split("[")[0].strip()
        if not base or base.endswith("["):
            raise ValueError(f"{path}:{lineno}: bad package name: {raw!r}")
        pinned[base] = version
    return pinned
