"""Reject local/private artifacts and unreviewable notebooks in Git content.

Default checks Git's staged blobs, not the working-tree versions.
--worktree audits tracked and unignored new files without staging them.
This is a conservative publishing check, not a comprehensive secret scanner.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from notebooks import check_pair

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT_FILES = {"CLAUDE.md", "CLAUDE.local.md", "state.json", "progress.md", "growth-notes.md",
                      "struggle-log.md", "discipline-log.md", "bonus-watches.md", "project-history.md"}
PRIVATE_PARTS = {".claude", ".local", ".remember", ".venv", "venv", "node_modules", ".vscode", ".idea",
                 ".ipynb_checkpoints", "__pycache__", ".pytest_cache", ".ruff_cache", ".benchmarks",
                 "checkpoints", "runs", "artifacts", "scratch", "github_contribution", "wandb", "lightning_logs", "mlruns"}
PRIVATE_NAMES = {"code_review.md", "flagged_questions.md", "confusion_notes.md"}
BULK_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npy", ".npz", ".parquet", ".pem", ".key", ".h5", ".hdf5", ".pkl", ".pickle"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|sk-ant-[A-Za-z0-9_-]{20,})")


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    if name in PRIVATE_ROOT_FILES or PRIVATE_PARTS.intersection(path.parts) or path.name in PRIVATE_NAMES:
        return True
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    return (bool(re.match(r"^month_\d+/data/", name)) or path.suffix.lower() in BULK_SUFFIXES or name.startswith(("reports/", "docs/maintenance/"))
            or bool(re.search(r"(?:^|/)data/(raw|processed)/|(?:^|/)results/raw/|(?:^|/)final_project/logs\.md$", name)))


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def audit(root: Path, *, worktree: bool = False) -> list[str]:
    args = ["ls-files", "-z", "--cached", "--others", "--exclude-standard"] if worktree else ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    names = sorted(set(git(root, *args).decode("utf-8").split("\0")) - {""})
    errors = []
    def read(name: str) -> bytes | None:
        if worktree:
            target = root / name
            if target.is_symlink():
                raise ValueError("symlinks require separate review")
            return target.read_bytes() if target.exists() else None
        item = subprocess.run(["git", "-C", str(root), "show", f":{name}"], capture_output=True)
        return item.stdout if item.returncode == 0 else None
    for name in names:
        if private_path(name):
            errors.append(f"{name}: local/private or bulk artifact must not be published")
            continue
        try:
            data = read(name)
            if data is None:
                continue
            if len(data) > 5 * 1024 * 1024:
                errors.append(f"{name}: exceeds the 5 MiB review limit; publish a manifest")
            if SECRET.search(data):
                errors.append(f"{name}: possible credential/private key; inspect locally (value withheld)")
            if name.endswith(".ipynb"):
                match = re.fullmatch(r"month_([3-9]|[1-9][0-9]+)/notebooks/(.+)\.ipynb", name)
                if not match:
                    errors.append(f"{name}: place publishable notebooks under month_N/notebooks")
                    continue
                number, relative = match.groups()
                mirror = f"month_{number}/src/month{number}/{relative}.py"
                source = read(mirror)
                for error in check_pair(json.loads(data), source.decode("utf-8") if source is not None else None):
                    errors.append(f"{name}: {error}")
            elif re.fullmatch(r"month_\d+/src/month\d+/.+\.py", name) and not name.endswith("/__init__.py"):
                # A changed paired export must match the notebook even if only .py was staged.
                match = re.fullmatch(r"month_(\d+)/src/month\d+/(.+)\.py", name)
                number, relative = match.groups()
                nbname = f"month_{number}/notebooks/{relative}.ipynb"
                nbdata = read(nbname)
                if nbdata is not None:
                    for error in check_pair(json.loads(nbdata), data.decode("utf-8")):
                        errors.append(f"{name}: {error}")
                else:
                    from notebooks import import_safety_errors
                    errors.extend(f"{name}: {error}" for error in import_safety_errors(data.decode("utf-8")))
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append(f"{name}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true")
    args = parser.parse_args()
    errors = audit(ROOT, worktree=args.worktree)
    for error in errors:
        print("FAIL " + error)
    print(f"Publishing check: {len(errors)} issue(s). Review the staged diff before committing.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
