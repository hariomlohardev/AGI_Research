"""Deterministic notebook-to-Python export, read-only checks, and explicit output cleaning.

Usage: python scripts/notebooks.py sync|check|clean --month 3 [--day WEEK DAY]
No command executes a notebook. Execute trusted smoke notebooks separately with nbclient.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_text(notebook: dict) -> str:
    import jupytext
    import nbformat
    node = nbformat.reads(json.dumps(notebook), as_version=4)
    nbformat.validate(node)
    return jupytext.writes(node, fmt="py:percent", notebook_metadata_filter="-all",
                           cell_metadata_filter="-all").replace("\r\n", "\n")


def source_path(notebook: Path, month_root: Path) -> Path:
    relative = notebook.resolve().relative_to((month_root / "notebooks").resolve())
    if relative.suffix != ".ipynb" or any(not part.isidentifier() for part in relative.with_suffix("").parts):
        raise ValueError("notebook folder/stem must be valid Python identifiers")
    number = int(month_root.name.removeprefix("month_"))
    return month_root / "src" / f"month{number}" / relative.with_suffix(".py")


def is_main_guard(node: ast.AST) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__" and not node.orelse)


def import_safety_errors(text: str) -> list[str]:
    """Conservative syntactic gate; never import or execute untrusted study code."""
    errors = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"invalid Python at line {exc.lineno}: {exc.msg}"]
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if is_main_guard(node):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
                continue
            except (ValueError, TypeError):
                pass
        errors.append(f"line {node.lineno}: put execution in a function and a __main__ guard")
    return errors


def notebook_errors(notebook: dict, *, clean: bool = True) -> list[str]:
    errors = []
    if notebook.get("metadata", {}).get("widgets"):
        errors.append("widget state must be removed before publication")
    for index, cell in enumerate(notebook.get("cells", []), 1):
        if cell.get("attachments"):
            errors.append(f"cell {index}: export attachments to reviewed result files")
        if cell.get("cell_type") != "code":
            continue
        code = cell.get("source", "")
        if isinstance(code, list):
            code = "".join(code)
        if re.search(r"(?m)^\s*[%!]|\bget_ipython\s*\(", code):
            errors.append(f"cell {index}: move shell/magic setup into a local setup notebook")
        if clean and (cell.get("outputs") or cell.get("execution_count") is not None):
            errors.append(f"cell {index}: clear outputs/execution count before publication")
    try:
        errors.extend(import_safety_errors(export_text(notebook)))
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(f"invalid notebook: {exc}")
    return errors


def clean_notebook(notebook: dict) -> None:
    metadata = notebook.setdefault("metadata", {})
    for key in list(metadata):
        if key not in {"kernelspec", "language_info", "jupytext"}:
            del metadata[key]
    for cell in notebook.get("cells", []):
        cell["metadata"] = {key: value for key, value in cell.get("metadata", {}).items() if key == "tags"}
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def check_pair(notebook: dict, source: str | None, *, clean: bool = True) -> list[str]:
    errors = notebook_errors(notebook, clean=clean)
    if source is None:
        errors.append("missing Python mirror; run notebooks.py sync")
    elif export_text(notebook) != source.replace("\r\n", "\n"):
        errors.append("Python mirror differs from notebook inputs; run sync and stage both")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sync", "check", "clean"])
    parser.add_argument("--month", type=int, default=3)
    parser.add_argument("--day", type=int, nargs=2, metavar=("WEEK", "DAY"))
    args = parser.parse_args()
    if args.month < 3 or (args.day and min(args.day) < 1):
        parser.error("use a positive Week/Day in Month 3 or later")
    month_root = ROOT / f"month_{args.month}"
    base = month_root / "notebooks"
    if args.day:
        base /= f"week_{args.day[0]}/day_{args.day[1]}"
    paths = sorted(base.rglob("*.ipynb")) if base.exists() else []
    if not paths:
        print("No notebooks in this scope yet; no learning checks were passed.")
        return 1 if args.day else 0
    failed = False
    for path in paths:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            source = source_path(path, month_root)
            if args.action == "clean":
                # Input is changed only by this explicitly mutating command.
                clean_notebook(notebook)
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"cleaned {path.relative_to(ROOT)}")
                continue
            errors = notebook_errors(notebook, clean=args.action == "check")
            if args.action == "sync" and not errors:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(export_text(notebook), encoding="utf-8", newline="\n")
            elif args.action == "check":
                errors = check_pair(notebook, source.read_text(encoding="utf-8") if source.exists() else None)
            if errors:
                failed = True
                print(f"FAIL {path.relative_to(ROOT)}: " + "; ".join(errors))
            else:
                print(f"OK {path.relative_to(ROOT)} -> {source.relative_to(ROOT)}")
        except (ValueError, OSError) as exc:
            failed = True
            print(f"FAIL {path.relative_to(ROOT)}: {exc}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
