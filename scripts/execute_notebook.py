"""Execute one trusted smoke notebook in a fresh kernel; keep output local."""
from __future__ import annotations

import argparse
import ast
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def smoke_config(notebook) -> bool:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            return False
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets):
                try:
                    config = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return False
                return isinstance(config, dict) and config.get("mode") in {"smoke", "check"}
    return False


def execute(source: Path, root: Path = ROOT, *, timeout: int = 180, kernel: str = "python3") -> Path:
    source = source.resolve()
    source.relative_to(root.resolve())
    if source.suffix != ".ipynb":
        raise ValueError("expected an .ipynb notebook inside the project")
    notebook = nbformat.read(source, as_version=4)
    if not smoke_config(notebook):
        raise ValueError("set a literal CONFIG with mode='smoke' or 'check'; full runs are deliberate separate work")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / ".local" / "notebook-runs" / stamp / source.name
    output.parent.mkdir(parents=True, exist_ok=False)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    client = NotebookClient(notebook, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(root)}}, allow_errors=False)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=180, help="maximum seconds per cell")
    parser.add_argument("--kernel", default="python3", help="kernel containing the exercise dependencies")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    try:
        output = execute(args.notebook, timeout=args.timeout, kernel=args.kernel)
    except Exception as exc:
        print(f"Smoke execution failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"Smoke execution passed; executed copy: {output.relative_to(ROOT)}")
    print("The source notebook and study progress were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
