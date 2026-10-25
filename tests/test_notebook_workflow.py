"""Contract tests for notebook export and staged publication, not ML exercise answers."""
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from curriculum_paths import day_paths
from notebooks import check_pair, clean_notebook, export_text, import_safety_errors, source_path
from check_public import audit, private_path


def notebook(code="def square(x):\n    return x * x\n\nif __name__ == '__main__':\n    assert square(3) == 9\n"):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# Test fixture"), nbformat.v4.new_code_cell(code)])


def test_month3_paths_separate_docs_code_tests_and_allow_eighth_day():
    paths = day_paths(3, 1, 8)
    assert paths["docs"] == "month_3/docs/days/week_1/day_8"
    assert paths["source"] == "month_3/src/month3/week_1/day_8"
    assert len({paths[k] for k in ("docs", "source", "tests", "notebooks")}) == 4
    assert day_paths(2, 1, 1)["tests"] == "month_2/week_1/day_1/code/tests"
    assert day_paths(3, 1, 1, "tiny-check")["tests"].endswith("micro_projects/tiny_check")


@pytest.mark.parametrize("values", [(0, 1, 1), (3, -1, 1), (True, 1, 1)])
def test_invalid_path_numbers_rejected(values):
    with pytest.raises(ValueError):
        day_paths(*values)


def test_path_traversal_slug_rejected():
    with pytest.raises(ValueError):
        day_paths(3, 1, 1, "../../private")


def test_export_preserves_definitions_and_does_not_execute_main():
    text = export_text(notebook())
    namespace = {"__name__": "fixture"}
    exec(compile(text, "fixture.py", "exec"), namespace)
    assert namespace["square"](4) == 16
    assert check_pair(notebook(), text) == []
    assert check_pair(notebook(), text.replace("x * x", "x + x"))


def test_export_rejects_eager_training_but_allows_explicit_main_guard():
    assert import_safety_errors("model = train_for_hours()\n")
    assert import_safety_errors("train_for_hours()\n")
    assert import_safety_errors("def broken(:\n")
    assert not import_safety_errors("CONFIG = {'mode': 'smoke'}\ndef run():\n    pass\nif __name__ == '__main__':\n    run()\n")


def test_cleaning_is_explicit_and_preserves_inputs():
    nb = notebook()
    original = nb.cells[1].source
    nb.cells[1].execution_count = 1
    nb.cells[1].outputs = [nbformat.v4.new_output("stream", text="local output", name="stdout")]
    nb.metadata["widgets"] = {"private": "state"}
    assert check_pair(nb, export_text(nb))
    clean_notebook(nb)
    assert nb.cells[1].source == original
    assert not nb.cells[1].outputs and nb.cells[1].execution_count is None
    assert check_pair(nb, export_text(nb)) == []


def test_magic_is_not_silently_exported_as_a_python_solution():
    assert any("magic" in e for e in check_pair(notebook("%pip install torch"), None))


def test_mirror_path_cannot_escape_notebook_tree(tmp_path):
    month = tmp_path / "month_3"
    assert source_path(month / "notebooks/week_1/day_8/tensors.ipynb", month) == month / "src/month3/week_1/day_8/tensors.py"
    with pytest.raises(ValueError):
        source_path(tmp_path / "outside.ipynb", month)
    with pytest.raises(ValueError):
        source_path(month / "notebooks/bad-name.ipynb", month)


@pytest.mark.parametrize("name", ["state.json", ".claude/skills/done/SKILL.md", "month_3/docs/days/week_1/day_1/code_review.md", ".env.secret", "month_3/checkpoints/best.pth", "month_3/data/raw/train.csv", "month_3/data/MNIST/train-images.gz", "wandb/run-1/config.yaml"])
def test_private_paths_blocked(name):
    assert private_path(name)


@pytest.mark.parametrize("name", [".env.example", "month_3/docs/reports/result.md", "month_3/tests/test_model.py", "month_3/notebooks/week_1/day_1/tensors.ipynb"])
def test_public_paths_allowed(name):
    assert not private_path(name)


def git(tmp_path, *args):
    subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)


def stage_pair(tmp_path):
    git(tmp_path, "init", "-q")
    nb = notebook()
    npath = tmp_path / "month_3/notebooks/week_1/day_1/example.ipynb"
    source = tmp_path / "month_3/src/month3/week_1/day_1/example.py"
    npath.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    npath.write_text(nbformat.writes(nb), encoding="utf-8")
    source.write_text(export_text(nb), encoding="utf-8")
    git(tmp_path, "add", "month_3")
    return npath, source


def test_public_check_reads_index_not_unstaged_working_copy(tmp_path):
    npath, source = stage_pair(tmp_path)
    assert audit(tmp_path) == []
    source.write_text("raise RuntimeError('unstaged')\n", encoding="utf-8")
    assert audit(tmp_path) == []
    git(tmp_path, "add", str(source))
    assert any("mirror differs" in e for e in audit(tmp_path))


def test_clean_worktree_cannot_hide_dirty_staged_notebook(tmp_path):
    npath, source = stage_pair(tmp_path)
    nb = json.loads(npath.read_text(encoding="utf-8"))
    nb["cells"][1]["execution_count"] = 10
    npath.write_text(json.dumps(nb), encoding="utf-8")
    git(tmp_path, "add", str(npath))
    clean_notebook(nb)
    npath.write_text(json.dumps(nb), encoding="utf-8")
    assert any("execution count" in e for e in audit(tmp_path))


def test_force_staged_private_file_is_rejected_and_no_value_is_reported(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "state.json").write_text('{"personal":"do-not-print"}', encoding="utf-8")
    git(tmp_path, "add", "state.json")
    errors = audit(tmp_path)
    assert errors and "do-not-print" not in str(errors)


def test_clean_preserves_markdown_attachments_for_explicit_review():
    nb = notebook()
    nb.cells[0]["attachments"] = {"figure.png": {"image/png": "fixture"}}
    clean_notebook(nb)
    assert nb.cells[0]["attachments"]
    assert any("attachments" in e for e in check_pair(nb, export_text(nb)))


def test_smoke_runner_requires_an_explicit_small_run_mode():
    from execute_notebook import smoke_config
    assert smoke_config(notebook("CONFIG = {'mode': 'smoke'}"))
    assert not smoke_config(notebook("CONFIG = {'mode': 'train'}"))
    assert not smoke_config(notebook("CONFIG = get_old_config()"))
