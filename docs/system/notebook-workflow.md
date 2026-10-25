# Notebook work and Python deliverables

This repository uses notebooks as the learning workspace and Python exports as reviewable, importable deliverables. The approved Month 3 V5 goals still govern what is learned. The new repository layout adds the Python mirrors the learner requested; it does not require separate implementations of the same exercise.

## File locations

| Kind | Month 3 path |
|---|---|
| Markdown lessons, tasks, derivations | month_3/docs/days/week_W/day_D/ |
| Implementation notebook | month_3/notebooks/week_W/day_D/task.ipynb |
| Mirrored Python input cells | month_3/src/month3/week_W/day_D/task.py |
| Independent tests | month_3/tests/week_W/day_D/test_task.py |
| Shared checked code | month_3/src/month3/shared/ |
| Curated report | month_3/docs/reports/ |
| Small public result tables | month_3/results/summary/ |
| Local datasets/checkpoints/runs | month_3/data/, checkpoints/, runs/ |

Use day_in_week in filenames. Week 1 Day 8 is valid; global_day is only the month-wide progress counter. Resolve paths with:

    python scripts/curriculum_paths.py 3 1 8

Earlier Month 2 files remain at their existing paths. Standard entry points such as README.md, CLAUDE.md and configuration files retain their conventional locations. Personal reviews/flags/confusion notes are ignored wherever the day document resolver places them.

## Environment

On Windows select the project interpreter explicitly:

    py -3.12 -m venv .venv
    .venv/Scripts/python.exe -m pip install -r requirements-dev.txt

On POSIX use python3 -m venv .venv and .venv/bin/python. In examples below, python means that environment's interpreter, not a Windows Store alias. Python 3.12 is the tested tooling/CI baseline. A local 3.12 tooling environment has been prepared. To open notebooks, select its kernel or a compatible configured Jupyter/Colab environment. JupyterLab is an optional editor installation, not needed for the checks below.

Install a compatible PyTorch/torchvision pair for the actual CPU/CUDA runtime using the [official installer](https://pytorch.org/get-started/locally/). Record the tested runtime's direct requirements in month_3/requirements-runtime.txt when the first data/model exercise is prepared; do not guess GPU wheels or blindly upgrade a working Colab environment. Requirements-dev.txt pins tested workflow tools separately from the legacy Month 2 requirements. Export package-version records without credentials or machine-specific paths. The kernel used for smoke execution must contain the exercise's actual dependencies.

## Daily implementation

1. Read the source day's goals and distinguish the core mechanism from routine library work. Put derivation, prediction, assumptions and questions in Markdown cells; put experiments in named functions with explicit configs and budgets.
2. Work in a valid .ipynb under notebooks/. All package code needed by tests must be defined there or in a deliberate shared module. Use NumPy/pandas/torchvision/sklearn for standard data operations, losses/optimizers/autograd for ordinary PyTorch training, and plotting tools for figures. Keep assigned manual backprop, convolution and recurrent-cell exercises yours to implement.
3. Keep imported modules safe: top-level imports, definitions and literal configs; call training, downloads, plotting and file writes only from functions behind a __main__ guard. Such guards run in a normal notebook's __main__ kernel but do not run when pytest imports an exported module. Function defaults, decorators and class bodies must also avoid expensive side effects. The static check is conservative, not a security sandbox.
4. Use a smoke mode with tiny data/few updates as the default, with explicit full-training/resume/analysis choices. Mark outputs with run ID, seed, mode, split, budget and source revision. Recreate model/optimizer after a class change. Do not carry old experiment state forward accidentally.
5. Export the notebook inputs and test the exported code:

       python scripts/notebooks.py sync --month 3 --day 1 1
       python -m pytest -c month_3/pyproject.toml month_3/tests/week_1/day_1 -q

6. Execute the day's trusted smoke/check notebook after a kernel restart, in its documented lightweight mode. Record assertions, numerical errors and required experiment evidence locally. Executing imports or a returning stub does not count as passing the assigned task. Full training needs its own evidence and budget; rebuilding a figure from saved metrics is a separate check.
7. Save durable configs/results/manifests, then clear notebook outputs for publication and verify consistency:

       python scripts/notebooks.py clean --month 3 --day 1 1
       python scripts/notebooks.py check --month 3 --day 1 1

The source notebook's inputs are authoritative. The sync command intentionally writes the .py representation; inspect a diff first if either representation was independently edited. Restore the intended notebook inputs before syncing. Jupytext's optional editor pairing is configured by month_3/jupytext.toml; the repository's deterministic exporter/check is the final authority. Routine .py files such as __init__.py and deliberate shared helpers are allowed without an artificial notebook mirror.

The template is local under .claude/templates/study.ipynb. It contains a real unsolved placeholder and is not a completed day. /i-am-in creates day materials lazily from the current state. The prepared Month 3 folder does not start Month 3 or alter Month 2 progress.

## Scientific and publishing evidence

Tests live separately and check behavior: reference gradients, axes, split boundaries, target alignment, masks, parameter registration or checkpoint parity as required. Avoid tests that merely duplicate the implementation or demand an accuracy winner. Keep honest negative/inconclusive outcomes.

Store checkpoints, dataset downloads and raw logs locally; publish small curated metrics/configs and a manifest explaining how to obtain/reproduce required assets. Keep class/vocabulary mappings, preprocessing, state_dict/config identity, metric denominators and source revision with every result. Before a public commit, read [publishing.md](publishing.md).

The export script never executes notebooks. Clean-kernel execution is a separate step, using Jupyter/Colab or nbclient with a bounded timeout and output destination under .local. /done checks both the imported Python behavior and required notebook evidence. Tooling CI checks structure, exports and publication rules; it does not certify a learner's math or full training result.

## Sources

The workflow adapts [Jupytext's paired notebook approach](https://pypi.org/project/jupytext/) and its [separate-folder configuration](https://raw.githubusercontent.com/mwouts/jupytext/v1.16.4/docs/config.md). The pinned working exporter is tested here; documentation examples from another version should be checked against the installed package.

A trusted lightweight notebook can be executed with python scripts/execute_notebook.py <notebook-path> --timeout 180. It requires a literal CONFIG mode of smoke/check, starts a fresh kernel and saves the executed copy under ignored .local/notebook-runs. The timeout is per cell, so keep each smoke section small. This command leaves the authoritative source notebook and progress unchanged, preserving outputs locally even when execution fails. It is not a sandbox for untrusted notebooks. To import month3 shared helpers, install the local package with python -m pip install -e month_3 in the selected kernel environment.

After /done verifies a completed day, its reviewed public docs/notebook/Python/tests are committed and pushed to main automatically under the learner's standing authorization. Completed micro-projects follow the same rule; unfinished scaffolds and private notes stay local. See publishing.md for branch handling and Git-only retries.
