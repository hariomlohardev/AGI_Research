# AI research learning portfolio

A personal path toward AI/AGI research through mathematical reasoning, implementation, controlled experiments and reproducible evidence. Month 2's classical-ML work is preserved. Month 3 is prepared around the approved V5 roadmap; its learning days have not been generated or marked complete by this setup.

- [Month 3 roadmap](month_3/docs/roadmap.md)
- [Month 3 folder guide](month_3/README.md)
- [Notebook and Python workflow](docs/system/notebook-workflow.md)
- [What is published](docs/system/publishing.md)
- [Month 2 minisklearn](month_2/minisklearn/README.md)
- [State schema](STATE_SCHEMA.md)

For each assigned mechanism, derive it, implement it and verify it. Use established libraries for routine data manipulation, preprocessing, metrics, plotting and ordinary training operations. Scientific claims should match the data, compute budget and uncertainty actually measured.

## Setup and checks

Create a Python 3.12 project environment and install requirements-dev.txt. Use that environment's interpreter; on Windows it is .venv/Scripts/python.exe. Install learning dependencies separately for the selected month/runtime. requirements.txt retains the legacy Month 2 dependencies. Configure the local hook with git config core.hooksPath .githooks.

    python -m pytest tests -q
    python scripts/notebooks.py check --month 3
    python scripts/run_curriculum_tests.py
    python scripts/check_public.py --worktree

The last two checks require the corresponding curriculum dependencies and reviewed source files. See the workflow for export, cleaning and clean-kernel execution. Public CI uses sanitized fixtures rather than personal progress files. The optional local dashboard lives under web/; its tests/build can run without personal state, while its live view needs locally initialized state.json.

The learning controller, coaching history, scratch experiments, datasets and model checkpoints stay local. Public artifacts include reviewed documentation, code, tests, clean notebooks, configs and small result summaries. This work builds evidence and research habits; roadmap completion alone does not establish professional research readiness or general intelligence.
