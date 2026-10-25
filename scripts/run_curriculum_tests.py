"""Run legacy day suites in isolation and Month 3+ suites with their own config."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failed = False
    count = 0
    for month in sorted(ROOT.glob("month_[0-9]*")):
        if not month.is_dir():
            continue
        number = int(month.name.removeprefix("month_"))
        if number < 3:
            suites = sorted(month.glob("week_*/day_*/code/tests")) + sorted(month.glob("week_*/day_*/micro_projects/*/code/tests"))
            for suite in suites:
                count += 1
                result = subprocess.run([sys.executable, "-m", "pytest", str(suite), "--import-mode=importlib", "-q"], cwd=suite.parent)
                failed |= result.returncode != 0
        elif any((month / "tests").rglob("test_*.py")):
            count += 1
            result = subprocess.run([sys.executable, "-m", "pytest", "-c", str(month / "pyproject.toml"), str(month / "tests"), "-q"], cwd=ROOT)
            failed |= result.returncode != 0
    print(f"Curriculum suites: {count}; failures present: {failed}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
