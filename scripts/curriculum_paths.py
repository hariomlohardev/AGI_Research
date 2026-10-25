"""Resolve Month 3+ study paths without creating files or changing progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def day_paths(month: int, week: int, day: int, slug: str | None = None) -> dict[str, str]:
    if any(type(value) is not int or value < 1 for value in (month, week, day)):
        raise ValueError("month, week and day must be positive integers")
    if slug is not None:
        import re
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            raise ValueError("micro-project slug must use lowercase words separated by hyphens")
    base = f"month_{month}"
    suffix = f"week_{week}/day_{day}"
    if month < 3:
        docs = f"{base}/{suffix}"
        if slug:
            docs += f"/micro_projects/{slug}"
        return {"docs": docs, "notebooks": f"{docs}/code", "source": f"{docs}/code",
                "tests": f"{docs}/code/tests", "final_project": f"{base}/final_project"}
    docs_suffix = suffix + (f"/micro_projects/{slug}" if slug else "")
    if slug:
        suffix += f"/micro_projects/{slug.replace(chr(45), chr(95))}"
    return {"docs": f"{base}/docs/days/{docs_suffix}",
            "notebooks": f"{base}/notebooks/{suffix}",
            "source": f"{base}/src/month{month}/{suffix}",
            "tests": f"{base}/tests/{suffix}",
            "final_project": f"{base}/docs/final_project"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("month", type=int)
    parser.add_argument("week", type=int)
    parser.add_argument("day", type=int, help="day_in_week, not global_day; Week 1 Day 8 is valid")
    parser.add_argument("--slug")
    args = parser.parse_args()
    try:
        print(json.dumps(day_paths(args.month, args.week, args.day, args.slug), indent=2))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
