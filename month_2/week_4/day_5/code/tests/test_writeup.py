"""Tests for Day 23 write-up deliverables (shape checks, not prose grades).

The deliverables are two markdown files, so there are no code stubs today:
- `month_2/minisklearn/README.md` (Problem 1, ships with the package)
- `month_2/week_4/day_5/blog_post.md` (Problem 2, published free)

Everything here fails until those files exist with the required shape —
each test is a task in coding_problems.md. Clarity, correctness of the
numbers, and honesty are graded by the learner (Problem 3) and `/done`'s
quiz, not by these assertions.
"""

import pathlib
import re

import pytest

DAYDIR = pathlib.Path(__file__).resolve().parents[1]
MONTH2 = pathlib.Path(__file__).resolve().parents[3]
README = MONTH2 / "minisklearn" / "README.md"
BLOG = DAYDIR / "blog_post.md"

MODELS = (
    "LinearRegression",
    "LogisticRegression",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "KMeans",
)


def _read(path):
    assert path.is_file(), f"missing deliverable: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"deliverable is empty: {path}"
    return text


def test_readme_has_title_and_sections():
    text = _read(README)
    assert re.search(r"^# .*\bminisklearn\b", text, re.M)
    lowered = text.lower()
    assert "install" in lowered
    assert "usage" in lowered or "example" in lowered


def test_readme_names_models_and_shows_imports():
    text = _read(README)
    assert sum(1 for name in MODELS if name in text) >= 3
    assert "import minisklearn" in text
    assert ".fit(" in text
    assert "```" in text  # runnable example in a fenced code block


def test_blog_post_has_headline_and_body():
    text = _read(BLOG)
    match = re.search(r"^# (.+)$", text, re.M)
    assert match is not None
    assert len(match.group(1).strip()) >= 20
    assert len(text.split()) >= 300


def test_blog_post_reports_real_numbers():
    text = _read(BLOG)
    assert "sklearn" in text.lower()
    assert re.search(r"\d", text) is not None
