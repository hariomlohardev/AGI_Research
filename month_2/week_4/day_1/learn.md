# Day 19 — Package Structure (Week 4 Day 1, Month 2)

**Topic:** turning five standalone scripts into one importable library — package layout, `__init__.py` as public interface, relative imports, uniform `.fit()`/`.predict()`
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> No math to derive today — build day. No dataset either: today's test data is the same tiny synthetic fragments from Weeks 1–3 (a few rows each — the point is wiring, not modeling). Code is stdlib-only; the package under construction is `month_2/minisklearn/`.

## Why this matters

Weeks 1–3 produced a pile of scripts that only run from their own folders. A pile of scripts isn't a portfolio piece and isn't usable — nobody can `import` your forest into their script. Today converts the pile into `minisklearn/`: five modules behind one front door, every model speaking `.fit(X, y)` / `.predict(X)`. That uniformity is load-bearing, not cosmetic — it's what lets Day 3's evaluation harness loop over all models with zero glue code, and Day 4 run the identical loop against real `sklearn`. Read `month_2/final_project/structure.md` once before coding: it's the 6-day map, and the skeleton section is today's spec. (And a habit note from the Day-18 review: relative imports between siblings, `from .tree import ...` — never a copied file, never a path hack inside the package.)

## Watch (in this order)

1. **What does '__init__.py' do in Python?** — Tech With Tim — https://www.youtube.com/watch?v=VEbuZox5qC4
   Concepts pick, code-on-screen throughout (not slides): modules & packages → `__init__` usage → relative imports & import errors → multiple packages. The chapters are today's syllabus in order, and the relative-imports-plus-errors section is the exact failure mode you'll hit wiring `minisklearn/` submodules together. Compact — main content ends around the 10-minute mark.

2. **What should you write into the __init__.py file? 2MinutesPy** — 2MinutesPy — https://www.youtube.com/watch?v=eUqLH1m92Yc
   The clean-API half, ~3 minutes total: re-exporting specific names (`from .cluster import KMeans` — the pattern behind `from minisklearn import KMeans`), wildcard control via `__all__`, package metadata. Watch immediately before writing `month_2/minisklearn/__init__.py` — it turns that file from "empty marker I'm told to create" into the deliberate public-interface decision it is.

3. **How to Package and Publish Python Code the Right Way** — ArjanCodes — https://www.youtube.com/watch?v=5KEObONUkik
   Backup pick, watch only the first half (~0:00–11:00: layout + building/installing so a package imports from anywhere); treat the PyPI-publishing tail as optional. Two honest caveats: it's the longest of the three (~20 min), and it frames things around `setup.py`, which is legacy in favor of `pyproject.toml` — take the layout and build/install *concepts*, don't copy the `setup.py` mechanics verbatim. Reach for it if `import minisklearn` won't work from a sibling script.

(All three verified live at generation time. Deliberately short/build-first picks after the Day-18 2/10 verdict — no 40-minute lectures today.)

## Math — none today

Build day per the source plan. The derivations are done and banked — today they get *used* (every formula from Weeks 1–3 now has to survive living inside a package with callers other than you).

## Code — see `coding_problems.md`

Move map (copies, never cuts) → normalize to `.fit/.predict` (+ `.predict_proba` on logistic; `KMeans.fit(X)` asymmetry documented) → `metrics.py` (`confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, new `f1_score`, binary labels, `0.0` zero-division conventions) → wire `__init__.py` (ten top-level names) → prove it from a *different* directory.

## Practice questions (source plan)

1. Why is a consistent `.fit()`/`.predict()` interface across all your models valuable, beyond just "looking professional"? (Think about what it lets you do with a loop over multiple models.)
2. What's the difference between a package `__init__.py` that's empty vs. one that explicitly imports and exposes your classes (e.g. `from .linear_model import LinearRegression`)? Which is more convenient for a user of your library, and why?
