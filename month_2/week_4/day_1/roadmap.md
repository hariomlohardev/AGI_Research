# Roadmap — Week 4 Day 1: Package Structure (`minisklearn`)

**Current:** Month 2, Week 4, Day 1 (global Day 19) — *stop writing isolated scripts; build the library*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (packaging intuition → `__init__.py` +
   imports mechanics, right before touching `minisklearn/`).
2. **No paper derivation today** — build day. Read `month_2/final_project/structure.md`
   once instead: it's the 6-day map and today's skeleton section is the spec.
3. **Move the models** (`coding_problems.md` Problem 1): copy (never cut —
   Weeks 1–3 are the archive) Day 13→`linear_model.py` (with Day 14's
   logistic), Day 16→`tree.py`, Day 17→`ensemble.py`, Day 18→`cluster.py`.
   Normalize to `.fit(X, y)` / `.predict(X)` (+ `.predict_proba` on logistic),
   rename/alias to sklearn-style names, sibling imports relative (`from .tree
   import ...`). Delete each MOVE MAP block as it lands.
4. **Write `metrics.py`** (Problem 2): adapt Day 14's trio, add `f1_score`,
   all five with the binary-label + zero-division conventions. Hand-check
   `[[2,1],[1,2]]` first, then trust the suite.
5. **Wire `__init__.py`** (Problem 3): explicit re-exports for all ten
   top-level names.
6. **Prove it from outside:** `cd` anywhere else and run
   `python -c "import minisklearn; m = minisklearn.linear_model.LogisticRegression(); ..."`
   — submodule path AND top-level names, fit + predict on a toy set. This is
   the day's Done when, not a nice-to-have.
7. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
8. **Run `/done`:** tests, then code review, explain-back, and a quiz on
   interface design (why uniformity buys you the model loop), packaging
   mechanics (`__init__.py`, relative imports), and the metric definitions —
   especially precision vs recall vs F1 and when each lies.

---

## How this ties to the final project (`minisklearn`)

Today *is* the final project, day one of six: the package you create is the
artifact Days 2–6 operate on (real data → eval → sklearn showdown → write-up
→ GitHub). The uniform `.fit/.predict` interface is load-bearing — Day 3's
evaluation harness loops over models without glue code, and Day 4 runs the
identical loop against real `sklearn`. Keep every Week 3 test green in its
home folder; the package copies are the ones allowed to evolve.

---

## Files in this day (and beyond it)

- `learn.md` — packaging theory + verified videos
- `coding_problems.md` — move map + metrics spec + `__init__` contract
- `roadmap.md` — execution order and final-project tie
- `code/tests/` — pytest suite (imports + interface uniformity + smoke fits + metrics hand values)
- `month_2/minisklearn/` — THE package (month-level: Days 2–6 build on it)
