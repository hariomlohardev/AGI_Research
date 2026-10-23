# Roadmap — Week 4 Day 4: Compare Against Real `sklearn`

**Current:** Month 2, Week 4, Day 4 (global Day 22) — *same split, same ruler, honest gaps*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today’s build

1. **No videos today:** the source plan explicitly calls this a validation
   day. Read `learn.md`’s protocol instead of looking for tutorials.
2. **Reload yesterday exactly** (`coding_problems.md` Problem 1):
   `load_same_split` → 889 rows, 7 features, approximately 38% positives.
3. **Build the supervised harness** (Problem 2, `code/compare.py`):
   `score_classifier`, `accuracy_gap`, and `gap_verdict`.
4. **Run all four pairs** (Problem 3):
   logistic, tree, forest, and K-Means against the corresponding
   `sklearn` models.
5. **Compare correctly:**
   - classifiers with your `minisklearn.metrics`;
   - K-Means with inertia, not label equality.
6. **Investigate one discrepancy fully:** change one cause at a time,
   record settings, and write the honest paragraph.
7. **Run the suite:** `cd month_2/week_4/day_4 && py -m pytest code/tests/ -v`
8. **Run `/done`:** tests, code review, explain-back, and a quiz on
   same-split discipline, likely default-driven gaps, flat optima, and why
   validation outranks any single algorithm.

---

## How this ties to the final project (`minisklearn`)

Today turns “my package works” into “my package agrees with the standard
implementation for understandable reasons.” Keep the exact models,
hyperparameters, split seed, metrics, inertia values, gaps, and paragraph:
Day 5 writes them up, and Day 6 publishes them.

---

## Files in this day (and beyond it)

- `learn.md` — validation protocol and default-driven gap sources
- `coding_problems.md` — exact-split reload → supervised harness → four-model run
- `roadmap.md` — execution order and final-project tie
- `code/compare.py` — your stub: same split, scoring, accuracy gaps, verdicts
- `code/tests/` — pytest suite: contracts + logistic integration + K-Means smoke test
