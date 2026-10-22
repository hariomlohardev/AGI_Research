# Roadmap — Week 4 Day 3: Evaluate (honest metrics on Titanic)

**Current:** Month 2, Week 4, Day 3 (global Day 21) — *a model that runs isn't a model that works*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (confusion-matrix mechanics first,
   then what precision/recall/F1 each punish — take the metric
   definitions, not anyone's library code).
2. **Paper first** (`coding_problems.md` Problem 1): derive all four
   metrics from TP/FP/TN/FN by hand on the `50/10/5/35` example. If the
   hand numbers don't match, the code you write next is just a faster
   way to be wrong.
3. **Helpers** (Problem 2, `code/evaluate.py`): `load_titanic_xy`
   (rebuild yesterday's pipeline from the raw CSV — disagreements with
   yesterday's arrays stop the day), `comparison_table`, and
   `misclassified_indices`.
4. **The real run** (Problem 3): stratified-split with your own Day-20
   `split.py` → train `LogisticRegression` + `RandomForestClassifier`
   (mind the Day-13 learning-rate lesson on unscaled Fare — confirm the
   loss falls) → score with `minisklearn.metrics` → print the table →
   read 5 misclassified rows and write the one-sentence pattern note.
5. **Run the suite:** `cd month_2/week_4/day_3 && py -m pytest code/tests/ -v`
6. **Run `/done`:** tests, then code review, explain-back, and a quiz on
   precision-vs-recall trade-offs, the accuracy illusion on imbalance,
   what F1 is mathematically, and what your 5 errors actually looked like.

---

## How this ties to the final project (`minisklearn`)

Today is the first honest report card: your package's models scored by
your package's metrics on real data. Day 4 reruns this exact table
against real `sklearn` — keep today's numbers and the split seed on
record, because "ours vs theirs" is only meaningful on identical test
sets. Day 5 writes up whatever the table says, flattering or not.

---

## Files in this day (and beyond it)

- `learn.md` — metric derivations + verified videos
- `coding_problems.md` — hand derivation → helpers → two-model run spec
- `roadmap.md` — execution order and final-project tie
- `code/evaluate.py` — your stub (rebuild + table + error indices)
- `code/tests/` — pytest suite (synthetic + pins + one integration run)
