# Roadmap — Week 4 Day 2: Apply to a Real Dataset (Titanic prep)

**Current:** Month 2, Week 4, Day 2 (global Day 20) — *your models meet the wild*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (steal the EDA checklist, not the
   pandas code — mirror each section on Titanic with your own helpers).
2. **No paper derivation today** — build day. Skim
   `month_2/final_project/structure.md` once to see where today's arrays
   get consumed (Days 3–6).
3. **EDA** (`coding_problems.md` Problem 1, `code/eda.py`): load →
   missing report (confirm 177/2 against the vendored pins — mismatch
   means your loader is wrong) → class balance (write down the
   always-predict-0 accuracy) → Age/Fare summaries.
4. **Impute + encode** (Problem 2, `code/encode.py`): parse → mean-fill
   Age (keep the mean — Day 4 needs it) → label/one-hot Sex/Embarked →
   your documented feature-selection call → fully-numeric `X`, `y`.
5. **Stratified split** (Problem 3, `code/split.py`): seeded local RNG,
   ratio from each class; assert totals sum back to 549/342 and each
   side sits near 38% positive.
6. **Prove it by printing:** shapes of all four arrays, positive rate
   per side, and one line per helper decision (fill value, mappings,
   dropped rows). This printout is the day's Done when, next to the
   suite.
7. **Run the suite:** `cd month_2/week_4/day_2 && py -m pytest code/tests/ -v`
8. **Run `/done`:** tests, then code review, explain-back, and a quiz on
   imbalance-vs-accuracy, imputation trade-offs, stratification
   rationale, and your encoding calls — especially the ones you defended
   in a comment.

---

## How this ties to the final project (`minisklearn`)

Today produces the fuel: the four arrays Days 3–6 burn. Day 3 trains
your models on them and scores with your metrics; Day 4 reruns the same
pipeline against real `sklearn` (your fill value and mappings must be
on record, or the comparison is rigged); Day 5 writes up whatever the
numbers say. Keep `code/data/titanic.csv` untouched — it's the shared
ground truth, pinned by the tests.

---

## Files in this day (and beyond it)

- `learn.md` — EDA theory + verified videos (long picks per your ask)
- `coding_problems.md` — EDA → impute/encode → split spec + CSV pins
- `roadmap.md` — execution order and final-project tie
- `code/eda.py`, `code/encode.py`, `code/split.py` — your stubs
- `code/data/titanic.csv` — vendored dataset (Days 2–6 share it)
- `code/tests/` — pytest suite (synthetic fixtures + CSV pins)
