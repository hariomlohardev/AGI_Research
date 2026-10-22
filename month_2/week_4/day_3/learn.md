# Day 21 — Evaluate: confusion matrix, accuracy/precision/recall/F1

**Why this matters (from the plan):** a model that "runs" isn't the same
as a model that "works." Today is entirely about honestly measuring what
you built — your `minisklearn` models, your `minisklearn` metrics, real
Titanic test data.

## Watch first

1. **Machine Learning Fundamentals: The Confusion Matrix** — StatQuest
   with Josh Starmer (~7 min) — https://www.youtube.com/watch?v=Kdsp6soqA7o
2. **Machine Learning Fundamentals: Sensitivity and Specificity** —
   StatQuest with Josh Starmer (short) — https://www.youtube.com/watch?v=vP06aMoz4v8
   (sensitivity *is* recall under another name; pairs with specificity to
   round out all four cells)
3. **ROC and AUC, Clearly Explained!** — StatQuest with Josh Starmer
   (~20 min, bonus — threshold thinking) — https://www.youtube.com/watch?v=4jRBRDbJemM

> Note: the plan's original confusion-matrix link (a StatQuest upload)
> turned out to be a removed video — it was swapped for the re-upload
> above, verified live at fix time against StatQuest's own official
> video index (statquest.org/video_index.html) plus a watch-page fetch
> with no unavailable/removed wording. The dead ID fails those same
> checks, so they're confirmed to discriminate. Runtimes are
> approximate — glance at the durations on the page. Take the metric definitions, not anyone's library code: you are
hand-rolling all of this in stdlib Python.

## The math (paper, Problem 1)

From the four cells — TP, FP, TN, FN:

- **Accuracy** = (TP + TN) / everything. Fraction exactly right.
- **Precision** = TP / (TP + FP). Of the predicted positives, how many
  were real? Punishes false alarms.
- **Recall** = TP / (TP + FN). Of the real positives, how many did you
  catch? Punishes misses. (StatQuest calls this sensitivity.)
- **F1** = 2·P·R / (P + R) — the *harmonic* mean of precision and
  recall. Harmonic, not arithmetic, so it collapses toward the worse of
  the two: a model with P = 1.0 and R = 0.0 scores F1 = 0, not 0.5.
  That's the whole point — one number that refuses to let either half
  hide.

Hand-work the `TN = 50, FP = 10, FN = 5, TP = 35` example in
`coding_problems.md` before touching code.

## The workflow (Problems 2–3, see `roadmap.md`)

Rebuild `(X, y)` from the raw CSV → stratified-split with your own
Day-20 `split.py` → train `LogisticRegression` + `RandomForestClassifier`
→ score with `minisklearn.metrics` → print the comparison table → read
5 misclassified rows and write the pattern sentence.

**Day-13 lesson, still loaded:** unscaled features (Fare reaches the
hundreds) plus the default `lr=0.01` diverged on Day 13. Use a smaller
`lr` (or standardize first) and confirm the loss actually falls before
you trust any metric in the table.
