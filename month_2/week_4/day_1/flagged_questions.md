# Flagged questions — Day 19 (Month 2 Week 4 Day 1)

## Q6 — metrics zero-division conventions (flagged 2026-10-14)

**Question:** Your metrics have zero-division conventions: precision is `0.0`
when there are no predicted positives, recall `0.0` when no actual positives,
F1 `0.0` when `P+R == 0`. Why return `0.0` in those cases instead of raising
an error or returning `None`?

**User's reason:** "i dont quiet understnad what you want from me" — asked for
clarification twice, then flagged. Not a claim the question is factually wrong;
a comprehension block on what the question was after.

**Note for review:** the intended point was loop-robustness (Day 3's
evaluate-every-model harness keeps running when one model predicts no
positives) plus honesty (`0.0` is the earned score — no correct positive
predictions were made). Revisit via `/spaced-review`.
