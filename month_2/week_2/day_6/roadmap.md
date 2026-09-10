# Roadmap — Week 2 Day 6: Review + Consolidation

**Current:** Month 2, Week 2, Day 6 (global Day 12) — *turn seven isolated functions into one retrievable story*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Retrieve first (30–45 min):** close your notes and write the seven formulas listed in `learn.md`. Mark the ones you cannot reproduce; those are today's review targets.
2. **Watch the verified review videos** in `learn.md`. Re-watch the concept you marked weakest first, then use the second video to connect the week's parameter-estimation and information-theory halves.
3. **Write the two summary paragraphs** from the practice questions before opening the editor: Bayes/MLE/MAP, then entropy/cross-entropy/KL/MI.
4. **Consolidate:** implement all seven functions in `code/week2_utils.py` without importing the earlier day modules. Keep validation explicit and put the full `ValueError` contract in every public docstring.
5. **Add the seven runnable assertions** under `if __name__ == "__main__":`; use tolerances for floating-point identities, never exact equality for derived floats.
6. **Run both checks:** `python code/week2_utils.py`, then `python -m pytest code/tests/ -v`.
7. **Run `/done`:** after the tests pass, `/done` will ask for today's explain-back and then run the Week 2 review against the stored `done_when` criterion.

---

## How this ties to the final project (`minisklearn`)

`week2_utils.py` is a deliberately small compatibility layer for the final project: MLE/MAP explain model fitting and regularization; entropy, cross-entropy, KL, and MI become the loss, evaluation, and feature-selection vocabulary used by `minisklearn`. The Week 3 decision tree will reuse entropy/information gain, while logistic regression will reuse binary cross-entropy. Consolidating now makes those later implementations deliberate reuse instead of disconnected copies.

---

## Files in this day

- `learn.md` — week synthesis, formulas, practice questions, verified videos
- `coding_problems.md` — consolidation task and complete contracts
- `roadmap.md` — execution order and final-project tie
- `code/week2_utils.py` — seven-function consolidation stub + runnable assertions
- `code/tests/` — pytest suite for the consolidated behavior
- `code_review.md` — written by `code-evaluator` during `/done`
