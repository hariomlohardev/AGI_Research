# Roadmap — Week 3 Day 3: Support Vector Machines & the Kernel Trick

**Current:** Month 2, Week 3, Day 3 (global Day 15) — *maximum margin instead of maximum likelihood*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` (SVM margins → the kernel trick).
2. **Derive on paper:** the margin picture (`w·x + b = 0`, support vectors),
   then the hinge loss `max(0, 1 − y·(w·x+b))` case-by-case, then the
   subgradient rule (violators pull, comfortable points don't).
3. **Implement** `code/svm.py` (`hinge_loss`, `hinge_subgradients`,
   `LinearSVMGD` with `fit`/`decision_function`/`predict` + `l2` support,
   `polynomial_kernel`, `add_radius_feature`).
4. **Run the demos:** `python code/svm.py` (clean separable fit with
   support-vector count, ring failure, kernel ordering, radius-feature
   rescue) — note the `{-1, +1}` label convention shift from Day 14.
5. **Plot (optional but recommended):** the `__main__` boundary + margin
   figure for the blobs, next to the ring scatter that defeats it.
6. **Run the suite:** `python -m pytest code/tests/ -v` from this day's folder.
7. **Run `/done`:** tests, then code review, explain-back, and a quiz on the
   hinge-vs-cross-entropy zero, support vectors, the kernel trick bargain,
   and the hand-built radius feature.

---

## How this ties to the final project (`minisklearn`)

This is Week 4's `cluster.py`-neighbour clarity plus a future `svm.py`
candidate: the `fit`/`predict` interface matches Days 13–14 exactly, so a
`LinearSVM` drops into the model loop unchanged. The deeper tie is the loss
contrast — after two days of likelihood losses (MSE, BCE), today is the first
margin loss, and Week 4's comparison table gets more interesting for it. The
ring-vs-radius lesson (features beat algorithms) returns on Week 4 Day 2 when
real-data encoding decisions matter more than model choice.

---

## Files in this day

- `learn.md` — margin/hinge/kernel theory + verified videos
- `coding_problems.md` — build spec with complete contracts
- `roadmap.md` — execution order and final-project tie
- `code/svm.py` — hinge + subgradient-GD classifier + kernel stub with runnable experiments
- `code/tests/` — pytest suite (SVM behaviour + kernel values + ring fail/rescue)
