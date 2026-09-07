# Roadmap — Week 2 Day 4: Cross-Entropy & KL Divergence

**Current:** Month 2, Week 2, Day 4 (global Day 10) — *the cost of using the wrong distribution = the loss function*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` first (start with 3Blue1Brown Part 2, ~32 min, then StatQuest Cross Entropy) — ~50 min for the two core videos, plus the short MI one if asymmetry still feels abstract. Focus on how `H(P, Q)` falls out of "compressing with the wrong model" and why the KL direction matters.
2. **Math on paper (20-30 min, do not skip):**
   - Write both formulas from memory: `H(P, Q) = -Σ P(x) log Q(x)` and `D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))`.
   - Hand-compute: `CE([0.5,0.5], [0.9,0.1]) ≈ 1.737`; `KL(P||Q) ≈ 0.737`, `KL(Q||P) ≈ 0.531` — confirm the two directions differ.
   - Reason through why `CE ≥ H` always (wrong model can only add cost) and why `CE − H = KL` makes the two minimisations equivalent.
3. **Code — `cross_entropy` + `kl_divergence`** (`coding_problems.md` Problems 1-2): the one-line formulas with `math.log`, the `q(x) == 0 where p(x) > 0` guard raising `ValueError`, `0 * log 0 = 0` convention where `p(x) == 0`. Confirm the three hand computations above.
4. **Code — `binary_log_loss`** (Problem 3): mean cross-entropy over samples, strict `(0, 1)` range on predictions. Leave the confident-wrong vs confident-correct prints in `__main__`.
5. **Experiments** (Problem 3): run the `__main__` demos — confirm the `3.322` vs `0.152` pair, the asymmetry print, and the `CE ≥ H` demo on at least two `(p, q)` pairs.
6. **Practice questions** (in `learn.md`) on paper before step 5 — if you can't do the `1.737` computation without code, you don't know the math yet.
7. **Commit & run `/done`:** `python -m pytest code/tests/ -v` must pass before the quiz. `/done` will quiz you on the CE hand value, the asymmetry argument, the confident-wrong loss, and the `CE − H = KL` identity.

---

## How this ties to the final project (`minisklearn`)

This day is the **loss-function implementation** day. `binary_log_loss` is the miniature version of the BCE loss Week 3 Day 2's Logistic Regression minimises, and Week 4's `minisklearn` evaluation reports cross-entropy alongside accuracy — today's functions (or their direct descendants) land in `metrics.py` / `linear_model.py`. The KL-asymmetry intuition also foreshadows Week 3 Day 4: information gain is a KL-shaped quantity, and the direction (split reality vs prior) is what makes it meaningful. Keep both files.

---

## Files in this day

- `learn.md` — what to learn + verified videos
- `coding_problems.md` — problem statements (intermediate scale)
- `code/cross_entropy.py` — stubs for `cross_entropy` + `kl_divergence`
- `code/log_loss.py` — stub for `binary_log_loss` + runnable `__main__` experiments
- `code/tests/` — pytest suite (must pass before `/done`)
- `further_reading.md` — appended only if you run `/for-read` (opt-in)
