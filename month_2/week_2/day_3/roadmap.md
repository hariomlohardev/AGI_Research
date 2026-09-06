# Roadmap — Week 2 Day 3: Shannon Entropy

**Current:** Month 2, Week 2, Day 3 (global Day 9) — *entropy = expected surprise = compressibility*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` first (start with StatQuest, ~16 min, then 3Blue1Brown Part 1, ~32 min) — 50 min total. Focus on how the formula `H = -Σ P log P` falls out of "expected surprise," and why uniform = maximum entropy. Skip Part 2 until tomorrow.
2. **Math on paper (20-30 min, do not skip):**
   - Write the entropy formula from memory: `H(X) = -Σ P(x) log P(x)`.
   - Hand-compute: fair coin → `1.0` bit; biased coin `[0.99, 0.01]` → `≈ 0.0808`; fair die → `log₂(6) ≈ 2.585`.
   - Reason through why uniform maximises entropy ("no information to help you guess" = maximum surprise) and why entropy → 0 as one outcome's probability → 1.
3. **Code — `entropy(pmf)`** (`coding_problems.md` Problem 1): the one-line formula with `math.log`, explicit `0 * log 0 = 0` handling, `ValueError` on bad pmfs. Confirm the three hand computations above.
4. **Code — `file_entropy`** (Problem 2): frequencies → your `entropy()`, char and word levels. Leave the compression comment.
5. **Experiments** (Problem 3): run the `__main__` demos — confirm repetitive < English < random, and print the char-vs-word pair on the same file with an explanation ready.
6. **Practice questions** (in `learn.md`) on paper before step 5 — if you can't do the fair-die computation without code, you don't know the math yet.
7. **Commit & run `/done`:** `pytest -v` must pass before the quiz. `/done` will quiz you on non-negativity, the uniform-maximum argument, the flat-limit intuition, and the char-vs-word comparison.

---

## How this ties to the final project (`minisklearn`)

This day is the **loss-function vocabulary** day. Tomorrow's cross-entropy `H(P,Q)` is the loss Week 3 Day 2's Logistic Regression minimises and the loss every neural net in Months 3+ trains on — and it is defined *in terms of* today's `H(P)`. Week 3 Day 4's Decision Trees split nodes by information gain = entropy before minus entropy after, so `entropy()` itself gets reused almost verbatim. Week 4's `minisklearn` evaluation will report cross-entropy alongside accuracy. Keep both files.

---

## Files in this day

- `learn.md` — what to learn + verified videos
- `coding_problems.md` — problem statements (intermediate scale)
- `code/entropy.py` — stub for `entropy`
- `code/file_entropy.py` — stub for `file_entropy` + runnable `__main__` experiments
- `code/tests/` — pytest suite (must pass before `/done`)
- `further_reading.md` — appended only if you run `/for-read` (opt-in)
