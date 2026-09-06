# Week 2 Day 3 — Shannon Entropy

**Topic:** Shannon entropy — the mathematical definition of "surprise" / uncertainty.

## Why this matters

Entropy is the foundation of every loss function that follows this week. Cross-entropy
(Day 4) is entropy plus a mismatch cost; KL divergence (Day 4) is the mismatch cost alone;
mutual information (Day 5) is entropy minus conditional entropy. It is also literally how
we measure compressibility — a low-entropy file compresses well, a high-entropy one doesn't.
Everything after today assumes you can compute `H(X) = -Σ P(x) log P(x)` in your sleep.

## What to learn today

### Math (on paper first, before code)

1. **Surprise first, entropy second.** For an outcome with probability `p`, define its
   "surprise" as `-log₂(p)`. Check the extremes: `p = 1` gives surprise 0 (no surprise —
   you knew it would happen), `p = 1/2` gives 1 bit, `p → 0` gives surprise → ∞.
2. **Entropy is expected surprise:** `H(X) = -Σ P(x) log₂ P(x)`.
3. **By hand:** entropy of a fair coin (`P = 0.5` → exactly 1 bit) and of a heavily
   biased coin (`P(heads) = 0.99`) — confirm numerically that the biased coin has
   *lower* entropy (less surprise, since you can already guess the outcome).
4. **Fair die by hand:** show it comes out to `log₂(6) ≈ 2.585` bits.
5. **Why uniform = maximum entropy.** No full proof needed, but reason it through:
   "no information to help you guess" is exactly "maximum surprise on average".
   (Tomorrow's KL divergence makes this precise: any tilt away from uniform is
   measurable distance from the max-entropy distribution.)

### Code

- `entropy(pmf)` — Shannon entropy of a discrete distribution, from scratch.
- `entropy_of_text(text, level)` — exact character- or word-level entropy of a text,
  with probabilities estimated from frequencies in the text itself.
- Experiments in `__main__`: repetitive vs varied vs random text; char level vs
  word level on the same file; a comment relating low entropy to compressibility.

See `coding_problems.md` for exact signatures and the practice questions.

## Watch first (guided — at least two)

1. **Entropy (for data science) Clearly Explained!!!** — StatQuest with Josh Starmer
   https://www.youtube.com/watch?v=YtebGVx-Fxw
   Foundations-first pick. Builds entropy up from "surprise" to the expected-value
   formula with no skipped steps, framed through the data-science lens (why impurity
   measures matter for models). Watch this one first — it gives the working definition
   you'll use all week.

2. **Solving Wordle using information theory** — 3Blue1Brown
   https://www.youtube.com/watch?v=v68zYyaEmEA
   Deep-intuition pick. Derives the entropy formula from first principles — why the
   log, why the expectation, what a "bit" actually buys you — through the concrete
   problem of choosing the best Wordle guess. This is the framing that makes
   cross-entropy click tomorrow: entropy as optimal encoding size / expected
   information gain.

## Practice questions (do on paper before coding)

1. Why is entropy always non-negative? (Hint: what does `-log P(x)` look like when `P(x)` is between 0 and 1?)
2. For a fair 6-sided die, compute the exact entropy by hand (it should come out to `log₂(6)`) — show why this makes sense (uniform distribution = maximum entropy for 6 outcomes).
3. What happens to entropy as one outcome's probability approaches 1 and all others approach 0? Explain in terms of "surprise."
4. Code check: run your text-entropy script on the same file at the *character* level vs. the *word* level — which one gives higher entropy, and why does that make sense given how language works?

---

## Done when

Your entropy calculator correctly reports higher entropy for more "random"-looking
text and lower entropy for repetitive text, and you can compute the entropy of a
fair die by hand. `pytest -v` passes.

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers
on entropy, coding theory, or maximum-entropy reasoning.
