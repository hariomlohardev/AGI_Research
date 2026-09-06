# Week 2 Day 3 — Shannon Entropy

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 9**

---

## Why this matters

Entropy is the mathematical definition of "surprise" or "uncertainty" — it's the foundation of every loss function that follows this week, and it's literally how we measure how compressible information is. Cross-entropy (tomorrow) is just entropy plus the cost of being wrong about the probabilities, and every classifier you train from Month 3 onward minimises it. The decision trees in Week 3 Day 4 will split nodes by *information gain*, which is entropy before minus entropy after. Learn this properly once and three later topics arrive pre-explained.

## What to learn today

### Math (on paper first, before code)

1. **The formula:**
   `H(X) = -Σ P(x) log P(x)` — expected surprise. Each outcome contributes its probability times its own surprise (`-log P(x)`); rare outcomes are very surprising but seldom happen, common ones barely surprise at all.
2. **Fair coin vs rigged coin:**
   Fair (`P(heads)=0.5`): `H = 1.0` bit — maximum uncertainty for 2 outcomes. Heavily biased (`P(heads)=0.99`): `H ≈ 0.0808` bits — almost no surprise, since you can already guess the outcome. Work both out by hand and confirm the biased coin is *lower*.
3. **Uniform maximises entropy:**
   No reason to favour any outcome means no information to help you guess, which is maximum surprise. A fair die gives `H = log₂(6) ≈ 2.585` bits — compute it by hand. (The full proof uses Lagrange multipliers; the intuition above is what's required today.)
4. **Entropy is non-negative, and zero means certainty:**
   When `P(x)` is between 0 and 1, `-log P(x) >= 0`, so every term contributes non-negative surprise. As one outcome's probability → 1, its surprise → 0 and all other terms vanish with their probabilities — total entropy → 0.

### Code

- `entropy(pmf, base=2.0)` — the one-line formula from scratch (`math.log` only).
- `file_entropy(path, level="char"/"word")` — estimate probabilities from within-file frequencies, then apply `entropy()`. Run it on repetitive vs varied vs random text and confirm the ordering.

See `coding_problems.md` for exact signatures, the char-vs-word experiment, and the compression comment.

---

## Watch first (guided — at least two)

> Search YouTube for the exact phrases below if links don't open — don't rely on autoplay.

1. **Entropy (for data science) Clearly Explained!!! — StatQuest with Josh Starmer**
   - https://www.youtube.com/watch?v=YtebGVx-Fxw (~16 min)
   - Builds the entropy formula from scratch with the biased-vs-fair-coin intuition (why 50/50 is maximally uncertain and a rigged coin carries almost no surprise) — maps directly onto today's `H(X)` hand calculations. Watch this one first.

2. **Reinventing Entropy | Compression is Intelligence Part 1 — 3Blue1Brown**
   - https://www.youtube.com/watch?v=l6DKRf-fAAM (~32 min)
   - Derives entropy from "how many bits do you actually need?" — the uniform-maximises-entropy idea and the entropy↔compression link your file-entropy experiments demonstrate. The visual treatment of expected code length is the best available for this material.
   - Note: 3Blue1Brown's Part 2 ("But what is cross-entropy?") is a teaser for *tomorrow's* topic — save it until Day 10.

---

## Practice questions (do on paper before coding)

1. Why is entropy always non-negative? (Hint: think about what `-log P(x)` looks like when `P(x)` is between 0 and 1.)
2. For a fair 6-sided die, compute the exact entropy by hand (it should come out to `log₂(6)`) — show why this makes sense (uniform distribution = maximum entropy for 6 outcomes).
3. What happens to entropy as one outcome's probability approaches 1 and all others approach 0? Explain in terms of "surprise."
4. Code check: run your text-entropy script on the same file at the *character* level vs. the *word* level — which one gives higher entropy, and why does that make sense given how language works?

---

## Done when

Your entropy calculator correctly reports higher entropy for more "random"-looking text and lower entropy for repetitive text, and you can compute the entropy of a fair die by hand.

---

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers on entropy, source coding, or maximum-entropy distributions.
