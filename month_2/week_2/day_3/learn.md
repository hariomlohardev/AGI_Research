# Week 2 Day 3 — Shannon Entropy

## Why this matters
Entropy is the mathematical definition of "surprise" or "uncertainty" — it's the foundation of every loss function that follows this week, and it's literally how we measure how compressible information is.

## Watch first (search YouTube)
- "StatQuest entropy clearly explained"
- "3Blue1Brown entropy" (if available — otherwise search "information theory entropy visual explanation")

## Math — work through by hand
- Write the entropy formula: `H(X) = -Σ P(x) log P(x)`.
- Work out entropy by hand for a fair coin (`P(heads)=0.5`) and for a heavily biased coin (`P(heads)=0.99`) — confirm numerically that the biased coin has *lower* entropy (less surprise, since you can already guess the outcome).
- Explain why entropy is maximized exactly when a distribution is uniform — you don't need a full proof, but reason through why "no information to help you guess" corresponds to maximum surprise.

## Code today
- Write an `entropy(pmf)` function from scratch.
- **The real task today:** write a script that computes the exact Shannon entropy of any text file, treating each character (or word, your choice — try both) as a random variable with probabilities estimated from its frequency in the file.
- Run it on a few different text files (e.g. a highly repetitive file vs. a genuinely varied piece of English text vs. a file of pure random characters) — confirm entropy increases with genuine "randomness"/unpredictability.
- Relate this to compression in a comment: explain why a file with lower entropy is more compressible.

## Practice questions
1. Why is entropy always non-negative? (Hint: think about what `-log P(x)` looks like when `P(x)` is between 0 and 1.)
2. For a fair 6-sided die, compute the exact entropy by hand (it should come out to `log₂(6)`) — show why this makes sense (uniform distribution = maximum entropy for 6 outcomes).
3. What happens to entropy as one outcome's probability approaches 1 and all others approach 0? Explain in terms of "surprise."
4. Code check: run your text-entropy script on the same file at the *character* level vs. the *word* level — which one gives higher entropy, and why does that make sense given how language works?

## Done when
Your entropy calculator correctly reports higher entropy for more "random"-looking text and lower entropy for repetitive text, and you can compute the entropy of a fair die by hand.