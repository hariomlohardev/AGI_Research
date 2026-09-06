# Day 9 Coding Problems — Shannon Entropy

Work the math in `learn.md` on paper **before** touching code. No `sklearn`,
`scipy.stats`, or `scipy.optimize` — everything from scratch (`math` only).

## Problem 1 — `entropy(pmf)` (in `code/entropy.py`)

```python
def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy H = -sum(p * log(p)) of a discrete distribution."""
```

- `pmf` is a sequence of probabilities that must sum to 1 (within floating-point
  tolerance, e.g. `math.isclose(..., abs_tol=1e-9)`). Raise `ValueError` on an empty
  pmf, a negative probability, a pmf that doesn't sum to 1, or `base <= 0`.
- **Zero probabilities:** use the standard convention `0 * log(0) = 0` — skip them,
  don't take `log(0)`. (Why is this the right convention? Think about what an
  impossible outcome contributes to expected surprise.)
- Default `base=2.0` reports bits. `base=math.e` should report nats. Sanity anchor:
  `entropy([0.5, 0.5])` is exactly `1.0`, and `entropy([0.5, 0.5], base=math.e)`
  is exactly `math.log(2)`.

## Problem 2 — text entropy (in `code/text_entropy.py`)

```python
def entropy_of_text(text: str, level: str = "char") -> float:
    """Exact Shannon entropy of a text, probabilities from its own frequencies."""
```

- `level="char"`: treat each character as an outcome, `P(c) = count(c)/len(text)`.
- `level="word"`: split on whitespace first, then `P(w) = count(w)/n_words`.
- Reuse your `entropy` from Problem 1 for the final computation — don't reimplement it.
- Raise `ValueError` on empty text or an unknown `level`.
- **Experiments** (keep runnable in `__main__`, with printed numbers):
  1. **Repetitive vs varied vs random:** e.g. `"aaaa..."` vs a paragraph of real
     English vs a uniform random string over some alphabet. Confirm entropy rises
     with genuine unpredictability.
  2. **Char level vs word level** on the same English text. Report which is higher
     and be ready to explain why in terms of how language works (this is practice
     question 4 — the numbers are the artifact, the explanation is for `/done`).

## Problem 3 — compression comment (a comment, not code you must run)

In `text_entropy.py`, leave a short comment block explaining: **why is a file with
lower entropy more compressible?** (Hint: entropy is the expected number of bits per
symbol under an optimal encoding — a skewed distribution lets frequent symbols use
short codes. Name the connection: what does a compressor exploit that entropy measures?)

---

### Running

```bash
pytest -v
```

Stubs raise `NotImplementedError` until you implement them.

### Done when (from the source plan)

Your entropy calculator correctly reports higher entropy for more "random"-looking
text and lower entropy for repetitive text, and you can compute the entropy of a
fair die by hand.
