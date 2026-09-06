# Day 9 — Shannon Entropy: coding problems

## Problem 1 — `entropy(pmf)` from scratch

```python
def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Shannon entropy H = -sum(p * log(p)) in the given log base (bits by default)."""
```

- Implement in `code/entropy.py` using only `math.log` — no `scipy.stats.entropy`, no `np.mean`-style shortcuts around the definition.
- `pmf` entries must be `>= 0` and sum to 1 (within `1e-9`); zero entries are allowed and contribute `0` (that is the `0 * log 0 = 0` convention — handle it explicitly, don't let `math.log(0)` blow up). Raise `ValueError` on empty input, any negative entry, a total that isn't `~1`, or `base <= 0`.
- Verify by hand first, then against your function: fair coin `[0.5, 0.5]` → exactly `1.0`; biased coin `[0.99, 0.01]` → `≈ 0.0808`; fair die `[1/6]*6` → `log2(6) ≈ 2.585`.

## Problem 2 — file entropy (the real task)

```python
def file_entropy(path: str | Path, level: str = "char") -> float:
    """Shannon entropy (bits) of a text file's token distribution.

    `level="char"` treats each character as the random variable;
    `level="word"` splits on whitespace and treats each word as the outcome.
    Probabilities are estimated from within-file frequencies (count / total).
    """
```

- Implement in `code/file_entropy.py`, built on your `entropy()` from Problem 1 (normalise the counts before calling it). Stdlib only (`collections.Counter`, `pathlib`).
- Raise `ValueError` on an empty file (no tokens to estimate from) or an unknown `level`.
- Leave a short comment in the file relating this to compression: why is a lower-entropy file more compressible? (Think: average bits per symbol needed.)

## Problem 3 — experiments (keep runnable under `__main__`)

1. **Ordering proof:** three texts — a highly repetitive one (e.g. `"ab" * 500`), a genuinely varied English paragraph, a pseudo-random string (fixed seed so it reruns identically). Print all three entropies. You must see repetitive < English < random.
2. **Char vs word:** run both levels on the same English file and print the pair. This is practice-question 4 — have an explanation ready for which is higher and why.

### Running

```bash
cd code && pytest -v
```

Stubs raise `NotImplementedError` until you implement them. Tests import from `code.entropy` and `code.file_entropy`.

### What not to use

- No `scipy.stats`, no `sklearn.metrics`.
- For the entropy math use `math.log` directly — the one-line formula is the whole point.

### Done when (from the source plan)

Your entropy calculator correctly reports higher entropy for more "random"-looking text and lower entropy for repetitive text, and you can compute the entropy of a fair die by hand.
