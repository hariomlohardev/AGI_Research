# Week 2 Day 3 — Shannon Entropy

## Coding Problems

### Problem 1: Entropy of a probability mass function
Write a function `entropy(pmf)` that takes a dictionary mapping outcomes to their probabilities and returns the Shannon entropy in bits (using base-2 logarithm).

- **Input**: `pmf: dict[float, float]` or `dict[str, float]` etc. – keys are outcomes, values are probabilities (should sum to 1, but you may normalize or assume they are valid).
- **Output**: `float` – entropy `H = -∑ p_i * log2(p_i)`. For any outcome with probability 0, the term contributes 0 (by convention `0 * log2(0) = 0`).
- **Requirements**:
  - Do not use `numpy` or `scipy`; use only `math.log2` or `math.log` with base conversion.
  - Raise `ValueError` if any probability is negative or if the sum of probabilities is not approximately 1 (within `1e-9`).
  - Handle empty pmf by raising `ValueError`.

### Problem 2: File entropy calculator
Write a script that computes the Shannon entropy of a text file, treating the file as a sequence of symbols.

- **Input**: a filename (string) and optionally a flag `--unit` that can be `char` (default) or `word`.
- **Behavior**:
  - Read the file as text (UTF-8).
  - If `--unit char`: treat each character (including whitespace and newline) as a symbol.
  - If `--unit word`: split on whitespace (using `.split()`) to get words; treat each word as a symbol.
  - Count frequencies of each symbol, compute probabilities, then compute entropy via the same formula as Problem 1.
  - Print the entropy in bits (as a float) to standard output.
- **Requirements**:
  - Do not use external libraries for counting; you may use `collections.Counter` or a plain dict.
  - Handle empty files (entropy = 0? or raise? We'll define entropy of empty file as 0).
  - Ensure the script can be run from the command line: `python file_entropy.py <filename> [--unit char|word]`.
  - Include a docstring and comment explaining why lower entropy means the file is more compressible (fewer bits needed per symbol on average).

### Problem 3: Verify properties (optional, for understanding)
These are not required to pass tests but are good to check:
- Confirm that entropy is always ≥ 0.
- Confirm that for a uniform distribution over `n` outcomes, entropy = `log2(n)`.
- Confirm that as one probability → 1 and others → 0, entropy → 0.
- Run your file entropy script on:
  - A file containing only the character `a` repeated 1000 times.
  - A file containing fair coin flips represented as `H` and `T` (e.g., `HTHT...`).
  - A file of uniformly random bytes (you can generate using `os.urandom` or `/dev/urandom` if available).
  - Observe how entropy changes.

## Stubs
Place the following stubs in `code/`:
- `entropy.py`: contains `def entropy(pmf: dict) -> float: ...`
- `file_entropy.py`: contains a `main()` that implements the script described above.

## Tests
We will provide tests in `code/tests/` that check:
- Correct entropy for known PMFs (fair coin, biased coin, uniform die).
- Handling of zero probabilities.
- Error on negative probabilities or probabilities not summing to 1.
- File entropy on simple known strings.
- Script runs and outputs a number.

You may use `numpy` only in tests for generating random data, but not in the implementation.