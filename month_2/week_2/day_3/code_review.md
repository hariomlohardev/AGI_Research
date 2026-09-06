# Code Review — Month 2, Week 2, Day 3 (Day 9): Shannon Entropy

Advisory only. All 13 tests pass; nothing here blocks completion.

## What is genuinely good

- `entropy.py:41` uses a generator inside `sum(...)` — `sum(p * ... for p in pmf if p > 0)`.
  This is the idiomatic form the two previous days missed. Keep this.
- The `0 * log 0 = 0` convention is handled explicitly with `if p > 0` rather than
  try/except around `math.log(0)`. Correct instinct.
- Every `raise ValueError` in both files carries a message. That is a step up from
  Day 8's bare `raise ValueError` with no message — noticed and appreciated.
- The `__main__` block in `file_entropy.py:71-98` is well constructed: seeded
  `random.Random(0)` so the "random" text reruns identically, `tempfile` so no
  stray files are left behind, and both required experiments (ordering + char-vs-word)
  are present and runnable.
- Module docstrings state the math (`H(X) = -sum ...`) and the compression link.
  Good habit for a curriculum where each file should be re-readable later in the month.

## Worth changing (real readability/maintainability items)

1. **Duplicate import — `entropy.py:11` and `entropy.py:13`.**
   `import math` appears twice. One of them is leftover from editing. Delete line 13.

2. **Guard clauses chained with `elif` — `entropy.py:32-39`.**
   The four validations are independent preconditions, not mutually exclusive branches,
   so `elif` misleads the reader into looking for a relationship between them.
   Since each branch raises, plain sequential `if` statements read better and make
   future reordering/deletion safe.

3. **Docstring promises `1e-9`, code enforces `0.001` — `entropy.py:20-21` vs `:38`.**
   The docstring says entries must sum to 1 "within 1e-9" but the check is
   `not abs(sum(pmf) - 1) < 0.001`. Pick one tolerance, use it in both places,
   preferably as a named constant. (Also consider `math.isclose` / `math.fsum`
   instead of hand-rolled `abs(sum(...) - 1)`; `sum` on floats is the noisier accumulator.)

4. **Misnamed helper `pdf` — `file_entropy.py:22-26`.**
   `pdf` universally means *probability density function* (continuous). This helper
   builds an empirical discrete distribution — it should be called `to_pmf`,
   `empirical_pmf`, or similar. It also has no return annotation and no docstring,
   unlike every other function in the two files. One line of docstring stating
   "counts normalised to sum to 1, order arbitrary" would remove all guesswork.

5. **Fragile import — `file_entropy.py:19`: `from entropy import entropy`.**
   This only resolves when the working directory happens to be `code/`. A reader running
   from the repo root gets an `ImportError` from the module under review while the tests
   pass. A package-relative import or a documented path assumption would make the file
   honest about how it is meant to be run.

6. **Redundant exception wrapping — `file_entropy.py:67-68`.**
   ```python
   except FileNotFoundError:
       raise FileNotFoundError(f"File is not found on the given file path {path}")
   ```
   This catches the informative stdlib error only to re-raise a less informative one
   (and without `from`, so the chain is implicit). The docstring on line 41 already
   promises "a missing file raises FileNotFoundError" — just let it propagate and
   delete the try/except.

7. **Char tokens derived from word tokens — `file_entropy.py:55-64`.**
   `words = file_content.split(" ")` on line 55, then lines 61-63 rebuild characters
   by iterating over `words`. That couples the two levels: any quirk of the word
   split silently changes the char distribution, and spaces vanish from
   the char alphabet without the reader being told. Derive `chars` directly from the
   file content (`Counter(file_content)` or equivalent) so each level has one obvious
   data source.

8. **Leftover scaffold in the module docstring — `file_entropy.py:9`.**
   `# TODO(student): write 1-2 sentences ...` is still in the shipped file even though
   lines 6-8 already contain a compression explanation. Delete the TODO line so the
   next reader does not wonder whether the explanation is still missing.

9. **Single-letter uppercase accumulator `H` — `entropy.py:41-43`.**
   `H` is fine on a whiteboard, but in a module that also discusses entropy in prose,
   a lowercase `entropy_bits` / `total` reads better and complies with PEP 8
   (uppercase = constants). Minor, but this is the returned value — it deserves a name.

## Minor nits (fix while you are in there)

- **Error-message wording.** Several messages are ungrammatical or imprecise:
  `:37` "one or more entry in pmf is negative" (entries), `:39` "The pmf not sum up to 1"
  (does not sum), `:35` "The base is less than 0" (also fires when `base == 0` — say
  "base must be positive"), `file_entropy.py:45` "Given level is ..." (the
  f-string interpolates user input into an error; naming the accepted values first
  reads better). Capitalisation is inconsistent (`"The pmf..."` vs `"one or more..."`
  vs `"The file..."`). None of these affect tests; all of them affect the next reader.
- **PEP 8 spacing slips** (`entropy.py:41` `p*math.log(p , base)`, `file_entropy.py:26`
  `count /len_data`, `:51` `open(path , encoding=...)`, `:50` `try :`, `:60`
  `elif level== "char":`). A formatter (black/ruff) would settle all of these at once.
- **Stray blank lines `file_entropy.py:47-49`.** Three blank lines plus an indented
  blank line with trailing whitespace inside the function body. One blank line suffices.
- **`split(" ")` vs `split()` — `file_entropy.py:55`.** The spec says "splits on
  whitespace". `split(" ")` splits on single spaces only: double spaces yield empty-string
  tokens and tabs/newlines are never split. Bare `split()` matches the spec.
- **`chars += list(word)` loop — `file_entropy.py:61-63`.** Works, but
  `chars.extend(word)` or a comprehension over the content is the idiomatic form;
  building a list only to hand it to `Counter` is also avoidable (`Counter` accepts
  any iterable directly).
- **`random` / `tempfile` imported at top level — `file_entropy.py:14-15`** but used
  only under `__main__`. Harmless; moving them inside the guard would signal they are
  experiment-only dependencies, not part of the library surface.
- **`__main__` repeats the `TemporaryDirectory` setup twice (`:85` and `:94`).**
  Fine at this size, but a tiny local helper would remove the duplication if a third
  experiment is ever added.

## One-line summary

Clean, well-documented first pass with a real improvement over Day 8 (messages on every
raise, generator-based sum). The highest-value fixes are the duplicate import, the
`pdf` name, the `1e-9`-vs-`0.001` mismatch, and letting `FileNotFoundError` propagate.
