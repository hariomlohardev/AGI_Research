# Code Review — Month 2 Day 7 (Week 2 Day 1) — Maximum Likelihood Estimation (MLE)

Language: python (intermediate) — Readable and tests pass. Below are style/readability/structure suggestions only.

## What went well
- Clear module docstrings that state the no-scipy/sklearn constraint and the math.
- Guard clauses for `sigma <= 0` and empty `data` at the top of `log_likelihood` and `mle_fit_gaussian_closed_form` — easy to follow.
- `mle_fit_gaussian` alias to keep `coding_problems.md` wording compatible is a nice touch.
- `if __name__ == "__main__"` demo blocks make manual sanity-checking easy.

## `log_likelihood.py`

1. **Error messages — typos and inconsistent phrasing (L34-37).**
   ```python
   raise ValueError("sigma cant be lessthan or equal to 0")
   raise ValueError("data is empy")
   ```
   Action: Fix spelling/punctuation and match style across files (compare `mle_fit_gaussian.py:34` which correctly says `"data is empty."`). Example:
   ```python
   raise ValueError("sigma must be > 0")
   raise ValueError("data must not be empty")
   ```
   Keep messages lower-sentence-case, no period in one file and period in another.

2. **Cryptic names `fst`/`scd`/`std`/`output` (L40-46).**
   `std` is actually variance (`sigma**2`), `fst` is the constant log-term, `scd` is the per-sample quadratic term. A reader has to reverse-engineer them.
   Action: Rename to intent:
   ```python
   var = sigma ** 2
   const_term = -0.5 * math.log(2 * math.pi * var)
   # inside loop: quad_term = -((x - mu)**2) / (2 * var)
   total_ll = 0.0  # instead of `output`
   ```

3. **Unpythonic loop `for i in range(len(data))` (L43-46).**
   Action: Iterate directly:
   ```python
   for x in data:
       quad = -((x - mu) ** 2) / (2 * var)
       total_ll += const_term + quad
   ```
   Eliminates `data[i]` indexing and the temporary `result`.

4. **Minor idiom / formatting (L34, L40-42).**
   - `if sigma <= 0 :` — remove space before `:`.
   - `if len(data) <= 0:` — idiomatic is `if not data:` or `if len(data) == 0:`. A length cannot be `< 0`.
   - `output = 0` — use `0.0` when accumulating floats.
   - `fst  =` double spaces; run `black` or `ruff format` to normalize.
   - `std = sigma ** 2` → `variance` or `var` as above avoids confusion with standard deviation.

## `mle_fit_gaussian.py`

1. **Typo `cureent_mu` and truncated names `log_likehd`/`mu_log_likehd` (L69-82).**
   Action: `current_mu`, `current_sigma`, `best_ll`, `best_mu_ll` etc. Consistent full words aid search. Also normalize spacing: `log_likelihood(data , cureent_mu ,sigma_hat )` → `log_likelihood(data, current_mu, sigma_hat)` and `return (mu_hat ,sigma_hat)` → `return mu_hat, sigma_hat`.

2. **Unnecessary list materialization (L37).**
   ```python
   var = sum([(x - mu_hat)**2 for x in data])/len(data)
   ```
   Action: Use a generator, no brackets, and consistent spacing:
   ```python
   var = sum((x - mu_hat) ** 2 for x in data) / len(data)
   sigma_hat = math.sqrt(var)  # clearer than `var ** (0.5)`, needs `import math`
   ```

3. **Grid search structure is two sequential 1-D sweeps, not the 2-D grid the docstring promises (L64-84).**
   The docstring says "steps x steps grid spanning mu_range x sigma_range" but the code fixes `sigma_hat = sigma_range[0]` to optimize mu, then fixes `mu_hat` to optimize sigma. This is surprising to a reader even though tests pass.
   Action: Either make the docstring match the sequential search, or implement the nested loop the docstring describes:
   ```python
   best_ll = float("-inf")
   for i in range(steps):
       mu = mu_range[0] + i * mu_step
       for j in range(steps):
           sigma = sigma_range[0] + j * sigma_step
           ll = log_likelihood(data, mu, sigma)
           ...
   ```
   Reusing `step` for both axes (L67, L75) also hides the intent — use `mu_step`/`sigma_step`.

4. **Off-by-one on grid and reuse of `step` (L67-68, L75-78).**
   `step = (max-min)/steps` with `range(steps)` never evaluates `max`. If inclusive bounds are intended, divide by `steps-1` or use `numpy.linspace` if allowed. Mention intent in a comment and avoid shadowing the name.

5. **Left-over dead code and whitespace (L88-90).**
   ```python
   # raise NotImplementedError
   ```
   and blank lines L87-92 should be deleted before commit. Same for trailing spaces.

6. **Import style (L12).**
   `from log_likelihood import log_likelihood` works because tests prepend `code/` to `sys.path`, but it is fragile as a package import. If this remains a script-style layout, add a comment explaining the `sys.path` trick, or use `from .log_likelihood import` if run as a module. Not blocking, just worth a comment for the next reader.

7. **Missing validation the docstring implies (L43-63).**
   Docstring says `sigma_range` both >0 and `steps` is resolution, but no `ValueError` is raised for `sigma_range[0] <=0` or `steps <=0`. For intermediate level, either validate or document that caller must ensure validity — keep behavior consistent with `log_likelihood`'s strict guards.

## Cross-cutting

- **Formatter:** Both files have inconsistent spacing around operators/commas and double spaces. Running `black`/`ruff format` once would normalize `L34`, `L42`, `L36`, `L70`, `L84`.
- **Error-message consistency:** Pick one style (`"data is empty"` vs `"data is empty."` vs `"data is empy"`) and reuse.
- **Type hints:** Good use of `Sequence[float]`/`Tuple[float,float]`. Consider `from __future__ import annotations` is already present — could use `tuple[float,float]` directly if targeting 3.10+.

These are all non-blocking style notes — functionality already verified by tests.
