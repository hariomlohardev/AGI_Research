# Code Review — Month 2, Week 2, Day 2 (global Day 8)

**Topic:** MAP Estimation & Priors · **Language:** python · **Level:** intermediate
**Scope:** style and readability only. Correctness is separately established — all 13 tests in
`code/tests/` pass. Nothing here blocks the day.
**Reviewed:** `code/map_fit_gaussian.py` (learner's work). `code/log_likelihood.py` is a carried-over
clean Day-1 reference and is used below only as the style benchmark. Test files are system-generated
and out of scope.

> Written inline rather than by the `code-evaluator` subagent — that agent failed twice with an
> auth error (`403 authentication_failed`), so the review was done directly instead of skipped.

## Worth changing

1. **`map_fit_gaussian.py:40,73-74,110,148` — `raise ValueError` with no message.** Four separate
   validation points all raise bare. The caller gets `ValueError` with nothing to debug from. Your own
   Day-1 code sitting in the same folder does this right — `log_likelihood.py:20` says
   `raise ValueError("sigma must be > 0")`. Match that.

2. **`map_fit_gaussian.py:40,110,148` — body on the same line as `if`.**
   `if prior_sigma <=0 : raise ValueError` — PEP 8 wants the body on its own line, and the space
   before `:` is non-standard. Small thing, but it hides the raise when scanning the function.

3. **`map_fit_gaussian.py:149` — `n = len(data)` is assigned and never used** in
   `map_fit_gaussian_grid`. Dead line; delete it.

4. **`map_fit_gaussian.py:159-160` — `np.arange` with a computed float step.** Two issues.
   The docstring at `:138` promises `mu_range` is *inclusive*, but `np.arange(a, b, step)` excludes
   `b`: with `mu_range=(-1, 8), steps=80` the grid runs `-1.0 … 7.8875` and never evaluates `8.0`.
   Separately, `arange` with a fractional step accumulates float error and its length isn't
   guaranteed. `np.linspace(mu_range[0], mu_range[1], steps)` fixes both at once — inclusive
   endpoints, exactly `steps` points, no drift. Tests pass here only because the true optimum sits
   mid-range; a search whose answer lands on the upper edge would silently miss it.

5. **`map_fit_gaussian.py:23` — `numpy` imported solely for the grid loop.** `coding_problems.md:83`
   scopes numpy to "data generation and the optional sweep plot"; the grid itself was meant to be
   plain Python (`range(steps)` plus arithmetic). Dropping numpy would also make the return types
   Python floats rather than `np.float64`.

6. **`map_fit_gaussian.py:78-79` — the `# MAP == L2 when ...` comment sits *below* `return Output`**
   (`:77`), so it reads like unreachable code. Move it above the return, or fold it into the
   docstring where a reader will actually find it.

7. **`map_fit_gaussian.py:113` — `sum([(x - mu_mle)**2 for x in data])` materialises a throwaway
   list.** Drop the brackets for a generator. Same nit as Day 7's review; costs nothing to fix.

## Nits

8. **`:116` `denomenator`** — misspelled (`denominator`). Same family as Day 7's `cureent_mu` /
   `log_likehd`.

9. **`:76` `Output`** — capitalised local. Python reserves leading capitals for classes; `out` or
   `log_post` reads better and says more.

10. **Spacing is inconsistent across the file.** `log_posterior(data , mu , sigma , prior_mu ,prior_sigma)`
    (`:161`) has spaces before commas and none after the last; `sigma_mle ** 2` vs `(prior_sigma**2)`
    mix within two lines of each other; `(-(0.5) * math.log(...))` (`:41`) is more parentheses than
    `-0.5 * math.log(...)` needs. One `ruff format` or `black` pass normalises the whole file and
    removes the trailing whitespace at `:156-157,167` and the stray double blank line at `:150-151`.

11. **`:41` is ~110 characters on one line.** `log_likelihood.py:24-27` splits the same computation
    into a named `const_term` plus the quadratic. Doing the same here would make the two halves of
    the log-PDF visible at a glance — which matters, because those two halves are exactly what the
    L2-penalty argument turns on.

12. **`:155-156` `mu_hat = 0` / `sigma_hat = 0`** as sentinels. If the grid were ever empty these zeros
    would be returned as if they were real answers. `None` would fail loudly instead. Can't happen with
    a valid range, so this is defensive taste rather than a bug.

## Credit where due

The docstrings you kept are accurate and the derivation comment at `:78-79` correctly states the
`lambda = 1/(2*prior_sigma^2)` correspondence. `log_posterior` is a clean two-term sum that delegates
validation instead of duplicating it, and the `__main__` block prints exactly the two experiments the
spec asked for with the expected values annotated inline — that annotation is a genuinely good habit.
