# Code Review — Month 2 Day 12 (Week 2 Review and Consolidation)

Scope: `month_2/week_2/day_6/code/week2_utils.py` (main file). `tests/test_week2_utils.py` used for context only. Style only — not correctness.

## 1. What was done well

For an intermediate learner consolidating a full week into one API, this is solid:

- Every public function has a docstring with an explicit `Raises: ValueError` contract. That is exactly the right habit for a utils module.
- Type hints throughout — `Sequence[float]`, `Sequence[Hashable]`, `tuple[float, float]` — and `Hashable`/`Sequence` correctly imported from `collections.abc` with `Counter` from `collections` (not `typing`). Correct module sources.
- Good stdlib-only idioms where it counts: `math.isclose` for sum-to-1 checks, `math.log(x, base)` for base selection, `Counter` for marginals in `mutual_information`.
- `kl_divergence` reuses `entropy` + `cross_entropy` instead of reimplementing the loop. Right DRY instinct.
- Correctly skips `p == 0` terms in `entropy` and guards `q[i] == 0 where p[i] > 0` in `cross_entropy` rather than letting `math.log(0)` throw.

## 2. Findings

1. `week2_utils.py:28` — Duplicated / wrong emptiness check. `len(likelihood) == 0` appears twice; `evidence` is a `float` and has no length, but the message claims to check it:
   `if len(prior) == 0 or len(likelihood) == 0 or len(likelihood) == 0:`
   Suggestion: `if len(prior) == 0 or len(likelihood) == 0:` and fix the message to name only the two sequences.
2. `week2_utils.py:29,32` — Error messages reference `evidence` length equality, which is impossible for a scalar. Line 32: `"The length of the 'prior' , 'likelihood' and 'evidence' is not equal"`. Suggestion: `"prior and likelihood must have the same length"` — drop `evidence` from both messages.
3. `week2_utils.py:31,157,219` — Non-idiomatic negated equality: `if not len(prior) == len(likelihood):`. Suggestion: `if len(prior) != len(likelihood):` (same fix at lines 157 and 219).
4. `week2_utils.py:37` vs docstring lines 20-24 — Docstring/code mismatch on zero likelihood. Docstring says raise if "any prior or likelihood is negative" (zero allowed), but code rejects zero: `if any(lh <= 0 for lh in likelihood)`. The test at `test_week2_utils.py:41` expects `bayes_update([0.5,0.5],[0.0,0.0],1.0)` to raise, but under the documented contract that should be the "posterior denominator is zero" case, not a negativity case. Suggestion: check `lh < 0` here and add the missing explicit denominator-zero check the docstring already promises.
5. `week2_utils.py:40` vs `week2_utils.py:131,166,169` — Inconsistent tolerance kind. `bayes_update` uses `rel_tol=1e-9`; all PMF checks elsewhere use `abs_tol=1e-9`. For probabilities near 0/1 these behave differently. Suggestion: pick one (the `abs_tol=1e-9` form used in `entropy`/`cross_entropy`) and use it everywhere.
6. `week2_utils.py:47-48,176,237` — Non-idiomatic `range(len(...))` loops. Example lines 47-48:
   `for i in range(len(prior)): posterior.append((prior[i] * likelihood[i])/evidence)`
   Suggestion: `posterior = [p * l / evidence for p, l in zip(prior, likelihood)]`. Same at line 176 (`for i in range(len(p))` → `for p_i, q_i in zip(p, q)`) and line 237 (`((x[i], y[i]) for i in range(len(x)))` → `Counter(zip(x, y))` directly).
7. `week2_utils.py:240` — Uppercase local violates PEP8: `MI = 0`. Suggestion: `mi = 0.0` or `mi_total = 0.0` (also fixes the int-accumulator-for-float issue shared with `output = 0` at lines 68, 136, 175).
8. `week2_utils.py:198-199` — Leading-underscore locals for ordinary temporaries: `_entropy`, `_cross_entropy`. Leading underscore signals module-private, not "temporary". Suggestion: `ent = entropy(p, base)` / `ce = cross_entropy(p, q, base)` or `entropy_val` / `cross_entropy_val`.
9. `week2_utils.py:160,163,241-242` — Cryptic / misleading loop names: `_p`/`_q` (underscore suggests unused/private) and `for a in count_x: for b in count_y:`. Suggestion: `any(v < 0 for v in p)`, and `for x_val in count_x: for y_val in count_y:` — or better, iterate `for (x_val, y_val), joint_count in pairs_count.items():` and avoid the Cartesian product entirely.
10. `week2_utils.py:114-115` — Cryptic abbreviation plus overlong line: `s_mu, s_sigma = mle_fit_gaussian(data)` then a ~110-char expression retyping `s_sigma ** 2` three times with inconsistent spacing (`prior_sigma ** 2` vs `s_sigma**2`). Suggestion:
   ```python
   sample_mean, sample_sigma = mle_fit_gaussian(data)
   like_var = sample_sigma ** 2
   prior_var = prior_sigma ** 2
   mu_map = (n * sample_mean / like_var + prior_mu / prior_var) / (n / like_var + 1 / prior_var)
   return mu_map
   ```
11. `week2_utils.py:198-200` — Duplicated validation via delegation (trade-off to note). `kl_divergence` calls `entropy(p)` then `cross_entropy(p, q)`, so `p`'s emptiness/negativity/sum-to-1/base are each validated twice. Reuse is good, but worth a comment acknowledging the double pass, or validate once and call private helpers.
12. `week2_utils.py:225-229` — Docstring/code mismatch on `TypeError`. Docstring lines 211-214 says a `TypeError` from an unhashable label "is allowed to propagate", but the code preempts it with two explicit checks (`isinstance(x, Sequence)`, `all(isinstance(item, Hashable) ...)`) that are never documented in `Raises:`. Suggestion: either document both `TypeError` cases or drop the preemptive `Hashable` scan and let `Counter` raise naturally as documented.
13. `week2_utils.py:8-10` — Import order within the `collections` group. `from collections import Counter` should sort before `from collections.abc import ...` (`collections` < `collections.abc`). Suggestion: swap lines 8 and 9.
14. `week2_utils.py:68-70,136-140,175-181` — Vague accumulator name `output` (used in three functions for three different quantities) plus recomputed loop invariant. Line 70 recomputes `-0.5 * math.log(2*math.pi*sigma**2)` per element. Suggestion: name by meaning (`total_ll`, `ent`, `ce`) and hoist the constant: `const = -0.5 * math.log(2 * math.pi * sigma ** 2)` before the loop.
15. `week2_utils.py:66` — f-string with no placeholder: `raise ValueError(f"the Standerd Deviation is either 0 or negative")`. Suggestion: drop the `f` prefix (linters flag this) and fix the typo while there.
16. `week2_utils.py:255-278` — Leftover `__main__` stub harness now duplicates `tests/test_week2_utils.py`. Not wrong, but dead weight in a consolidated deliverable — every assertion there is covered by pytest. Suggestion: delete the block or shrink it to a one-line smoke demo.

## 3. Minor nits (spelling / PEP8 spacing / message style)

- Typos in messages: line 29 `eigther` → `either`; lines 35, 38 `valude` → `value`; lines 41, 132, 167, 170 `doesnot` → `does not`; line 44 `evedence` → `evidence`; line 66 `Standerd` → `standard`; line 130 `prorbabilites` → `probabilities`; lines 173, 223 `the base is can't be` → `the base can't be`.
- Grammar/article: lines 35, 38 `an valude` → `a value`; consider sentence-case + no trailing spaces consistently (lines 44, 110 have trailing spaces; most messages start lowercase, line 32 starts uppercase).
- Space before comma/colon throughout: `1 ,sum` (lines 40, 166, 169), `p , base` (line 139), `mu_hat , sigma_hat` (lines 92, 114), `x[i] , y[i]` (line 237), `) :` (lines 34, 37, 110, 127). Run `ruff`/`black` or a PEP8 pass — these alone are ~15 spots.
- Missing spaces around operators: line 37 `lh <= 0` written without spaces; line 89 `sum(data)/n`; line 90 `(x-mu_hat)** 2`; line 250 `joint_count/ n`, `p_y  =` (double space). Lines 68-70 also need spaces around `*` and `/` per PEP8.
- Long lines >88 chars: lines 115, 228. Break them.
- Stray blank line inside `bayes_update` (lines 26-27) and the triple blank at lines 51-53. One blank separates logical blocks inside a function; two separate top-level defs.
- Runtime message leaks markdown (line 179): `"``q[i] == 0`` where ``p[i] > 0``"`. Suggestion: plain English — `"q is zero where p is positive"`.
