# Growth Notes

Short log of style observations per day. A theme is only promoted to a "pattern" after the same kind of issue appears on 3+ separate days.

## Single-day observations (not yet patterns)

- **Day 7 — Month 2 Week 2 Day 1 (Maximum Likelihood Estimation, python, intermediate):** Variable-naming clarity noted — cryptic locals `fst`/`scd`/`std` in `log_likelihood.py:41-45` and typo/truncated names `cureent_mu`/`log_likehd` in `mle_fit_gaussian.py:69-81`. Also `range(len(data))` indexing instead of direct iteration `for x in data`, error-message typos (`"data is empy"` `log_likelihood.py:37`, `"sigma cant be lessthan..."` `log_likelihood.py:35`), inconsistent punctuation/spacing, unnecessary list in `sum([...] for ...)` `mle_fit_gaussian.py:37`, and dead commented code ` # raise NotImplementedError`. No recurring pattern claimed yet — first observation of this type in available history.

- **Day 8 — Month 2 Week 2 Day 2 (MAP Estimation & Priors, python, intermediate):** Error handling is the main item — four bare `raise ValueError` with no message (`map_fit_gaussian.py:40,73-74,110,148`), inconsistent with the learner's own Day-1 style (`log_likelihood.py:20` does include messages). Also single-line `if ... : raise` bodies, a dead `n = len(data)` (`:149`), `np.arange` with a computed float step where the docstring promises inclusive endpoints (`:159-160`, so `mu_range[1]` is never evaluated), numpy pulled in for a grid the spec scoped to plain Python (`:23`), the `# MAP == L2` comment stranded below a `return` (`:78-79`), `sum([...])` list instead of a generator (`:113`), and `denomenator` misspelt (`:116`). No pattern claimed — see count note below.

**Second-occurrence counts (day 7 + day 8, still below the 3-day threshold):** identifier/message typos (`cureent_mu`, `log_likehd`, `"data is empy"` → `denomenator`) — 2 days; `sum([...])` list where a generator would do — 2 days; grid-bounds handling that misses an endpoint (`step = (max-min)/steps` with a range that stops short) — 2 days. Recorded as counts only, not as trends. If any of these appears again on a third day, promote it below.

## Themes (promoted after 3+ days)

_None yet — need 3 occurrences of the same category before promoting._
