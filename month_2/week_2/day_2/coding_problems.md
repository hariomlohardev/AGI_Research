# Week 2 Day 2 — MAP Estimation & Priors — Coding Problems

Scale: **intermediate** (source plan as-is). Python, no `sklearn`/`scipy.stats` — build from scratch. Reuses yesterday's `log_likelihood`.

---

## Problem 1 — Log-prior & log-posterior (the MAP objective)

Build the two extra log terms that turn MLE into MAP.

**Signatures**
```python
def log_prior_gaussian(mu: float, prior_mu: float, prior_sigma: float) -> float:
    """Log PDF of mu under N(prior_mu, prior_sigma)."""
def log_posterior(data: Sequence[float], mu: float, sigma: float,
                 prior_mu: float, prior_sigma: float) -> float:
    """log P(data|mu,sigma) + log P(mu).  Uses log_likelihood + log_prior_gaussian."""
```

- `log_prior_gaussian` is exactly the Gaussian log-PDF for a *single* value `mu` (same formula as `log_likelihood` but with `n=1` and `x=mu`). Raise `ValueError` if `prior_sigma <= 0`.
- `log_posterior` must call your Day-1 `log_likelihood` (do not re-implement it) and add the prior. `sigma` is still the *likelihood* sigma — the prior here is only on `mu` (this is the standard "Gaussian prior on mu" setup the hand-derivation uses).
- Show algebraically in a code comment that `log_posterior = log_likelihood - (mu - prior_mu)^2 / (2 prior_sigma^2) + const`, and that when `prior_mu=0` this is `log_likelihood - lambda*mu^2` with `lambda = 1/(2 prior_sigma^2)` — i.e. **L2-regularised MLE**. Leave this comment in the file (`# MAP == L2 when...`).

### Why this exists
This is the "MAP is regularisation" connection — the prior term is exactly a weight-decay penalty.

## Problem 2 — `map_fit_gaussian` (two ways, must agree)

Write a MAP fitter that returns `(mu_map, sigma_map)` maximising the log-posterior. Priors are only on `mu`; `sigma_map` is still the usual MLE-style variance estimate (keeps the problem focused).

**2a — Closed-form (weighted average — derive on paper first)**
```python
def map_fit_gaussian_closed_form(data: Sequence[float],
                                 prior_mu: float,
                                 prior_sigma: float) -> tuple[float, float]:
```
- Let `n = len(data)`, `mu_mle = mean(data)`, `sigma_mle^2 = mean((x - mu_mle)^2)`. Treat `sigma_mle` as the known `sigma` while solving for `mu_map` (this matches the "known variance" derivation you do by hand; it isolates the prior's effect on `mu`).
- Then `mu_map = (n * mu_mle / sigma_mle^2 + prior_mu / prior_sigma^2) / (n / sigma_mle^2 + 1 / prior_sigma^2)`. Derive this from `d/dmu log_posterior = 0` (see `learn.md` Math §3) — do not just copy the line, work it once on paper.
- Return `(mu_map, sigma_mle)` — sigma is unchanged (the prior only pulls `mu`). Raise `ValueError` on empty data or `prior_sigma <= 0`.

**2b — Numerical optimisation (grid search over the posterior)**
```python
def map_fit_gaussian_grid(data: Sequence[float],
                          prior_mu: float,
                          prior_sigma: float,
                          mu_range: tuple[float,float],
                          sigma_range: tuple[float,float],
                          steps: int = 100) -> tuple[float, float]:
```
- Brute-force `steps x steps` grid over `mu_range x sigma_range`, evaluate `log_posterior` at each point, return the pair with highest posterior. This is the *honest* MAP when no closed form is available — valuable precisely because most MAP problems have no closed form.
- Grid must actually maximise the *posterior* (not the likelihood). A common bug is calling `log_likelihood` here — tests will catch it by checking that strong priors actually pull the grid optimum toward `prior_mu`.

**Required agreement:** On synthetic data with moderate prior, `|mu_closed - mu_grid| < 0.15` and `|sigma_closed - sigma_grid| < 0.15`.

> Convenience alias (so tests written for either name work): `map_fit_gaussian = map_fit_gaussian_closed_form`.

## Problem 3 — Required experiments (runnable, not just tested)

Keep these as `if __name__ == "__main__":` blocks (in `map_fit_gaussian.py` or a separate `experiment.py`) — they are the artifacts the "Done when" clause refers to.

1. **3-point outlier robustness (MLE vs MAP):** Generate or hand-pick 3 data points that are deliberate outliers relative to a reasonable prior (e.g. prior `N(0,1)` but data `≈[8, 9, 7]`). Print `mu_mle` vs `mu_map` with `prior_mu=0, prior_sigma=1`. MAP should visibly sit *between* the raw sample mean and the prior — demonstrate the robustness claim.

2. **Prior-strength sweep (the intuition plot):** Fix a small dataset (e.g. `data=[0.5, -0.3, 0.2]`, `prior_mu=5`). Vary `prior_sigma` from `10.0` (very weak) down to `0.03` (very strong) — e.g. `[10, 5, 2, 1, 0.5, 0.1, 0.03]` — compute `mu_map` each time and print (or `matplotlib` plot) the sequence. It must move monotonically from "≈ MLE" to "≈ prior". Note *how tight* the prior has to get before it wins: those 3 points have `sigma^2_mle ≈ 0.109`, so the data's precision `n/sigma^2 ≈ 27.6`, and the prior's `1/prior_sigma^2` has to clear that — at `prior_sigma=0.5` it is worth only `4`, and `mu_map` is still `0.75`, nowhere near `5`. That comparison of precisions is the thing `/done` will ask you to explain.

3. **Flat-prior check (conceptual — log it as a comment):** Observe that as `prior_sigma → ∞` (e.g. `1e6`), `mu_map → mu_mle` exactly. Write one comment line stating: "MAP → MLE under flat prior because `1/prior_sigma^2 → 0`." No extra code needed.

4. **L1 vs L2 thought experiment (also a comment, not code you must run):** In `map_fit_gaussian.py` leave a short comment block predicting: if the prior were *Laplace* (double-exponential) instead of Gaussian, what would the penalty look like (`|mu - prior_mu|` instead of squared), and what effect would you expect — sparser / shrinkage-to-exactly-prior behaviour vs Gaussian's smooth pull? This is practice-question 4; the comment is the artifact.

---

### Running

```bash
pytest -v
```

Stubs raise `NotImplementedError` until you implement them. Tests import from `code.map_fit_gaussian` and `code.log_likelihood`.

### What not to use

- No `sklearn`, `scipy.stats`, `scipy.optimize`.
- For the *fitter logic* do not use `np.mean`/`np.var` — compute the mean/variance yourself (using `np.random.normal` only to *generate* synthetic data is fine).
- You may use `numpy`/`matplotlib` only for data generation and the optional sweep plot.

### Done when (from the source plan)

You can explain without notes why **L2 regularisation is secretly a Gaussian prior**, and your `map_fit_gaussian` visibly pulls estimates toward the prior more strongly as `prior_sigma` shrinks (equivalently, as `prior_strength = 1/prior_sigma^2` grows).
