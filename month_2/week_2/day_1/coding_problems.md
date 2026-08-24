# Week 2 Day 1 — Maximum Likelihood Estimation (MLE) — Coding Problems

Scale: **intermediate** (source plan as-is, no extra scaffolding). Python, no `sklearn`/`scipy.stats` — build from scratch.

---

## Problem 1 — `log_likelihood(data, mu, sigma)` (from scratch)

Write a function that returns the log-likelihood of `data` under a Gaussian `N(mu, sigma)`.

- **Signature:** `log_likelihood(data: list[float] | np.ndarray, mu: float, sigma: float) -> float`
- **Formula:** `LL = sum_i log N(x_i | mu, sigma)` where `N(x|mu,sigma) = (1 / (sigma*sqrt(2*pi))) * exp(-0.5*((x-mu)/sigma)^2)`. Implement the Gaussian PDF yourself — no `scipy.stats.norm`.
- **Constraints:** `sigma > 0` must be enforced (raise `ValueError` if not). Work in log-space directly; do not compute the product then log it (it underflows).
- **Verification:** Against your hand-derived log-likelihood on a tiny dataset (e.g. `data=[0,1,2]`, `mu=1, sigma=1`) compute the answer by hand and compare.

## Problem 2 — `mle_fit_gaussian` (two ways, same answer)

Write an MLE fitter that returns `(mu_hat, sigma_hat)` maximising the log-likelihood.

**2a — Closed-form (what you derived by hand):**
- `mle_fit_gaussian_closed_form(data) -> tuple[float, float]`
- `mu_hat = mean(data)`, `sigma_hat = sqrt(mean((x - mu_hat)^2))` (the *biased* MLE variance, dividing by `n` not `n-1`). Derive this from `d/dmu log L = 0` and `d/d sigma^2 log L = 0` on paper first, then implement.

**2b — Numerical optimisation (grid search or Month-1 gradient ascent):**
- `mle_fit_gaussian_grid(data, mu_range: tuple[float,float], sigma_range: tuple[float,float], steps: int = 100) -> tuple[float, float]`
- Loop over a `steps x steps` grid of `(mu, sigma)` values, evaluate `log_likelihood` at each, return the pair with highest LL. Alternative: reuse your Month-1 gradient-descent to *maximise* LL (negate for minimisation). Either is acceptable — document which you chose.

**Required comparison:**
- Generate a synthetic dataset with known truth (e.g. `true_mu=5, true_sigma=2, n=500, seed=0` using `np.random.normal` only for *data generation*, not for fitting).
- Run both fitters on the same data and assert they agree within tolerance: `|mu_closed - mu_grid| < 0.1` and `|sigma_closed - sigma_grid| < 0.1` (grid is coarse; tighter if you implement gradient ascent). Also assert both are close to the true parameters.

## Problem 3 — Sample-size experiment (Law of Large Numbers)

Not a graded test function, but a required runnable experiment (keep it as `if __name__ == "__main__"` in `mle_fit_gaussian.py` or a separate `experiment.py`):

- Run `mle_fit_gaussian_closed_form` on `n=5` vs `n=5000` (same true `mu, sigma`), print both estimates and comment in code why the `n=5` estimate varies much more — connect to LLN.
- This is what the practice question 4 asks you to reason about; your comment is the artifact.

---

### Running

```bash
pytest -v
```

Stubs raise `NotImplementedError` until you implement them. Tests import from `code.log_likelihood` and `code.mle_fit_gaussian`.

### What not to use

- No `sklearn`, `scipy.stats`, `np.mean`/`np.var` for the *fitter logic* itself (using `np.mean` only to generate synthetic data is fine; the fitter must compute the mean itself).
- No `scipy.optimize` — the point is to feel optimisation yourself.
