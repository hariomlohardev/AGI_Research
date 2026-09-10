# Day 12 — Review + Consolidation: coding problems

## Problem 1 — consolidate the week's utilities

Create `code/week2_utils.py` containing these seven from-scratch functions. Do not import `sklearn`, `scipy.stats`, or the implementations from Days 7–11; the point is retrieval and consolidation. Standard-library helpers such as `math` and `collections.Counter` are allowed.

```python
def bayes_update(prior: Sequence[float], likelihood: Sequence[float], evidence: float) -> list[float]:
    """Return normalized posterior probabilities for multiple hypotheses."""

def log_likelihood(data: Sequence[float], mu: float, sigma: float) -> float:
    """Return the Gaussian log-likelihood of the observations."""

def mle_fit_gaussian(data: Sequence[float]) -> tuple[float, float]:
    """Return the Gaussian MLE `(mu_hat, sigma_hat)`."""

def map_fit_gaussian(data: Sequence[float], prior_mu: float, prior_sigma: float) -> float:
    """Return the MAP estimate of `mu` under a Gaussian prior."""

def entropy(pmf: Sequence[float], base: float = 2.0) -> float:
    """Return Shannon entropy in the selected logarithm base."""

def cross_entropy(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Return cross-entropy H(P, Q) in the selected logarithm base."""

def kl_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Return D_KL(P || Q) in the selected logarithm base."""

def mutual_information(x: Sequence[Hashable], y: Sequence[Hashable], base: float = 2.0) -> float:
    """Return empirical mutual information for paired discrete observations."""
```

The exact signatures may be adjusted slightly if your earlier implementations use a consistent naming choice, but keep the public behavior below.

### Full validation contracts

Every public function's docstring must include its complete `Raises: ValueError` contract — this is part of today's consolidation quality bar.

- `bayes_update`: raise for empty sequences, unequal lengths, negative priors/likelihoods, a prior that does not sum to 1, non-positive evidence, or a posterior denominator of zero.
- `log_likelihood`: raise for empty data or `sigma <= 0`.
- `mle_fit_gaussian`: raise for empty data; a one-point dataset may return `sigma_hat = 0.0` because that is the Gaussian MLE, but document that consequence.
- `map_fit_gaussian`: raise for empty data or `prior_sigma <= 0`.
- `entropy`: raise for empty PMFs, negative probabilities, PMFs not summing to 1 within `1e-9`, or `base <= 0` / `base == 1`; skip zero-probability terms.
- `cross_entropy` and `kl_divergence`: raise for empty inputs, unequal lengths, negative entries, either PMF not summing to 1 within `1e-9`, `base <= 0` / `base == 1`, or `q[i] == 0` where `p[i] > 0`.
- `mutual_information`: raise for empty inputs, unequal lengths, or `base <= 0` / `base == 1`; labels must be hashable because they are counted as dictionary keys.

### Required assertions in `__main__`

Add one small assertion for each function, then keep the file runnable with:

```bash
cd month_2/week_2/day_6 && python code/week2_utils.py
```

Use these checks as the minimum:

1. Bayes: a two-hypothesis update returns a normalized posterior and favors the hypothesis with the larger likelihood.
2. Gaussian MLE: `data = [1.0, 2.0, 3.0]` returns `mu_hat == 2.0` and `sigma_hat == sqrt(2/3)`.
3. MAP: the estimate lies between the data mean and `prior_mu`, and moving `prior_sigma` smaller pulls it nearer the prior.
4. Entropy: a fair binary PMF gives `1.0` bit.
5. Cross-entropy/KL: for `P=[0.5,0.5]`, `Q=[0.9,0.1]`, verify `H(P,Q) - H(P) == D_KL(P||Q)` within tolerance.
6. MI: a perfectly copied binary label gives `1.0` bit, while an independent pair gives `0.0`.

### Running tests

```bash
cd month_2/week_2/day_6 && python -m pytest code/tests/ -v
```

The provided tests are intentionally written against the stubs and should pass after you implement the consolidated file. The tests cover the core values, cross-function identities, and the documented error boundaries; they are not a substitute for the two written summary paragraphs.

### Done when (from the source plan)

`week2_utils.py` runs cleanly top to bottom with all assertions passing, and you have written the two summary paragraphs without checking notes.
