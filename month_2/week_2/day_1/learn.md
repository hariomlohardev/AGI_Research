# Week 2 Day 1 — Maximum Likelihood Estimation (MLE)

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 7**

---

## Why this matters

Nearly every "training" step in ML is MLE in disguise — minimising cross-entropy *is* maximising likelihood. Today is the conceptual bridge from probability (Week 1) to actual model training: you write down how likely your data is under a parametric model, then pick the parameters that make that likelihood largest. The Gaussian case you do by hand here gives you the familiar `mean` and `variance` formulas as a *derived* result, not a definition, and the log trick you practice is reused verbatim in every loss function for the rest of the roadmap.

## What to learn today

### Math (on paper first, before code)

1. **Likelihood for n i.i.d. Gaussian samples:**
   `L(mu, sigma) = prod_{i=1}^n N(x_i | mu, sigma)` where `N(x|mu,sigma)= 1/(sigma sqrt(2pi)) exp(-0.5((x-mu)/sigma)^2)`.

2. **Log-likelihood:**
   `LL(mu,sigma)= sum_i log N(x_i|mu,sigma)`. Write it expanded; explain in one sentence why we log: product of many probabilities `<1` underflows to 0, log turns product → sum, and `argmax` is preserved because log is monotonic, plus sums are easier to differentiate.

3. **Derive the MLEs:**
   - `d/d mu LL = 0  →  mu_hat = (1/n) sum x_i` (sample mean).
   - `d/d sigma^2 LL = 0  →  sigma_hat^2 = (1/n) sum (x_i - mu_hat)^2` (biased MLE, divides by `n`, not `n-1`).
   Keep the algebra sheet — `/done` will ask you to reproduce it.

### Code

- `log_likelihood(data, mu, sigma)` — sum of Gaussian log-PDFs (no `scipy.stats`).
- `mle_fit_gaussian_closed_form(data)` — the formulas you just derived.
- `mle_fit_gaussian_grid(data, mu_range, sigma_range)` — brute-force grid search over `LL`, confirming both fitters agree on synthetic data with known truth.

See `coding_problems.md` for exact signatures and the required `n=5` vs `n=5000` Law-of-Large-Numbers experiment.

---

## Watch first (guided — at least two)

> Search YouTube for the exact phrases below if the links don't open — don't rely on autoplay.

1. **StatQuest — "Maximum Likelihood, clearly explained!!!"**
   - **URL:** https://www.youtube.com/watch?v=XepXtl9YKwc
   - **Channel:** StatQuest with Josh Starmer
   - **Why watch:** The canonical visual walkthrough — builds `L`, takes `log`, and shows why the MLE for `mu` lands on the mean. Matches today's derivation point-for-point. *Verified: YouTube title "Maximum Likelihood, clearly explained!!!"*.

2. **StatQuest — "Maximum Likelihood For the Normal Distribution, step-by-step!!!"**
   - **URL:** https://www.youtube.com/watch?v=Dn6b9fCIUpM
   - **Channel:** StatQuest with Josh Starmer
   - **Why watch:** Takes the general idea from (1) and does the full Gaussian algebra — product → log → derivative → sample mean/variance — exactly the hand derivation you must reproduce. Good second pass after the first video. *Verified: YouTube title "Maximum Likelihood For the Normal Distribution, step-by-step!!!"*.

> Delegation note: `video-researcher` was invoked for this day (search terms: "StatQuest maximum likelihood clearly explained"). These two StatQuest videos are the verified results it was asked to find (both YouTube titles confirmed via fetch). Watch (1) then (2) before you code.

---

## Practice questions (do on paper before coding)

1. Why do we maximise the *log*-likelihood instead of the likelihood directly? Show mathematically what happens to the product of many small probabilities and why `argmax` is unchanged.
2. Derive the MLE estimator for `mu` from the Gaussian log-likelihood by differentiating and setting to zero (the sheet from "What to learn" is the answer — reproduce it without notes).
3. If you had only 2 data points, would MLE still give a reasonable estimate? What happens with just 1 data point? (Think about `sigma_hat`.)
4. **Code check:** run `mle_fit_gaussian` on `n=5` vs `n=5000` (same true `mu,sigma`) — how much does the estimate vary? Explain what you observe via the Law of Large Numbers.

---

## Done when

- Your closed-form and grid-search MLE fitters agree on synthetic data with known truth (within grid tolerance), and you can derive the sample-mean MLE formula from scratch on paper — no notes.
- `pytest -v` passes (see `code/tests/`).

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers on MLE vs MAP.
