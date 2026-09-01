# Roadmap — Week 2 Day 2: MAP Estimation & Priors

**Current:** Month 2, Week 2, Day 2 (global Day 8) — *MLE + prior = MAP = regularisation*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` first (start with StatQuest MAP) — 30-40 min. Focus on how MAP = `P(data|θ)·P(θ)` differs from MLE = `P(data|θ)` and why a Gaussian prior *is* L2.
2. **Math on paper (30-40 min, do not skip):**
   - Write Bayes in parameter form: `P(θ|data) ∝ P(data|θ)·P(θ)`.
   - Write MLE objective vs MAP objective side-by-side.
   - For a Gaussian prior `P(μ)=N(prior_mu, prior_sigma²)`, expand `log P(μ|data) = log P(data|μ,σ) + log P(μ) + const`, show the prior term becomes `-((μ-prior_mu)²)/(2 prior_sigma²)` — a quadratic penalty. When `prior_mu=0`, note this is `-λ μ²` with `λ=1/(2 prior_sigma²)`.
   - Derive `d/dμ log_posterior = 0 → μ_map = (n·μ_mle/σ² + prior_mu/τ²)/(n/σ²+1/τ²)` (the weighted average). Keep the sheet — `/done` will ask you to reproduce why this collapses to MLE when `τ→∞`.
3. **Code — log_prior & log_posterior** (`coding_problems.md` Problem 1): implement `log_prior_gaussian` and `log_posterior` on top of yesterday's `log_likelihood`. Leave the `# MAP == L2` comment.
4. **Code — `map_fit_gaussian` two ways** (Problem 2): closed-form weighted average vs grid search over the *posterior*. Confirm they agree within `0.15`.
5. **Experiments** (Problem 3): run the three required `__main__` demos — 3-point outlier MLE vs MAP, prior-strength sweep (`prior_sigma 10→0.5`), and the flat-prior observation. Keep the L1-vs-L2 prediction comment.
6. **Practice questions** (in `learn.md`) on paper before you open the editor for step 5 — if you can't answer `MAP → MLE under flat prior` without code, you don't know the math yet.
7. **Commit & run `/done`:** `pytest -v` must pass before the quiz. `/done` will quiz you on flat priors, MAP stability, the Gaussian-L2 link, and the Laplace/L1 prediction.

---

## How this ties to the final project (`minisklearn`)

This day is the **regularisation** day. Week 3 Day 1's Linear Regression will add `λ||w||²` to its loss — that `λ` *is* `1/(2 prior_sigma²)` from today. Week 3 Day 2's Logistic Regression will do the same. Week 4's `minisklearn` library will expose a `regularization` argument whose "mystery math" is exactly the MAP derivation you write today. Keep `map_fit_gaussian.py` — Week 4's comparison table will let you swap MLE vs MAP fits and quantify the gain on small noisy tabular data.

---

## Files in this day

- `learn.md` — what to learn + verified videos
- `coding_problems.md` — problem statements (intermediate scale)
- `code/map_fit_gaussian.py` — stubs for `log_prior_gaussian`, `log_posterior`, `map_fit_gaussian_closed_form` / `_grid`
- `code/tests/` — pytest suite (must pass before `/done`)
- `further_reading.md` — appended only if you run `/for-read` (opt-in)
