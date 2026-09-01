# Week 2 Day 2 — MAP Estimation & Priors

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 8**

---

## Why this matters

MAP is where Bayes' theorem (Week 1) meets MLE (yesterday) — it's the bridge from "pure frequentist fitting" to Bayesian thinking, and it's mathematically *identical* to L2/L1 regularisation. That one connection unlocks why `weight_decay` in every neural net is not a hack but a prior. With tiny or outlier-heavy data, MLE overfits wildly; MAP stabilises it by blending what the data says with what you believed before seeing the data. Varying the prior's strength from "I barely care" to "I trust my prior over the data" and watching the estimate slide is the single most concrete way to *feel* what a prior does.

## What to learn today

### Math (on paper first, before code)

1. **Bayes in parameter form:**
   `P(θ|data) ∝ P(data|θ) · P(θ)` — posterior ∝ likelihood × prior. Write it with `θ = μ` for today's Gaussian case.

2. **MLE vs MAP side-by-side:**
   - MLE: `θ_hat = argmax_θ log P(data|θ)`
   - MAP: `θ_hat = argmax_θ [log P(data|θ) + log P(θ)]`
   In words: MAP adds a *penalty* (the log-prior) to the MLE objective. When the prior is flat/uniform, that term is constant, so MAP collapses to MLE.

3. **Gaussian prior → L2:**
   Let `P(μ)=N(prior_mu, prior_sigma²)`. Then
   `log P(μ) = -0.5·log(2π prior_sigma²) - (μ - prior_mu)²/(2 prior_sigma²)`.
   So `log_posterior = log_likelihood - (μ - prior_mu)²/(2 prior_sigma²) + const`.
   If `prior_mu=0`, this is `log_likelihood - λ·μ²` with `λ = 1/(2 prior_sigma²)` — exactly L2-regularised MLE / weight decay. Derive this line yourself; leave the `# MAP == L2 when...` comment in `map_fit_gaussian.py` as your artifact.
   From `d/dμ log_posterior = 0` you get the weighted average
   `μ_map = (n·μ_mle/σ² + prior_mu/τ²) / (n/σ² + 1/τ²)` where `τ = prior_sigma` and `σ²` is the likelihood variance. As `τ → ∞` (flat prior), `1/τ² → 0` and `μ_map → μ_mle`.

### Code

- `log_prior_gaussian(mu, prior_mu, prior_sigma)` + `log_posterior(data, mu, sigma, prior_mu, prior_sigma)` — built on yesterday's `log_likelihood`.
- `map_fit_gaussian_closed_form(data, prior_mu, prior_sigma)` — the weighted-average formula above.
- `map_fit_gaussian_grid(data, prior_mu, prior_sigma, mu_range, sigma_range)` — brute-force grid over the *posterior* (not just likelihood), confirming both fitters agree.

See `coding_problems.md` for exact signatures, the 3-point outlier experiment, and the prior-strength sweep that proves "weak → MLE, strong → prior".

---

## Watch first (guided — at least two)

> Search YouTube for the exact phrases below if links don't open — don't rely on autoplay.

1. **(ML 6.1) Maximum a posteriori (MAP) estimation — mathematicalmonk**
   - **URL:** https://www.youtube.com/watch?v=kkhdIriddSI
   - **Channel:** mathematicalmonk
   - **Why watch:** The cleanest white-board derivation of MAP vs MLE — writes `P(θ|data) ∝ P(data|θ)P(θ)`, takes logs, and shows exactly why a flat prior recovers MLE and a Gaussian prior adds a quadratic penalty. Matches the hand algebra you must reproduce (*Verified via YouTube oEmbed — title "(ML 6.1) Maximum a posteriori (MAP) estimation"* — found via DuckDuckGo search for "MAP estimation youtube").

2. **Bayesian Linear Regression and Maximum a Posteriori (MAP) Estimate — Steve Brunton**
   - **URL:** https://www.youtube.com/watch?v=wdWHbYdhfG8
   - **Channel:** Steve Brunton (Eigensteve)
   - **Why watch:** Shows MAP in a regression setting — the same "likelihood + prior = regularised fit" idea but with real data, and explicitly calls out how the prior stabilises estimates when data are scarce/outliers. Directly motivates the 3-point outlier experiment in `coding_problems.md` (*Verified via YouTube oEmbed — title "Bayesian Linear Regression and Maximum a Posteriori (MAP) Estimate"*).

3. **Regularization Part 1: Ridge (L2) Regression — StatQuest with Josh Starmer** *(bonus, directly ties to Day 2's punchline)*
   - **URL:** https://www.youtube.com/watch?v=Q81RR3yKn30
   - **Channel:** StatQuest with Josh Starmer
   - **Why watch:** Watch after the two MAP videos to cement the "Gaussian prior = L2 = weight decay" connection — StatQuest shows Ridge's `λ·slope²` penalty visually, which is the same `λ = 1/(2 prior_sigma²)` term you just derived (*Verified via YouTube oEmbed — title "Regularization Part 1: Ridge (L2) Regression"*).

> Delegation note: `video-researcher` was delegated for this day (suggested search: "StatQuest MAP maximum a posteriori clearly explained"). The three videos above are verified via YouTube oEmbed/DuckDuckGo search as fallback while the subagent's WebSearch was timing out — all titles confirmed via `https://www.youtube.com/oembed?url=…` on 2026-09-01.

---

## Practice questions (do on paper before coding)

1. If your prior is completely flat (uniform, no preference), show algebraically that MAP reduces exactly to MLE (hint: what is `log P(θ)` for a uniform prior?).
2. Why does MAP tend to be more stable than MLE when you have very little data? Give one-sentence intuition ("prior acts like extra pseudo-observations") then justify with the weighted-average formula's `n/σ²` vs `1/τ²` terms.
3. Explain the Gaussian-prior ↔ L2 connection: why is "regularisation" secretly Bayesian? Derive the `λ = 1/(2 prior_sigma²)` line without looking at `coding_problems.md`.
4. **Code check (thought experiment, leave as comment):** If the prior were Laplace `P(μ) ∝ exp(-|μ - prior_mu|/b)` (double-exponential), what penalty would appear in `log_posterior`? What effect would you expect vs Gaussian — sparser/shrink-to-exactly-prior behaviour vs smooth pull? (*Write your 1-2 sentence prediction in `map_fit_gaussian.py`.*)

---

## Done when

You can explain without notes why **L2 regularisation is secretly a Gaussian prior**, and your `map_fit_gaussian` demo visibly pulls estimates toward the prior more strongly as `prior_sigma` shrinks (equivalently `prior_strength = 1/prior_sigma²` grows) — weak → ≈ MLE, strong → ≈ prior. `pytest -v` passes.

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers on MAP vs MLE, priors, or L1/Laplace shrinkage.
