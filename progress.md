# Progress

Human-readable log of months/weeks/days — spans all months. `state.json` is the source of truth; this file is the readable shadow.

---

## Month 1 — Math Foundations + Autograd Engine — ✅ Already done (outside this system)

> Per `D:\Roadmap\firefox\roadmap-v2-free.md`, Month 1 (Gram-Schmidt / SVD / PCA, Jacobians, chain rule, hand-built autograd, MLP, tweaked Adam) was completed **before** this curriculum-system tracking was adopted. No `state.json` entries were created for it — it is recorded here for continuity only.

---

## Month 2 — Probability, Information Theory & Classical ML From Scratch
- **Started:** 2026-08-24
- **Language:** `python` · **Skill level:** `intermediate` · **Test command:** `pytest` (default)
- **Current:** Month 2, Day 15 (Week 3 Day 3 — `current`)
- **Source plan:** `D:\Roadmap\firefox\months\month-2-detailed.md`

### Weeks & days

**Week 1 — Probability & Bayesian Inference** — `review_status: pending`
- Done when: Bayes' theorem is the single most important idea in this week — everything else (Naive Bayes, MLE, MAP, even how LLMs sample the next token) traces back to it. By the end of this week you should be able to derive Bayes' theorem from scratch, on paper, without looking it up.
- Day 1 (global 1) — **done** — Bayes' Theorem & Conditional Probability _(via `/skip-to`)_
- Day 2 (global 2) — **done** — Naive Bayes Classifier _(via `/skip-to`)_
- Day 3 (global 3) — **done** — Random Variables & Distributions (PMF, PDF, CDF) _(via `/skip-to`)_
- Day 4 (global 4) — **done** — Expected Value & Variance _(via `/skip-to`)_
- Day 5 (global 5) — **done** — Covariance Matrices _(via `/skip-to`)_
- Day 6 (global 6) — **done** — Gaussian, Bernoulli, Multinomial + CLT _(via `/skip-to`)_

**Week 2 — Maximum Likelihood, MAP & Information Theory** — `review_status: done`
- Done when: Learn how models actually learn parameters from data (MLE/MAP), then learn the mathematical vocabulary of "surprise" and "difference between distributions" (entropy, cross-entropy, KL divergence) — this is the exact math behind every loss function you'll use for the rest of this roadmap.
- Day 1 (global 7) — **done** — Maximum Likelihood Estimation (MLE) — confidence: `strong`, difficulty: `medium`, time: 150m
- Day 2 (global 8) — **done** — MAP Estimation & Priors — confidence: `struggled`, difficulty: `hard`, time: not reported
- Day 3 (global 9) — **done** — Shannon Entropy — confidence: `shaky`, difficulty: `easy`, time: not reported
- Day 4 (global 10) — **done** — Cross-Entropy & KL Divergence — confidence: `shaky`, difficulty: `easy`, time: not reported
- Day 5 (global 11) — **done** — Mutual Information — confidence: `shaky`, difficulty: `medium`, time: 2–3 hours (self-reported range)
- Day 6 (global 12) — **done** — Review + Consolidation — confidence: `struggled`, difficulty: easy-to-medium (self-reported as a range, so `difficulty` stays `null`), time: 4–5 hours (self-reported range, so `time_spent_minutes` stays `null`)
- Week 2 review: **passed** — MLE vs MAP objectives plus uniform-prior coincidence, and `H(P,Q) = H(P) + D_KL(P||Q)` with `H(P)` constant in `Q`.

**Week 3 — Classical ML From Scratch** — `review_status: pending`
- Done when: Build the algorithms that ran the world before deep learning — and understand that they're not "outdated," they're still the right tool for a huge fraction of real-world problems (tabular data, small datasets, interpretability requirements).
- Day 1 (global 13) — **done** — Linear Regression (Gradient Descent) — confidence: `struggled`, difficulty: medium-to-hard (self-reported as a range, so `difficulty` stays `null`), time: about 6–8 hours (self-reported range, so `time_spent_minutes` stays `null`) — has flagged questions (Q4 L2↔prior) to revisit
- Day 2 (global 14) — **done** — Logistic Regression — confidence: `shaky`, difficulty: not reported as a single value ("not that much hard" in feedback, so `difficulty` stays `null`), time: 360m (self-reported "roughly 6 hours")
- Day 3 (global 15) — `current` — Support Vector Machines & the Kernel Trick
- Day 4 (global 16) — pending — Decision Trees (ID3 Algorithm)
- Day 5 (global 17) — pending — Random Forests
- Day 6 (global 18) — pending — K-Means Clustering

**Week 4 — Month 2 Capstone** — `review_status: pending`
- Done when: Stop writing isolated scripts. Package everything from Weeks 1–3 into one real, reusable, tested library — and use it on a genuine dataset the way you'd use sklearn.
- Day 1 (global 19) — pending — Package Structure (`minisklearn`)
- Day 2 (global 20) — pending — Apply to a Real Dataset (EDA, imputation, encoding, split)
- Day 3 (global 21) — pending — Evaluate (confusion matrix, accuracy/precision/recall/F1)
- Day 4 (global 22) — pending — Compare Against Real `sklearn`
- Day 5 (global 23) — pending — Write It Up (README + blog post)
- Day 6 (global 24) — pending — Push to GitHub + Rest

> Notes: This month has **no explicitly labelled Rest Days** — every day above is a real study day (status `pending`/`current`). Rest will be real on Month 2 Day 6 only in the sense of lighter load. Badges / streak / confusion notes / flags all start empty — they'll fill as you run `/done`, `/confused`, `/spaced-review`.
