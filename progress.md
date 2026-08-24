# Progress

Human-readable log of months/weeks/days — spans all months. `state.json` is the source of truth; this file is the readable shadow.

---

## Month 1 — Math Foundations + Autograd Engine — ✅ Already done (outside this system)

> Per `D:\Roadmap\firefox\roadmap-v2-free.md`, Month 1 (Gram-Schmidt / SVD / PCA, Jacobians, chain rule, hand-built autograd, MLP, tweaked Adam) was completed **before** this curriculum-system tracking was adopted. No `state.json` entries were created for it — it is recorded here for continuity only.

---

## Month 2 — Probability, Information Theory & Classical ML From Scratch
- **Started:** 2026-08-24
- **Language:** `python` · **Skill level:** `intermediate` · **Test command:** `pytest` (default)
- **Current:** Month 2, Day 1 (Week 1 Day 1 — `current`)
- **Source plan:** `D:\Roadmap\firefox\months\month-2-detailed.md`

### Weeks & days

**Week 1 — Probability & Bayesian Inference** — `review_status: pending`
- Done when: Bayes' theorem is the single most important idea in this week — everything else (Naive Bayes, MLE, MAP, even how LLMs sample the next token) traces back to it. By the end of this week you should be able to derive Bayes' theorem from scratch, on paper, without looking it up.
- Day 1 (global 1) — **current** — Bayes' Theorem & Conditional Probability
- Day 2 (global 2) — pending — Naive Bayes Classifier
- Day 3 (global 3) — pending — Random Variables & Distributions (PMF, PDF, CDF)
- Day 4 (global 4) — pending — Expected Value & Variance
- Day 5 (global 5) — pending — Covariance Matrices
- Day 6 (global 6) — pending — Gaussian, Bernoulli, Multinomial + CLT

**Week 2 — Maximum Likelihood, MAP & Information Theory** — `review_status: pending`
- Done when: Learn how models actually learn parameters from data (MLE/MAP), then learn the mathematical vocabulary of "surprise" and "difference between distributions" (entropy, cross-entropy, KL divergence) — this is the exact math behind every loss function you'll use for the rest of this roadmap.
- Day 1 (global 7) — pending — Maximum Likelihood Estimation (MLE)
- Day 2 (global 8) — pending — MAP Estimation & Priors
- Day 3 (global 9) — pending — Shannon Entropy
- Day 4 (global 10) — pending — Cross-Entropy & KL Divergence
- Day 5 (global 11) — pending — Mutual Information
- Day 6 (global 12) — pending — Review + Consolidation

**Week 3 — Classical ML From Scratch** — `review_status: pending`
- Done when: Build the algorithms that ran the world before deep learning — and understand that they're not "outdated," they're still the right tool for a huge fraction of real-world problems (tabular data, small datasets, interpretability requirements).
- Day 1 (global 13) — pending — Linear Regression (Gradient Descent Version)
- Day 2 (global 14) — pending — Logistic Regression
- Day 3 (global 15) — pending — Support Vector Machines & the Kernel Trick
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
