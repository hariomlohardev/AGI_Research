# Month 2 — Final Project: `minisklearn` — A From-Scratch Classical ML Library

**Goal:** turn a folder of standalone scripts into one real, importable, tested Python package you can pip-install locally and publish to GitHub as a portfolio piece.

This capstone is not a throwaway exercise — it is the proof that every hand-derived formula from Weeks 1–3 actually works on messy, real data. By the end of Week 4 you will have validated your implementations against the industry standard (`scikit-learn`) on the same train/test split and written honestly about where they matched and where they diverged.

## Why this project matters

- **Before deep learning, this was machine learning.** Linear/logistic regression, Naive Bayes, decision trees, random forests, and K-Means still handle a huge share of real-world tabular and small-data work — and interviewers know it.
- **It reuses all three weeks deliberately:**
  - Week 1 (Bayes, distributions, variance/covariance, CLT) → Naive Bayes, K-Means, and the probabilistic grounding for everything after.
  - Week 2 (MLE/MAP, entropy, cross-entropy, KL, MI) → the loss functions (BCE/MSE) and regularisation insight (MAP ↔ L2) behind linear/logistic regression, plus the entropy → information gain → ID3 chain.
  - Week 3 (the five classical models) → the package itself.

## Package skeleton (created on Week 4 Day 1)

```
minisklearn/
  __init__.py
  linear_model.py      # LinearRegression (GD), LogisticRegression (sigmoid + BCE)
  naive_bayes.py       # (optional) Naive Bayes spam filter — week 1 artifact, exposable here
  tree.py              # DecisionTreeClassifier (ID3) — reuses your entropy() from Week 2 Day 3
  ensemble.py          # RandomForestClassifier — bagging + feature-randomness on top of ID3
  cluster.py           # KMeans — assignment/update loop until convergence
  metrics.py           # accuracy, precision, recall, f1_score, confusion_matrix
  utils.py             # (optional) week2_utils.py consolidation — entropy, kl_divergence, etc.
```

Every public class follows a consistent `sklearn`-like API:

```python
model = LinearRegression()          # or LogisticRegression(), DecisionTreeClassifier(), ...
model.fit(X_train, y_train)
preds = model.predict(X_test)       # + predict_proba where relevant
```

That consistency is load-bearing: it lets you loop over models in an evaluation harness without rewriting glue code.

## How Month 2's weeks feed Week 4

| Week 4 Day | What happens | Which prior day it pays off |
|---|---|---|
| **Day 1 — Package structure** | Move Week 3 models into `minisklearn/`, add `metrics.py`, write docstrings | Week 3 Days 1–6 |
| **Day 2 — Apply to a real dataset** | Pick a real Kaggle CSV (mixed numeric/categorical, few thousand rows), EDA, handle missing values, encode categoricals, stratified train/test split | Week 1 Day 4–5 (mean/variance/cov intuition for EDA), Week 2 Day 5 (MI for feature selection, optional) |
| **Day 3 — Evaluate** | Train ≥2 `minisklearn` models, build confusion matrix + accuracy/precision/recall/F1 from scratch, inspect 5 misclassified examples | Week 2 Day 4 (cross-entropy = BCE), Week 3 Day 2 (precision/recall) |
| **Day 4 — Compare against real `sklearn`** | Same split through `sklearn`'s equivalent models; side-by-side table; debug one discrepancy fully | Entire month — this *is* the validation |
| **Day 5 — Write it up** | `README.md` + free blog post (Medium/dev.to/GitHub Pages) — hook, one surprising insight (MAP ↔ L2 is a strong candidate), honest numbers | Week 2 Day 2 |
| **Day 6 — Push to GitHub + rest** | Cleanup, accurate `requirements.txt` (`numpy`, `matplotlib`, optional `pandas`), clean commit history, public repo | — |

## Stretch / personalisation

- Expose `NaiveBayes` from `naive_bayes.py` as a first-class classifier too — your spam filter from Week 1 deserves the same `.fit/.predict` treatment.
- Add a one-command demo script: `python -m minisklearn.demo --dataset titanic --model random_forest`.
- Consider publishing the blog post's comparison table as the README's headline — real numbers beat adjectives.

## Done when

`import minisklearn` works from a separate script, `model.fit/predict` runs on a real Kaggle dataset, your `sklearn` comparison table is honest and explainable, and the repo + blog post are public. See Week 4 Day-by-day `Done when` lines for the per-day bar.
