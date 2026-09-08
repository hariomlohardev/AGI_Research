# Week 2 Day 5 — Mutual Information

**Mode:** guided · **Language:** python · **Skill:** intermediate · **Global Day 11**

---

## Why this matters

Entropy measured surprise. Cross-entropy measured the cost of the wrong model. KL measured how far apart two distributions sit. Mutual information puts those tools to work on the most practical question so far: **how much does knowing one variable tell you about another?** That question is feature selection (which columns of your dataset are worth keeping?), and it's the conceptual ancestor of attention mechanisms you'll meet in Month 4/5 ("how much does this token tell you about that token?"). It also settles a score from Week 1 Day 5: you learned zero covariance doesn't mean independence — today you get the measure that actually *does* detect independence.

## What to learn today

### Math (on paper first, before code)

1. **MI as uncertainty reduction:**
   `I(X; Y) = H(X) − H(X|Y)` — your uncertainty about `X` minus your uncertainty about `X` *after* learning `Y`. Hand-compute both cases with binary variables:
   - Perfectly correlated (`X = Y`, fair coin): `H(X) = 1`, and once you know `Y` there is nothing left to be surprised about, `H(X|Y) = 0` → `I = 1` bit. Knowing `Y` removes *all* uncertainty.
   - Independent (two separate fair coins): knowing `Y` removes nothing, `H(X|Y) = H(X) = 1` → `I = 0`.
2. **MI as a KL divergence:**
   `I(X; Y) = D_KL(P(X,Y) || P(X)P(Y))` — how far the true joint distribution sits from the joint you'd see *if X and Y were independent*. Read it as Day 4's "extra bits from using the wrong distribution," where the wrong distribution is the independence assumption. Consequence: `I(X; Y) = 0` exactly when the joint equals the product of the marginals — i.e. exactly when `X` and `Y` are independent. (Compare Week 1 Day 5: covariance 0 never promised you this.)
3. **MI is never negative:**
   Falls straight out of Day 4 — KL divergences can't go below zero (Gibbs), and MI *is* a KL. Intuition: information can be useless, but it can't hurt. Learning `Y` never *increases* your uncertainty about `X`.
4. **MI vs covariance — the `Y = X²` case:**
   Take `X = −4..4` (symmetric) and `Y = X²`. Covariance is *exactly* 0 (positive and negative `X` cancel), yet `Y` plainly constrains `X` — and MI reports `≈ 2.28` bits, most of `H(X) ≈ 3.17`. Covariance only sees linear co-movement; MI sees any dependence. This is why today's ranking tool beats correlation for feature selection.

### Code

- `mutual_information(x, y, base=2.0)` — empirical MI between two discrete label sequences: count the joint and marginals straight from the paired samples, then the KL-shaped sum. Works with any hashable labels (ints, strings, binned values).
- `rank_features_by_mi(features, y, base=2.0)` — MI of every feature column against the label, sorted best-first. The miniature feature-selection tool.
- `__main__` experiments in `feature_selection.py`: informative-vs-noise ranking on synthetic data, plus the `Y = X²` covariance-0-vs-MI-high demo.

See `coding_problems.md` for exact signatures, the full `ValueError` contracts, and the hand values.

---

## Watch first (guided — at least two)

> Search YouTube for the exact phrases below if links don't open — don't rely on autoplay.

1. **Mutual Information, Clearly Explained!!! — StatQuest with Josh Starmer**
   - https://www.youtube.com/watch?v=eJIp_mgVLwE (~16 min)
   - Watch this one first. Builds MI up from the entropy intuition you already have from Days 3–4, and frames MI against R² — a fit measure that, like covariance, only sees numeric/linear relationships. Maps directly onto today's "MI catches what covariance misses" point.
2. **Information Theory Tutorial: Mutual Information — Complexity Explorer (Santa Fe Institute)**
   - https://www.youtube.com/watch?v=d7AUaut6hso (~13 min)
   - The rigorous complement to StatQuest. Part of a dedicated information-theory series — the right place to cement the `I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))` formulation and the independence connection (`I = 0` iff independent) after StatQuest gives you the intuition.
3. **An introduction to mutual information — Ben Lambert** (optional short refresher)
   - https://www.youtube.com/watch?v=U9h1xkNELvY (~8.5 min)
   - Equation-first and compact — reinforces the `I(X;Y) = H(X) − H(X|Y)` "uncertainty reduction" identity. Good recap watch after you've done the hand math, or the fallback if you're short on time.

---

## Practice questions (do on paper before coding)

1. Why is mutual information always ≥ 0? Connect this back to KL divergence's non-negativity from Day 4 — no new proof needed, just the right pointer.
2. If `I(X;Y) = 0`, does that mean `X` and `Y` are independent? Compare this carefully to the covariance question from Week 1 Day 5 — MI = 0 is a stronger, more general condition than covariance 0. Say exactly what each one does and doesn't guarantee.
3. Give a real-world pair of variables where covariance would be near 0 but mutual information would be high (hint: any symmetric nonlinear relationship, like `Y = X²` — find one outside physics textbooks, e.g. in business, biology, or everyday life).
4. Code check: construct the `Y = X²` example in code over symmetric integer `X`, compute both covariance and mutual information — confirm covariance is ~0 while MI correctly detects the strong nonlinear relationship.

---

## Done when

Your feature-selection tool correctly ranks a synthetic "informative" feature above a synthetic "noise" feature, and you can explain in one sentence why MI catches nonlinear relationships that covariance misses.

---

## Further reading

Not bundled — opt-in only. Run `/for-read` if you want supplementary articles/papers on MI estimation for continuous variables (binning vs k-NN estimators), MI-based feature selection (mRMR), or the MI–attention connection.
