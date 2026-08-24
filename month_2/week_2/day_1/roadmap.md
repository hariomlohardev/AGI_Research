# Roadmap — Week 2 Day 1: Maximum Likelihood Estimation (MLE)

**Current:** Month 2, Week 2, Day 1 (global Day 7) — *MLE: the bridge from probability to training*
**Mode:** guided · **Language:** python · **Skill:** intermediate

---

## Today's build

1. **Watch the videos** in `learn.md` first (start with StatQuest MLE) — 30-45 min. Take notes on why we log the likelihood and how the Gaussian MLE lands on the sample mean.
2. **Math on paper (30 min, do not skip):**
   - Write `L(mu,sigma)=prod_i N(x_i|mu,sigma)` for n i.i.d. Gaussian samples.
   - Take `log` → `LL = sum_i log N(...)`; explain in one sentence why (numerical stability + sum is easier to differentiate than product).
   - Differentiate LL w.r.t `mu`, set to 0, solve → `mu_hat = mean(x)`. Do the same for `sigma^2` → biased variance. Keep the sheet — you need it for the code and the quiz in `/done`.
3. **Code — `log_likelihood`** (see `coding_problems.md` Problem 1): implement Gaussian PDF + sum of log-PDFs. Verify against your hand calculation on `[0,1,2]`.
4. **Code — `mle_fit_gaussian` two ways** (Problem 2): closed-form vs grid-search (or Month-1 GD). Confirm they agree on synthetic data with known truth.
5. **Experiment** (Problem 3): `n=5` vs `n=5000` — print both fits and write the LLN comment.
6. **Practice questions** (in `learn.md`) on paper before you open the editor for step 5 — if you can't answer without code, you don't know the math yet.
7. **Commit & run `/done`:** tests must pass before the quiz. `/done` will quiz you on MLE vs MAP, log-likelihood, and the sample-size effect.

---

## How this ties to the final project (`minisklearn`)

This day's `log_likelihood` and MLE fitter are the conceptual engine behind **every** learner Week 3 builds: minimising cross-entropy/MSE *is* maximising likelihood. Week 3 Day 1–2's Linear/Logistic Regression will maximise the same LL you implement today. The MAP regularisation insight from tomorrow (Week 2 Day 2) then slots in as "MLE + prior = weight decay." Today's closed-form vs numerical comparison is also your first `sklearn` validation rehearsal — Week 4 Day 4 does the same comparison on real data.

---

## Files in this day

- `learn.md` — what to learn + verified videos
- `coding_problems.md` — problem statements (intermediate scale)
- `code/log_likelihood.py` — stub for `log_likelihood`
- `code/mle_fit_gaussian.py` — stubs for closed-form + grid fitters
- `code/tests/` — pytest suite (must pass before `/done`)
- `further_reading.md` — appended only if you run `/for-read` (opt-in)
