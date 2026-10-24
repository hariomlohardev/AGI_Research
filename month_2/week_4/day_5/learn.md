# Day 23 — Write It Up: README + blog post (Week 4 Day 5, Month 2)

**Topic:** turning the `minisklearn` build into a portfolio artifact — a usable README and an honest public write-up
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> Writing day, not a learning day. The source plan explicitly says
> **"Watch first: none today."** There are therefore no videos below by
> design. Today's work is judgment and prose, not new math.

## Why this matters (from the plan)

Nobody can see the six days of derivations and debugging that got you
here — only what you publish. Today turns private work into portfolio:
a README a stranger can actually use, and a blog post that shows you
understood what happened, flattering numbers or not.

## The README (`month_2/minisklearn/README.md`)

A usable README answers four questions, in this order:

1. **What is this?** One paragraph: from-scratch classical ML library,
   which models, stdlib-only (plus `scikit-learn` for validation).
2. **What's implemented?** The model list with one line each
   (`LinearRegression`, `LogisticRegression`, `DecisionTreeClassifier`,
   `RandomForestClassifier`, `KMeans`, `metrics`).
3. **How do I install/import it?** Exact commands and the import lines
   (submodule path + top-level names — Day 19's front door).
4. **Show me it working.** A short runnable example on the Titanic data:
   load → split → fit → predict → one metric. Copy-pasteable, no
   missing steps.

Then the honesty table: your Day-22 side-by-side numbers against
`sklearn`, with the one-line verdict per model. Real numbers beat
adjectives — and an explained gap beats a hidden one.

## The blog post (`blog_post.md` in this day's folder, then published free)

Suggested structure (from the plan):

- **Hook:** the single most interesting thing you learned — the
  MAP↔L2 (regularization) connection is the strong candidate, but pick
  the one that's genuinely yours.
- **What you built:** the package, the dataset, the pipeline — brief,
  concrete.
- **The surprise:** one real insight you didn't expect (a derivation
  that clicked, a default that explained a gap, a metric that lied).
- **The numbers:** the `sklearn` comparison table with actual values,
  including where you diverged and why.
- Publish free (Medium / dev.to / GitHub Pages / personal site), then
  link it from the README.

## Where the numbers come from

- Dataset: the vendored Day-20 file
  (`month_2/week_4/day_2/code/data/titanic.csv`) — 889 rows, 7 features,
  ~38% positive.
- Split and metrics: Day 21's pipeline (`test_ratio=0.2, seed=0`) and
  your `minisklearn.metrics`.
- Comparison values: your Day-22 run — same split, same ruler. Do not
  invent or "clean up" numbers; report what the table said.

## Practice questions (source plan)

1. If a stranger with no context read only your README, could they get
   your package running and call one model on their own data in under 2
   minutes? If not, what's missing?
2. What's the single most interesting thing you learned this month that
   you'd genuinely want to explain to someone else? (This should become
   the hook/opening of your blog post.)

## Done when (from the source plan)

Your README is genuinely usable by someone who's never seen your code,
and your blog post is published somewhere public and free.
