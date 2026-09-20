# Day 16 — Decision Trees (ID3 Algorithm) (Week 3 Day 4, Month 2)

**Topic:** recursive ID3 decision trees from scratch — entropy, information
gain, depth control
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> All data today is small, categorical, and embedded inline (a 6-row hand
> example plus the classic 14-row play-tennis table) — see
> `coding_problems.md`. No external dataset is fetched. Labels are arbitrary
> hashable values (`"yes"`/`"no"`, `0`/`1`, …), not the `{0, 1}` / `{-1, +1}`
> of Days 13–15 — trees never do arithmetic on labels, they only count them.

## Why this matters

Days 13–15 all trained by numerical optimisation (gradient descent on
MSE/BCE/hinge). Today there is no gradient, no loss curve, no learning
rate: the model is built by recursively asking "which feature reduces my
uncertainty the most?" — and the uncertainty measure is Week 2 Day 3's
`entropy()`, reused almost verbatim. That is the direct payoff for math
you learned a week ago. Tomorrow's Random Forests are just bagged crowds
of today's trees, and Week 4's `tree.py` / `ensemble.py` ship today's
class behind an sklearn-like wrapper.

## Watch (in this order)

1. **StatQuest: Decision Trees** — StatQuest with Josh Starmer —
   https://www.youtube.com/watch?v=7VeUPuFGJHk
   Intuition-first pick: how a tree chooses splits from impurity (Gini and
   entropy / information gain) — the direct continuation of Week 2's
   entropy into today's `information_gain` task. Also covers why deep
   trees overfit, teeing up the deep-vs-shallow demo and tomorrow's
   Random Forests.
2. **Let's Write a Decision Tree Classifier from Scratch — Machine
   Learning Recipes #8** — Google for Developers (Josh Gordon) —
   https://www.youtube.com/watch?v=LDRbO9a6XPU
   Implementation-leaning pick: builds a small tree classifier from
   scratch in Python — the find-the-best-split loop plus recursive
   predict structure mirrors today's coding tasks almost one-to-one.
   Short; watch right before coding.

(Both verified live at generation time via YouTube metadata. No third,
long-lecture pick met the never-fabricate verification bar this session
— video search was down at generation time — so the gain-ratio and
threshold-split gaps below are worked by hand instead. `/for-read` can
supply a full lecture later.)

## Math — work through by hand

1. **Information gain formula.** Write
   `IG(S, feature) = H(S) − Σ (|S_v|/|S|)·H(S_v)` — parent entropy minus
   the size-weighted child entropies. Say in one sentence what `IG = 0`
   means (the feature tells you nothing) and what maximal IG means (the
   split purifies completely).
2. **The 6-row hand example.** Labels `[1,1,1,0,0,0]` so `H(S) = 1.0`
   bit. Feature A splits `[1,1,1] | [0,0,0]` (both pure, `H = 0`) so
   `IG(A) = 1.0`. Feature B splits `[1,1,0] | [1,0,0]` (each
   `H ≈ 0.918`) so `IG(B) ≈ 0.082`. ID3 picks A — quota: reproduce both
   numbers from memory by `/done`.
3. **Pure nodes stop.** A pure node has `H = 0`; any further split
   produces pure children with weighted entropy `0`, so its gain is
   exactly `0` — there is nothing left to gain, literally. State this
   without notes (practice question 2).
4. **Gain's bias, and today's scope.** One paragraph, in your own words:
   raw IG favours many-valued attributes (an ID column gets maximal gain
   by memorising one row per branch) — C4.5's gain-ratio fixes this by
   normalising; we stay plain-ID3 today and note the flaw rather than
   patch it. Related: continuous features need sorted-midpoint threshold
   search (CART); today's tree splits on exact categorical values only,
   by design — Tennis-scale data, not binned floats.

## Code — see `coding_problems.md`

`entropy` → `information_gain` → `majority_label` → `DecisionTreeID3`
(`fit`/`predict`/`to_dict` + `max_depth`/`min_samples_split`) →
`format_tree` (readable print of the learned splits).

## Practice questions (source plan)

1. Work through your hand example above fully — show the entropy and
   information gain calculations for at least 2 candidate features, and
   state which one ID3 selects.
2. Why does ID3 stop growing a branch when a node is "pure" (all same
   label)? What would the information gain of any further split be in
   that case?
3. What's a real risk of letting a decision tree grow without any depth
   limit or stopping condition (think about what you saw in Month 1's
   discussion of overfitting)?
4. Code check: add a `max_depth` parameter to your tree and train the
   same dataset with `max_depth=1` vs. unlimited depth — compare the
   resulting trees and their training accuracy.
