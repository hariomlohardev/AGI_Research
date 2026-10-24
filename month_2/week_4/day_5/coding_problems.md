# Day 23 — Write It Up: coding problems

Scale: **intermediate** (source plan as-is). No new code today — the
deliverables are two markdown files, and the tests check their shape,
not their literary merit.

> The governing constraint is honesty: every number in both documents
> must trace back to your actual Day-21/Day-22 runs. A documented gap
> is a result; a missing gap is a hole.

---

## Problem 1 — README for `minisklearn` (`month_2/minisklearn/README.md`)

Write the package README at `month_2/minisklearn/README.md` (package
level, not inside this day's folder — it ships with the library).
It must contain:

1. A top-level `#` title naming `minisklearn`.
2. A **what-is-this** paragraph (from-scratch, which models).
3. An **installation/import** section with real commands and at least
   one `import minisklearn` line plus a `.fit(` call in a fenced code
   block.
4. A **usage example** a stranger can follow: load → split → fit →
   predict → one metric.
5. The **model list** — at least three of `LinearRegression`,
   `LogisticRegression`, `DecisionTreeClassifier`,
   `RandomForestClassifier`, `KMeans` named explicitly.
6. Your **Day-22 comparison table** (or its headline numbers) with the
   per-model verdict.

## Problem 2 — Blog post (`blog_post.md` in this day's folder)

Write `month_2/week_4/day_5/blog_post.md`, then publish it free
(Medium / dev.to / GitHub Pages / personal site) and link it from the
README. It must contain:

1. A title/`#` heading that works as the hook (non-trivial — at least
   20 characters of actual headline, not "My blog post").
2. At least 300 words of real content.
3. Your genuine insight — the MAP↔L2 connection is the suggested
   candidate, but the test only checks that the post names `sklearn`
   and reports at least one numeric result (a digit somewhere in the
   comparison section).
4. The honest numbers: both implementations' results and at least one
   explained divergence.

## Problem 3 — Self-review (manual, the two source-plan questions)

1. Hand your README to the "stranger test": read it as someone who's
   never seen your code. Can they install, import, and run one model in
   under 2 minutes? List every missing step you find and fix it.
2. Read your post's opening back: is the hook the thing you'd
   genuinely want to explain to someone? If the opening could belong to
   any ML post, rewrite it until it could only be yours.

### Running tests

```bash
cd month_2/week_4/day_5 && py -m pytest code/tests/ -v
```

Tests check the deliverables' shape only: README exists with the
required sections, imports, model names, and a code block; the post
exists with a real headline, 300+ words, an `sklearn` mention, and at
least one reported number. Taste, clarity, and honesty are yours —
and `/done`'s quiz will ask about them.

### Done when (from the source plan)

Your README is genuinely usable by someone who's never seen your code,
and your blog post is published somewhere public and free.
