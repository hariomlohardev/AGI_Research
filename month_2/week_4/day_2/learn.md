# Day 20 — Apply to a Real Dataset (Week 4 Day 2, Month 2)

**Topic:** EDA on real tabular data (missingness, class balance,
distributions) → mean-imputation + categorical encoding → stratified
train/test split, all hand-rolled in stdlib
**Mode:** guided · **Language:** python · **Skill level:** intermediate

> Build day, no derivations. The dataset is vendored at
> `code/data/titanic.csv` — 891 rows, 12 columns
> (`PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked`),
> verified at generation time: **Age 177 missing, Embarked 2 missing**,
> target **342 survived / 549 not** (~38% positive), `Sex` female/male,
> `Pclass` 1/2/3. `csv` + `math` + `random` only — no pandas. Days 2–6
> share this file; keep your pipeline re-runnable.

## Why this matters

Every model so far has met only tiny synthetic data that you designed to
behave. Today it meets the wild: blanks where numbers should be, words
where numbers must go, and a 62/38 class split that will punish anyone
who quotes accuracy without thinking (Day 3 grades exactly that). The
four verbs — **look, fill, encode, split** — are the unglamorous 80% of
every real project, and the arrays you finish with (`X_train, X_test,
y_train, y_test`) are what Days 3–6 operate on. Get them honestly clean
today or every number next week is suspect.

## Watch (in this order)

All four use pandas on screen — that's fine and expected. You are not
watching for *how* (you hand-roll every step in `csv`/`math`/`random`
per `coding_problems.md`); you are watching for **what to check and
why**: which gaps to hunt, which distributions to distrust, what order
to do things in. Steal the checklist, not the code.

1. **Exploratory Data Analysis Tutorial | Basics of EDA with Python** — Great Learning — https://www.youtube.com/watch?v=05w2322DHUI
   The big one (~4.5 hours): end-to-end EDA in Python — missing-value
   checks, distributions, feature inspection. Primary watch, picked for
   your stated big-video preference. Mirror each section on Titanic as
   you go: when they check missingness, you run `missing_counts`; when
   they plot a distribution, you run `numeric_summary`.

2. **Exploratory Data Analysis (EDA) Using Python | Exploratory Data Analysis Project** — Simplilearn — https://www.youtube.com/watch?v=clblk_NwEU8
   Framed as an EDA *project* on real data rather than isolated
   concepts — the closest mirror to today's topic. Watch for the
   shape of the work: explore first, find the gaps, prepare for
   modeling last. Surfaced independently in two separate searches.

3. **Data Analysis with Python Course - Numpy, Pandas, Data Visualization** — freeCodeCamp.org — https://www.youtube.com/watch?v=GPVsHOlRBBI
   Full-length course (several hours) on the whole pipeline. Backup,
   not required: reach for it if a step's *concept* feels thin (e.g.
   why a distribution check matters) rather than its mechanics.

4. **Exploratory Data Analysis in Pandas | Python Pandas Tutorials** — Alex The Analyst — https://www.youtube.com/watch?v=Liv6eeb1VfE
   The focused companion: distributions, grouping, sanity checks in one
   tight lesson — maps directly onto "check class balance and feature
   distributions." Watch if the big videos blur together and you want
   one concrete technique demo.

(All four verified live at generation time. Deliberately long picks
after your 4/10 verdict and the big-video ask — no 3-minute shorts
today.)

## Math — none today

Build day per the source plan. The one piece of arithmetic that matters
— why a 62/38 split makes accuracy lie — is Practice Q1, and Day 3
collects on it.

## Code — see `coding_problems.md`

EDA (`eda.py`: load → missing report → balance → summaries) → impute +
encode (`encode.py`: parse → mean-fill → label/one-hot, your documented
feature-selection call) → stratified split by hand (`split.py`, seeded
local RNG) → clean `X_train, X_test, y_train, y_test`, no missing.

## Practice questions (source plan)

1. Why does class imbalance in your dataset matter for how you interpret accuracy later — connect this back to Week 3 Day 2's precision/recall discussion?
2. If a numeric feature has some missing values, what are two different strategies you could use to fill them, and what's a downside of each?
3. Why is stratified splitting (preserving class proportions in both train and test sets) important when a dataset is imbalanced?
