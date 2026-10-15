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

1. **Exploratory Data Analysis in Pandas | Python Pandas Tutorials** — Alex The Analyst — https://www.youtube.com/watch?v=Liv6eeb1VfE (~32 min)
   The core pick: a focused, right-sized EDA walkthrough on a real
   dataset (distributions, groupbys, sanity checks). Mirror each step
   on Titanic as you go: when they check a distribution, you run
   `numeric_summary`; when they count categories, you run
   `class_balance`.

2. **A Gentle Introduction to Pandas Data Analysis (on Kaggle)** — Rob Mulla — https://www.youtube.com/watch?v=_Eb0utIRdkw (~39 min)
   Second perspective, closest to the plan's Kaggle angle — taught on
   Kaggle on a real dataset by a Kaggle Grandmaster. Watch for EDA
   mechanics: missing-value hunts, value counts (your class balance),
   distributions.

3. **Data Cleaning in Pandas | Python Pandas Tutorials** — Alex The Analyst — https://www.youtube.com/watch?v=bDhvCp3_lYw (~39 min)
   The prep half: finding and handling missing values — feeds directly
   into your mean-imputation step. Same series as #1, so no
   context-switching cost.

4. **Machine Learning Tutorial Python - 6: Dummy Variables & One Hot Encoding** — codebasics — https://www.youtube.com/watch?v=9yl6-HEY7_s (~22 min)
   The encoding half: dummy variables and one-hot encoding, short and
   tightly scoped — watch right before writing `label_encode`/`one_hot`.

(All four verified live at generation time. Right-sized picks per your
correction — long videos reserved for complex topics; nothing
course-length today. Honest gap: none demos a hand-written stratified
split, so that piece rests on your coding tasks — fitting, since the
day specifies stdlib-only splitting anyway.)

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
